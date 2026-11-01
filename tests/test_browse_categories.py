"""Tests for the offline category directory."""

import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from openlens.browse_categories import (build_directory, write_report,
                                        register_latest_results, report_after_training)


class CategoryDirectoryTests(TestCase):
    def test_training_refreshes_one_report_and_resets_stale_lower_results(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            upper, lower = root / "upper.csv", root / "lower.csv"
            upper.write_text(
                "video_id,title,watch_count,category_label,predicted_label,review_status\n"
                "a,Song,2,,Entertainment,review_needed\n", encoding="utf-8")
            lower.write_text(
                "video_id,parent_category,primary_label,predicted_primary_label,review_status\n"
                "a,Entertainment,,음악,review_needed\n", encoding="utf-8")
            report = report_after_training(upper, project_root=root)
            self.assertIsNotNone(report)
            registry = root / "data/predictions/latest-results.json"
            self.assertEqual(json.loads(registry.read_text())["subcategories"], {})
            self.assertEqual(report_after_training(lower, parent="Entertainment", project_root=root), report)
            self.assertIn("<summary>음악", report.read_text())
            self.assertIn("Entertainment", json.loads(registry.read_text())["subcategories"])

            # Starting a new upper run must not display older lower assignments.
            report_after_training(upper, project_root=root)
            self.assertEqual(json.loads(registry.read_text())["subcategories"], {})
            self.assertNotIn("<summary>음악", report.read_text())

            old_report, old_registry = report.read_bytes(), registry.read_bytes()
            lower.write_text(
                "video_id,parent_category,primary_label,predicted_primary_label,review_status\n"
                "other,Entertainment,,음악,review_needed\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                register_latest_results(upper, [lower], project_root=root)
            self.assertEqual(report.read_bytes(), old_report)
            self.assertEqual(registry.read_bytes(), old_registry)

    def test_hierarchy_escapes_titles_and_keeps_unclassified_videos(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            upper = root / "upper.csv"
            lower = root / "lower.csv"
            with upper.open("w", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerow(("video_id", "title", "watch_count", "category_label",
                                 "predicted_label", "review_status"))
                writer.writerow(("music-1", "<script>bad</script>", 2, "Entertainment",
                                 "Entertainment", "labeled"))
                writer.writerow(("science-1", "Science title", 1, "", "Tech & Science",
                                 "review_needed"))
            with lower.open("w", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerow(("video_id", "parent_category", "primary_label",
                                 "predicted_primary_label", "review_status", "secondary_labels"))
                writer.writerow(("music-1", "Entertainment", "", "음악", "suggested", ""))
            tree, skipped = build_directory(upper, [lower])
            self.assertEqual(skipped, 0)
            self.assertEqual(len(tree["Entertainment"]["음악"]), 1)
            self.assertEqual(len(tree["Tech & Science"]["하위 미분류"]), 1)
            output = root / "report.html"
            write_report(tree, output)
            html = output.read_text(encoding="utf-8")
            self.assertIn("&lt;script&gt;bad&lt;/script&gt;", html)
            self.assertNotIn("<script>bad</script>", html)
            with self.assertRaises(FileExistsError):
                write_report(tree, output)
