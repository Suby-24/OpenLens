"""Tests for the offline category directory."""

import csv
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from openlens.browse_categories import build_directory, write_report


class CategoryDirectoryTests(TestCase):
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
