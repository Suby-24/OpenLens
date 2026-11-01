"""Check that automatic suggestions never masquerade as human ground truth."""

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from openlens.review_predictions import read_rows, review, save_rows


class ReviewTests(TestCase):
    def test_human_corrections_are_preserved_and_suggestions_are_explicit(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source, labels, output = (root / name for name in ("source.csv", "labels.csv", "out.csv"))
            fields = ["video_id", "title", "category_label", "predicted_label", "confidence", "review_status"]
            rows = [dict(zip(fields, row)) for row in [
                ("a", "오버워치 겐지", "", "Sports", "0.5", "review_needed"),
                ("b", "음악 가사", "", "Entertainment", "0.4", "review_needed"),
                ("c", "https://www.youtube.com/watch?v=c", "", "Sports", "0.4", "review_needed"),
            ]]
            save_rows(source, fields, rows)
            save_rows(labels, ["video_id", "category_label"], [{"video_id": "b", "category_label": "Lifestyle & Culture"}])
            review(source, labels, output)
            _, result = read_rows(output)
            self.assertEqual(result[0]["predicted_label"], "Entertainment")
            self.assertEqual(result[0]["category_label"], "")
            self.assertEqual(result[0]["confidence"], "")
            self.assertEqual(result[0]["original_confidence"], "0.5")
            self.assertEqual(result[0]["review_status"], "ai_suggested")
            self.assertEqual(result[1]["category_label"], "Lifestyle & Culture")
            self.assertEqual(result[1]["review_status"], "labeled")
            self.assertEqual(result[2]["review_status"], "review_needed")
