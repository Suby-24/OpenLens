import csv
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

import joblib
import numpy as np

from openlens.subcategory import classify_parent, sample_parent, select_diverse_samples


class SubcategoryTests(TestCase):
    def test_samples_cover_center_to_edge(self) -> None:
        vectors = np.arange(40, dtype=float).reshape(-1, 1)
        selected = select_diverse_samples(list(range(40)), vectors, np.arange(40),
                                          [f"video-{i}" for i in range(40)], 20)
        self.assertEqual(len(selected), 20)
        self.assertEqual(len(set(selected)), 20)
        self.assertEqual([sum(start <= index < start + 10 for index in selected)
                          for start in (0, 10, 20, 30)], [5, 5, 5, 5])

    def test_sample_then_train_from_manual_primary_labels(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "parent-predictions.csv"
            with source.open("w", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerow(("video_id", "title", "watch_count", "category_label",
                                 "predicted_label", "confidence", "review_status"))
                for number in range(6):
                    writer.writerow((f"video-{number}", f"title {number}", 1, "",
                                     "Entertainment", "0.4", "review_needed"))
                writer.writerow(("other", "science title", 1, "", "Tech & Science",
                                 "0.9", "suggested"))

            vectors = np.array([[0.0, 0.0], [0.1, 0.1], [0.2, 0.0],
                                [5.0, 5.0], [5.1, 5.0], [5.0, 5.1]])
            with patch("openlens.subcategory.embed_titles", return_value=vectors):
                clusters, labels = sample_parent(source, "Entertainment", root, k=2,
                                                 per_cluster=2)

            with clusters.open(newline="", encoding="utf-8-sig") as file:
                all_rows = list(csv.DictReader(file))
            self.assertEqual(len(all_rows), 6)
            self.assertTrue(all(row["parent_review_status"] == "review_needed"
                                for row in all_rows))

            with labels.open(newline="", encoding="utf-8-sig") as file:
                sample_rows = list(csv.DictReader(file))
            self.assertEqual(len(sample_rows), 4)
            for row in sample_rows:
                row["primary_label"] = "음악" if row["cluster_id"] == "0" else "영화"
            sample_rows[0]["secondary_labels"] = "아이돌"
            with labels.open("w", newline="", encoding="utf-8-sig") as file:
                writer = csv.DictWriter(file, fieldnames=sample_rows[0])
                writer.writeheader()
                writer.writerows(sample_rows)

            output = root / "subcategory-predictions.csv"
            model_path = root / "subcategory-model.joblib"
            with patch("openlens.subcategory.embed_titles", return_value=vectors):
                reviewed, total = classify_parent(clusters, labels, "Entertainment",
                                                  output, model_path)
            self.assertEqual((reviewed, total), (4, 6))
            with output.open(newline="", encoding="utf-8-sig") as file:
                predictions = list(csv.DictReader(file))
            self.assertEqual(len(predictions), 6)
            self.assertEqual(sum(row["review_status"] == "labeled" for row in predictions), 4)
            self.assertEqual(sum(bool(row["secondary_labels"]) for row in predictions), 1)
            self.assertEqual(joblib.load(model_path)["parent_category"], "Entertainment")
