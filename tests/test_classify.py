import csv
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

import numpy as np

from openlens.classify import classify, read_external_labels, read_training_data
from openlens.takeout import WatchEvent, write_watch_events_csv
from openlens.training_data import build_training_data


class ExternalTrainingTests(TestCase):
    def test_external_titles_train_but_do_not_become_profile_rows(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = write_watch_events_csv([
                WatchEvent("personal-a", "음악 공연", "https://example.org/a", datetime(2026, 1, 1, tzinfo=timezone.utc)),
                WatchEvent("personal-b", "축구 경기", "https://example.org/b", datetime(2026, 1, 1, tzinfo=timezone.utc)),
            ], root / "watch.csv")
            labels = root / "labels.csv"
            with labels.open("w", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerow(("video_id", "title", "category_label"))
                writer.writerow(("personal-a", "음악 공연", "Entertainment"))
                writer.writerow(("personal-b", "축구 경기", "Sports"))
            external = root / "external.csv"
            with external.open("w", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerow(("video_id", "title", "category_label", "label_status"))
                writer.writerow(("external-a", "기후 변화", "Environment", "proposed"))
            self.assertEqual(read_external_labels(external), [])
            output = root / "predictions.csv"
            vectors = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]])
            with patch("openlens.classify.embed_titles", return_value=vectors):
                classify(source, labels, output, external_labels=external, allow_proposed_external=True)
            with output.open(newline="", encoding="utf-8-sig") as file:
                rows = list(csv.DictReader(file))
            self.assertEqual({row["video_id"] for row in rows}, {"personal-a", "personal-b"})

            combined = root / "training-with-external.csv"
            build_training_data(labels, external, combined)
            personal, external_rows = read_training_data(combined, allow_proposed=True)
            self.assertEqual(len(personal), 2)
            self.assertEqual(len(external_rows), 1)
            combined_output = root / "predictions-combined.csv"
            with patch("openlens.classify.embed_titles", return_value=vectors):
                classify(source, None, combined_output, training_data=combined,
                         allow_proposed_external=True)
            with combined_output.open(newline="", encoding="utf-8-sig") as file:
                combined_rows = list(csv.DictReader(file))
            self.assertEqual({row["video_id"] for row in combined_rows}, {"personal-a", "personal-b"})
