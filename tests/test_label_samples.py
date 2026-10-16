from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

import numpy as np

from openlens.cluster import ClusteredVideo, ClusterSummary, ClusteringResult
from openlens.label_samples import select_label_samples, write_label_samples


class LabelSamplesTests(TestCase):
    def test_nearest_items_are_selected_and_labels_remain_blank(self) -> None:
        now = datetime(2026, 10, 16, tzinfo=timezone.utc)
        result = ClusteringResult(
            window_start=now,
            window_end=now,
            total_watch_events=4,
            videos=(
                ClusteredVideo("a", "첫 영상", 0, 1),
                ClusteredVideo("b", "둘째 영상", 0, 2),
                ClusteredVideo("c", "셋째 영상", 0, 1),
            ),
            clusters=(ClusterSummary(0, 3, 4, ("첫 영상",)),),
            embedding_vectors=np.array([[0.0], [1.0], [10.0]]),
        )
        samples = select_label_samples(result, per_cluster=2)
        self.assertEqual([sample.video_id for sample in samples], ["b", "a"])
        with TemporaryDirectory() as directory:
            path = Path(directory) / "samples.csv"
            write_label_samples(samples, path)
            content = path.read_text(encoding="utf-8-sig")
            self.assertIn("category_label,review_note", content)
            self.assertIn("b,둘째 영상,2", content)
            with self.assertRaises(FileExistsError):
                write_label_samples(samples, path)
