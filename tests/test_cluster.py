from datetime import datetime, timedelta, timezone
from unittest import TestCase

from openlens.cluster import cluster_watch_events
from openlens.takeout import WatchEvent


class ClusterTests(TestCase):
    def setUp(self) -> None:
        self.now = datetime(2020, 1, 10, tzinfo=timezone.utc)
        titles = {
            "a": "운동 근력 운동",
            "b": "운동 달리기 훈련",
            "c": "경제 금융 시장",
            "d": "경제 주식 시장",
        }
        self.events = [
            WatchEvent(video_id, title, f"https://example.org/{video_id}", self.now)
            for video_id, title in titles.items()
        ]

    def test_repeat_watches_change_consumption_not_clusters(self) -> None:
        baseline = cluster_watch_events(self.events, k=2)
        repeated = cluster_watch_events(self.events + [self.events[0]] * 4, k=2)

        self.assertEqual(
            [(video.video_id, video.cluster_id) for video in baseline.videos],
            [(video.video_id, video.cluster_id) for video in repeated.videos],
        )
        self.assertEqual(len(repeated.videos), 4)
        self.assertEqual(repeated.total_watch_events, 8)
        self.assertEqual(sum(cluster.watch_events for cluster in repeated.clusters), 8)
        self.assertEqual(next(video.watch_count for video in repeated.videos if video.video_id == "a"), 5)

    def test_old_watches_are_excluded_from_counts_and_vectors(self) -> None:
        old = WatchEvent("old", "오래된 여행 영상", "https://example.org/old", self.now - timedelta(days=91))
        result = cluster_watch_events(self.events + [old], k=2, days=90)

        self.assertEqual(result.total_watch_events, 4)
        self.assertEqual(len(result.videos), 4)
        self.assertNotIn("old", {video.video_id for video in result.videos})

    def test_invalid_k_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "k must be at least 2"):
            cluster_watch_events(self.events, k=1)
        with self.assertRaisesRegex(ValueError, "distinct video titles"):
            cluster_watch_events(self.events, k=5)
