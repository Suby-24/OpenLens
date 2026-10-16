import sys
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

import numpy as np

from openlens.cluster import EMBEDDING_MODEL, cluster_watch_events, embed_titles
from openlens.takeout import WatchEvent


class ClusterTests(TestCase):
    def setUp(self) -> None:
        def fake_embeddings(titles: list[str]) -> np.ndarray:
            vectors = {
                "운동 근력 운동": [1.0, 0.0],
                "운동 달리기 훈련": [0.9, 0.1],
                "경제 금융 시장": [-1.0, 0.0],
                "경제 주식 시장": [-0.9, 0.1],
            }
            return np.array(
                [vectors[title] for title in titles]
            )

        embed_patch = patch("openlens.cluster.embed_titles", side_effect=fake_embeddings)
        self.mock_embed = embed_patch.start()
        self.addCleanup(embed_patch.stop)
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
        self.assertEqual(self.mock_embed.call_count, 2)
        self.assertTrue(all(len(call.args[0]) == 4 for call in self.mock_embed.call_args_list))

    def test_old_watches_are_excluded_from_counts_and_vectors(self) -> None:
        old = WatchEvent("old", "오래된 여행 영상", "https://example.org/old", self.now - timedelta(days=91))
        result = cluster_watch_events(self.events + [old], k=2, days=90)

        self.assertEqual(result.total_watch_events, 4)
        self.assertEqual(len(result.videos), 4)
        self.assertNotIn("old", {video.video_id for video in result.videos})
        self.assertEqual(len(self.mock_embed.call_args.args[0]), 4)

    def test_invalid_k_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "k must be at least 2"):
            cluster_watch_events(self.events, k=1)
        with self.assertRaisesRegex(ValueError, "distinct video titles"):
            cluster_watch_events(self.events, k=5)

    def test_embedding_uses_clustering_prompt_once_per_unique_title(self) -> None:
        calls = []

        class FakeSentenceTransformer:
            def __init__(self, model_name: str) -> None:
                calls.append(model_name)

            def encode(self, titles: list[str], **options: object) -> np.ndarray:
                calls.append((titles, options))
                return np.array([[1.0, 0.0], [0.0, 1.0]])

        with patch.dict(
            sys.modules,
            {"sentence_transformers": SimpleNamespace(SentenceTransformer=FakeSentenceTransformer)},
        ):
            vectors = embed_titles(["운동 식단", "경제 뉴스"])

        self.assertEqual(vectors.shape, (2, 2))
        self.assertEqual(calls[0], EMBEDDING_MODEL)
        self.assertEqual(calls[1][0], ["운동 식단", "경제 뉴스"])
        self.assertEqual(calls[1][1]["prompt_name"], "Clustering")
        self.assertTrue(calls[1][1]["normalize_embeddings"])
