from datetime import datetime, timezone
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

import numpy as np

from openlens.cluster import cluster_watch_events
from openlens.takeout import WatchEvent
from openlens.visualize import project_titles, write_cluster_report


class VisualizationTests(TestCase):
    def test_offline_report_uses_synthetic_titles(self) -> None:
        watched_at = datetime(2020, 1, 1, tzinfo=timezone.utc)
        titles = ["운동 근력", "운동 달리기", "경제 금융", "경제 시장"]
        events = [
            WatchEvent(str(index), title, f"https://example.org/{index}", watched_at)
            for index, title in enumerate(titles)
        ]
        events.append(events[0])
        with patch("openlens.cluster.embed_titles") as mock_embed:
            mock_embed.return_value = np.array(
                [[1.0, 0.0], [0.9, 0.1], [-1.0, 0.0], [-0.9, 0.1]]
            )
            result = cluster_watch_events(events, k=2)

        self.assertEqual(len(project_titles(result)), 4)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "report.html"
            write_cluster_report(result, path)
            html = path.read_text(encoding="utf-8")

        self.assertIn("OpenLens 클러스터링 결과", html)
        self.assertIn("시청 횟수", html)
        # Plotly serializes non-ASCII strings as JSON escape sequences.
        encoded_title = json.dumps("운동 근력", ensure_ascii=True)[1:-1]
        self.assertTrue(encoded_title in html, "synthetic title missing from report")
        # The Plotly bundle may contain CDN URLs as text, but no external script is loaded.
        self.assertTrue('<script src="https://' not in html, "external script in report")
