from datetime import timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from zipfile import ZipFile

from openlens.takeout import read_watch_events


class TakeoutParserTests(TestCase):
    def test_reads_watch_event_and_ignores_other_links(self) -> None:
        html = """<div class="outer-cell">
          <div class="content-cell mdl-typography--body-1">
            Watched <a href="https://www.youtube.com/watch?v=abc123">운동과 수학</a><br>
            <a href="https://www.youtube.com/channel/example">채널</a><br>
            Jan 1, 2020, 9:00:00\u202fAM GMT+00:00<br>
          </div>
        </div>"""
        with TemporaryDirectory() as directory:
            path = Path(directory) / "takeout.zip"
            with ZipFile(path, "w") as archive:
                archive.writestr("Takeout/YouTube and YouTube Music/history/watch-history.html", html)
            events = read_watch_events(path)

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].video_id, "abc123")
        self.assertEqual(events[0].title, "운동과 수학")
        self.assertEqual(events[0].watched_at.utcoffset(), timedelta(0))

    def test_missing_history_is_reported(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "empty.zip"
            with ZipFile(path, "w") as archive:
                archive.writestr("other.txt", "")
            with self.assertRaisesRegex(ValueError, "Expected one watch-history.html"):
                read_watch_events(path)
