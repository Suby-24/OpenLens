from datetime import timedelta
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from zipfile import ZipFile

from openlens.takeout import (
    merge_watch_events,
    read_chrome_navigation_events,
    read_watch_events,
    write_watch_events_csv,
)


class TakeoutParserTests(TestCase):
    def test_korean_history_and_merged_csv(self) -> None:
        html = '''<div class="outer-cell"><div class="content-cell mdl-typography--body-1">
        시청함 <a href="https://www.youtube.com/watch?v=abc123">운동 영상</a><br>
        2026. 10. 16. 오전 11:05:58 GMT+04:00<br></div></div>'''
        with TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first.zip"
            second = root / "second.zip"
            for path in (first, second):
                with ZipFile(path, "w") as archive:
                    archive.writestr("Takeout/YouTube 및 YouTube Music/시청 기록/시청 기록.html", html)
            merged = merge_watch_events(first, second)
            output = write_watch_events_csv(merged, root / "merged.csv")
            loaded = read_watch_events(output)
        self.assertEqual(len(merged), 1)
        self.assertEqual(loaded, merged)
        self.assertEqual(loaded[0].watched_at.utcoffset(), timedelta(hours=4))

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

    def test_chrome_session_navigation_from_folder_is_sorted(self) -> None:
        history = {
            "Browser History": [],
            "Typed Url": [],
            "Session": [
                {"tab": {"navigation": [
                    {"virtual_url": "https://example.org/later", "title": "  Later  page ",
                     "timestamp_msec": 1_577_836_801_000},
                    {"virtual_url": "https://example.org/earlier", "title": "Earlier",
                     "timestamp_msec": 1_577_836_800_000},
                    {"virtual_url": "https://example.org/invalid", "timestamp_msec": None},
                ]}},
            ],
        }
        with TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "History.json").write_text(json.dumps(history), encoding="utf-8")
            events = read_chrome_navigation_events(path)

        self.assertEqual(len(events), 2)
        self.assertEqual(events[0].url, "https://example.org/earlier")
        self.assertEqual(events[1].title, "Later page")
        self.assertEqual(events[0].visited_at.utcoffset(), timedelta(0))
        self.assertEqual(events[0].source, "session_navigation")

    def test_chrome_history_can_be_read_from_zip(self) -> None:
        history = {"Session": [{"tab": {"navigation": [
            {"virtual_url": "https://example.org/", "title": "Example",
             "timestamp_msec": 1_577_836_800_000},
        ]}}]}
        with TemporaryDirectory() as directory:
            path = Path(directory) / "takeout.zip"
            with ZipFile(path, "w") as archive:
                archive.writestr("Takeout/Chrome/History.json", json.dumps(history))
            events = read_chrome_navigation_events(path)
        self.assertEqual(len(events), 1)

    def test_chrome_history_requires_session_list(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "History.json"
            path.write_text('{"Session": {}}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Session list"):
                read_chrome_navigation_events(path)
