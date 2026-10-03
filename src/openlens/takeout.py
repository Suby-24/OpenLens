"""Read YouTube watch history from a user-provided Google Takeout archive."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from html.parser import HTMLParser
from pathlib import Path
import re
from urllib.parse import parse_qs, urlparse
from zipfile import ZipFile


WATCH_HISTORY_SUFFIX = "YouTube and YouTube Music/history/watch-history.html"
DATE_PATTERN = re.compile(r"[A-Z][a-z]{2} \d{1,2}, \d{4}, \d{1,2}:\d{2}:\d{2}\s*[AP]M GMT[+-]\d{2}:\d{2}")


@dataclass(frozen=True)
class WatchEvent:
    video_id: str
    title: str
    url: str
    watched_at: datetime


def _video_id(url: str) -> str | None:
    parsed = urlparse(url)
    host = parsed.netloc.lower().removeprefix("www.")
    if host == "youtube.com" and parsed.path == "/watch":
        return parse_qs(parsed.query).get("v", [None])[0]
    if host == "youtube.com" and parsed.path.startswith("/shorts/"):
        return parsed.path.split("/")[2]
    if host == "youtu.be":
        return parsed.path.strip("/").split("/")[0]
    return None


class _WatchHistoryHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.depth = 0
        self.outer_depth: int | None = None
        self.content_depth: int | None = None
        self.link_depth: int | None = None
        self.link_url: str | None = None
        self.link_text: list[str] = []
        self.parts: list[str] = []
        self.events: list[WatchEvent] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_dict = dict(attrs)
        if tag == "div":
            self.depth += 1
            classes = set((attrs_dict.get("class") or "").split())
            if "outer-cell" in classes:
                self.outer_depth = self.depth
                self.content_depth = None
                self.link_url = None
                self.link_text = []
                self.parts = []
            elif self.outer_depth is not None and "content-cell" in classes and "mdl-typography--body-1" in classes and "mdl-typography--text-right" not in classes:
                self.content_depth = self.depth
        elif tag == "a" and self.content_depth is not None and self.link_url is None:
            self.link_url = attrs_dict.get("href")
            self.link_depth = self.depth
        elif tag == "br" and self.content_depth is not None:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag == "a":
            self.link_depth = None
        elif tag == "div":
            if self.content_depth == self.depth:
                self.content_depth = None
            if self.outer_depth == self.depth:
                self._finish_entry()
                self.outer_depth = None
            self.depth -= 1

    def handle_data(self, data: str) -> None:
        if self.content_depth is not None:
            self.parts.append(data)
            if self.link_depth is not None:
                self.link_text.append(data)

    def _finish_entry(self) -> None:
        if not self.link_url:
            return
        video_id = _video_id(self.link_url)
        if not video_id:
            return
        title = " ".join("".join(self.link_text).split())
        if not title:
            return
        match = DATE_PATTERN.search("".join(self.parts).replace("\u202f", " ").replace("\xa0", " "))
        if not match:
            return
        watched_at = datetime.strptime(match.group().replace("GMT", ""), "%b %d, %Y, %I:%M:%S %p %z")
        self.events.append(WatchEvent(video_id, title, self.link_url, watched_at))


def read_watch_events(archive_path: str | Path) -> list[WatchEvent]:
    """Return dated YouTube watch events without extracting files to disk."""
    with ZipFile(archive_path) as archive:
        matches = [name for name in archive.namelist() if name.endswith(WATCH_HISTORY_SUFFIX)]
        if len(matches) != 1:
            raise ValueError(f"Expected one watch-history.html, found {len(matches)}")
        html = archive.read(matches[0]).decode("utf-8")
    parser = _WatchHistoryHTMLParser()
    parser.feed(html)
    return parser.events
