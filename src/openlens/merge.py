"""Merge local YouTube Takeout exports into one private input file."""

from __future__ import annotations

import argparse
from pathlib import Path

from .takeout import merge_watch_events, write_watch_events_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge YouTube Takeout watch histories locally")
    parser.add_argument("sources", nargs="+", type=Path, help="YouTube Takeout ZIP or merged CSV files")
    parser.add_argument("--output", required=True, type=Path, help="Private merged CSV path")
    args = parser.parse_args()
    if args.output.resolve() in {source.resolve() for source in args.sources}:
        parser.error("Output must differ from every input file")
    try:
        events = merge_watch_events(*args.sources)
    except ValueError as exc:
        parser.error(str(exc))
    path = write_watch_events_csv(events, args.output)
    print(f"통합 시청 이벤트: {len(events):,}건")
    if events:
        print(f"기간: {events[0].watched_at.date()} ~ {events[-1].watched_at.date()}")
    print(f"로컬 파일: {path}")
    print("개인 시청 기록이므로 GitHub에 올리거나 공유하지 마세요.")


if __name__ == "__main__":
    main()
