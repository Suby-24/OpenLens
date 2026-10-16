"""Small, privacy-preserving commands for the first Takeout experiment."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import timedelta

from .takeout import read_watch_events


def main() -> None:
    parser = argparse.ArgumentParser(prog="openlens")
    parser.add_argument("archive", help="Path to a YouTube Takeout ZIP or merged CSV file")
    parser.add_argument("--days", type=int, default=90, help="Look-back window (default: 90)")
    args = parser.parse_args()
    if args.days <= 0:
        parser.error("--days must be positive")

    events = read_watch_events(args.archive)
    if not events:
        print("분석 가능한 YouTube 시청 이벤트가 없습니다.")
        return
    latest = max(event.watched_at for event in events)
    recent = [event for event in events if event.watched_at >= latest - timedelta(days=args.days)]
    print(f"전체 시청 이벤트: {len(events):,}건")
    print(f"전체 고유 영상: {len({event.video_id for event in events}):,}개")
    print(f"기록 기간: {min(event.watched_at for event in events).date()} ~ {latest.date()}")
    print(f"최근 {args.days}일 시청 이벤트: {len(recent):,}건")
    print(f"최근 {args.days}일 고유 영상: {len({event.video_id for event in recent}):,}개")
    print("월별 시청 이벤트:")
    monthly = Counter(event.watched_at.strftime("%Y-%m") for event in events)
    for month, count in sorted(monthly.items(), reverse=True):
        print(f"  {month}: {count:,}건")


if __name__ == "__main__":
    main()
