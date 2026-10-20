"""Collect public YouTube search candidates for human category review.

Search matches are proposed labels, never verified ground truth.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
from pathlib import Path
from urllib.parse import urlencode


QUERIES = {
    "Politics": ["한국 정치 국회 선거 분석", "대통령 정책 정당 정치 해설", "parliament election politics explained", "government policy political analysis"],
    "Economy & Business": ["경제 주식 금리 투자 분석", "기업 경영 창업 경제 설명", "economics finance investing explained", "business markets economy lecture"],
    "Society & Crime": ["사회 사건 범죄 뉴스 분석", "법원 재판 경찰 사건 보도", "crime investigation court news explained", "social issues labor law documentary"],
    "International & World": ["국제 정세 외교 세계 뉴스", "세계 분쟁 국가 관계 설명", "world geopolitics international relations explained", "global affairs world news analysis"],
    "Tech & Science": ["인공지능 반도체 과학 기술 설명", "로봇 우주 과학 강의", "technology science AI explained", "computer science robotics lecture"],
    "Health & Wellness": ["건강 의학 질병 설명", "운동 식단 건강 전문가 강의", "health medicine nutrition explained", "exercise physiology medical lecture"],
    "Environment": ["기후 변화 환경 생태 다큐", "탄소 배출 지구온난화 환경 설명", "climate change environment documentary", "biodiversity ecology environmental science"],
}


def _videos(query: str) -> list[tuple[str, str]]:
    # Fetch one public search page and extract video titles and IDs.
    url = "https://www.youtube.com/results?" + urlencode({"search_query": query})
    page = subprocess.run(
        ["curl", "-fsSL", "--max-time", "25", "-A", "Mozilla/5.0", url],
        check=True, capture_output=True, text=True,
    ).stdout
    match = re.search(r"var ytInitialData = (\{.*?\});</script>", page)
    if not match:
        raise ValueError("YouTube search page format changed")
    data = json.loads(match.group(1))
    found: list[tuple[str, str]] = []

    def visit(value: object) -> None:
        # Search nested page data for video result objects.
        if isinstance(value, dict):
            video = value.get("videoRenderer")
            if isinstance(video, dict):
                title = video.get("title", {}).get("runs", [{}])[0].get("text", "")
                video_id = video.get("videoId", "")
                if title and video_id:
                    found.append((video_id, title))
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(data)
    return found


def collect(output: Path, *, target: int = 50) -> dict[str, int]:
    # Gather unique examples per category as unverified label candidates.
    if target < 1:
        raise ValueError("target must be positive")
    rows: list[tuple[str, str, str, str, str, str]] = []
    seen: set[str] = set()
    counts: dict[str, int] = {}
    for category, queries in QUERIES.items():
        count = 0
        for query in queries:
            for video_id, title in _videos(query):
                if video_id in seen or count >= target:
                    continue
                seen.add(video_id)
                rows.append((video_id, title, f"https://www.youtube.com/watch?v={video_id}",
                             category, "proposed", query))
                count += 1
            if count >= target:
                break
        counts[category] = count
    output.parent.mkdir(parents=True, exist_ok=True)
    # Save candidates for manual verification before normal training.
    with output.open("x", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(("video_id", "title", "video_url", "category_label", "label_status", "source_query"))
        writer.writerows(rows)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect external video candidates for category review")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--target", type=int, default=50)
    args = parser.parse_args()
    counts = collect(args.output, target=args.target)
    for category, count in counts.items():
        print(f"{category}: {count}")
    print(f"Candidate file: {args.output}")


if __name__ == "__main__":
    main()
