"""Create a private, offline HTML directory from category prediction CSVs."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from html import escape
from pathlib import Path
from urllib.parse import quote


def read_rows(path: Path, required: set[str]) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames or not required <= set(reader.fieldnames):
            raise ValueError(f"Missing required columns in {path}: {', '.join(sorted(required))}")
        return list(reader)


def build_directory(upper_path: Path, sub_paths: list[Path]) -> tuple[dict, int]:
    """Match lower predictions to upper predictions by video ID."""
    upper = read_rows(upper_path, {"video_id", "title", "watch_count", "category_label",
                                   "predicted_label", "review_status"})
    lower: dict[str, dict[str, str]] = {}
    for path in sub_paths:
        for row in read_rows(path, {"video_id", "parent_category", "primary_label",
                                    "predicted_primary_label", "review_status"}):
            video_id = row["video_id"].strip()
            if not video_id or video_id in lower:
                raise ValueError(f"Missing or duplicate subcategory video ID: {video_id}")
            lower[video_id] = row

    directory: dict[str, dict[str, list[dict[str, str]]]] = defaultdict(lambda: defaultdict(list))
    seen: set[str] = set()
    skipped = 0
    for row in upper:
        video_id = row["video_id"].strip()
        if not video_id or video_id in seen:
            raise ValueError(f"Missing or duplicate upper-category video ID: {video_id}")
        seen.add(video_id)
        parent = row["category_label"].strip() or row["predicted_label"].strip() or "상위 미분류"
        child_row = lower.get(video_id)
        if child_row and child_row["parent_category"].strip() != parent:
            skipped += 1
            child_row = None
        child = (child_row["primary_label"].strip() or child_row["predicted_primary_label"].strip()) if child_row else "하위 미분류"
        directory[parent][child].append({
            "video_id": video_id, "title": row["title"], "watch_count": row["watch_count"],
            "upper_status": row["review_status"],
            "lower_status": child_row["review_status"] if child_row else "not_run",
            "secondary": child_row.get("secondary_labels", "") if child_row else "",
        })
    skipped += len(set(lower) - seen)
    return directory, skipped


def write_report(directory: dict, output: Path, skipped: int = 0) -> Path:
    """Write a standalone, escaped HTML tree without replacing older reports."""
    if output.exists():
        raise FileExistsError(f"Report already exists: {output}")
    total = sum(len(videos) for children in directory.values() for videos in children.values())
    parts = []
    for parent, children in sorted(directory.items()):
        parent_count = sum(len(videos) for videos in children.values())
        child_parts = []
        for child, videos in sorted(children.items(), key=lambda item: (-len(item[1]), item[0])):
            items = []
            for video in sorted(videos, key=lambda item: (-int(item["watch_count"]), item["title"])):
                url = "https://www.youtube.com/watch?v=" + quote(video["video_id"], safe="")
                title = escape(video["title"] or video["video_id"])
                secondary = f" · 보조: {escape(video['secondary'])}" if video["secondary"] else ""
                items.append(
                    f'<li><a href="{url}" target="_blank" rel="noopener noreferrer">{title}</a>'
                    f'<span class="meta"> · {escape(video["watch_count"])}회'
                    f' · 상위 {escape(video["upper_status"])}'
                    f' · 하위 {escape(video["lower_status"])}{secondary}</span></li>'
                )
            child_parts.append(
                f'<details class="child"><summary>{escape(child)} <span>{len(videos):,}개</span></summary>'
                f'<ul>{"".join(items)}</ul></details>'
            )
        parts.append(
            f'<details class="parent"><summary>{escape(parent)} <span>{parent_count:,}개</span></summary>'
            f'<div class="children">{"".join(child_parts)}</div></details>'
        )
    warning = (f'<p class="warning">상위 결과와 맞지 않거나 없는 하위 결과 {skipped:,}건은 제외했습니다.</p>'
               if skipped else "")
    document = f"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>OpenLens 카테고리 탐색</title>
<style>
body {{max-width:1100px;margin:32px auto;padding:0 20px;font:15px/1.6 system-ui,sans-serif;color:#202124;background:#fafafa}}
h1 {{font-size:1.6rem}} .note,.meta {{color:#60656b}} .warning {{color:#a34321}}
.parent,.child {{margin:8px 0;background:white;border:1px solid #ddd;border-radius:8px}}
summary {{cursor:pointer;padding:10px 14px;font-weight:600}} summary span {{float:right;color:#60656b;font-weight:400}}
.children {{padding:0 14px 12px 24px}} ul {{margin:0;padding:0 18px 12px 36px}} li {{padding:4px 0;overflow-wrap:anywhere}}
a {{color:#1759a5;text-decoration:none}} a:hover {{text-decoration:underline}}
@media(max-width:600px) {{body {{margin:16px auto;padding:0 12px}} .meta {{display:block}}}}
</style></head><body>
<h1>OpenLens 카테고리 탐색</h1>
<p class="note">상위 카테고리 {len(directory):,}개 · 영상 {total:,}개. 이름을 눌러 하위 카테고리와 영상을 펼쳐 보세요.
예측 결과는 정답이 아니며, <code>review_needed</code>는 검토가 필요합니다.</p>
{warning}
<main>{''.join(parts)}</main>
</body></html>
"""
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as file:
        file.write(document)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Browse private category predictions as an HTML tree")
    parser.add_argument("upper", type=Path, help="Upper-category prediction CSV")
    parser.add_argument("--subcategory", type=Path, action="append", default=[],
                        help="Subcategory prediction CSV; repeat for multiple parents")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        directory, skipped = build_directory(args.upper, args.subcategory)
        write_report(directory, args.output, skipped)
    except (ValueError, FileExistsError) as exc:
        parser.error(str(exc))
    print(f"로컬 보고서: {args.output}")
    print("개인 시청 제목이 포함되므로 GitHub에 올리거나 공유하지 마세요.")


if __name__ == "__main__":
    main()
