"""Build one source-aware training CSV without changing either input file."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


FIELDS = ("video_id", "title", "category_label", "source", "label_status", "video_url")


def build_training_data(personal: Path, external: Path, output: Path, *, verify_external: bool = False) -> dict[str, int]:
    rows: list[tuple[str, ...]] = []
    seen: set[str] = set()
    with personal.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames or not {"video_id", "title", "category_label"} <= set(reader.fieldnames):
            raise ValueError("Personal CSV needs video_id,title,category_label")
        for row in reader:
            video_id, title, label = (row[key].strip() for key in ("video_id", "title", "category_label"))
            if not video_id or not title or not label:
                continue
            if video_id in seen:
                raise ValueError(f"Duplicate personal video ID: {video_id}")
            seen.add(video_id)
            rows.append((video_id, title, label, "personal", "verified",
                         f"https://www.youtube.com/watch?v={video_id}"))
    personal_count = len(rows)
    with external.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames or not {"video_id", "title", "category_label", "label_status"} <= set(reader.fieldnames):
            raise ValueError("External CSV needs video_id,title,category_label,label_status")
        for row in reader:
            video_id, title, label = (row[key].strip() for key in ("video_id", "title", "category_label"))
            status = row["label_status"].strip()
            if not video_id or not title or not label or status not in {"proposed", "verified"}:
                continue
            if verify_external:
                status = "verified"
            if video_id in seen:
                raise ValueError(f"Duplicate video ID across sources: {video_id}")
            seen.add(video_id)
            rows.append((video_id, title, label, "external", status,
                         f"https://www.youtube.com/watch?v={video_id}"))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(FIELDS)
        writer.writerows(rows)
    return {"personal": personal_count, "external": len(rows) - personal_count}


def main() -> None:
    parser = argparse.ArgumentParser(description="Combine personal and external training rows")
    parser.add_argument("personal", type=Path)
    parser.add_argument("external", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify-external", action="store_true",
                        help="Mark external labels verified after human review")
    args = parser.parse_args()
    if args.output.resolve() in {args.personal.resolve(), args.external.resolve()}:
        parser.error("Output must be a new file")
    try:
        counts = build_training_data(args.personal, args.external, args.output,
                                     verify_external=args.verify_external)
    except (ValueError, FileExistsError) as exc:
        parser.error(str(exc))
    print(f"개인 검증 레이블 {counts['personal']}개 + 외부 후보 {counts['external']}개")
    print(f"학습 데이터: {args.output}")


if __name__ == "__main__":
    main()
