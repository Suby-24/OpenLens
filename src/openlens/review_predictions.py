"""Audit prediction titles and propose labels with explicit provenance."""

from __future__ import annotations

import argparse
import csv
import re
import shutil
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


def normalized(text: str) -> str:
    return unicodedata.normalize("NFKC", text).casefold().strip()


def title_proposal(title: str) -> tuple[str, str, str]:
    """Return conservative title-based suggestions, never content verification."""
    text = normalized(title)
    if not text or text.startswith("http"):
        return "", "", "Title unavailable; video information needed"
    # Specific formats take priority over people or broad topic words.
    rules = [
        (r"grand beatbox battle", "Entertainment", "공연·뮤지컬"),
        (r"\b(lyrics?|playlist|official audio|official mv|music video)\b|가사|플레이리스트|\[mv\]|ost\b", "Entertainment", "음악"),
        (r"리센느|rescene|프로미스나인|fromis.?9|하츠투하츠|hearts2hearts|하투하|엔믹스|nmixx|에스파|aespa|아일릿|\billit\b|피프티피프티|fifty.?fifty|베이비몬스터|babymonster|아이돌|걸그룹|#아이브|#ive\b|#kpop", "Entertainment", "아이돌"),
        (r"오버워치|overwatch|발로란트|valorant|배틀그라운드|\bpubg\b|pubgmobile|클래시로얄|clashroyale|클래시오브클랜|클오클|리니지|\bcs2\b|pokemonunite|둠피스트|정크랫|겐지|#gaming\b", "Entertainment", "게임 콘텐츠"),
        (r"#드라마|#신병|스파이더맨|#마블|#영화", "Entertainment", ""),
        (r"#예능|라디오스타|런닝맨|핑계고|코미디빅리그|지락실|개그맨|무한도전", "Entertainment", "예능"),
        (r"먹방|mukbang|레시피|요리법", "Lifestyle & Culture", ""),
        (r"#헬스|피트니스|근비대|광배|견갑|벤치프레스|스쿼트|데드리프트", "Health & Wellness", ""),
        (r"손흥민|#축구|#야구|#테니스|#농구|#배구|#volleyball|챔피언스리그|\bnba\b", "Sports", ""),
        (r"갤럭시.{0,12}(살까|구매|리뷰|사지)|갤럭시 워치|반도체|인공지능|\bgpt[- ]?\d|골전도스피커", "Tech & Science", ""),
    ]
    for pattern, parent, child in rules:
        match = re.search(pattern, text)
        if match:
            return parent, child, f"Title cue: {match.group(0)}; full content not verified"
    return "", "", "Insufficient title evidence; video information needed"


def read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        return list(reader.fieldnames or []), list(reader)


def save_rows(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def review(predictions: Path, labels: Path, output: Path, *, lower: bool = False) -> Counter:
    """Preserve human corrections and annotate every remaining row for follow-up."""
    fields, rows = read_rows(predictions)
    _, label_rows = read_rows(labels)
    target = "primary_label" if lower else "category_label"
    predicted = "predicted_primary_label" if lower else "predicted_label"
    manual = {row["video_id"]: row for row in label_rows if row.get(target, "").strip()}
    extra = ["original_prediction", "original_confidence", "review_method", "review_reason",
             "proposed_parent", "proposed_label", "reviewed_at"]
    fields += [field for field in extra if field not in fields]
    counts: Counter = Counter()
    for row in rows:
        row["original_prediction"] = row.get("original_prediction", row[predicted])
        row["original_confidence"] = row.get("original_confidence", row.get("confidence", ""))
        row["reviewed_at"] = datetime.now(timezone.utc).isoformat()
        parent, child, reason = title_proposal(row["title"])
        row["proposed_parent"] = parent
        row["proposed_label"] = child if lower else parent
        if row["video_id"] in manual or row.get(target, "").strip():
            if row["video_id"] in manual:
                row[target] = manual[row["video_id"]][target].strip()
                if lower:
                    row["secondary_labels"] = manual[row["video_id"]].get("secondary_labels", "")
            row["review_method"] = "existing_human_label"
            row["review_reason"] = "Preserved existing human label"
            row["review_status"] = "labeled"
        else:
            proposal = child if lower else parent
            if lower and parent and parent != row["parent_category"]:
                row["review_method"] = "parent_review_needed"
                row["review_status"] = "review_needed"
                row["review_reason"] = f"Possible wrong parent: {parent}. {reason}"
            elif proposal:
                row[predicted] = proposal
                # A model probability cannot describe a rule-based replacement.
                row["confidence"] = ""
                row["review_status"] = "ai_suggested"
                row["review_method"] = "title_rule"
                row["review_reason"] = reason
            else:
                row["review_method"] = "insufficient_evidence"
                row["review_status"] = "review_needed"
                row["review_reason"] = reason
        counts[row["review_method"]] += 1
    save_rows(output, fields, rows)
    unresolved = [row for row in rows if row["review_method"] != "existing_human_label"]
    save_rows(output.with_name(output.stem + "-review-queue.csv"), fields, unresolved)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("labels", type=Path)
    parser.add_argument("--lower", action="store_true")
    args = parser.parse_args()
    backup = args.predictions.with_suffix(".before-review.csv")
    if backup.exists():
        parser.error(f"Backup already exists: {backup}")
    shutil.copy2(args.predictions, backup)
    print(review(args.predictions, args.labels, args.predictions, lower=args.lower))
    print(f"Backup: {backup}")


if __name__ == "__main__":
    main()
