"""Train a category classifier from reviewed YouTube titles."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from .cluster import embed_titles
from .takeout import read_watch_events


def read_labels(path: Path) -> dict[str, str]:
    labels: dict[str, str] = {}
    with path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames or not {"video_id", "category_label"} <= set(reader.fieldnames):
            raise ValueError("Label CSV needs video_id and category_label columns")
        for row in reader:
            video_id, label = row["video_id"].strip(), row["category_label"].strip()
            if not video_id or not label:
                continue
            if video_id in labels and labels[video_id] != label:
                raise ValueError(f"Conflicting labels for video_id: {video_id}")
            labels[video_id] = label
    return labels


def classify(source: Path, label_file: Path, output: Path, *, days: int = 90) -> None:
    if days < 1:
        raise ValueError("days must be positive")
    events = read_watch_events(source)
    if not events:
        raise ValueError("No watch events found")
    from datetime import timedelta
    cutoff = max(event.watched_at for event in events) - timedelta(days=days)
    recent = [event for event in events if event.watched_at >= cutoff]
    titles = {event.video_id: event.title for event in recent}
    counts = Counter(event.video_id for event in recent)
    labels = read_labels(label_file)
    known = sorted(set(titles) & set(labels))
    classes = Counter(labels[video_id] for video_id in known)
    if len(classes) < 2:
        raise ValueError("At least two labeled categories are needed")
    ids = sorted(titles)
    vectors = embed_titles([titles[video_id] for video_id in ids])
    index = {video_id: i for i, video_id in enumerate(ids)}
    x = np.asarray([vectors[index[video_id]] for video_id in known])
    y = [labels[video_id] for video_id in known]
    if min(classes.values()) >= 2 and len(known) >= 10:
        test_count = max(len(classes), round(len(known) * 0.2))
        if len(known) - test_count >= len(classes):
            x_train, x_test, y_train, y_test = train_test_split(
                x, y, test_size=test_count, random_state=42, stratify=y
            )
            evaluation_model = LogisticRegression(max_iter=1000, class_weight="balanced")
            evaluation_model.fit(x_train, y_train)
            print(classification_report(y_test, evaluation_model.predict(x_test), zero_division=0))
    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(x, y)
    probabilities = model.predict_proba(vectors)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(("video_id", "title", "watch_count", "category_label", "predicted_label", "confidence", "review_status"))
        for video_id, scores in zip(ids, probabilities):
            best = int(np.argmax(scores))
            confidence = float(scores[best])
            writer.writerow((video_id, titles[video_id], counts[video_id], labels.get(video_id, ""),
                             model.classes_[best], f"{confidence:.4f}",
                             "labeled" if video_id in labels else "suggested" if confidence >= 0.8 else "review_needed"))
    print(f"학습 레이블: {len(known)}개 / 카테고리: {len(classes)}개 / 출력 영상: {len(ids)}개")
    print(f"결과: {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify YouTube titles using reviewed labels")
    parser.add_argument("source", type=Path)
    parser.add_argument("labels", type=Path)
    parser.add_argument("--days", type=int, default=90)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        classify(args.source, args.labels, args.output, days=args.days)
    except (ValueError, FileExistsError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
