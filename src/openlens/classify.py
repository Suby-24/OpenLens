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


def read_external_labels(path: Path, *, allow_proposed: bool = False) -> list[tuple[str, str, str]]:
    """Read external titles for training only; never use them as watch events."""
    rows: list[tuple[str, str, str]] = []
    with path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        required = {"video_id", "title", "category_label", "label_status"}
        if not reader.fieldnames or not required <= set(reader.fieldnames):
            raise ValueError("External CSV needs video_id,title,category_label,label_status")
        for row in reader:
            status = row["label_status"].strip()
            if status == "verified" or (allow_proposed and status == "proposed"):
                if row["video_id"].strip() and row["title"].strip() and row["category_label"].strip():
                    rows.append((row["video_id"].strip(), row["title"].strip(), row["category_label"].strip()))
    return rows


def read_training_data(path: Path, *, allow_proposed: bool = False) -> tuple[dict[str, str], list[tuple[str, str, str]]]:
    """Read one training file while keeping personal and external origins distinct."""
    personal: dict[str, str] = {}
    external: list[tuple[str, str, str]] = []
    seen: set[str] = set()
    with path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        required = {"video_id", "title", "category_label", "source", "label_status"}
        if not reader.fieldnames or not required <= set(reader.fieldnames):
            raise ValueError("Training CSV needs video_id,title,category_label,source,label_status")
        for row in reader:
            video_id = row["video_id"].strip()
            label = row["category_label"].strip()
            if not video_id or not label:
                continue
            if video_id in seen:
                raise ValueError(f"Duplicate training video ID: {video_id}")
            seen.add(video_id)
            source = row["source"].strip()
            status = row["label_status"].strip()
            if source == "personal" and status == "verified":
                personal[video_id] = label
            elif source == "external" and (status == "verified" or (allow_proposed and status == "proposed")):
                title = row["title"].strip()
                if title:
                    external.append((video_id, title, label))
            elif source not in {"personal", "external"}:
                raise ValueError(f"Unknown training source: {source}")
    return personal, external


def classify(source: Path, label_file: Path | None, output: Path, *, days: int = 90,
             external_labels: Path | None = None, training_data: Path | None = None,
             allow_proposed_external: bool = False) -> None:
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
    if training_data:
        if label_file or external_labels:
            raise ValueError("Use either --training-data or separate label files")
        labels, external = read_training_data(training_data, allow_proposed=allow_proposed_external)
    else:
        if label_file is None:
            raise ValueError("A label CSV or --training-data is required")
        labels = read_labels(label_file)
        external = read_external_labels(external_labels, allow_proposed=allow_proposed_external) if external_labels else []
    known = sorted(set(titles) & set(labels))
    external = [row for row in external if row[0] not in labels]
    classes = Counter([labels[video_id] for video_id in known] + [row[2] for row in external])
    if len(classes) < 2:
        raise ValueError("At least two labeled categories are needed")
    ids = sorted(titles)
    all_vectors = embed_titles([titles[video_id] for video_id in ids] + [row[1] for row in external])
    vectors = all_vectors[:len(ids)]
    external_vectors = all_vectors[len(ids):]
    index = {video_id: i for i, video_id in enumerate(ids)}
    x = np.asarray([vectors[index[video_id]] for video_id in known])
    y = [labels[video_id] for video_id in known]
    personal_classes = Counter(y)
    if len(personal_classes) >= 2 and min(personal_classes.values()) >= 2 and len(known) >= 10:
        test_count = max(len(personal_classes), round(len(known) * 0.2))
        if len(known) - test_count >= len(personal_classes):
            x_train, x_test, y_train, y_test = train_test_split(
                x, y, test_size=test_count, random_state=42, stratify=y
            )
            if external:
                x_train = np.concatenate((x_train, external_vectors))
                y_train = list(y_train) + [row[2] for row in external]
            evaluation_model = LogisticRegression(max_iter=1000, class_weight="balanced")
            evaluation_model.fit(x_train, y_train)
            print(classification_report(y_test, evaluation_model.predict(x_test), zero_division=0))
    if external:
        x = np.concatenate((x, external_vectors))
        y = list(y) + [row[2] for row in external]
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
    print(f"개인 학습 레이블: {len(known)}개 / 외부 학습 레이블: {len(external)}개 / 카테고리: {len(classes)}개")
    print(f"예측 대상: 개인 시청 영상 {len(ids)}개")
    print(f"결과: {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify YouTube titles using reviewed labels")
    parser.add_argument("source", type=Path)
    parser.add_argument("labels", nargs="?", type=Path)
    parser.add_argument("--days", type=int, default=90)
    parser.add_argument("--external-labels", type=Path)
    parser.add_argument("--training-data", type=Path,
                        help="Combined training CSV with source and label_status columns")
    parser.add_argument("--allow-proposed-external", action="store_true",
                        help="Include unverified search-based external labels in training")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        classify(args.source, args.labels, args.output, days=args.days,
                 external_labels=args.external_labels,
                 training_data=args.training_data,
                 allow_proposed_external=args.allow_proposed_external)
    except (ValueError, FileExistsError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
