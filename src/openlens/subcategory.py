"""Discover and classify subcategories within one top-level category."""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression

from .cluster import EMBEDDING_MODEL, embed_titles
from .sampling import select_diverse_samples
from .subcategory_taxonomy import SUBCATEGORIES



CLUSTER_FIELDS = (
    "parent_category", "cluster_id", "video_id", "title", "watch_count",
    "parent_confidence", "parent_review_status", "parent_assignment_source",
    "distance_to_center",
)
LABEL_FIELDS = CLUSTER_FIELDS + ("rank", "primary_label", "secondary_labels", "review_note")
PREDICTION_FIELDS = CLUSTER_FIELDS + (
    "primary_label", "predicted_primary_label", "confidence", "review_status",
    "secondary_labels",
)


@dataclass(frozen=True)
class ParentVideo:
    video_id: str
    title: str
    watch_count: int
    confidence: str
    review_status: str
    assignment_source: str

# 부모 카테고리에 해당하는 비디오를 읽어오는 함수. 확인된 부모 레이블이 있는 경우 이를 사용하고, 그렇지 않으면 예측된 레이블을 사용합니다.
def read_parent_videos(source: Path, parent: str) -> list[ParentVideo]:
    """Use confirmed parent labels when present, otherwise use predictions."""
    if parent not in SUBCATEGORIES:
        raise ValueError(f"Unknown parent category: {parent}")
    videos: list[ParentVideo] = []
    seen: set[str] = set()
    with source.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        required = {"video_id", "title", "watch_count", "category_label",
                    "predicted_label", "confidence", "review_status"}
        if not reader.fieldnames or not required <= set(reader.fieldnames):
            raise ValueError("Source must be a category-predictions CSV")
        for row in reader:
            confirmed = row["category_label"].strip()
            assigned = confirmed or row["predicted_label"].strip()
            if assigned != parent:
                continue
            video_id = row["video_id"].strip()
            if not video_id or not row["title"].strip() or video_id in seen:
                raise ValueError("Parent category has a missing or duplicate video ID/title")
            seen.add(video_id)
            videos.append(ParentVideo(
                video_id=video_id,
                title=row["title"].strip(),
                watch_count=int(row["watch_count"]),
                confidence=row["confidence"].strip(),
                review_status=row["review_status"].strip(),
                assignment_source="confirmed" if confirmed else "predicted",
            ))
    return sorted(videos, key=lambda video: video.video_id)

# 부모 카테고리에 해당하는 비디오를 모아 새로운 파일을 생성
def write_csv(path: Path, fields: tuple[str, ...], rows: list[dict[str, object]]) -> None:
    """Create a private CSV without replacing an earlier review file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sample_parent(source: Path, parent: str, output_dir: Path, *, k: int = 5,
                  per_cluster: int = 20) -> tuple[Path, Path]:
    """Cluster one parent category and create examples for manual labeling."""
    if k < 2 or per_cluster < 1:
        raise ValueError("k must be at least 2 and per_cluster must be positive")
    videos = read_parent_videos(source, parent)
    if len({video.title for video in videos}) < k:
        raise ValueError(f"{parent} needs at least {k} distinct titles")

    slug = re.sub(r"[^a-z0-9]+", "-", parent.lower()).strip("-")
    clusters_path = output_dir / f"{slug}-clusters-k{k}-diverse.csv"
    labels_path = output_dir / f"{slug}-label-samples-k{k}-diverse.csv"
    if clusters_path.exists() or labels_path.exists():
        raise FileExistsError("Cluster or label sample CSV already exists")

    # Embed each title once, then fit K-Means within this parent category.
    # 부모 카테고리에 속한 비디오 클러스터링을 다시함.
    vectors = embed_titles([video.title for video in videos])
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_ids = model.fit_predict(vectors)
    distances = model.transform(vectors)

    cluster_rows: list[dict[str, object]] = []
    label_rows: list[dict[str, object]] = []
    for index, video in enumerate(videos):
        cluster_id = int(cluster_ids[index])
        cluster_rows.append({
            "parent_category": parent,
            "cluster_id": cluster_id,
            "video_id": video.video_id,
            "title": video.title,
            "watch_count": video.watch_count,
            "parent_confidence": video.confidence,
            "parent_review_status": video.review_status,
            "parent_assignment_source": video.assignment_source,
            "distance_to_center": f"{distances[index, cluster_id]:.6f}",
        })

    # Sample across four distance bands, avoiding near-duplicates within each band.
    video_ids = [video.video_id for video in videos]
    for cluster_id in range(k):
        members = [i for i, label in enumerate(cluster_ids) if label == cluster_id]
        selected = select_diverse_samples(members, vectors, distances[:, cluster_id],
                                          video_ids, per_cluster)
        for rank, index in enumerate(selected, start=1):
            label_rows.append({
                **cluster_rows[index], "rank": rank, "primary_label": "",
                "secondary_labels": "", "review_note": "",
            })

    write_csv(clusters_path, CLUSTER_FIELDS, cluster_rows)
    write_csv(labels_path, LABEL_FIELDS, label_rows)
    return clusters_path, labels_path


def read_reviewed_labels(path: Path, parent: str) -> dict[str, tuple[str, str]]:
    """Read only manually entered primary labels and optional secondary labels."""
    allowed = set(SUBCATEGORIES[parent])
    reviewed: dict[str, tuple[str, str]] = {}
    with path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        required = {"parent_category", "video_id", "primary_label", "secondary_labels"}
        if not reader.fieldnames or not required <= set(reader.fieldnames):
            raise ValueError("Label CSV is missing subcategory review columns")
        for row in reader:
            primary = row["primary_label"].strip()
            if not primary:
                continue
            video_id = row["video_id"].strip()
            if row["parent_category"].strip() != parent or not video_id:
                raise ValueError("Reviewed label has the wrong parent or no video ID")
            secondary = [label.strip() for label in row["secondary_labels"].split("|") if label.strip()]
            if primary not in allowed or any(label not in allowed or label == primary for label in secondary):
                raise ValueError(f"Invalid subcategory label for video ID: {video_id}")
            if video_id in reviewed:
                raise ValueError(f"Duplicate reviewed video ID: {video_id}")
            reviewed[video_id] = (primary, "|".join(dict.fromkeys(secondary)))
    return reviewed

# 클러스터링 이후에 분류작업
def classify_parent(clusters_path: Path, labels_path: Path, parent: str,
                    output: Path, model_output: Path) -> tuple[int, int]:
    """Train on reviewed primary labels and predict the rest of one parent."""
    if parent not in SUBCATEGORIES:
        raise ValueError(f"Unknown parent category: {parent}")
    if output.resolve() == model_output.resolve() or output.exists() or model_output.exists():
        raise FileExistsError("Prediction or model path already exists or overlaps")
    with clusters_path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames or not set(CLUSTER_FIELDS) <= set(reader.fieldnames):
            raise ValueError("Cluster CSV is missing required columns")
        rows = list(reader)
    if not rows or any(row["parent_category"].strip() != parent for row in rows):
        raise ValueError("Cluster CSV has no rows or contains another parent category")
    ids = [row["video_id"].strip() for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Cluster CSV contains duplicate video IDs")
    reviewed = read_reviewed_labels(labels_path, parent)
    if not set(reviewed) <= set(ids):
        raise ValueError("A reviewed video is missing from the cluster CSV")
    if len({primary for primary, _ in reviewed.values()}) < 2:
        raise ValueError("At least two reviewed primary subcategories are needed")

    # Re-embed titles and fit a parent-specific classifier on reviewed examples.
    vectors = embed_titles([row["title"] for row in rows])
    train_indices = [index for index, video_id in enumerate(ids) if video_id in reviewed]
    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(vectors[train_indices], [reviewed[ids[i]][0] for i in train_indices])
    probabilities = model.predict_proba(vectors)

    # Keep secondary labels only where a human supplied them; do not invent them.
    predictions: list[dict[str, object]] = []
    for row, video_id, scores in zip(rows, ids, probabilities):
        best = int(np.argmax(scores))
        confidence = float(scores[best])
        predictions.append({
            **row,
            "primary_label": reviewed.get(video_id, ("", ""))[0],
            "predicted_primary_label": model.classes_[best],
            "confidence": f"{confidence:.4f}",
            "review_status": "labeled" if video_id in reviewed else (
                "suggested" if confidence >= 0.8 else "review_needed"
            ),
            "secondary_labels": reviewed.get(video_id, ("", ""))[1],
        })

    write_csv(output, PREDICTION_FIELDS, predictions)
    model_output.parent.mkdir(parents=True, exist_ok=True)
    with model_output.open("xb") as file:
        joblib.dump({
            "classifier": model,
            "parent_category": parent,
            "embedding_model": EMBEDDING_MODEL,
            "embedding_dimensions": int(vectors.shape[1]),
            "scikit_learn_version": sklearn.__version__,
        }, file)
    return len(reviewed), len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Explore subcategories within one parent")
    commands = parser.add_subparsers(dest="command", required=True)
    sample = commands.add_parser("sample", help="Cluster one parent and create a review CSV")
    sample.add_argument("source", type=Path, help="Top-level category predictions CSV")
    sample.add_argument("--parent", choices=SUBCATEGORIES, required=True)
    sample.add_argument("--k", type=int, default=5)
    sample.add_argument("--per-cluster", type=int, default=20)
    sample.add_argument("--output-dir", type=Path, default=Path("data/subcategories"))
    classify = commands.add_parser("classify", help="Train on reviewed primary labels")
    classify.add_argument("clusters", type=Path)
    classify.add_argument("labels", type=Path)
    classify.add_argument("--parent", choices=SUBCATEGORIES, required=True)
    classify.add_argument("--output", type=Path, required=True)
    classify.add_argument("--model-output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "sample":
            clusters, labels = sample_parent(args.source, args.parent, args.output_dir,
                                             k=args.k, per_cluster=args.per_cluster)
            print(f"전체 군집 파일: {clusters}")
            print(f"레이블링 샘플: {labels}")
            print(f"허용되는 주 레이블: {', '.join(SUBCATEGORIES[args.parent])}")
        else:
            reviewed, total = classify_parent(args.clusters, args.labels, args.parent,
                                              args.output, args.model_output)
            print(f"검증된 하위 레이블: {reviewed}개 / 분류한 영상: {total}개")
            print(f"예측 결과: {args.output}")
            print(f"학습 모델: {args.model_output}")
    except (ValueError, FileExistsError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
