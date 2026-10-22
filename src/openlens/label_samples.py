"""Create a private, human-labelable sample from clustering results."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .cluster import ClusteringResult, cluster_watch_events
from .sampling import select_diverse_samples
from .takeout import read_watch_events


@dataclass(frozen=True)
class LabelSample:
    cluster_id: int
    rank: int
    video_id: str
    title: str
    watch_count: int
    distance_to_center: float


def select_label_samples(result: ClusteringResult, *, per_cluster: int = 20) -> list[LabelSample]:
    """Sample each cluster across center-distance bands with embedding diversity."""
    if per_cluster < 1:
        raise ValueError("per_cluster must be positive")
    samples: list[LabelSample] = []
    video_ids = [video.video_id for video in result.videos]
    for cluster in result.clusters:
        members = [index for index, video in enumerate(result.videos)
                   if video.cluster_id == cluster.cluster_id]
        center = result.embedding_vectors[members].mean(axis=0)
        distances = np.linalg.norm(result.embedding_vectors - center, axis=1)
        selected = select_diverse_samples(members, result.embedding_vectors, distances,
                                          video_ids, per_cluster)
        for rank, index in enumerate(selected, start=1):
            video = result.videos[index]
            samples.append(LabelSample(
                cluster_id=cluster.cluster_id,
                rank=rank,
                video_id=video.video_id,
                title=video.title,
                watch_count=video.watch_count,
                distance_to_center=float(distances[index]),
            ))
    return samples


def write_label_samples(samples: list[LabelSample], output: Path) -> Path:
    """Create (never overwrite) a CSV with blank labels for manual review."""
    # Leave category and note fields empty for a reviewer to fill in.
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(("cluster_id", "rank", "video_id", "title", "watch_count",
                         "distance_to_center", "category_label", "review_note"))
        for sample in samples:
            writer.writerow((sample.cluster_id, sample.rank, sample.video_id, sample.title,
                             sample.watch_count, f"{sample.distance_to_center:.6f}", "", ""))
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample clustered videos for manual labeling")
    parser.add_argument("source", type=Path, help="Merged YouTube watch-events CSV or Takeout ZIP")
    parser.add_argument("--k", type=int, default=11)
    parser.add_argument("--days", type=int, default=90)
    parser.add_argument("--per-cluster", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = cluster_watch_events(read_watch_events(args.source), k=args.k, days=args.days)
        samples = select_label_samples(result, per_cluster=args.per_cluster)
        output = write_label_samples(samples, args.output)
    except (ValueError, FileExistsError) as exc:
        parser.error(str(exc))
    print(f"기간: {result.window_start.date()} ~ {result.window_end.date()}")
    print(f"클러스터: {len(result.clusters)}개 / 레이블링 샘플: {len(samples)}개")
    print(f"로컬 파일: {output}")
    print("category_label 열에 영상별 카테고리를 직접 입력하세요. 기존 파일은 덮어쓰지 않습니다.")


if __name__ == "__main__":
    main()
