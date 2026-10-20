"""Cluster unique YouTube titles with EmbeddingGemma and K-Means."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans

from .takeout import WatchEvent, read_watch_events

EMBEDDING_MODEL = "google/embeddinggemma-300m"


def embed_titles(titles: list[str]) -> np.ndarray:
    """Encode titles locally with EmbeddingGemma's clustering prompt."""
    # Load the embedding model and turn each title into a normalized vector.
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EMBEDDING_MODEL)
    vectors = np.asarray(
        model.encode(
            titles,
            prompt_name="Clustering",
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
    )
    if vectors.ndim != 2 or vectors.shape[0] != len(titles):
        raise ValueError("Embedding model returned an unexpected vector shape")
    if not np.isfinite(vectors).all() or np.any(np.linalg.norm(vectors, axis=1) == 0):
        raise ValueError("Embedding model returned invalid vectors")
    return vectors


@dataclass(frozen=True)
class ClusteredVideo:
    video_id: str
    title: str
    cluster_id: int
    watch_count: int


@dataclass(frozen=True)
class ClusterSummary:
    cluster_id: int
    unique_videos: int
    watch_events: int
    representative_titles: tuple[str, ...]


@dataclass(frozen=True)
class ClusteringResult:
    window_start: datetime
    window_end: datetime
    total_watch_events: int
    videos: tuple[ClusteredVideo, ...]
    clusters: tuple[ClusterSummary, ...]
    embedding_vectors: np.ndarray = field(repr=False, compare=False)


def cluster_watch_events(
    events: list[WatchEvent], *, k: int = 11, days: int = 90
) -> ClusteringResult:
    """Cluster each video once; count every watch only in consumption totals.

    The look-back window ends at the newest event in the supplied Takeout file.
    EmbeddingGemma encodes each unique title once. Clusters have no automatic
    category names; the embedding vectors are retained for the 2D report.
    """
    if k < 2:
        raise ValueError("k must be at least 2")
    if days < 1:
        raise ValueError("days must be positive")
    if not events:
        raise ValueError("No watch events were supplied")

    window_end = max(event.watched_at for event in events)
    window_start = window_end - timedelta(days=days)
    recent = [event for event in events if event.watched_at >= window_start]

    # Count repeated watches, but keep one title per video for clustering.
    watch_counts = Counter(event.video_id for event in recent)
    # Sorting makes the input order reproducible even if Takeout changes order.
    latest_title = {
        event.video_id: event.title
        for event in sorted(recent, key=lambda event: event.watched_at)
    }
    video_ids = sorted(latest_title)
    titles = [latest_title[video_id] for video_id in video_ids]
    if len(set(titles)) < k:
        raise ValueError("k cannot exceed the number of distinct video titles")

    matrix = embed_titles(titles)

    # Assign every unique video to one of the requested K-Means clusters.
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(matrix)
    if len(set(labels)) != k:
        raise ValueError("Fewer than k distinct text groups were found")

    videos = tuple(
        ClusteredVideo(video_id, title, int(label), watch_counts[video_id])
        for video_id, title, label in zip(video_ids, titles, labels)
    )

    distances = model.transform(matrix)
    clusters = []
    # Summarize each cluster and pick titles closest to its center.
    for cluster_id in range(k):
        members = [index for index, label in enumerate(labels) if label == cluster_id]
        nearest = sorted(members, key=lambda index: distances[index, cluster_id])[:3]
        clusters.append(
            ClusterSummary(
                cluster_id=cluster_id,
                unique_videos=len(members),
                watch_events=sum(videos[index].watch_count for index in members),
                representative_titles=tuple(titles[index] for index in nearest),
            )
        )

    return ClusteringResult(
        window_start=window_start,
        window_end=window_end,
        total_watch_events=len(recent),
        videos=videos,
        clusters=tuple(clusters),
        embedding_vectors=matrix,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Cluster local YouTube Takeout titles")
    parser.add_argument("archive", type=Path, help="YouTube Takeout ZIP or merged CSV path")
    parser.add_argument("--k", type=int, default=11, help="Number of clusters (default: 11)")
    parser.add_argument("--days", type=int, default=90, help="Look-back days (default: 90)")
    args = parser.parse_args()

    try:
        result = cluster_watch_events(read_watch_events(args.archive), k=args.k, days=args.days)
    except ValueError as exc:
        parser.error(str(exc))

    print(f"기간: {result.window_start.date()} ~ {result.window_end.date()}")
    print(f"고유 영상: {len(result.videos):,}개 / 시청 이벤트: {result.total_watch_events:,}건")
    for cluster in result.clusters:
        print(
            f"\n클러스터 {cluster.cluster_id}: "
            f"고유 영상 {cluster.unique_videos:,}개, 시청 {cluster.watch_events:,}회"
        )
        for title in cluster.representative_titles:
            print(f"  대표 제목: {title}")


if __name__ == "__main__":
    main()
