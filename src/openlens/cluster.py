"""First experiment: cluster unique YouTube titles with TF-IDF and K-Means."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

from .takeout import WatchEvent, read_watch_events


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
    top_terms: tuple[str, ...]
    representative_titles: tuple[str, ...]


@dataclass(frozen=True)
class ClusteringResult:
    window_start: datetime
    window_end: datetime
    total_watch_events: int
    videos: tuple[ClusteredVideo, ...]
    clusters: tuple[ClusterSummary, ...]


def cluster_watch_events(
    events: list[WatchEvent], *, k: int, days: int = 90
) -> ClusteringResult:
    """Cluster each video once; count every watch only in consumption totals.

    The look-back window ends at the newest event in the supplied Takeout file.
    TF-IDF uses scikit-learn's default word tokenizer; no Korean-specific
    morphology or automatic cluster naming is applied in this baseline.
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

    vectorizer = TfidfVectorizer()
    try:
        matrix = vectorizer.fit_transform(titles)
    except ValueError as exc:
        raise ValueError("Titles contain no words recognized by default TF-IDF") from exc
    if (matrix.getnnz(axis=1) == 0).any():
        raise ValueError("Some titles have no words recognized by default TF-IDF")

    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(matrix)
    if len(set(labels)) != k:
        raise ValueError("Fewer than k distinct text groups were found")

    videos = tuple(
        ClusteredVideo(video_id, title, int(label), watch_counts[video_id])
        for video_id, title, label in zip(video_ids, titles, labels)
    )

    terms = vectorizer.get_feature_names_out()
    distances = model.transform(matrix)
    clusters = []
    for cluster_id in range(k):
        members = [index for index, label in enumerate(labels) if label == cluster_id]
        nearest = sorted(members, key=lambda index: distances[index, cluster_id])[:3]
        top_indices = model.cluster_centers_[cluster_id].argsort()[::-1][:8]
        clusters.append(
            ClusterSummary(
                cluster_id=cluster_id,
                unique_videos=len(members),
                watch_events=sum(videos[index].watch_count for index in members),
                top_terms=tuple(str(terms[index]) for index in top_indices),
                representative_titles=tuple(titles[index] for index in nearest),
            )
        )

    return ClusteringResult(
        window_start=window_start,
        window_end=window_end,
        total_watch_events=len(recent),
        videos=videos,
        clusters=tuple(clusters),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Cluster local YouTube Takeout titles")
    parser.add_argument("archive", type=Path, help="YouTube Takeout ZIP path")
    parser.add_argument("--k", type=int, required=True, help="Number of clusters to test")
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
        print(f"  주요 단어: {', '.join(cluster.top_terms)}")
        for title in cluster.representative_titles:
            print(f"  대표 제목: {title}")


if __name__ == "__main__":
    main()
