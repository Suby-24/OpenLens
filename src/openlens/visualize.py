"""Create an offline, local HTML report for a clustering experiment."""

from __future__ import annotations

import argparse
from html import escape
from math import sqrt
from pathlib import Path

import plotly.graph_objects as go
from plotly.colors import qualitative
from sklearn.decomposition import PCA

from .cluster import ClusteringResult, cluster_watch_events
from .takeout import read_watch_events


def project_titles(result: ClusteringResult) -> list[tuple[float, float]]:
    """Project the same embedding vectors used by K-Means to 2D for display."""
    coordinates = PCA(n_components=2).fit_transform(result.embedding_vectors)
    return [(float(x), float(y)) for x, y in coordinates]


def write_cluster_report(result: ClusteringResult, output: Path) -> Path:
    """Save an offline HTML report; the file contains private video titles."""
    coordinates = project_titles(result)
    palette = qualitative.Plotly

    scatter = go.Figure()
    for cluster in result.clusters:
        members = [
            (video, coordinates[index])
            for index, video in enumerate(result.videos)
            if video.cluster_id == cluster.cluster_id
        ]
        scatter.add_trace(
            go.Scattergl(
                x=[point[0] for _, point in members],
                y=[point[1] for _, point in members],
                mode="markers",
                name=f"그룹 {cluster.cluster_id}",
                marker={
                    "color": palette[cluster.cluster_id % len(palette)],
                    "size": [min(26, 8 + 3 * sqrt(video.watch_count)) for video, _ in members],
                    "opacity": 0.76,
                },
                customdata=[
                    [escape(video.title, quote=True), video.watch_count]
                    for video, _ in members
                ],
                hovertemplate=(
                    "%{customdata[0]}<br>시청 %{customdata[1]}회"
                    "<extra></extra>"
                ),
            )
        )
    scatter.update_layout(
        title="영상 제목 분포",
        xaxis_title="PCA 축 1",
        yaxis_title="PCA 축 2",
        legend_title="K-Means 그룹",
        height=620,
        hovermode="closest",
    )

    labels = [f"그룹 {cluster.cluster_id}" for cluster in result.clusters]
    bars = go.Figure()
    bars.add_trace(
        go.Bar(name="고유 영상", x=labels, y=[c.unique_videos for c in result.clusters])
    )
    bars.add_trace(
        go.Bar(name="시청 횟수", x=labels, y=[c.watch_events for c in result.clusters])
    )
    bars.update_layout(
        title="그룹별 콘텐츠와 소비량",
        xaxis_title="K-Means 그룹 번호",
        yaxis_title="개수 / 횟수",
        barmode="group",
        height=420,
    )

    scatter_html = scatter.to_html(full_html=False, include_plotlyjs=True)
    bars_html = bars.to_html(full_html=False, include_plotlyjs=False)
    document = f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>OpenLens 클러스터링 결과</title>
  <style>
    body {{ max-width: 1200px; margin: 32px auto; padding: 0 18px;
           font-family: system-ui, sans-serif; color: #202124; }}
    .note {{ color: #555; line-height: 1.5; }}
  </style>
</head>
<body>
  <h1>OpenLens 클러스터링 결과</h1>
  <p>기간: {result.window_start.date()} ~ {result.window_end.date()} ·
     고유 영상: {len(result.videos):,}개 · 시청 이벤트: {result.total_watch_events:,}건</p>
  <p class="note">각 점은 고유 영상 하나이며 크기는 시청 횟수입니다. 점에 마우스를 올리면
     제목과 횟수를 볼 수 있습니다. 점의 위치는 클러스터링에 사용한 EmbeddingGemma 벡터를
     PCA로 2차원에 투영한 것으로,
     거리가 실제 K-Means의 원래 고차원 거리와 정확히 같지는 않습니다.
     그룹 번호는 확정된 주제명이 아닙니다.</p>
  {scatter_html}
  {bars_html}
</body>
</html>
"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(document, encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a private local cluster report")
    parser.add_argument("archive", type=Path, help="YouTube Takeout ZIP or merged CSV path")
    parser.add_argument("--k", type=int, default=11, help="Number of clusters (default: 11)")
    parser.add_argument("--days", type=int, default=90, help="Look-back days (default: 90)")
    args = parser.parse_args()

    try:
        result = cluster_watch_events(read_watch_events(args.archive), k=args.k, days=args.days)
    except ValueError as exc:
        parser.error(str(exc))

    project_root = Path(__file__).resolve().parents[2]
    output = project_root / "artifacts" / f"clusters-k{args.k}-days{args.days}.html"
    write_cluster_report(result, output)
    print(f"로컬 보고서: {output}")
    print("이 HTML에는 개인 시청 제목이 포함됩니다. GitHub에 올리거나 공유하지 마세요.")


if __name__ == "__main__":
    main()
