"""Compare K-Means choices on one shared set of title embeddings."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import timedelta
from html import escape
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from .cluster import embed_titles
from .takeout import read_watch_events


def compare(source: Path, output: Path, *, days: int = 90, sample_size: int = 1000) -> None:
    # Select recent unique videos and embed their titles once for every k.
    events = read_watch_events(source)
    if not events or days < 1:
        raise ValueError("Need watch events and a positive day window")
    end = max(event.watched_at for event in events)
    recent = [event for event in events if event.watched_at >= end - timedelta(days=days)]
    titles_by_id = {event.video_id: event.title for event in sorted(recent, key=lambda e: e.watched_at)}
    ids = sorted(titles_by_id)
    titles = [titles_by_id[video_id] for video_id in ids]
    if len(set(titles)) < 12:
        raise ValueError("Need at least 12 distinct titles")
    matrix = embed_titles(titles)
    coordinates = PCA(n_components=2, random_state=42).fit_transform(matrix)
    watches = Counter(event.video_id for event in recent)
    # Reuse one PCA projection so the k=5..12 plots stay comparable.
    charts = make_subplots(rows=2, cols=4, subplot_titles=[f"k={k}" for k in range(5, 13)])
    focus = go.Figure()
    metrics: list[tuple[int, float, float]] = []
    palette = ["#4477AA", "#EE6677", "#228833", "#CCBB44", "#66CCEE", "#AA3377", "#BBBBBB", "#332288", "#44AA99", "#DDCC77", "#AA4499", "#117733"]
    # Fit each k independently and record inertia and silhouette scores.
    for k in range(5, 13):
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(matrix)
        silhouette = silhouette_score(matrix, labels, sample_size=min(sample_size, len(ids)), random_state=42)
        metrics.append((k, float(model.inertia_), float(silhouette)))
        row, col = (k - 5) // 4 + 1, (k - 5) % 4 + 1
        charts.add_trace(go.Scattergl(
            x=coordinates[:, 0], y=coordinates[:, 1], mode="markers", showlegend=False,
            marker={"color": [palette[int(label) % len(palette)] for label in labels], "size": 5, "opacity": 0.65},
            customdata=[[escape(title, quote=True), watches[video_id], int(label)] for video_id, title, label in zip(ids, titles, labels)],
            hovertemplate="%{customdata[0]}<br>시청 %{customdata[1]}회 · 군집 %{customdata[2]}<extra></extra>",
        ), row=row, col=col)
        # Build a larger view for the currently selected k.
        if k == 11:
            for cluster_id in range(k):
                members = [i for i, label in enumerate(labels) if label == cluster_id]
                focus.add_trace(go.Scattergl(
                    x=coordinates[members, 0], y=coordinates[members, 1],
                    mode="markers", name=f"군집 {cluster_id} · {len(members):,}개",
                    marker={"color": palette[cluster_id], "size": 8, "opacity": 0.7},
                    customdata=[[escape(titles[i], quote=True), watches[ids[i]]] for i in members],
                    hovertemplate="%{customdata[0]}<br>시청 %{customdata[1]}회<extra></extra>",
                ))
    charts.update_layout(title="k=5~12 군집 분포 · 동일한 PCA 좌표", height=1050)
    focus.update_layout(title="k=11 상세 분포 · 점에 마우스를 올려 제목 확인", height=850,
                        xaxis_title="PCA 축 1", yaxis_title="PCA 축 2", legend_title="군집과 영상 수",
                        hovermode="closest")
    scores = go.Figure()
    # Show both the elbow curve and the gain from adding one cluster.
    scores.add_trace(go.Scatter(x=[m[0] for m in metrics], y=[m[1] for m in metrics], mode="lines+markers", name="Inertia (엘보우)", yaxis="y1"))
    scores.add_trace(go.Scatter(x=[m[0] for m in metrics], y=[m[2] for m in metrics], mode="lines+markers", name="실루엣", yaxis="y2"))
    scores.update_layout(title="엘보우와 실루엣", xaxis={"title": "k", "dtick": 1}, yaxis={"title": "Inertia"}, yaxis2={"title": "실루엣", "overlaying": "y", "side": "right"}, height=450)
    gains = [metrics[i-1][1] - metrics[i][1] for i in range(1, len(metrics))]
    gain_chart = go.Figure(go.Bar(x=[m[0] for m in metrics[1:]], y=gains,
                                  text=[f"{gain:.1f}" for gain in gains], textposition="outside"))
    gain_chart.update_layout(title="k를 1 늘렸을 때 Inertia 감소량", xaxis={"title": "새 k", "dtick": 1},
                             yaxis_title="감소량", height=340)
    rows = "".join(f"<tr><td>{k}</td><td>{inertia:.2f}</td><td>{sil:.4f}</td><td>{(1-inertia/metrics[0][1])*100:.1f}%</td></tr>" for k, inertia, sil in metrics)
    # Build an interactive report whose tooltips contain private titles.
    html = f"""<!doctype html><html lang="ko"><meta charset="utf-8"><title>OpenLens k 비교</title>
<style>body{{font-family:system-ui,sans-serif;max-width:1450px;margin:auto;padding:24px}}table{{border-collapse:collapse}}td,th{{padding:7px 16px;border:1px solid #ccc}}</style>
<h1>클러스터 수 비교</h1><p>기간 {end.date()} 기준 최근 {days}일 · 고유 영상 {len(ids):,}개.</p>
<p>실루엣은 원래 임베딩 공간에서 고정된 최대 {sample_size:,}개 영상으로 계산했습니다. 점 분포는 비교를 위해 모든 k에서 같은 PCA 투영을 씁니다. 색과 군집 번호는 k 간 직접 대응하지 않습니다.</p>
<p>Inertia는 k가 커지면 계속 감소합니다. 이 데이터에는 뚜렷한 단일 꺾임이 없으므로 아래 감소량과 실루엣, 실제 군집 내용을 함께 보세요. PCA에서 점이 겹쳐 보여도 원래 고차원 공간에서는 다를 수 있습니다.</p>
{scores.to_html(full_html=False, include_plotlyjs=True)}
{gain_chart.to_html(full_html=False, include_plotlyjs=False)}
<table><tr><th>k</th><th>Inertia</th><th>실루엣</th><th>k=5 대비 Inertia 감소</th></tr>{rows}</table>
{focus.to_html(full_html=False, include_plotlyjs=False)}
{charts.to_html(full_html=False, include_plotlyjs=False)}</html>"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")
    for k, inertia, sil in metrics:
        print(f"k={k}: inertia={inertia:.2f}, silhouette={sil:.4f}")
    print(f"로컬 보고서: {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare k=5..12 with one set of embeddings")
    parser.add_argument("source", type=Path)
    parser.add_argument("--days", type=int, default=90)
    parser.add_argument("--silhouette-sample", type=int, default=1000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.silhouette_sample < 2:
        parser.error("--silhouette-sample must be at least 2")
    compare(args.source, args.output, days=args.days, sample_size=args.silhouette_sample)


if __name__ == "__main__":
    main()
