"""Evaluate K-Means on the combined, source-aware labeled training CSV."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from html import escape
from pathlib import Path

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score

from .cluster import embed_titles


def compare(source: Path, output: Path) -> None:
    # Keep only human-verified training rows from the combined CSV.
    with source.open(newline="", encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))
    required = {"video_id", "title", "category_label", "source", "label_status"}
    if not rows or not required <= rows[0].keys():
        raise ValueError("Expected a combined training CSV with source and labels")
    rows = [row for row in rows if row["label_status"] == "verified" and row["title"] and row["category_label"]]
    if len(rows) < 12 or len({row["video_id"] for row in rows}) != len(rows):
        raise ValueError("Need at least 12 unique verified videos")
    titles = [row["title"] for row in rows]
    true_labels = [row["category_label"] for row in rows]
    # Embed titles once, then use the same coordinates in every plot.
    matrix = embed_titles(titles)
    points = PCA(n_components=2, random_state=42).fit_transform(matrix)
    palette = ["#4477AA", "#EE6677", "#228833", "#CCBB44", "#66CCEE", "#AA3377", "#BBBBBB", "#332288", "#44AA99", "#DDCC77", "#AA4499", "#117733"]
    panels = make_subplots(rows=2, cols=4, subplot_titles=[f"k={k}" for k in range(5, 13)])
    metrics = []
    cluster_details = {}
    # Compare cluster separation with agreement against reviewed categories.
    for k in range(5, 13):
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(matrix)
        silhouette = float(silhouette_score(matrix, labels))
        ari = float(adjusted_rand_score(true_labels, labels))
        nmi = float(normalized_mutual_info_score(true_labels, labels))
        metrics.append((k, float(model.inertia_), silhouette, ari, nmi))
        cluster_details[k] = [Counter(true_labels[i] for i, label in enumerate(labels) if label == cluster)
                              for cluster in range(k)]
        subplot_row, subplot_col = (k - 5) // 4 + 1, (k - 5) % 4 + 1
        panels.add_trace(go.Scattergl(
            x=points[:, 0], y=points[:, 1], mode="markers", showlegend=False,
            marker={"color": [palette[int(label) % len(palette)] for label in labels], "size": 7, "opacity": 0.75},
            customdata=[[escape(row["title"], quote=True), row["source"], row["category_label"], int(label)]
                        for row, label in zip(rows, labels)],
            hovertemplate="%{customdata[0]}<br>%{customdata[1]} · 정답 %{customdata[2]} · 군집 %{customdata[3]}<extra></extra>",
        ), row=subplot_row, col=subplot_col)
    panels.update_layout(title="k=5~12 군집 분포 · 동일한 PCA 좌표", height=950)
    elbow = go.Figure(go.Scatter(x=[m[0] for m in metrics], y=[m[1] for m in metrics], mode="lines+markers"))
    elbow.update_layout(title="엘보우 · Inertia", xaxis={"title": "k", "dtick": 1}, yaxis_title="Inertia", height=380)
    quality = go.Figure()
    for index, name in ((2, "실루엣"), (3, "ARI"), (4, "NMI")):
        quality.add_trace(go.Scatter(x=[m[0] for m in metrics], y=[m[index] for m in metrics],
                                     mode="lines+markers", name=name))
    quality.update_layout(title="군집 분리도와 실제 카테고리 일치도", xaxis={"title": "k", "dtick": 1}, height=400)
    selected_k = 11
    # Show the reviewed category mix inside each selected cluster.
    detail = "".join(f"<tr><td>{i}</td><td>{sum(counts.values())}</td><td>{escape(', '.join(f'{name}: {count}' for name, count in counts.most_common()))}</td></tr>"
                     for i, counts in enumerate(cluster_details[selected_k]))
    score_rows = "".join(f"<tr><td>{k}</td><td>{inertia:.2f}</td><td>{sil:.4f}</td><td>{ari:.4f}</td><td>{nmi:.4f}</td></tr>"
                         for k, inertia, sil, ari, nmi in metrics)
    source_counts = Counter(row["source"] for row in rows)
    # Build an HTML report that includes title-level training examples.
    html = f"""<!doctype html><html lang="ko"><meta charset="utf-8"><title>OpenLens 통합 학습 데이터 군집 평가</title>
<style>body{{font-family:system-ui,sans-serif;max-width:1450px;margin:auto;padding:24px}}table{{border-collapse:collapse}}td,th{{padding:7px 16px;border:1px solid #ccc;text-align:left}}</style>
<h1>개인·외부 통합 학습 데이터 군집 평가</h1><p>검증된 영상 {len(rows)}개 · 개인 {source_counts['personal']}개 · 외부 {source_counts['external']}개.</p>
<p>실루엣은 원래 임베딩 공간의 전체 영상으로 계산합니다. ARI·NMI는 사람이 검증한 카테고리와 군집의 일치도를 나타내며, 1에 가까울수록 일치합니다. PCA 점 분포는 표시용입니다.</p>
{elbow.to_html(full_html=False, include_plotlyjs=True)}{quality.to_html(full_html=False, include_plotlyjs=False)}
<table><tr><th>k</th><th>Inertia</th><th>실루엣</th><th>ARI</th><th>NMI</th></tr>{score_rows}</table>
<h2>선택한 k={selected_k}의 군집별 실제 레이블</h2><table><tr><th>군집</th><th>영상 수</th><th>검증된 카테고리</th></tr>{detail}</table>
{panels.to_html(full_html=False, include_plotlyjs=False)}</html>"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")
    for k, inertia, sil, ari, nmi in metrics:
        print(f"k={k}: inertia={inertia:.2f}, silhouette={sil:.4f}, ARI={ari:.4f}, NMI={nmi:.4f}")
    print(f"선택한 k={selected_k}")
    for i, counts in enumerate(cluster_details[selected_k]):
        print(f"  군집 {i}: {dict(counts.most_common())}")
    print(f"로컬 보고서: {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare k=5..12 on a combined labeled training CSV")
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    compare(args.source, args.output)


if __name__ == "__main__":
    main()
