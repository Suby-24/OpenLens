"""Select varied labeling examples from clustered embedding vectors."""

from __future__ import annotations

import numpy as np


def select_diverse_samples(members: list[int], vectors: np.ndarray, center_distances: np.ndarray,
                           video_ids: list[str], limit: int) -> list[int]:
    """Cover four center-distance bands, then spread selections within each band."""
    ranked = sorted(members, key=lambda i: (center_distances[i], video_ids[i]))
    order = {index: position for position, index in enumerate(ranked)}
    bands = [list(band) for band in np.array_split(ranked, 4)]
    selected: list[int] = []
    while len(selected) < min(limit, len(ranked)):
        progressed = False
        for band in bands:
            if not band or len(selected) >= limit:
                continue
            if selected:
                # Prefer candidates far from their nearest already selected neighbor.
                chosen = max(band, key=lambda i: (
                    min(float(np.linalg.norm(vectors[i] - vectors[j])) for j in selected),
                    -order[i],
                ))
            else:
                chosen = band[len(band) // 2]
            selected.append(chosen)
            band.remove(chosen)
            progressed = True
        if not progressed:
            break
    return selected
