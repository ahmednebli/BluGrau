"""Topology comparison between original and generated graphs (Figure 2 of the paper)."""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from .centrality import compute_centralities
from .config import TOPOLOGY_MEASURES


def graph_means(adjacency: np.ndarray, measures: Sequence[str] = TOPOLOGY_MEASURES, n_jobs: int = 1) -> np.ndarray:
    """Mean over nodes of every (per-graph min-max normalized) centrality: (G, len(measures))."""
    return compute_centralities(adjacency, measures, n_jobs=n_jobs).mean(axis=1)


def compare(original: np.ndarray, generated: np.ndarray, measures: Sequence[str] = TOPOLOGY_MEASURES,
            n_jobs: int = 1) -> pd.DataFrame:
    """Mean +- standard error over graphs for both groups and the relative difference."""
    a, b = graph_means(original, measures, n_jobs), graph_means(generated, measures, n_jobs)
    rows = []
    for k, m in enumerate(measures):
        mo, mg = a[:, k].mean(), b[:, k].mean()
        rows.append({
            "measure": m,
            "original_mean": mo, "original_se": a[:, k].std(ddof=1) / np.sqrt(len(a)),
            "generated_mean": mg, "generated_se": b[:, k].std(ddof=1) / np.sqrt(len(b)),
            "relative_diff_pct": 100.0 * (mg - mo) / mo if mo != 0 else float("nan"),
            "n_original": len(a), "n_generated": len(b),
        })
    return pd.DataFrame(rows)
