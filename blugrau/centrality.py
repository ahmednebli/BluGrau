"""Node centralities used as topological roles and for the topology evaluation.

A graph is a weighted, directed NetworkX graph built from the adjacency matrix. Negative
entries and the diagonal are treated as absent edges. For path-based measures (betweenness,
closeness, harmonic, load) the connection weight is a *strength*, so the path length of an
edge is its inverse, ``1 / weight``.

Every measure is min-max normalized across the nodes of the graph, so that measures on
different scales contribute comparably to the blueprint (the paper does not state a scaling).
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from typing import Dict, Iterable, List, Sequence

import networkx as nx
import numpy as np

EPS = 1e-12


def _graph(A: np.ndarray) -> nx.DiGraph:
    A = np.clip(A, 0.0, None).copy()
    np.fill_diagonal(A, 0.0)
    G = nx.from_numpy_array(A, create_using=nx.DiGraph)
    for _, _, d in G.edges(data=True):
        d["dist"] = 1.0 / d["weight"]
    return G


def _as_vector(values: Dict[int, float], n: int) -> np.ndarray:
    return np.array([values[i] for i in range(n)], dtype=np.float64)


def _spectral_radius(A: np.ndarray) -> float:
    return float(np.max(np.abs(np.linalg.eigvals(A))))


def _strength(G, A):
    return _as_vector(dict(G.degree(weight="weight")), len(A))


def _betweenness(G, A):
    return _as_vector(nx.betweenness_centrality(G, weight="dist"), len(A))


def _closeness(G, A):
    return _as_vector(nx.closeness_centrality(G, distance="dist"), len(A))


def _harmonic(G, A):
    return _as_vector(nx.harmonic_centrality(G, distance="dist"), len(A))


def _eigenvector(G, A):
    return np.abs(_as_vector(nx.eigenvector_centrality_numpy(G, weight="weight"), len(A)))


def _pagerank(G, A):
    return _as_vector(nx.pagerank(G, weight="weight"), len(A))


def _hits_hub(G, A):
    # HITS hub scores = principal left singular vector of the adjacency matrix
    u, _, _ = np.linalg.svd(np.clip(A, 0, None))
    return np.abs(u[:, 0])


def _hits_authority(G, A):
    _, _, vt = np.linalg.svd(np.clip(A, 0, None))
    return np.abs(vt[0])


def _katz(G, A):
    rho = _spectral_radius(np.clip(A, 0, None))
    alpha = 0.5 / rho if rho > 0 else 0.1      # alpha < 1 / spectral radius guarantees convergence
    return _as_vector(nx.katz_centrality_numpy(G, alpha=alpha, weight="weight"), len(A))


def _load(G, A):
    return _as_vector(nx.load_centrality(G, weight="dist"), len(A))


MEASURES = {
    "strength": _strength, "betweenness": _betweenness, "closeness": _closeness,
    "harmonic": _harmonic, "eigenvector": _eigenvector, "pagerank": _pagerank,
    "hits_hub": _hits_hub, "hits_authority": _hits_authority, "katz": _katz, "load": _load,
}


def _minmax(v: np.ndarray) -> np.ndarray:
    return (v - v.min()) / (v.max() - v.min() + EPS)


def graph_centralities(A: np.ndarray, names: Sequence[str]) -> np.ndarray:
    """Return an (N, len(names)) array of per-graph min-max normalized centralities."""
    G = _graph(A)
    cols = [_minmax(MEASURES[name](G, np.clip(A, 0, None))) for name in names]
    return np.stack(cols, axis=1)


def _worker(args):
    return graph_centralities(*args)


def compute_centralities(adjacency: np.ndarray, names: Iterable[str], n_jobs: int = 1) -> np.ndarray:
    """Centralities for a stack of graphs: (G, N, K)."""
    names = list(names)
    unknown = [n for n in names if n not in MEASURES]
    if unknown:
        raise ValueError(f"unknown centralities {unknown}; available: {sorted(MEASURES)}")
    jobs: List = [(A, names) for A in adjacency]
    if n_jobs > 1:
        with ProcessPoolExecutor(max_workers=n_jobs) as pool:
            out = list(pool.map(_worker, jobs))
    else:
        out = [_worker(j) for j in jobs]
    return np.stack(out, axis=0)
