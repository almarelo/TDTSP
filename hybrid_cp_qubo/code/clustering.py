"""
Step 2 — Cluster-first decomposition.

Partitions Melgarejo noTW visits into clusters for Approach 2:

* If ``p <= max_cluster_size``: one cluster with all visits, ``use_qubo=False``.
  The CP master will solve the full time-dependent instance (FIFO / Dij(t)).
* If ``p > max_cluster_size``: K-means on an MDS embedding of the static
  travel-time slice at ``t0``, ``use_qubo=True``. Each cluster is later
  solved as a static TSP-QUBO; the CP master owns global timing.

The depot (visit 0) is included in clustering like any other visit.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from sklearn.cluster import KMeans
from sklearn.manifold import MDS

try:
    from .td_data import TravelTimeMatrix, TDTSPInstance
except ImportError:  # `python clustering.py`
    from td_data import TravelTimeMatrix, TDTSPInstance


@dataclass(frozen=True)
class ClusterResult:
    """Outcome of the cluster-first decomposition."""
    clusters: List[List[int]]  # visit indices per cluster
    use_qubo: bool             # False => CP master solves the whole instance
    coords: np.ndarray         # (p, 2) embedding used for K-means
    t0: float                  # reference time for the static distance slice
    max_cluster_size: int

    @property
    def n_clusters(self) -> int:
        return len(self.clusters)

    @property
    def cluster_sizes(self) -> List[int]:
        return [len(c) for c in self.clusters]


def _static_distance_matrix(
    matrix: TravelTimeMatrix,
    instance: TDTSPInstance,
    t0: float,
) -> np.ndarray:
    """``p x p`` travel-time slice at ``t0`` over the instance visit vertices."""
    return matrix.slice_at(t0, instance.vertices)


def _embed_mds(distance_matrix: np.ndarray, random_state: int = 42) -> np.ndarray:
    """
    Embed a (possibly asymmetric) distance matrix into 2-D via classical MDS.

    Symmetrize first: MDS expects a symmetric dissimilarity matrix.
    """
    n = distance_matrix.shape[0]
    if n == 1:
        return np.zeros((1, 2), dtype=float)
    if n == 2:
        d = float(distance_matrix[0, 1] + distance_matrix[1, 0]) / 2.0
        return np.array([[0.0, 0.0], [max(d, 1e-6), 0.0]])

    dm = 0.5 * (distance_matrix + distance_matrix.T)
    np.fill_diagonal(dm, 0.0)
    # Guard against numerical junk
    dm = np.maximum(dm, 0.0)

    mds = MDS(
        n_components=2,
        metric="precomputed",
        init="classical_mds",
        random_state=random_state,
        normalized_stress="auto",
        max_iter=300,
        n_init=1,
    )
    return mds.fit_transform(dm)


def cluster_visits(
    instance: TDTSPInstance,
    matrix: TravelTimeMatrix,
    max_cluster_size: int = 10,
    t0: Optional[float] = None,
    random_state: int = 42,
) -> ClusterResult:
    """
    Cluster-first decomposition of an instance.

    Parameters
    ----------
    instance :
        Melgarejo noTW instance.
    matrix :
        Matching ``TravelTimeMatrix``.
    max_cluster_size :
        Target maximum visits per cluster (default 10).
    t0 :
        Reference departure time for the static distance slice.
        Defaults to ``instance.start_time``.
    random_state :
        Seed for MDS and K-means.

    Returns
    -------
    ClusterResult
        ``use_qubo`` is False when ``p <= max_cluster_size`` (CP-only path).
    """
    if max_cluster_size < 1:
        raise ValueError(f"max_cluster_size must be >= 1; got {max_cluster_size}")

    p = instance.p
    if t0 is None:
        t0 = instance.start_time

    dm = _static_distance_matrix(matrix, instance, t0)
    coords = _embed_mds(dm, random_state=random_state)

    # Small instance: CP master owns the full TD-TSP; no static QUBO.
    if p <= max_cluster_size:
        return ClusterResult(
            clusters=[list(range(p))],
            use_qubo=False,
            coords=coords,
            t0=float(t0),
            max_cluster_size=max_cluster_size,
        )

    n_clusters = int(np.ceil(p / max_cluster_size))
    n_clusters = min(n_clusters, p)

    km = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=10,
    )
    labels = km.fit_predict(coords)

    clusters: List[List[int]] = [[] for _ in range(n_clusters)]
    for visit_idx, lab in enumerate(labels):
        clusters[int(lab)].append(visit_idx)
    clusters = [c for c in clusters if c]
    clusters = _enforce_max_cluster_size(
        clusters, coords, max_cluster_size, random_state=random_state
    )

    return ClusterResult(
        clusters=clusters,
        use_qubo=True,
        coords=coords,
        t0=float(t0),
        max_cluster_size=max_cluster_size,
    )


def _enforce_max_cluster_size(
    clusters: List[List[int]],
    coords: np.ndarray,
    max_cluster_size: int,
    *,
    random_state: int = 42,
) -> List[List[int]]:
    """Split any cluster larger than ``max_cluster_size`` via local K-means."""
    out: List[List[int]] = []
    for cluster in clusters:
        queue = [list(cluster)]
        while queue:
            cur = queue.pop()
            if len(cur) <= max_cluster_size:
                out.append(cur)
                continue
            k = int(np.ceil(len(cur) / max_cluster_size))
            k = min(k, len(cur))
            sub_coords = coords[np.array(cur, dtype=int)]
            km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
            labs = km.fit_predict(sub_coords)
            parts: List[List[int]] = [[] for _ in range(k)]
            for local_i, lab in enumerate(labs):
                parts[int(lab)].append(cur[local_i])
            for part in parts:
                if not part:
                    continue
                if len(part) > max_cluster_size and len(part) < len(cur):
                    queue.append(part)
                elif len(part) > max_cluster_size:
                    # Degenerate geometry: fall back to contiguous chunks
                    for i in range(0, len(part), max_cluster_size):
                        out.append(part[i : i + max_cluster_size])
                else:
                    out.append(part)
    return out


def _self_test() -> None:
    print("Step 2 self-test: cluster-first decomposition")
    print("-" * 60)

    mx = TravelTimeMatrix.load("matrix00")

    # n=10 <= max 10 => CP-only, single cluster
    inst10 = TDTSPInstance.load(10, 1)
    r10 = cluster_visits(inst10, mx, max_cluster_size=10)
    print(f"  inst_10_1: p={inst10.p}  n_clusters={r10.n_clusters}  "
          f"sizes={r10.cluster_sizes}  use_qubo={r10.use_qubo}")
    assert r10.use_qubo is False
    assert r10.n_clusters == 1
    assert r10.clusters[0] == list(range(10))
    assert 0 in r10.clusters[0]  # depot included

    # n=20 > max 10 => QUBO path, 2 clusters
    inst20 = TDTSPInstance.load(20, 1)
    r20 = cluster_visits(inst20, mx, max_cluster_size=10)
    print(f"  inst_20_1: p={inst20.p}  n_clusters={r20.n_clusters}  "
          f"sizes={r20.cluster_sizes}  use_qubo={r20.use_qubo}")
    assert r20.use_qubo is True
    assert r20.n_clusters >= 2
    assert max(r20.cluster_sizes) <= 10
    flat = sorted(v for c in r20.clusters for v in c)
    assert flat == list(range(20))
    assert sum(0 in c for c in r20.clusters) == 1  # depot in exactly one

    # Hard cap: n=10 with max 5 must not produce a cluster larger than 5
    r10_mc5 = cluster_visits(inst10, mx, max_cluster_size=5)
    print(f"  inst_10_1 mc5: n_clusters={r10_mc5.n_clusters}  "
          f"sizes={r10_mc5.cluster_sizes}  use_qubo={r10_mc5.use_qubo}")
    assert r10_mc5.use_qubo is True
    assert max(r10_mc5.cluster_sizes) <= 5

    print("  OK")


if __name__ == "__main__":
    _self_test()
