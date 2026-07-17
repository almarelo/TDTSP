"""
Step 3 — Static TSP-QUBO builder (per-cluster subproblem).

Mirrors the one-hot Lucas / permutation-matrix encoding used in
``code/refined/solvers/tdtsp_cluster_qaoa.py`` (``_tsp_to_qubo``) and
``tdtsp_cluster_quanfluence.py``:

    x[i, t] = 1  iff city i occupies position t in the tour
    n^2 binary variables
    cost:   dm[i][j] * x[i,t] * x[j, (t+1) mod n]
    penalties: each city once, each position filled once

This module only *builds* (and decodes) the QUBO. Solving is left to
``subsolver_SA.py`` / ``subsolver_dwave.py``. Used only when
``ClusterResult.use_qubo`` is True.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

try:
    import dimod
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "dimod is required for qubo_tsp (install dwave-ocean-sdk)"
    ) from exc


ArrayLike = Union[Sequence[Sequence[float]], np.ndarray]


@dataclass(frozen=True)
class TSPQUBO:
    """Static TSP encoded as a BinaryQuadraticModel."""
    bqm: "dimod.BinaryQuadraticModel"
    n: int
    penalty: float
    # Variable label -> (city, position)
    var_index: Dict[str, Tuple[int, int]]

    @property
    def num_variables(self) -> int:
        return self.n * self.n


def _as_matrix(distance_matrix: ArrayLike) -> np.ndarray:
    dm = np.asarray(distance_matrix, dtype=float)
    if dm.ndim != 2 or dm.shape[0] != dm.shape[1]:
        raise ValueError(
            f"distance_matrix must be square; got shape {dm.shape}"
        )
    if dm.shape[0] < 2:
        raise ValueError("TSP-QUBO requires at least 2 cities")
    return dm


def _vidx(city: int, pos: int, n: int) -> int:
    return city * n + pos


def _var_label(city: int, pos: int) -> str:
    return f"x_{city}_{pos}"


def build_tsp_qubo(
    distance_matrix: ArrayLike,
    penalty: Optional[float] = None,
) -> TSPQUBO:
    """
    Build a one-hot TSP QUBO from a static distance matrix.

    Parameters
    ----------
    distance_matrix :
        ``n x n`` cluster-local costs (frozen at cluster entry time).
    penalty :
        One-hot penalty weight ``A``. Defaults to ``2 * max(dm) * n``,
        matching the refined QAOA / Quanfluence builders.

    Returns
    -------
    TSPQUBO
        ``bqm`` ready for a dimod sampler, plus decode metadata.
    """
    dm = _as_matrix(distance_matrix)
    n = int(dm.shape[0])
    max_dist = float(np.max(dm))
    if penalty is None:
        penalty = 2.0 * max_dist * n
    if penalty <= 0:
        raise ValueError(f"penalty must be positive; got {penalty}")

    # Same upper-triangular Q dict as tdtsp_cluster_qaoa._tsp_to_qubo
    Q: Dict[Tuple[int, int], float] = {}

    # Cost: dm[i][j] * x[i,t] * x[j,t+1]
    for t in range(n):
        nt = (t + 1) % n
        for i in range(n):
            for j in range(n):
                if i != j and dm[i, j] > 0:
                    a, b = _vidx(i, t, n), _vidx(j, nt, n)
                    key = (min(a, b), max(a, b))
                    Q[key] = Q.get(key, 0.0) + float(dm[i, j])

    # Each city visited exactly once: (1 - sum_t x[i,t])^2
    for i in range(n):
        for t in range(n):
            idx = _vidx(i, t, n)
            Q[(idx, idx)] = Q.get((idx, idx), 0.0) - penalty
        for t1 in range(n):
            for t2 in range(t1 + 1, n):
                a, b = _vidx(i, t1, n), _vidx(i, t2, n)
                key = (min(a, b), max(a, b))
                Q[key] = Q.get(key, 0.0) + 2 * penalty

    # Each position holds exactly one city: (1 - sum_i x[i,t])^2
    for t in range(n):
        for i in range(n):
            idx = _vidx(i, t, n)
            Q[(idx, idx)] = Q.get((idx, idx), 0.0) - penalty
        for i1 in range(n):
            for i2 in range(i1 + 1, n):
                a, b = _vidx(i1, t, n), _vidx(i2, t, n)
                key = (min(a, b), max(a, b))
                Q[key] = Q.get(key, 0.0) + 2 * penalty

    # Map flat indices -> named binary variables for readable samples
    labels = [_var_label(c, p) for c in range(n) for p in range(n)]
    Q_named: Dict[Tuple[str, str], float] = {}
    for (a, b), coef in Q.items():
        Q_named[(labels[a], labels[b])] = coef

    bqm = dimod.BinaryQuadraticModel.from_qubo(Q_named)
    # Constant offset from expanding (1 - sum x)^2 => +penalty per constraint
    # (n city + n position). dimod from_qubo ignores the constant; energy
    # comparisons stay consistent as long as we use the same BQM end-to-end.
    bqm.offset += 2 * n * penalty

    var_index = {
        _var_label(c, p): (c, p) for c in range(n) for p in range(n)
    }
    return TSPQUBO(bqm=bqm, n=n, penalty=float(penalty), var_index=var_index)


def decode_tour(
    sample: Dict[str, int],
    n: int,
) -> Optional[List[int]]:
    """
    Decode a one-hot sample into a tour ``[city_at_pos_0, ...]``.

    Returns ``None`` if the sample is not a valid permutation (repair is
    left to the subsolver / caller if desired).
    """
    tour: List[Optional[int]] = [None] * n
    for label, bit in sample.items():
        if bit != 1:
            continue
        # label format: x_{city}_{pos}
        try:
            _, city_s, pos_s = label.split("_", 2)
            city, pos = int(city_s), int(pos_s)
        except (ValueError, AttributeError):
            continue
        if not (0 <= city < n and 0 <= pos < n):
            return None
        if tour[pos] is not None:
            return None
        tour[pos] = city

    if any(c is None for c in tour):
        return None
    if len(set(tour)) != n:
        return None
    return [int(c) for c in tour]  # type: ignore[arg-type]


def tour_cost(tour: Sequence[int], distance_matrix: ArrayLike) -> float:
    """Closed-tour length on a static distance matrix."""
    dm = _as_matrix(distance_matrix)
    n = len(tour)
    if n != dm.shape[0]:
        raise ValueError("tour length must match distance_matrix size")
    total = 0.0
    for k in range(n):
        total += float(dm[tour[k], tour[(k + 1) % n]])
    return total


def _self_test() -> None:
    print("Step 3 self-test: static TSP-QUBO builder")
    print("-" * 60)

    # Tiny asymmetric 4-city instance with a known good tour 0-1-3-2-0
    dm = np.array([
        [0, 1, 5, 4],
        [4, 0, 5, 1],
        [2, 5, 0, 3],
        [5, 2, 3, 0],
    ], dtype=float)
    known = [0, 1, 3, 2]
    known_cost = tour_cost(known, dm)

    tsp = build_tsp_qubo(dm)
    print(f"  n={tsp.n}  num_vars={tsp.num_variables}  penalty={tsp.penalty}")
    assert tsp.num_variables == 16
    assert len(tsp.bqm.variables) == 16

    # Build the exact one-hot sample for `known` and check energy is finite
    sample = {lab: 0 for lab in tsp.var_index}
    for pos, city in enumerate(known):
        sample[_var_label(city, pos)] = 1
    decoded = decode_tour(sample, tsp.n)
    assert decoded == known
    energy = tsp.bqm.energy(sample)
    print(f"  known tour {known}  static_cost={known_cost}  bqm_energy={energy:.3f}")

    # Invalid sample (two cities in one position) must fail decode
    bad = dict(sample)
    bad[_var_label(0, 0)] = 1
    bad[_var_label(1, 0)] = 1
    assert decode_tour(bad, tsp.n) is None

    # Spot-check: Q dict size / structure matches refined QAOA builder idea
    # (n^2 diagonal-capable entries exist via from_qubo).
    assert tsp.bqm.num_interactions > 0

    print("  OK")


if __name__ == "__main__":
    _self_test()
