"""
Step 4b — D-Wave Leap Hybrid CQM subsolver for static cluster TSPs.

Matches the encoding used in ``code/refined/solvers/tdtsp_cluster_dwave.py``
and ``code/autonomous/tdtsp_dwave.py``:

    binary x[i,t] = 1 iff city i at position t
    objective: dm[i][j] * x[i,t] * x[j,(t+1) mod n]
    hard constraints: each city once, each position once
    sampler: LeapHybridCQMSampler

Credentials: ``DWAVE_API_TOKEN`` (repo standard) or ``DWAVE_API_KEY``
(loaded from ``.env`` / ``env`` at the repo root).
"""
from __future__ import annotations

import os
import sys
from io import StringIO
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from dotenv import load_dotenv

try:
    from .qubo_tsp import decode_tour, tour_cost
except ImportError:  # `python subsolver_dwave.py`
    from qubo_tsp import decode_tour, tour_cost  # type: ignore

ArrayLike = Union[Sequence[Sequence[float]], np.ndarray]

REPO_ROOT = Path(__file__).resolve().parents[2]
_sampler_cache: Dict[str, object] = {}


def _load_token() -> Optional[str]:
    load_dotenv(REPO_ROOT / ".env")
    load_dotenv(REPO_ROOT / "env")
    return os.getenv("DWAVE_API_TOKEN") or os.getenv("DWAVE_API_KEY")


def build_tsp_cqm(distance_matrix: ArrayLike):
    """
    Build a ConstrainedQuadraticModel for a static TSP.

    Same formulation as ``tdtsp_cluster_dwave._solve_cluster_cqm``.
    """
    import dimod

    dm = np.asarray(distance_matrix, dtype=float)
    if dm.ndim != 2 or dm.shape[0] != dm.shape[1]:
        raise ValueError(f"distance_matrix must be square; got {dm.shape}")
    n = int(dm.shape[0])
    if n < 2:
        raise ValueError("TSP-CQM requires at least 2 cities")

    cqm = dimod.ConstrainedQuadraticModel()
    x = [[dimod.Binary(f"x_{i}_{t}") for t in range(n)] for i in range(n)]

    objective = 0
    for t in range(n):
        nt = (t + 1) % n
        for i in range(n):
            for j in range(n):
                if i != j:
                    objective += float(dm[i, j]) * x[i][t] * x[j][nt]
    cqm.set_objective(objective)

    for i in range(n):
        cqm.add_constraint(
            sum(x[i][t] for t in range(n)) == 1,
            label=f"visit_city_{i}",
        )
    for t in range(n):
        cqm.add_constraint(
            sum(x[i][t] for i in range(n)) == 1,
            label=f"position_{t}",
        )
    return cqm, n


def solve_tsp_dwave(
    distance_matrix: ArrayLike,
    *,
    time_limit: float = 5.0,
    api_token: Optional[str] = None,
) -> Dict:
    """
    Solve a static cluster TSP with Leap Hybrid CQM.

    Returns
    -------
    dict with keys
        ``tour``, ``cost``, ``feasible``, ``backend``, ``time_limit``.
    """
    from dwave.system import LeapHybridCQMSampler

    dm = np.asarray(distance_matrix, dtype=float)
    n = int(dm.shape[0])
    if n == 2:
        tour = [0, 1]
        return {
            "tour": tour,
            "cost": tour_cost(tour, dm),
            "feasible": True,
            "backend": "DWave-HybridCQM",
            "time_limit": 0.0,
        }

    token = api_token or _load_token()
    if not token:
        raise ValueError(
            "D-Wave token not found. Set DWAVE_API_TOKEN or DWAVE_API_KEY "
            "in .env or env at the repo root."
        )

    cqm, n = build_tsp_cqm(dm)

    sampler = _sampler_cache.get("cqm_sampler")
    if sampler is None:
        sampler = LeapHybridCQMSampler(token=token)
        _sampler_cache["cqm_sampler"] = sampler

    old_stdout = sys.stdout
    sys.stdout = StringIO()
    try:
        sampleset = sampler.sample_cqm(cqm, time_limit=time_limit)
    finally:
        sys.stdout = old_stdout

    feasible = sampleset.filter(lambda s: s.is_feasible)
    used_feasible = len(feasible) > 0
    sample = feasible.first.sample if used_feasible else sampleset.first.sample

    tour = decode_tour(dict(sample), n)
    if tour is None:
        # Match cluster-dwave fallback: identity order in local indices
        tour = list(range(n))
        return {
            "tour": tour,
            "cost": tour_cost(tour, dm),
            "feasible": False,
            "backend": "DWave-HybridCQM",
            "time_limit": float(time_limit),
        }

    return {
        "tour": tour,
        "cost": tour_cost(tour, dm),
        "feasible": used_feasible,
        "backend": "DWave-HybridCQM",
        "time_limit": float(time_limit),
    }


def _self_test(live: bool = True) -> None:
    print("Step 4b self-test: D-Wave Hybrid CQM TSP subsolver")
    print("-" * 60)

    dm = np.array([
        [0, 1, 5, 4],
        [4, 0, 5, 1],
        [2, 5, 0, 3],
        [5, 2, 3, 0],
    ], dtype=float)
    known = [0, 1, 3, 2]
    known_cost = tour_cost(known, dm)

    cqm, n = build_tsp_cqm(dm)
    assert n == 4
    assert len(cqm.constraints) == 8  # 4 city + 4 position
    print(f"  CQM built: n={n}, constraints={len(cqm.constraints)}, "
          f"vars={len(cqm.variables)}")

    token = _load_token()
    if not token:
        print("  SKIP live solve: no DWAVE_API_TOKEN / DWAVE_API_KEY")
        print("  OK (structure only)")
        return
    if not live:
        print("  SKIP live solve: live=False")
        print("  OK (structure only)")
        return

    print("  calling LeapHybridCQMSampler (small 4-city, time_limit=5s)...")
    result = solve_tsp_dwave(dm, time_limit=5.0)
    print(f"  known tour {known} cost={known_cost}")
    print(f"  D-Wave     {result['tour']} cost={result['cost']} "
          f"feasible={result['feasible']}")
    assert result["tour"] is not None
    assert len(result["tour"]) == 4
    print("  OK")


if __name__ == "__main__":
    _self_test(live=True)
