"""
Step 4a — Local Simulated Annealing subsolver for static TSP-QUBOs.

Uses ``dwave.samplers.SimulatedAnnealingSampler`` on the BQM from
``qubo_tsp.build_tsp_qubo``. No cloud credentials required.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Union

import numpy as np

try:
    from .qubo_tsp import (
        TSPQUBO,
        build_tsp_qubo,
        decode_tour,
        tour_cost,
    )
except ImportError:  # `python subsolver_SA.py`
    from qubo_tsp import (  # type: ignore
        TSPQUBO,
        build_tsp_qubo,
        decode_tour,
        tour_cost,
    )

ArrayLike = Union[Sequence[Sequence[float]], np.ndarray]


def solve_tsp_sa(
    distance_matrix: ArrayLike,
    *,
    num_reads: int = 200,
    num_sweeps: int = 1000,
    seed: Optional[int] = 42,
    tsp: Optional[TSPQUBO] = None,
) -> Dict:
    """
    Solve a static cluster TSP with simulated annealing.

    Returns
    -------
    dict with keys
        ``tour`` (local city indices), ``cost`` (static closed length),
        ``energy``, ``num_reads``, ``feasible``.
    """
    from dwave.samplers import SimulatedAnnealingSampler

    if tsp is None:
        tsp = build_tsp_qubo(distance_matrix)
    n = tsp.n

    if n == 2:
        tour = [0, 1]
        return {
            "tour": tour,
            "cost": tour_cost(tour, distance_matrix),
            "energy": None,
            "num_reads": 0,
            "feasible": True,
            "backend": "SA",
        }

    sampler = SimulatedAnnealingSampler()
    sampleset = sampler.sample(
        tsp.bqm,
        num_reads=num_reads,
        num_sweeps=num_sweeps,
        seed=seed,
    )

    best_tour: Optional[List[int]] = None
    best_cost = float("inf")
    best_energy = None

    for sample, energy in sampleset.data(["sample", "energy"]):
        tour = decode_tour(dict(sample), n)
        if tour is None:
            continue
        cost = tour_cost(tour, distance_matrix)
        if cost < best_cost:
            best_cost = cost
            best_tour = tour
            best_energy = float(energy)

    if best_tour is None:
        # Fallback: identity order if no feasible sample
        best_tour = list(range(n))
        best_cost = tour_cost(best_tour, distance_matrix)
        return {
            "tour": best_tour,
            "cost": best_cost,
            "energy": best_energy,
            "num_reads": num_reads,
            "feasible": False,
            "backend": "SA",
        }

    return {
        "tour": best_tour,
        "cost": float(best_cost),
        "energy": best_energy,
        "num_reads": num_reads,
        "feasible": True,
        "backend": "SA",
    }


def _self_test() -> None:
    print("Step 4a self-test: Simulated Annealing TSP subsolver")
    print("-" * 60)

    dm = np.array([
        [0, 1, 5, 4],
        [4, 0, 5, 1],
        [2, 5, 0, 3],
        [5, 2, 3, 0],
    ], dtype=float)
    known = [0, 1, 3, 2]
    known_cost = tour_cost(known, dm)

    result = solve_tsp_sa(dm, num_reads=300, num_sweeps=2000, seed=42)
    print(f"  known tour {known} cost={known_cost}")
    print(f"  SA tour    {result['tour']} cost={result['cost']} "
          f"feasible={result['feasible']} energy={result['energy']}")
    assert result["feasible"]
    assert abs(result["cost"] - known_cost) < 1e-9 or result["cost"] <= known_cost + 1e-9
    # With this tiny instance SA should recover the optimum
    assert abs(result["cost"] - known_cost) < 1e-9
    print("  OK")


if __name__ == "__main__":
    _self_test()
