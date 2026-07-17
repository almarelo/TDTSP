"""
End-to-end orchestrator for the hybrid CP + QUBO Approach.

Behaviour
---------
1. Load Melgarejo noTW instance + travel-time matrix.
2. Cluster-first decomposition (``cluster_visits``).
3. If ``use_qubo=False`` (p <= max_cluster_size): CP-SAT solves the full
   TD-TSP (true FIFO ``Dij(t)``, objective = return time).
4. If ``use_qubo=True``: static QUBO/CQM per cluster (SA or D-Wave), then
   CP-SAT stitches cluster cycles with timed ``Dij(t)`` (optional refine).
5. Write a JSON record under ``results/hybrid_cp_qubo/``.

CLI
---
    python code/hybrid_cp_qubo/solver.py --size 10 --index 1 --matrix matrix00
    python code/hybrid_cp_qubo/solver.py --size 20 --index 1 --matrix matrix00 \\
        --qubo-backend sa
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "code") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "code"))

try:
    from .td_data import TravelTimeMatrix, TDTSPInstance
    from .clustering import cluster_visits
    from .cp_base import solve_tdtsp_cp
    from .cp_master import solve_cluster_qubo_cp
except ImportError:  # `python solver.py`
    from td_data import TravelTimeMatrix, TDTSPInstance
    from clustering import cluster_visits
    from cp_base import solve_tdtsp_cp
    from cp_master import solve_cluster_qubo_cp

DEFAULT_RESULTS_DIR = REPO_ROOT / "results" / "hybrid_cp_qubo"


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def solve(
    size: int,
    index: int,
    matrix_name: str = "matrix00",
    *,
    max_cluster_size: int = 10,
    cp_time_limit: float = 60.0,
    qubo_backend: str = "sa",
    refine_passes: int = 1,
    results_dir: Optional[Path] = None,
    write_json: bool = True,
) -> Dict[str, Any]:
    """
    Run Approach 2 on one Melgarejo noTW instance.

    Parameters
    ----------
    qubo_backend :
        ``'sa'`` or ``'dwave'`` — used when ``use_qubo=True``.
    refine_passes :
        Extra QUBO re-solves after updating cluster entry times from the
        stitched tour (0 = single pass).
    """
    wall_start = _iso_now()
    matrix = TravelTimeMatrix.load(matrix_name)
    instance = TDTSPInstance.load(size, index)
    clusters = cluster_visits(
        instance, matrix, max_cluster_size=max_cluster_size
    )

    record: Dict[str, Any] = {
        "approach": "hybrid_cp_qubo",
        "instance": instance.name,
        "matrix": matrix_name,
        "n": instance.p,
        "vertices": instance.vertices,
        "start_time": instance.start_time,
        "max_cluster_size": max_cluster_size,
        "n_clusters": clusters.n_clusters,
        "cluster_sizes": clusters.cluster_sizes,
        "clusters": clusters.clusters,
        "use_qubo": clusters.use_qubo,
        "qubo_backend": qubo_backend,
        "objective": "return_time",
        "wall_clock_start_iso": wall_start,
    }

    if not clusters.use_qubo:
        cp_result = solve_tdtsp_cp(
            instance,
            matrix,
            time_limit_seconds=cp_time_limit,
        )
        record.update({
            "path": "cp_only",
            "status": cp_result["status"],
            "solver": cp_result["solver"],
            "tour_open": cp_result.get("tour_open"),
            "return_time": cp_result.get("return_time"),
            "return_time_oracle": cp_result.get("return_time_oracle"),
            "objective_bound": cp_result.get("objective_bound"),
            "solve_time_seconds": cp_result.get("solve_time_seconds"),
            "cp_time_limit_seconds": cp_time_limit,
            "arrivals": cp_result.get("arrivals"),
            "departures": cp_result.get("departures"),
            "edge_travel_times": (
                cp_result.get("timing", {}) or {}
            ).get("edge_travel_times"),
        })
    else:
        hybrid = solve_cluster_qubo_cp(
            instance,
            matrix,
            clusters,
            qubo_backend=qubo_backend,
            cp_time_limit=cp_time_limit,
            refine_passes=refine_passes,
        )
        record.update({
            "path": "cluster_qubo_cp",
            "status": hybrid.get("status"),
            "solver": hybrid.get("solver"),
            "tour_open": hybrid.get("tour_open"),
            "return_time": hybrid.get("return_time"),
            "return_time_oracle": hybrid.get("return_time_oracle"),
            "objective_bound": hybrid.get("objective_bound"),
            "solve_time_seconds": hybrid.get("solve_time_seconds"),
            "qubo_time_seconds": hybrid.get("qubo_time_seconds"),
            "cp_time_limit_seconds": cp_time_limit,
            "refine_passes": refine_passes,
            "refine_pass_used": hybrid.get("refine_pass"),
            "cluster_cycles": hybrid.get("cluster_cycles"),
            "cluster_static_costs": hybrid.get("cluster_static_costs"),
            "cluster_entry_times": hybrid.get("entry_times"),
            "rotations": hybrid.get("rotations"),
            "arrivals": hybrid.get("arrivals"),
            "departures": hybrid.get("departures"),
            "edge_travel_times": (
                hybrid.get("timing", {}) or {}
            ).get("edge_travel_times"),
            "history": hybrid.get("history"),
        })

    record["wall_clock_end_iso"] = _iso_now()

    if write_json:
        out_dir = Path(results_dir) if results_dir else DEFAULT_RESULTS_DIR
        out_dir.mkdir(parents=True, exist_ok=True)
        suffix = "cp" if not clusters.use_qubo else f"qubo_{qubo_backend}"
        out_path = out_dir / (
            f"{Path(instance.name).stem}_{matrix_name}_{suffix}.json"
        )
        out_path.write_text(json.dumps(record, indent=2) + "\n")
        record["results_path"] = str(out_path)

    return record


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Hybrid CP+QUBO TD-TSP solver (Melgarejo noTW)"
    )
    p.add_argument("--size", type=int, default=10, help="Instance size p")
    p.add_argument("--index", type=int, default=1, help="Instance index")
    p.add_argument(
        "--matrix", default="matrix00",
        help="Travel-time matrix name (matrix00 / matrix10 / matrix20)",
    )
    p.add_argument(
        "--max-cluster-size", type=int, default=10,
        help="Clusters only if p > this (default 10 => CP-only for n=10)",
    )
    p.add_argument(
        "--cp-time-limit", type=float, default=60.0,
        help="CP-SAT wall-time limit in seconds",
    )
    p.add_argument(
        "--qubo-backend", choices=("sa", "dwave"), default="sa",
        help="Per-cluster static TSP backend when use_qubo=True",
    )
    p.add_argument(
        "--refine-passes", type=int, default=1,
        help="Extra QUBO passes after updating cluster entry times",
    )
    p.add_argument(
        "--results-dir", type=Path, default=None,
        help=f"Output directory (default: {DEFAULT_RESULTS_DIR})",
    )
    p.add_argument(
        "--no-write", action="store_true",
        help="Do not write a JSON result file",
    )
    return p


def main(argv: Optional[list] = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    print("Hybrid CP+QUBO orchestrator")
    print("-" * 60)
    print(f"  instance : inst_{args.size}_{args.index}.txt")
    print(f"  matrix   : {args.matrix}")
    print(f"  max_cluster_size: {args.max_cluster_size}")
    print(f"  qubo_backend    : {args.qubo_backend}")

    record = solve(
        size=args.size,
        index=args.index,
        matrix_name=args.matrix,
        max_cluster_size=args.max_cluster_size,
        cp_time_limit=args.cp_time_limit,
        qubo_backend=args.qubo_backend,
        refine_passes=args.refine_passes,
        results_dir=args.results_dir,
        write_json=not args.no_write,
    )

    print(f"  path     : {record.get('path')}")
    print(f"  use_qubo : {record.get('use_qubo')}")
    print(f"  status   : {record.get('status')}")
    print(f"  tour     : {record.get('tour_open')}")
    print(f"  return_time: {record.get('return_time')}")
    if record.get("results_path"):
        print(f"  wrote    : {record['results_path']}")
    if record.get("status") not in ("OPTIMAL", "FEASIBLE"):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
