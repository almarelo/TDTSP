"""
Hybrid CP + cluster-subsolver Melgarejo experiments.

Protocol
--------
* ``p <= 10``: solve the **full** instance with CP-SAT only (``cp_base``).
* ``p > 10``: MDS + K-means clustering with ``max_cluster_size=5``, solve
  each cluster's *static* TSP with one of the backends below, then stitch
  with timed CP-SAT (``cp_master.stitch_clusters_cp``).

Cluster backends (p > 10 only)
------------------------------
1. ``cp``          — OR-Tools CP-SAT static TSP
2. ``gurobi``      — Gurobi MTZ (refined)
3. ``dwave_sa``    — D-Wave SimulatedAnnealingSampler on BQM (local)
4. ``dwave_leap``  — Leap Hybrid CQM (refined LeapHybridCQMSampler)
5. ``qaoa``        — AWS Braket QAOA (refined, SV1 by default)
6. ``quanfluence`` — Quanfluence Ising (refined)

Default suite (5 Melgarejo noTW instances, matrix00)
----------------------------------------------------
    inst_10_1, inst_10_2, inst_20_1, inst_20_2, inst_30_1

Examples
--------
    python hybrid_cp_qubo/code/experiments.py
    python hybrid_cp_qubo/code/experiments.py --local-only
    python hybrid_cp_qubo/code/experiments.py --instances 20:1 20:2
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from dotenv import load_dotenv
from ortools.sat.python import cp_model

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_ROOT.parent
LOCAL_CODE_DIR = Path(__file__).resolve().parent
REFINED_DIR = REPO_ROOT / "code" / "refined"

for path in (LOCAL_CODE_DIR, REFINED_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

load_dotenv(REPO_ROOT / ".env")
load_dotenv(REPO_ROOT / "env")

try:
    from .td_data import TravelTimeMatrix, TDTSPInstance
    from .clustering import cluster_visits
    from .cp_base import solve_tdtsp_cp
    from .cp_master import stitch_clusters_cp, _cluster_entry_times_from_tour
    from .subsolver_SA import solve_tsp_sa
except ImportError:  # `python experiments.py`
    from td_data import TravelTimeMatrix, TDTSPInstance
    from clustering import cluster_visits
    from cp_base import solve_tdtsp_cp
    from cp_master import stitch_clusters_cp, _cluster_entry_times_from_tour
    from subsolver_SA import solve_tsp_sa

from solvers.tdtsp_cluster_gurobi import _solve_cluster_mtz
from solvers.tdtsp_cluster_dwave import _dwave_solve_cluster
from solvers.tdtsp_cluster_qaoa import _solve_cluster_qaoa
from solvers.tdtsp_cluster_quanfluence import (
    _solve_cluster_quanfluence,
    _QuanfluenceClient,
)

DEFAULT_RESULTS_DIR = PACKAGE_ROOT / "results"
DEFAULT_MAX_CLUSTER_SIZE = 5
CP_FULL_SIZE_LIMIT = 10

# (size, index) — covers CP-only (p<=10) and hybrid (p>10) regimes
DEFAULT_SUITE: List[Tuple[int, int]] = [
    (10, 1),
    (10, 2),
    (20, 1),
    (20, 2),
    (30, 1),
]

CLUSTER_BACKENDS = (
    "cp",
    "gurobi",
    "dwave_sa",
    "dwave_leap",
    "qaoa",
    "quanfluence",
)
LOCAL_CLUSTER_BACKENDS = ("cp", "gurobi", "dwave_sa")
CLOUD_CLUSTER_BACKENDS = ("dwave_leap", "qaoa", "quanfluence")


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_melgarejo_reference(
    size: int,
    index: int,
    matrix_name: str = "matrix00",
) -> Optional[Dict[str, Any]]:
    """Parse published return-time from ``data/Melgarejo/results_noTW.txt``."""
    path = PACKAGE_ROOT / "data" / "Melgarejo" / "results_noTW.txt"
    if not path.is_file():
        return None
    key_inst = f"inst_{size}_{index}.txt"
    key_mx = f"{matrix_name}.txt"
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 3:
            continue
        if parts[0] != key_inst or parts[1] != key_mx:
            continue
        raw = parts[2]
        optimal = raw.endswith("*")
        try:
            value = float(raw.rstrip("*"))
        except ValueError:
            return None
        if value < 0:
            return {"return_time": None, "optimal": False, "raw": raw}
        return {"return_time": value, "optimal": optimal, "raw": raw}
    return None


def _closed_static_cost(
    cycle: Sequence[int],
    full_dm: Sequence[Sequence[float]],
) -> float:
    n = len(cycle)
    if n <= 1:
        return 0.0
    return float(
        sum(full_dm[cycle[k]][cycle[(k + 1) % n]] for k in range(n))
    )


def _sub_dm(
    cluster_indices: Sequence[int],
    full_dm: Sequence[Sequence[float]],
) -> List[List[float]]:
    n = len(cluster_indices)
    return [
        [float(full_dm[cluster_indices[i]][cluster_indices[j]]) for j in range(n)]
        for i in range(n)
    ]


def _solve_cluster_cp(
    cluster_indices: List[int],
    full_dm: List[List[float]],
    *,
    time_limit: float = 10.0,
) -> List[int]:
    """Static asymmetric TSP on a cluster via OR-Tools CP-SAT + AddCircuit."""
    n = len(cluster_indices)
    if n <= 2:
        return list(cluster_indices)

    dm = _sub_dm(cluster_indices, full_dm)
    # Integer costs (Melgarejo matrices are integer-valued)
    costs = [[int(round(dm[i][j])) for j in range(n)] for i in range(n)]

    model = cp_model.CpModel()
    arc: Dict[Tuple[int, int], Any] = {}
    circuit = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            lit = model.NewBoolVar(f"a_{i}_{j}")
            arc[i, j] = lit
            circuit.append((i, j, lit))
    model.AddCircuit(circuit)
    model.Minimize(sum(costs[i][j] * arc[i, j] for i, j in arc))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit)
    solver.parameters.num_search_workers = 4
    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return list(cluster_indices)

    # Reconstruct tour from selected arcs
    nxt = {i: j for (i, j), lit in arc.items() if solver.Value(lit)}
    tour = [0]
    while len(tour) < n:
        tour.append(nxt[tour[-1]])
    return [cluster_indices[i] for i in tour]


def _solve_cluster_sa(
    cluster_indices: List[int],
    full_dm: List[List[float]],
    *,
    num_reads: int = 200,
) -> List[int]:
    n = len(cluster_indices)
    if n <= 2:
        return list(cluster_indices)
    dm = _sub_dm(cluster_indices, full_dm)
    result = solve_tsp_sa(dm, num_reads=num_reads)
    local = result.get("tour")
    if not result.get("feasible") or local is None or len(local) != n:
        return list(cluster_indices)
    return [cluster_indices[i] for i in local]


def _make_cluster_solver(
    backend: str,
    *,
    cp_cluster_time_limit: float = 10.0,
    dwave_time_limit: float = 5.0,
    sa_num_reads: int = 200,
    qaoa_device: str = "sv1",
    qaoa_p: int = 2,
    qaoa_shots: int = 200,
) -> Callable[[List[int], List[List[float]]], List[int]]:
    backend = backend.lower()

    if backend == "cp":
        def solve(cluster_indices, full_dm):
            return _solve_cluster_cp(
                cluster_indices, full_dm, time_limit=cp_cluster_time_limit
            )
        return solve

    if backend == "gurobi":
        def solve(cluster_indices, full_dm):
            return _solve_cluster_mtz(cluster_indices, full_dm)
        return solve

    if backend == "dwave_sa":
        def solve(cluster_indices, full_dm):
            return _solve_cluster_sa(
                cluster_indices, full_dm, num_reads=sa_num_reads
            )
        return solve

    if backend == "dwave_leap":
        token = os.getenv("DWAVE_API_TOKEN") or os.getenv("DWAVE_API_KEY")
        if not token:
            raise ValueError(
                "D-Wave token not found (DWAVE_API_TOKEN or DWAVE_API_KEY)"
            )
        cache: Dict[str, object] = {}

        def solve(cluster_indices, full_dm):
            return _dwave_solve_cluster(
                cluster_indices, full_dm, dwave_time_limit, token, cache
            )
        return solve

    if backend == "qaoa":
        aws_key = os.getenv("AWS_ACCESS_KEY_ID")
        aws_secret = os.getenv("AWS_SECRET_ACCESS_KEY")
        if not aws_key or not aws_secret:
            raise ValueError("AWS credentials not found for QAOA backend")
        s3_bucket = os.getenv("BRAKET_S3_BUCKET")
        s3_prefix = os.getenv("BRAKET_S3_PREFIX", "tasks")
        capture: List[Dict] = []

        def solve(cluster_indices, full_dm):
            return _solve_cluster_qaoa(
                cluster_indices,
                full_dm,
                qaoa_p,
                qaoa_shots,
                qaoa_device,
                s3_bucket,
                s3_prefix,
                aws_key,
                aws_secret,
                capture,
                False,
            )
        return solve

    if backend == "quanfluence":
        client = _QuanfluenceClient()

        def solve(cluster_indices, full_dm):
            return _solve_cluster_quanfluence(cluster_indices, full_dm, client)
        return solve

    raise ValueError(
        f"Unknown cluster backend {backend!r}; choose from {CLUSTER_BACKENDS}"
    )


def solve_clusters(
    instance: TDTSPInstance,
    matrix: TravelTimeMatrix,
    clusters: Sequence[Sequence[int]],
    entry_times: Sequence[float],
    backend: str,
    **solver_kwargs: Any,
) -> Dict[str, Any]:
    if len(clusters) != len(entry_times):
        raise ValueError("clusters and entry_times length mismatch")

    solve_fn = _make_cluster_solver(backend, **solver_kwargs)
    cycles: List[List[int]] = []
    static_costs: List[float] = []
    t0 = time.perf_counter()

    for visits, t_entry in zip(clusters, entry_times):
        visits = list(visits)
        full_dm = matrix.slice_at(t_entry, instance.vertices).tolist()
        tour = solve_fn(visits, full_dm)
        if sorted(tour) != sorted(visits):
            tour = list(visits)
        cycles.append(tour)
        static_costs.append(_closed_static_cost(tour, full_dm))

    return {
        "cycles": cycles,
        "static_costs": static_costs,
        "entry_times": list(entry_times),
        "backend": backend.lower(),
        "cluster_solve_seconds": time.perf_counter() - t0,
    }


def run_cp_full(
    size: int,
    index: int,
    matrix_name: str = "matrix00",
    *,
    cp_time_limit: float = 60.0,
    results_dir: Optional[Path] = None,
    write_json: bool = True,
) -> Dict[str, Any]:
    """Full-instance CP-SAT path for ``p <= 10``."""
    wall_start = _iso_now()
    matrix = TravelTimeMatrix.load(matrix_name)
    instance = TDTSPInstance.load(size, index)
    ref = load_melgarejo_reference(size, index, matrix_name)

    t0 = time.perf_counter()
    cp_result = solve_tdtsp_cp(
        instance, matrix, time_limit_seconds=cp_time_limit
    )
    wall_s = time.perf_counter() - t0

    record: Dict[str, Any] = {
        "experiment": "hybrid_cp_cluster_backends",
        "path": "cp_full",
        "instance": instance.name,
        "matrix": matrix_name,
        "n": instance.p,
        "cluster_backend": None,
        "max_cluster_size": None,
        "objective": "return_time",
        "status": cp_result.get("status"),
        "solver": cp_result.get("solver"),
        "tour_open": cp_result.get("tour_open"),
        "return_time": cp_result.get("return_time"),
        "return_time_oracle": cp_result.get("return_time_oracle"),
        "objective_bound": cp_result.get("objective_bound"),
        "solve_time_seconds": cp_result.get("solve_time_seconds"),
        "wall_seconds": wall_s,
        "reference": ref,
        "gap_to_reference": None,
        "wall_clock_start_iso": wall_start,
        "wall_clock_end_iso": _iso_now(),
    }
    if (
        record["return_time"] is not None
        and ref
        and ref.get("return_time") is not None
    ):
        record["gap_to_reference"] = (
            float(record["return_time"]) - float(ref["return_time"])
        ) / float(ref["return_time"])

    if write_json:
        out_dir = Path(results_dir) if results_dir else DEFAULT_RESULTS_DIR
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / (
            f"{Path(instance.name).stem}_{matrix_name}_cp_full.json"
        )
        path.write_text(json.dumps(record, indent=2) + "\n")
        record["results_path"] = str(path)

    return record


def run_hybrid_cluster(
    size: int,
    index: int,
    matrix_name: str = "matrix00",
    *,
    backend: str,
    max_cluster_size: int = DEFAULT_MAX_CLUSTER_SIZE,
    cp_time_limit: float = 60.0,
    refine_passes: int = 1,
    results_dir: Optional[Path] = None,
    write_json: bool = True,
    **solver_kwargs: Any,
) -> Dict[str, Any]:
    """Cluster (max size 5) → static cluster solve → timed CP stitch."""
    wall_start = _iso_now()
    matrix = TravelTimeMatrix.load(matrix_name)
    instance = TDTSPInstance.load(size, index)
    ref = load_melgarejo_reference(size, index, matrix_name)
    clusters = cluster_visits(
        instance, matrix, max_cluster_size=max_cluster_size
    )

    record: Dict[str, Any] = {
        "experiment": "hybrid_cp_cluster_backends",
        "path": "cluster_hybrid",
        "instance": instance.name,
        "matrix": matrix_name,
        "n": instance.p,
        "cluster_backend": backend.lower(),
        "max_cluster_size": max_cluster_size,
        "n_clusters": clusters.n_clusters,
        "cluster_sizes": clusters.cluster_sizes,
        "clusters": clusters.clusters,
        "objective": "return_time",
        "reference": ref,
        "wall_clock_start_iso": wall_start,
    }

    if instance.p <= CP_FULL_SIZE_LIMIT:
        record.update({
            "status": "SKIPPED",
            "message": (
                f"p={instance.p} <= {CP_FULL_SIZE_LIMIT}; use cp_full path"
            ),
            "return_time": None,
        })
        record["wall_clock_end_iso"] = _iso_now()
        return record

    entry_times = [instance.start_time] * len(clusters.clusters)
    history: List[Dict[str, Any]] = []
    best: Optional[Dict[str, Any]] = None
    t_wall0 = time.perf_counter()

    for pass_id in range(1 + max(0, refine_passes)):
        cluster_sol = solve_clusters(
            instance,
            matrix,
            clusters.clusters,
            entry_times,
            backend,
            **solver_kwargs,
        )
        stitch = stitch_clusters_cp(
            instance,
            matrix,
            cluster_sol["cycles"],
            time_limit_seconds=cp_time_limit,
        )
        history.append({
            "pass": pass_id,
            "entry_times": list(entry_times),
            "cycles": cluster_sol["cycles"],
            "static_costs": cluster_sol["static_costs"],
            "cluster_solve_seconds": cluster_sol["cluster_solve_seconds"],
            "stitch_status": stitch["status"],
            "return_time": stitch.get("return_time"),
            "tour_open": stitch.get("tour_open"),
            "stitch_time_seconds": stitch.get("solve_time_seconds"),
        })

        if stitch.get("tour_open") is None:
            break

        if best is None or (
            stitch["return_time"] is not None
            and (
                best.get("return_time") is None
                or stitch["return_time"] < best["return_time"]
            )
        ):
            best = {
                **stitch,
                "cluster_cycles": cluster_sol["cycles"],
                "cluster_static_costs": cluster_sol["static_costs"],
                "cluster_solve_seconds": cluster_sol["cluster_solve_seconds"],
                "entry_times": list(entry_times),
                "refine_pass": pass_id,
            }

        if pass_id >= refine_passes:
            break
        entry_times = _cluster_entry_times_from_tour(
            stitch["tour_open"],
            stitch["arrivals"],
            clusters.clusters,
        )

    wall_s = time.perf_counter() - t_wall0

    if best is None:
        record.update({
            "status": history[-1]["stitch_status"] if history else "ERROR",
            "return_time": None,
            "tour_open": None,
            "history": history,
            "wall_seconds": wall_s,
            "solver": f"cluster-{backend} + CP-SAT stitch",
        })
    else:
        gap = None
        if (
            best.get("return_time") is not None
            and ref
            and ref.get("return_time") is not None
        ):
            gap = (
                float(best["return_time"]) - float(ref["return_time"])
            ) / float(ref["return_time"])
        record.update({
            "status": best.get("status"),
            "solver": f"cluster-{backend} + CP-SAT timed stitch",
            "tour_open": best.get("tour_open"),
            "return_time": best.get("return_time"),
            "return_time_oracle": best.get("return_time_oracle"),
            "objective_bound": best.get("objective_bound"),
            "solve_time_seconds": best.get("solve_time_seconds"),
            "cluster_solve_seconds": best.get("cluster_solve_seconds"),
            "cluster_cycles": best.get("cluster_cycles"),
            "cluster_static_costs": best.get("cluster_static_costs"),
            "cluster_entry_times": best.get("entry_times"),
            "rotations": best.get("rotations"),
            "refine_pass_used": best.get("refine_pass"),
            "history": history,
            "wall_seconds": wall_s,
            "gap_to_reference": gap,
        })

    record["wall_clock_end_iso"] = _iso_now()

    if write_json:
        out_dir = Path(results_dir) if results_dir else DEFAULT_RESULTS_DIR
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / (
            f"{Path(instance.name).stem}_{matrix_name}_"
            f"cluster_{backend.lower()}.json"
        )
        path.write_text(json.dumps(record, indent=2) + "\n")
        record["results_path"] = str(path)

    return record


def run_instance(
    size: int,
    index: int,
    matrix_name: str = "matrix00",
    *,
    backends: Sequence[str] = CLUSTER_BACKENDS,
    max_cluster_size: int = DEFAULT_MAX_CLUSTER_SIZE,
    cp_time_limit: float = 60.0,
    refine_passes: int = 1,
    results_dir: Optional[Path] = None,
    write_json: bool = True,
    continue_on_error: bool = True,
    **solver_kwargs: Any,
) -> Dict[str, Any]:
    """
    Run the protocol for one instance.

    p <= 10 → CP full only.
    p > 10  → every requested cluster backend.
    """
    out: Dict[str, Any] = {
        "instance": f"inst_{size}_{index}.txt",
        "matrix": matrix_name,
        "n": size,
        "runs": {},
    }

    if size <= CP_FULL_SIZE_LIMIT:
        print(f"\n=== inst_{size}_{index}  path=cp_full ===")
        try:
            rec = run_cp_full(
                size,
                index,
                matrix_name,
                cp_time_limit=cp_time_limit,
                results_dir=results_dir,
                write_json=write_json,
            )
        except Exception as exc:
            rec = {
                "status": "ERROR",
                "path": "cp_full",
                "error": f"{type(exc).__name__}: {exc}",
                "return_time": None,
            }
            print(f"  ERROR: {rec['error']}")
            if not continue_on_error:
                raise
        out["runs"]["cp_full"] = {
            "status": rec.get("status"),
            "return_time": rec.get("return_time"),
            "gap_to_reference": rec.get("gap_to_reference"),
            "wall_seconds": rec.get("wall_seconds"),
            "reference": rec.get("reference"),
            "results_path": rec.get("results_path"),
            "error": rec.get("error"),
        }
        print(
            f"  status={rec.get('status')}  "
            f"return_time={rec.get('return_time')}  "
            f"ref={((rec.get('reference') or {}).get('raw'))}  "
            f"gap={rec.get('gap_to_reference')}"
        )
        return out

    for backend in backends:
        print(
            f"\n=== inst_{size}_{index}  path=cluster  "
            f"backend={backend}  max_cluster_size={max_cluster_size} ===",
            flush=True,
        )
        try:
            rec = run_hybrid_cluster(
                size,
                index,
                matrix_name,
                backend=backend,
                max_cluster_size=max_cluster_size,
                cp_time_limit=cp_time_limit,
                refine_passes=refine_passes,
                results_dir=results_dir,
                write_json=write_json,
                **solver_kwargs,
            )
        except Exception as exc:
            rec = {
                "experiment": "hybrid_cp_cluster_backends",
                "path": "cluster_hybrid",
                "instance": f"inst_{size}_{index}.txt",
                "matrix": matrix_name,
                "n": size,
                "cluster_backend": backend,
                "status": "ERROR",
                "error": f"{type(exc).__name__}: {exc}",
                "return_time": None,
                "reference": load_melgarejo_reference(size, index, matrix_name),
                "wall_clock_end_iso": _iso_now(),
            }
            print(f"  ERROR: {rec['error']}", flush=True)
            if write_json:
                out_dir = Path(results_dir) if results_dir else DEFAULT_RESULTS_DIR
                out_dir.mkdir(parents=True, exist_ok=True)
                err_path = out_dir / (
                    f"inst_{size}_{index}_{matrix_name}_"
                    f"cluster_{backend}.json"
                )
                err_path.write_text(json.dumps(rec, indent=2) + "\n")
                rec["results_path"] = str(err_path)
            if not continue_on_error:
                raise

        out["runs"][backend] = {
            "status": rec.get("status"),
            "return_time": rec.get("return_time"),
            "gap_to_reference": rec.get("gap_to_reference"),
            "wall_seconds": rec.get("wall_seconds"),
            "n_clusters": rec.get("n_clusters"),
            "cluster_sizes": rec.get("cluster_sizes"),
            "reference": rec.get("reference"),
            "results_path": rec.get("results_path"),
            "error": rec.get("error"),
        }
        print(
            f"  status={rec.get('status')}  "
            f"return_time={rec.get('return_time')}  "
            f"clusters={rec.get('cluster_sizes')}  "
            f"gap={rec.get('gap_to_reference')}"
        )

    return out


def build_suite_summary(instance_summaries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate per-instance runs into a compact comparison table."""
    rows = []
    for inst in instance_summaries:
        for backend, run in inst.get("runs", {}).items():
            rows.append({
                "instance": inst["instance"],
                "n": inst["n"],
                "matrix": inst["matrix"],
                "backend": backend,
                "status": run.get("status"),
                "return_time": run.get("return_time"),
                "reference": (run.get("reference") or {}).get("raw"),
                "gap_to_reference": run.get("gap_to_reference"),
                "wall_seconds": run.get("wall_seconds"),
                "cluster_sizes": run.get("cluster_sizes"),
                "error": run.get("error"),
            })

    # Best hybrid backend per large instance
    best_by_instance: Dict[str, Any] = {}
    for row in rows:
        if row["backend"] == "cp_full":
            continue
        key = row["instance"]
        rt = row["return_time"]
        if rt is None:
            continue
        cur = best_by_instance.get(key)
        if cur is None or rt < cur["return_time"]:
            best_by_instance[key] = row

    return {
        "experiment": "hybrid_cp_cluster_backends",
        "protocol": {
            "cp_full_when": f"p <= {CP_FULL_SIZE_LIMIT}",
            "cluster_when": f"p > {CP_FULL_SIZE_LIMIT}",
            "max_cluster_size": DEFAULT_MAX_CLUSTER_SIZE,
            "cluster_backends": list(CLUSTER_BACKENDS),
        },
        "wall_clock_end_iso": _iso_now(),
        "table": rows,
        "best_hybrid_by_instance": best_by_instance,
        "instances": instance_summaries,
    }


def run_suite(
    instances: Sequence[Tuple[int, int]] = DEFAULT_SUITE,
    matrix_name: str = "matrix00",
    *,
    backends: Sequence[str] = CLUSTER_BACKENDS,
    max_cluster_size: int = DEFAULT_MAX_CLUSTER_SIZE,
    cp_time_limit: float = 60.0,
    refine_passes: int = 1,
    results_dir: Optional[Path] = None,
    write_json: bool = True,
    continue_on_error: bool = True,
    **solver_kwargs: Any,
) -> Dict[str, Any]:
    out_dir = Path(results_dir) if results_dir else DEFAULT_RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Melgarejo hybrid experiment suite")
    print("-" * 60)
    print(f"  instances         : {list(instances)}")
    print(f"  matrix            : {matrix_name}")
    print(f"  cp_full if p <=   : {CP_FULL_SIZE_LIMIT}")
    print(f"  max_cluster_size  : {max_cluster_size}")
    print(f"  cluster backends  : {list(backends)}")

    summaries: List[Dict[str, Any]] = []
    suite_start = _iso_now()
    for size, index in instances:
        summaries.append(
            run_instance(
                size,
                index,
                matrix_name,
                backends=backends,
                max_cluster_size=max_cluster_size,
                cp_time_limit=cp_time_limit,
                refine_passes=refine_passes,
                results_dir=out_dir,
                write_json=write_json,
                continue_on_error=continue_on_error,
                **solver_kwargs,
            )
        )

    summary = build_suite_summary(summaries)
    summary["wall_clock_start_iso"] = suite_start
    summary["matrix"] = matrix_name

    if write_json:
        path = out_dir / f"suite_summary_{matrix_name}.json"
        path.write_text(json.dumps(summary, indent=2) + "\n")
        summary["summary_path"] = str(path)
        print(f"\nSuite summary: {path}")

    print("\nResults table")
    print("-" * 60)
    for row in summary["table"]:
        print(
            f"  {row['instance']:16s}  {row['backend']:12s}  "
            f"rt={row['return_time']}  ref={row['reference']}  "
            f"gap={row['gap_to_reference']}  "
            f"t={row['wall_seconds']}"
        )
    return summary


def _parse_instances(raw: Sequence[str]) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    for item in raw:
        m = re.fullmatch(r"(\d+):(\d+)", item.strip())
        if not m:
            raise argparse.ArgumentTypeError(
                f"Bad instance spec {item!r}; use SIZE:INDEX (e.g. 20:1)"
            )
        out.append((int(m.group(1)), int(m.group(2))))
    return out


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Melgarejo hybrid experiments: CP-full for p<=10; "
            "cluster(max=5)+backends for p>10"
        )
    )
    p.add_argument(
        "--instances",
        nargs="+",
        default=None,
        help="SIZE:INDEX list (default: 10:1 10:2 20:1 20:2 30:1)",
    )
    p.add_argument("--matrix", default="matrix00")
    p.add_argument(
        "--backends",
        nargs="+",
        default=None,
        choices=list(CLUSTER_BACKENDS),
        help="Cluster backends for p>10 (default: all six)",
    )
    p.add_argument(
        "--local-only",
        action="store_true",
        help="Only cp / gurobi / dwave_sa (skip cloud backends)",
    )
    p.add_argument(
        "--max-cluster-size", type=int, default=DEFAULT_MAX_CLUSTER_SIZE
    )
    p.add_argument("--cp-time-limit", type=float, default=60.0)
    p.add_argument("--refine-passes", type=int, default=1)
    p.add_argument("--dwave-time-limit", type=float, default=5.0)
    p.add_argument("--sa-num-reads", type=int, default=200)
    p.add_argument("--qaoa-device", default="sv1", choices=("sv1", "rigetti"))
    p.add_argument("--qaoa-p", type=int, default=2)
    p.add_argument("--qaoa-shots", type=int, default=200)
    p.add_argument("--results-dir", type=Path, default=None)
    p.add_argument("--no-write", action="store_true")
    p.add_argument("--stop-on-error", action="store_true")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    instances = (
        _parse_instances(args.instances)
        if args.instances
        else list(DEFAULT_SUITE)
    )
    if args.local_only:
        backends = list(LOCAL_CLUSTER_BACKENDS)
    elif args.backends:
        backends = list(args.backends)
    else:
        backends = list(CLUSTER_BACKENDS)

    summary = run_suite(
        instances,
        args.matrix,
        backends=backends,
        max_cluster_size=args.max_cluster_size,
        cp_time_limit=args.cp_time_limit,
        refine_passes=args.refine_passes,
        results_dir=args.results_dir,
        write_json=not args.no_write,
        continue_on_error=not args.stop_on_error,
        dwave_time_limit=args.dwave_time_limit,
        sa_num_reads=args.sa_num_reads,
        qaoa_device=args.qaoa_device,
        qaoa_p=args.qaoa_p,
        qaoa_shots=args.qaoa_shots,
    )

    ok = any(
        r.get("status") in ("OPTIMAL", "FEASIBLE")
        for inst in summary.get("instances", [])
        for r in inst.get("runs", {}).values()
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
