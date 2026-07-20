"""
Hybrid CP pipeline on original NYC JSON instances (``data/instances/``).

Protocol (as requested)
-----------------------
* All sizes: 5, 10, 25, 50, 100
* All time slots: morning_peak, midday, evening_peak, night
* Cluster backends: cp, gurobi, dwave_sa, dwave_leap
* ``max_cluster_size = 15``
  - ``n <= 15`` → full CP-SAT on the timed instance
  - ``n > 15``  → cluster → static cluster solve → CP timed stitch

Travel model: hourly stepwise ``Dij(t) = base[i,j] * multiplier[slot(hour)]``
built by ``json_instances.py``. Objective = Melgarejo-style return time
(arrival back at depot); tour duration = return_time - start_time.

Examples
--------
    python hybrid_cp_qubo/code/experiments_original.py
    python hybrid_cp_qubo/code/experiments_original.py --sizes 5 10 --slots morning_peak
    python hybrid_cp_qubo/code/experiments_original.py --backends cp gurobi
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_ROOT.parent
LOCAL_CODE_DIR = Path(__file__).resolve().parent

if str(LOCAL_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(LOCAL_CODE_DIR))

from dotenv import load_dotenv

load_dotenv(REPO_ROOT / ".env")
load_dotenv(REPO_ROOT / "env")

from clustering import cluster_visits
from cp_base import solve_tdtsp_cp
from cp_master import stitch_clusters_cp, _cluster_entry_times_from_tour
from json_instances import load_hybrid, list_json_instances
from experiments import (
    CLUSTER_BACKENDS,
    solve_clusters,
)

DEFAULT_RESULTS_DIR = PACKAGE_ROOT / "results" / "original"
DEFAULT_SIZES = [5, 10, 25, 50, 100]
DEFAULT_SLOTS = ["morning_peak", "midday", "evening_peak", "night"]
DEFAULT_BACKENDS = ["cp", "gurobi", "dwave_sa", "dwave_leap"]
DEFAULT_MAX_CLUSTER_SIZE = 15


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_cp_full_json(
    size: int,
    slot_name: str,
    *,
    cp_time_limit: float = 60.0,
    results_dir: Optional[Path] = None,
    write_json: bool = True,
) -> Dict[str, Any]:
    matrix, instance, meta = load_hybrid(size, slot_name)
    wall0 = time.perf_counter()
    cp_result = solve_tdtsp_cp(
        instance, matrix, time_limit_seconds=cp_time_limit
    )
    wall_s = time.perf_counter() - wall0

    start = float(instance.start_time)
    rt = cp_result.get("return_time")
    duration = None if rt is None else float(rt) - start

    record: Dict[str, Any] = {
        "experiment": "hybrid_cp_original_json",
        "path": "cp_full",
        "dataset": "data/instances",
        "instance_file": f"tdtsp_n{size}.json",
        "n": size,
        "slot": meta["slot_name"],
        "slot_label": meta["slot_label"],
        "slot_multiplier": meta["slot_multiplier"],
        "cluster_backend": None,
        "max_cluster_size": None,
        "status": cp_result.get("status"),
        "solver": cp_result.get("solver"),
        "tour_open": cp_result.get("tour_open"),
        "return_time": rt,
        "tour_duration_seconds": duration,
        "start_time_seconds": start,
        "solve_time_seconds": cp_result.get("solve_time_seconds"),
        "wall_seconds": wall_s,
        "objective_bound": cp_result.get("objective_bound"),
        "meta": meta,
        "wall_clock_start_iso": _iso_now(),
        "wall_clock_end_iso": _iso_now(),
    }

    if write_json:
        out_dir = Path(results_dir) if results_dir else DEFAULT_RESULTS_DIR
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"tdtsp_n{size}_{slot_name}_cp_full.json"
        path.write_text(json.dumps(record, indent=2) + "\n")
        record["results_path"] = str(path)
    return record


def run_hybrid_json(
    size: int,
    slot_name: str,
    backend: str,
    *,
    max_cluster_size: int = DEFAULT_MAX_CLUSTER_SIZE,
    cp_time_limit: float = 60.0,
    refine_passes: int = 0,
    results_dir: Optional[Path] = None,
    write_json: bool = True,
    **solver_kwargs: Any,
) -> Dict[str, Any]:
    matrix, instance, meta = load_hybrid(size, slot_name)
    clusters = cluster_visits(
        instance, matrix, max_cluster_size=max_cluster_size
    )
    wall0 = time.perf_counter()

    record: Dict[str, Any] = {
        "experiment": "hybrid_cp_original_json",
        "path": "cluster_hybrid",
        "dataset": "data/instances",
        "instance_file": f"tdtsp_n{size}.json",
        "n": size,
        "slot": meta["slot_name"],
        "slot_label": meta["slot_label"],
        "slot_multiplier": meta["slot_multiplier"],
        "cluster_backend": backend.lower(),
        "max_cluster_size": max_cluster_size,
        "n_clusters": clusters.n_clusters,
        "cluster_sizes": clusters.cluster_sizes,
        "clusters": clusters.clusters,
        "meta": meta,
        "wall_clock_start_iso": _iso_now(),
    }

    if not clusters.use_qubo:
        # Should not happen when caller routes n<=max to cp_full, but be safe
        wall_s = time.perf_counter() - wall0
        record.update({
            "status": "SKIPPED",
            "message": f"n={size} <= max_cluster_size={max_cluster_size}",
            "return_time": None,
            "wall_seconds": wall_s,
            "wall_clock_end_iso": _iso_now(),
        })
        return record

    entry_times = [instance.start_time] * len(clusters.clusters)
    history: List[Dict[str, Any]] = []
    best: Optional[Dict[str, Any]] = None

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
            "cycles": cluster_sol["cycles"],
            "static_costs": cluster_sol["static_costs"],
            "cluster_solve_seconds": cluster_sol["cluster_solve_seconds"],
            "stitch_status": stitch.get("status"),
            "return_time": stitch.get("return_time"),
            "tour_open": stitch.get("tour_open"),
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
                "refine_pass": pass_id,
            }
        if pass_id >= refine_passes:
            break
        entry_times = _cluster_entry_times_from_tour(
            stitch["tour_open"],
            stitch["arrivals"],
            clusters.clusters,
        )

    wall_s = time.perf_counter() - wall0
    start = float(instance.start_time)

    if best is None:
        record.update({
            "status": history[-1]["stitch_status"] if history else "ERROR",
            "return_time": None,
            "tour_duration_seconds": None,
            "tour_open": None,
            "history": history,
            "wall_seconds": wall_s,
            "solver": f"cluster-{backend} + CP-SAT stitch",
        })
    else:
        rt = best.get("return_time")
        record.update({
            "status": best.get("status"),
            "solver": f"cluster-{backend} + CP-SAT timed stitch",
            "tour_open": best.get("tour_open"),
            "return_time": rt,
            "tour_duration_seconds": (
                None if rt is None else float(rt) - start
            ),
            "start_time_seconds": start,
            "solve_time_seconds": best.get("solve_time_seconds"),
            "cluster_solve_seconds": best.get("cluster_solve_seconds"),
            "cluster_cycles": best.get("cluster_cycles"),
            "cluster_static_costs": best.get("cluster_static_costs"),
            "refine_pass_used": best.get("refine_pass"),
            "history": history,
            "wall_seconds": wall_s,
        })

    record["wall_clock_end_iso"] = _iso_now()

    if write_json:
        out_dir = Path(results_dir) if results_dir else DEFAULT_RESULTS_DIR
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / (
            f"tdtsp_n{size}_{slot_name}_cluster_{backend.lower()}.json"
        )
        path.write_text(json.dumps(record, indent=2) + "\n")
        record["results_path"] = str(path)
    return record


def run_suite(
    sizes: Sequence[int] = DEFAULT_SIZES,
    slots: Sequence[str] = DEFAULT_SLOTS,
    backends: Sequence[str] = DEFAULT_BACKENDS,
    *,
    max_cluster_size: int = DEFAULT_MAX_CLUSTER_SIZE,
    cp_time_limit: float = 60.0,
    refine_passes: int = 0,
    results_dir: Optional[Path] = None,
    write_json: bool = True,
    continue_on_error: bool = True,
    **solver_kwargs: Any,
) -> Dict[str, Any]:
    out_dir = Path(results_dir) if results_dir else DEFAULT_RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    available = {p.stem: p for p in list_json_instances(sizes=list(sizes))}
    print("Original JSON hybrid experiment suite", flush=True)
    print("-" * 60, flush=True)
    print(f"  sizes             : {list(sizes)}", flush=True)
    print(f"  slots             : {list(slots)}", flush=True)
    print(f"  backends (n>{max_cluster_size}): {list(backends)}", flush=True)
    print(f"  max_cluster_size  : {max_cluster_size}", flush=True)
    print(f"  results           : {out_dir}", flush=True)

    table: List[Dict[str, Any]] = []
    suite_start = _iso_now()

    for size in sizes:
        key = f"tdtsp_n{size}"
        if key not in available:
            print(f"\n=== n={size} SKIPPED (file missing) ===", flush=True)
            continue

        for slot_name in slots:
            if size <= max_cluster_size:
                print(
                    f"\n=== n={size} slot={slot_name} path=cp_full ===",
                    flush=True,
                )
                try:
                    rec = run_cp_full_json(
                        size,
                        slot_name,
                        cp_time_limit=cp_time_limit,
                        results_dir=out_dir,
                        write_json=write_json,
                    )
                except Exception as exc:
                    rec = {
                        "status": "ERROR",
                        "error": f"{type(exc).__name__}: {exc}",
                        "return_time": None,
                        "tour_duration_seconds": None,
                        "n": size,
                        "slot": slot_name,
                        "cluster_backend": "cp_full",
                    }
                    print(f"  ERROR: {rec['error']}", flush=True)
                    if not continue_on_error:
                        raise
                row = {
                    "n": size,
                    "slot": slot_name,
                    "backend": "cp_full",
                    "status": rec.get("status"),
                    "return_time": rec.get("return_time"),
                    "tour_duration_seconds": rec.get("tour_duration_seconds"),
                    "wall_seconds": rec.get("wall_seconds"),
                    "cluster_sizes": None,
                    "error": rec.get("error"),
                    "results_path": rec.get("results_path"),
                }
                table.append(row)
                print(
                    f"  status={row['status']}  duration={row['tour_duration_seconds']}  "
                    f"return_time={row['return_time']}",
                    flush=True,
                )
                continue

            for backend in backends:
                print(
                    f"\n=== n={size} slot={slot_name} "
                    f"backend={backend} max_cluster={max_cluster_size} ===",
                    flush=True,
                )
                try:
                    rec = run_hybrid_json(
                        size,
                        slot_name,
                        backend,
                        max_cluster_size=max_cluster_size,
                        cp_time_limit=cp_time_limit,
                        refine_passes=refine_passes,
                        results_dir=out_dir,
                        write_json=write_json,
                        **solver_kwargs,
                    )
                except Exception as exc:
                    rec = {
                        "status": "ERROR",
                        "error": f"{type(exc).__name__}: {exc}",
                        "return_time": None,
                        "tour_duration_seconds": None,
                        "n": size,
                        "slot": slot_name,
                        "cluster_backend": backend,
                    }
                    print(f"  ERROR: {rec['error']}", flush=True)
                    if write_json:
                        err_path = out_dir / (
                            f"tdtsp_n{size}_{slot_name}_"
                            f"cluster_{backend}.json"
                        )
                        err_path.write_text(
                            json.dumps({
                                **rec,
                                "experiment": "hybrid_cp_original_json",
                                "wall_clock_end_iso": _iso_now(),
                            }, indent=2)
                            + "\n"
                        )
                        rec["results_path"] = str(err_path)
                    if not continue_on_error:
                        raise

                row = {
                    "n": size,
                    "slot": slot_name,
                    "backend": backend,
                    "status": rec.get("status"),
                    "return_time": rec.get("return_time"),
                    "tour_duration_seconds": rec.get("tour_duration_seconds"),
                    "wall_seconds": rec.get("wall_seconds"),
                    "cluster_sizes": rec.get("cluster_sizes"),
                    "error": rec.get("error"),
                    "results_path": rec.get("results_path"),
                }
                table.append(row)
                print(
                    f"  status={row['status']}  duration={row['tour_duration_seconds']}  "
                    f"clusters={row['cluster_sizes']}",
                    flush=True,
                )

    summary = {
        "experiment": "hybrid_cp_original_json",
        "dataset": "data/instances",
        "protocol": {
            "cp_full_when": f"n <= {max_cluster_size}",
            "cluster_when": f"n > {max_cluster_size}",
            "max_cluster_size": max_cluster_size,
            "backends": list(backends),
            "slots": list(slots),
            "sizes": list(sizes),
        },
        "wall_clock_start_iso": suite_start,
        "wall_clock_end_iso": _iso_now(),
        "table": table,
    }
    if write_json:
        path = out_dir / "suite_summary_original.json"
        path.write_text(json.dumps(summary, indent=2) + "\n")
        summary["summary_path"] = str(path)
        print(f"\nSuite summary: {path}", flush=True)

    print("\nResults table", flush=True)
    print("-" * 60, flush=True)
    for row in table:
        print(
            f"  n={row['n']:<4} {row['slot']:<14} {row['backend']:<12} "
            f"dur={row['tour_duration_seconds']}  "
            f"status={row['status']}",
            flush=True,
        )
    return summary


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Hybrid CP on original data/instances JSON TD-TSP "
            "(max cluster size 15)"
        )
    )
    p.add_argument("--sizes", nargs="+", type=int, default=DEFAULT_SIZES)
    p.add_argument("--slots", nargs="+", default=DEFAULT_SLOTS)
    p.add_argument(
        "--backends",
        nargs="+",
        default=DEFAULT_BACKENDS,
        choices=[b for b in CLUSTER_BACKENDS if b in DEFAULT_BACKENDS]
        + list(DEFAULT_BACKENDS),
    )
    p.add_argument(
        "--max-cluster-size", type=int, default=DEFAULT_MAX_CLUSTER_SIZE
    )
    p.add_argument("--cp-time-limit", type=float, default=60.0)
    p.add_argument("--refine-passes", type=int, default=0)
    p.add_argument("--dwave-time-limit", type=float, default=5.0)
    p.add_argument("--sa-num-reads", type=int, default=200)
    p.add_argument("--results-dir", type=Path, default=None)
    p.add_argument("--no-write", action="store_true")
    p.add_argument("--stop-on-error", action="store_true")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    # Deduplicate / validate backends against known set
    backends = []
    for b in args.backends:
        if b not in DEFAULT_BACKENDS:
            # allow only the four requested, but experiments.solve_clusters
            # supports more — keep requested default list
            pass
        backends.append(b)

    summary = run_suite(
        sizes=args.sizes,
        slots=args.slots,
        backends=backends or DEFAULT_BACKENDS,
        max_cluster_size=args.max_cluster_size,
        cp_time_limit=args.cp_time_limit,
        refine_passes=args.refine_passes,
        results_dir=args.results_dir,
        write_json=not args.no_write,
        continue_on_error=not args.stop_on_error,
        dwave_time_limit=args.dwave_time_limit,
        sa_num_reads=args.sa_num_reads,
    )
    ok = any(
        r.get("status") in ("OPTIMAL", "FEASIBLE")
        for r in summary.get("table", [])
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
