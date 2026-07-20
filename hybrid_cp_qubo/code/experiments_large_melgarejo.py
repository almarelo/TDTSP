"""
Full CP vs hybrid on the largest Melgarejo noTW instances.

Compares wall time and return_time for:
  * ``cp_full``  — CP-SAT on the entire timed instance
  * ``hybrid``   — cluster (max size 5) + static cluster-CP + timed CP stitch

Default suite: inst_{50,100}_{1,2} / matrix00, CP budget 300 s each path.

    python hybrid_cp_qubo/code/experiments_large_melgarejo.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
LOCAL_CODE_DIR = Path(__file__).resolve().parent
if str(LOCAL_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(LOCAL_CODE_DIR))

from experiments import run_cp_full, run_hybrid_cluster, load_melgarejo_reference

DEFAULT_RESULTS_DIR = PACKAGE_ROOT / "results" / "large_melgarejo"
DEFAULT_SUITE: List[Tuple[int, int]] = [
    (50, 1),
    (50, 2),
    (100, 1),
    (100, 2),
]


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_comparison(
    size: int,
    index: int,
    matrix_name: str = "matrix00",
    *,
    cp_time_limit: float = 300.0,
    max_cluster_size: int = 5,
    hybrid_backend: str = "cp",
    refine_passes: int = 0,
    results_dir: Optional[Path] = None,
    write_json: bool = True,
) -> Dict[str, Any]:
    out_dir = Path(results_dir) if results_dir else DEFAULT_RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    ref = load_melgarejo_reference(size, index, matrix_name)

    print(
        f"\n{'=' * 60}\n"
        f"inst_{size}_{index}  matrix={matrix_name}  "
        f"ref={((ref or {}).get('raw'))}\n"
        f"{'=' * 60}",
        flush=True,
    )

    print(f"\n--- cp_full (time_limit={cp_time_limit}s) ---", flush=True)
    t0 = time.perf_counter()
    cp_rec = run_cp_full(
        size,
        index,
        matrix_name,
        cp_time_limit=cp_time_limit,
        results_dir=out_dir,
        write_json=write_json,
    )
    # Prefer measured wall; rewrite filename tag already cp_full
    cp_wall = cp_rec.get("wall_seconds")
    if cp_wall is None:
        cp_wall = time.perf_counter() - t0
    print(
        f"  status={cp_rec.get('status')}  return_time={cp_rec.get('return_time')}  "
        f"wall={cp_wall:.2f}s",
        flush=True,
    )

    print(
        f"\n--- hybrid cluster-{hybrid_backend} "
        f"(max_cluster={max_cluster_size}, stitch_limit={cp_time_limit}s) ---",
        flush=True,
    )
    t1 = time.perf_counter()
    hy_rec = run_hybrid_cluster(
        size,
        index,
        matrix_name,
        backend=hybrid_backend,
        max_cluster_size=max_cluster_size,
        cp_time_limit=cp_time_limit,
        refine_passes=refine_passes,
        results_dir=out_dir,
        write_json=write_json,
    )
    hy_wall = hy_rec.get("wall_seconds")
    if hy_wall is None:
        hy_wall = time.perf_counter() - t1
    print(
        f"  status={hy_rec.get('status')}  return_time={hy_rec.get('return_time')}  "
        f"clusters={hy_rec.get('cluster_sizes')}  wall={hy_wall:.2f}s",
        flush=True,
    )

    cp_rt = cp_rec.get("return_time")
    hy_rt = hy_rec.get("return_time")
    ref_rt = (ref or {}).get("return_time")

    def _gap(rt: Any) -> Optional[float]:
        if rt is None or ref_rt is None:
            return None
        return (float(rt) - float(ref_rt)) / float(ref_rt)

    comparison = {
        "experiment": "large_melgarejo_cp_vs_hybrid",
        "instance": f"inst_{size}_{index}.txt",
        "matrix": matrix_name,
        "n": size,
        "reference": ref,
        "cp_time_limit_seconds": cp_time_limit,
        "hybrid": {
            "backend": hybrid_backend,
            "max_cluster_size": max_cluster_size,
            "refine_passes": refine_passes,
        },
        "cp_full": {
            "status": cp_rec.get("status"),
            "return_time": cp_rt,
            "wall_seconds": cp_wall,
            "gap_to_reference": _gap(cp_rt),
            "objective_bound": cp_rec.get("objective_bound"),
            "results_path": cp_rec.get("results_path"),
            "error": cp_rec.get("error"),
        },
        "hybrid_run": {
            "status": hy_rec.get("status"),
            "return_time": hy_rt,
            "wall_seconds": hy_wall,
            "gap_to_reference": _gap(hy_rt),
            "cluster_sizes": hy_rec.get("cluster_sizes"),
            "n_clusters": hy_rec.get("n_clusters"),
            "cluster_solve_seconds": hy_rec.get("cluster_solve_seconds"),
            "results_path": hy_rec.get("results_path"),
            "error": hy_rec.get("error"),
        },
        "winner_quality": (
            "tie"
            if cp_rt is not None and hy_rt is not None and cp_rt == hy_rt
            else (
                "cp_full"
                if cp_rt is not None and (hy_rt is None or cp_rt < hy_rt)
                else (
                    "hybrid"
                    if hy_rt is not None
                    else "none"
                )
            )
        ),
        "winner_speed": (
            "tie"
            if abs(float(cp_wall) - float(hy_wall)) < 1e-6
            else ("cp_full" if float(cp_wall) < float(hy_wall) else "hybrid")
        ),
        "wall_clock_end_iso": _iso_now(),
    }

    if write_json:
        path = out_dir / f"compare_inst_{size}_{index}_{matrix_name}.json"
        path.write_text(json.dumps(comparison, indent=2) + "\n")
        comparison["results_path"] = str(path)
        print(f"  wrote {path}", flush=True)

    return comparison


def run_suite(
    instances: Sequence[Tuple[int, int]] = DEFAULT_SUITE,
    matrix_name: str = "matrix00",
    **kwargs: Any,
) -> Dict[str, Any]:
    out_dir = Path(kwargs.get("results_dir") or DEFAULT_RESULTS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    start = _iso_now()
    print("Large Melgarejo: full CP vs hybrid", flush=True)
    print("-" * 60, flush=True)
    print(f"  instances : {list(instances)}", flush=True)
    print(f"  matrix    : {matrix_name}", flush=True)
    print(f"  cp_limit  : {kwargs.get('cp_time_limit', 300)}s", flush=True)
    print(
        f"  hybrid    : max_cluster={kwargs.get('max_cluster_size', 5)} "
        f"backend={kwargs.get('hybrid_backend', 'cp')}",
        flush=True,
    )

    for size, index in instances:
        rows.append(
            run_comparison(size, index, matrix_name, **kwargs)
        )

    summary = {
        "experiment": "large_melgarejo_cp_vs_hybrid",
        "matrix": matrix_name,
        "wall_clock_start_iso": start,
        "wall_clock_end_iso": _iso_now(),
        "protocol": {
            "cp_time_limit_seconds": kwargs.get("cp_time_limit", 300),
            "max_cluster_size": kwargs.get("max_cluster_size", 5),
            "hybrid_backend": kwargs.get("hybrid_backend", "cp"),
            "refine_passes": kwargs.get("refine_passes", 0),
        },
        "comparisons": rows,
        "table": [
            {
                "instance": r["instance"],
                "n": r["n"],
                "reference": (r.get("reference") or {}).get("raw"),
                "cp_status": r["cp_full"]["status"],
                "cp_return_time": r["cp_full"]["return_time"],
                "cp_wall_s": r["cp_full"]["wall_seconds"],
                "cp_gap": r["cp_full"]["gap_to_reference"],
                "hy_status": r["hybrid_run"]["status"],
                "hy_return_time": r["hybrid_run"]["return_time"],
                "hy_wall_s": r["hybrid_run"]["wall_seconds"],
                "hy_gap": r["hybrid_run"]["gap_to_reference"],
                "hy_clusters": r["hybrid_run"].get("cluster_sizes"),
                "winner_quality": r["winner_quality"],
                "winner_speed": r["winner_speed"],
            }
            for r in rows
        ],
    }
    path = out_dir / f"suite_summary_large_{matrix_name}.json"
    path.write_text(json.dumps(summary, indent=2) + "\n")
    summary["summary_path"] = str(path)

    print("\nComparison table", flush=True)
    print("-" * 60, flush=True)
    for row in summary["table"]:
        print(
            f"  {row['instance']:16s}  ref={row['reference']}\n"
            f"    CP     status={row['cp_status']:10s}  "
            f"rt={row['cp_return_time']}  wall={row['cp_wall_s']:.1f}s  "
            f"gap={row['cp_gap']}\n"
            f"    Hybrid status={row['hy_status']:10s}  "
            f"rt={row['hy_return_time']}  wall={row['hy_wall_s']:.1f}s  "
            f"gap={row['hy_gap']}  clusters={row['hy_clusters']}\n"
            f"    winners: quality={row['winner_quality']}  "
            f"speed={row['winner_speed']}",
            flush=True,
        )
    print(f"\nSummary: {path}", flush=True)
    return summary


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="Full CP vs hybrid on largest Melgarejo instances"
    )
    p.add_argument(
        "--instances",
        nargs="+",
        default=None,
        help="SIZE:INDEX list (default 50:1 50:2 100:1 100:2)",
    )
    p.add_argument("--matrix", default="matrix00")
    p.add_argument("--cp-time-limit", type=float, default=300.0)
    p.add_argument("--max-cluster-size", type=int, default=5)
    p.add_argument(
        "--hybrid-backend",
        default="cp",
        choices=("cp", "gurobi", "dwave_sa", "dwave_leap"),
    )
    p.add_argument("--refine-passes", type=int, default=0)
    p.add_argument("--results-dir", type=Path, default=None)
    args = p.parse_args(argv)

    if args.instances:
        instances = []
        for item in args.instances:
            a, b = item.split(":")
            instances.append((int(a), int(b)))
    else:
        instances = list(DEFAULT_SUITE)

    summary = run_suite(
        instances,
        args.matrix,
        cp_time_limit=args.cp_time_limit,
        max_cluster_size=args.max_cluster_size,
        hybrid_backend=args.hybrid_backend,
        refine_passes=args.refine_passes,
        results_dir=args.results_dir,
    )
    ok = any(
        (r["cp_full"].get("return_time") is not None)
        or (r["hybrid_run"].get("return_time") is not None)
        for r in summary["comparisons"]
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
