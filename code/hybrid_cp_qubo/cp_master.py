"""
CP master for large instances: cluster QUBO + timed stitching.

Pipeline when ``use_qubo=True``:
    1. Freeze each cluster's costs at an entry time (initially start_time).
    2. Solve each cluster as a static TSP (SA BQM or D-Wave Hybrid CQM).
    3. CP-SAT master builds a global timed tour that:
         - respects each cluster's QUBO cycle up to rotation (block arcs),
         - starts at the depot,
         - propagates true Melgarejo ``Dij(t)``,
         - minimizes return time.
    4. Optional second pass: re-freeze clusters at the discovered entry
       times, re-solve QUBOs, stitch again.

For the CP-only full TD-TSP (no QUBO) see ``cp_base.py``.
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional, Sequence

import numpy as np
from ortools.sat.python import cp_model

try:
    from .td_data import TravelTimeMatrix, TDTSPInstance, TDCostModel
    from .clustering import ClusterResult
    from .subsolver_SA import solve_tsp_sa
    from .subsolver_dwave import solve_tsp_dwave
except ImportError:  # `python cp_master.py`
    from td_data import TravelTimeMatrix, TDTSPInstance, TDCostModel
    from clustering import ClusterResult
    from subsolver_SA import solve_tsp_sa
    from subsolver_dwave import solve_tsp_dwave


def _greedy_tour(dm: np.ndarray) -> List[int]:
    n = dm.shape[0]
    tour = [0]
    remaining = set(range(1, n))
    while remaining:
        cur = tour[-1]
        nxt = min(remaining, key=lambda j: dm[cur, j])
        tour.append(nxt)
        remaining.remove(nxt)
    return tour


def solve_cluster_tours(
    instance: TDTSPInstance,
    matrix: TravelTimeMatrix,
    clusters: Sequence[Sequence[int]],
    entry_times: Sequence[float],
    *,
    qubo_backend: str = "sa",
    sa_num_reads: int = 200,
    dwave_time_limit: float = 5.0,
) -> Dict:
    """
    Solve a static TSP for each cluster at its frozen entry time.

    Returns
    -------
    dict with ``cycles`` (visit-index cycles), ``local_tours``,
    ``static_costs``, ``entry_times``, ``backend``, ``qubo_time_seconds``.
    """
    if len(clusters) != len(entry_times):
        raise ValueError("clusters and entry_times length mismatch")

    cost = TDCostModel(matrix, instance)
    cycles: List[List[int]] = []
    local_tours: List[List[int]] = []
    static_costs: List[float] = []
    backend = qubo_backend.lower()
    t0 = time.perf_counter()

    for visits, t_entry in zip(clusters, entry_times):
        visits = list(visits)
        if len(visits) == 1:
            cycles.append(visits)
            local_tours.append([0])
            static_costs.append(0.0)
            continue
        if len(visits) == 2:
            local = [0, 1]
            cycles.append([visits[0], visits[1]])
            local_tours.append(local)
            dm = cost.static_cluster_matrix(visits, t_entry)
            static_costs.append(float(dm[0, 1] + dm[1, 0]))
            continue

        dm = cost.static_cluster_matrix(visits, t_entry)
        if backend == "sa":
            result = solve_tsp_sa(dm, num_reads=sa_num_reads)
            local = result["tour"]
            feasible = result["feasible"]
        elif backend == "dwave":
            result = solve_tsp_dwave(dm, time_limit=dwave_time_limit)
            local = result["tour"]
            feasible = result["feasible"]
        else:
            raise ValueError(f"Unknown qubo_backend={qubo_backend!r}")

        if not feasible or local is None or len(local) != len(visits):
            local = _greedy_tour(dm)

        cycle = [visits[i] for i in local]
        cycles.append(cycle)
        local_tours.append(list(local))
        # closed static cost in local indexing
        static_costs.append(
            float(sum(dm[local[k], local[(k + 1) % len(local)]]
                      for k in range(len(local))))
        )

    return {
        "cycles": cycles,
        "local_tours": local_tours,
        "static_costs": static_costs,
        "entry_times": list(entry_times),
        "backend": backend,
        "qubo_time_seconds": time.perf_counter() - t0,
    }


def stitch_clusters_cp(
    instance: TDTSPInstance,
    matrix: TravelTimeMatrix,
    cluster_cycles: Sequence[Sequence[int]],
    *,
    time_limit_seconds: float = 60.0,
    num_workers: int = 8,
    log_search_progress: bool = False,
) -> Dict:
    """
    CP-SAT global tour: timed circuit + QUBO cycle block constraints.

    Each ``cluster_cycles[c]`` is a cyclic visit order from the QUBO.
    CP chooses a rotation of each cycle (contiguous block) and the
    inter-cluster connections, while minimizing return time under ``Dij(t)``.
    """
    p = instance.p
    cycles = [list(c) for c in cluster_cycles]
    flat = sorted(v for c in cycles for v in c)
    if flat != list(range(p)):
        raise ValueError(
            "cluster_cycles must partition visit indices 0..p-1"
        )

    verts = instance.vertices
    services = [int(v.service) for v in instance.visits]
    start_time = int(instance.start_time)
    step = int(matrix.d)
    n_steps = int(matrix.m)
    horizon = int(matrix.horizon + matrix.tensor.max() * p + sum(services))

    profiles: List[List[List[int]]] = [
        [
            [int(matrix.tensor[verts[i], verts[j], s]) for s in range(n_steps)]
            for j in range(p)
        ]
        for i in range(p)
    ]
    max_tt = max(max(max(row) for row in dest) for dest in profiles)

    model = cp_model.CpModel()

    arc: Dict[tuple, cp_model.IntVar] = {}
    circuit_arcs = []
    for i in range(p):
        for j in range(p):
            if i == j:
                continue
            lit = model.NewBoolVar(f"arc_{i}_{j}")
            arc[i, j] = lit
            circuit_arcs.append((i, j, lit))
    model.AddCircuit(circuit_arcs)

    arrive = [model.NewIntVar(0, horizon, f"arr_{i}") for i in range(p)]
    depart = [model.NewIntVar(0, horizon, f"dep_{i}") for i in range(p)]
    return_time = model.NewIntVar(0, horizon, "return_time")

    model.Add(arrive[0] == 0)
    model.Add(depart[0] == start_time)
    for i in range(1, p):
        model.Add(depart[i] == arrive[i] + services[i])

    bucket = [model.NewIntVar(0, n_steps - 1, f"bucket_{i}") for i in range(p)]
    for i in range(p):
        raw = model.NewIntVar(0, horizon // step + 1, f"raw_bucket_{i}")
        model.AddDivisionEquality(raw, depart[i], step)
        model.AddMinEquality(bucket[i], [raw, model.NewConstant(n_steps - 1)])

    pos = [model.NewIntVar(0, p - 1, f"pos_{i}") for i in range(p)]
    model.Add(pos[0] == 0)
    for i in range(p):
        for j in range(1, p):
            if i == j:
                continue
            model.Add(pos[j] == pos[i] + 1).OnlyEnforceIf(arc[i, j])

    for pred, succ in instance.precedences:
        model.Add(pos[pred] < pos[succ])

    for i in range(p):
        for j in range(p):
            if i == j:
                continue
            lit = arc[i, j]
            tt = model.NewIntVar(0, max_tt, f"tt_{i}_{j}")
            model.AddElement(bucket[i], profiles[i][j], tt)
            if j == 0:
                model.Add(return_time == depart[i] + tt).OnlyEnforceIf(lit)
            else:
                model.Add(arrive[j] == depart[i] + tt).OnlyEnforceIf(lit)

    # --- QUBO block constraints (rotation of each cluster cycle) ----------
    # Each entry is a list of (rotation_index, bool_var) options.
    rotation_options: List[List[tuple]] = []
    for cid, cycle in enumerate(cycles):
        k = len(cycle)
        options = []
        if k == 1:
            lit = model.NewBoolVar(f"rot_{cid}_0")
            model.Add(lit == 1)
            options.append((0, lit))
        else:
            for r in range(k):
                path = cycle[r:] + cycle[:r]
                # Depot cluster: only the rotation that starts at visit 0.
                if 0 in cycle and path[0] != 0:
                    continue
                lit = model.NewBoolVar(f"rot_{cid}_{r}")
                options.append((r, lit))
                for i in range(k - 1):
                    u, v = path[i], path[i + 1]
                    model.Add(arc[u, v] == 1).OnlyEnforceIf(lit)
            if not options:
                raise ValueError(
                    f"Cluster {cid} cycle {cycle} has no valid rotation"
                )
            model.AddExactlyOne([lit for _, lit in options])
        rotation_options.append(options)

    model.Minimize(return_time)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit_seconds)
    solver.parameters.num_search_workers = int(num_workers)
    solver.parameters.log_search_progress = bool(log_search_progress)

    status = solver.Solve(model)
    status_name = solver.StatusName(status)

    out: Dict = {
        "status": status_name,
        "solver": "CP-SAT stitch (OR-Tools) + cluster QUBO",
        "n": p,
        "return_time": None,
        "tour_open": None,
        "solve_time_seconds": solver.WallTime(),
        "objective_bound": solver.BestObjectiveBound(),
        "time_limit_seconds": float(time_limit_seconds),
        "rotations": None,
    }

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return out

    succ = {}
    for i in range(p):
        for j in range(p):
            if i == j:
                continue
            if solver.Value(arc[i, j]) == 1:
                succ[i] = j
                break

    tour = [0]
    seen = {0}
    cur = 0
    for _ in range(p - 1):
        cur = succ[cur]
        if cur in seen:
            break
        tour.append(cur)
        seen.add(cur)

    if len(tour) != p or set(tour) != set(range(p)):
        out["status"] = "FEASIBLE_BUT_TOUR_DECODE_FAILED"
        out["return_time"] = int(solver.Value(return_time))
        return out

    rotations = []
    for options in rotation_options:
        chosen = None
        for r, lit in options:
            if solver.Value(lit) == 1:
                chosen = r
                break
        rotations.append(chosen)

    cp_return = int(solver.Value(return_time))
    cost = TDCostModel(matrix, instance)
    timing = cost.evaluate_tour(tour)

    out.update({
        "tour_open": tour,
        "return_time": cp_return,
        "return_time_oracle": timing["return_time"],
        "timing": timing,
        "arrivals": timing["arrivals"],
        "departures": timing["departures"],
        "rotations": rotations,
    })
    return out


def _cluster_entry_times_from_tour(
    tour_open: Sequence[int],
    arrivals: Sequence[float],
    clusters: Sequence[Sequence[int]],
) -> List[float]:
    """
    Entry time of a cluster = arrival at the first cluster city in the tour.

    ``arrivals`` from ``TDCostModel.evaluate_tour`` is indexed by position
    in ``tour_with_return`` (length p+1). For position k in tour_open,
    arrival is ``arrivals[k]``.
    """
    pos_of = {visit: k for k, visit in enumerate(tour_open)}
    entry_times = []
    for cluster in clusters:
        first_pos = min(pos_of[v] for v in cluster)
        entry_times.append(float(arrivals[first_pos]))
    return entry_times


def solve_cluster_qubo_cp(
    instance: TDTSPInstance,
    matrix: TravelTimeMatrix,
    cluster_result: ClusterResult,
    *,
    qubo_backend: str = "sa",
    cp_time_limit: float = 60.0,
    refine_passes: int = 1,
    sa_num_reads: int = 200,
    dwave_time_limit: float = 5.0,
) -> Dict:
    """
    Full hybrid path: QUBO per cluster → CP stitch → optional refine.
    """
    clusters = cluster_result.clusters
    entry_times = [instance.start_time] * len(clusters)
    history = []

    best: Optional[Dict] = None

    for pass_id in range(1 + max(0, refine_passes)):
        qubo = solve_cluster_tours(
            instance,
            matrix,
            clusters,
            entry_times,
            qubo_backend=qubo_backend,
            sa_num_reads=sa_num_reads,
            dwave_time_limit=dwave_time_limit,
        )
        stitch = stitch_clusters_cp(
            instance,
            matrix,
            qubo["cycles"],
            time_limit_seconds=cp_time_limit,
        )
        pass_record = {
            "pass": pass_id,
            "entry_times": list(entry_times),
            "cycles": qubo["cycles"],
            "static_costs": qubo["static_costs"],
            "qubo_time_seconds": qubo["qubo_time_seconds"],
            "stitch_status": stitch["status"],
            "return_time": stitch.get("return_time"),
            "tour_open": stitch.get("tour_open"),
            "stitch_time_seconds": stitch.get("solve_time_seconds"),
        }
        history.append(pass_record)

        if stitch.get("tour_open") is None:
            break

        if best is None or (
            stitch["return_time"] is not None
            and (best.get("return_time") is None
                 or stitch["return_time"] < best["return_time"])
        ):
            best = {
                **stitch,
                "qubo_backend": qubo["backend"],
                "cluster_cycles": qubo["cycles"],
                "cluster_static_costs": qubo["static_costs"],
                "qubo_time_seconds": qubo["qubo_time_seconds"],
                "entry_times": list(entry_times),
                "refine_pass": pass_id,
            }

        if pass_id >= refine_passes:
            break

        # Update entry times from the stitched tour for the next QUBO freeze
        entry_times = _cluster_entry_times_from_tour(
            stitch["tour_open"],
            stitch["arrivals"],
            clusters,
        )

    if best is None:
        return {
            "status": history[-1]["stitch_status"] if history else "ERROR",
            "solver": "cluster QUBO + CP stitch",
            "return_time": None,
            "tour_open": None,
            "history": history,
        }

    best["history"] = history
    best["solver"] = (
        f"Cluster-{qubo_backend.upper()} QUBO + CP-SAT stitch"
    )
    return best


def _self_test() -> None:
    print("Self-test: CP master (cluster QUBO + timed stitch)")
    print("-" * 60)

    try:
        from .clustering import cluster_visits
    except ImportError:
        from clustering import cluster_visits

    mx = TravelTimeMatrix.load("matrix00")
    inst = TDTSPInstance.load(20, 1)
    cr = cluster_visits(inst, mx, max_cluster_size=10)
    print(f"  {inst.name}: p={inst.p} clusters={cr.cluster_sizes} "
          f"use_qubo={cr.use_qubo}")
    assert cr.use_qubo

    result = solve_cluster_qubo_cp(
        inst, mx, cr,
        qubo_backend="sa",
        cp_time_limit=60.0,
        refine_passes=1,
    )
    print(f"  status      : {result['status']}")
    print(f"  return_time : {result.get('return_time')}")
    print(f"  oracle      : {result.get('return_time_oracle')}")
    print(f"  tour        : {result.get('tour_open')}")
    print(f"  passes      : {len(result.get('history', []))}")
    assert result["status"] in ("OPTIMAL", "FEASIBLE")
    assert result["tour_open"] is not None
    assert abs(result["return_time"] - result["return_time_oracle"]) < 1e-6
    print("  OK")


if __name__ == "__main__":
    _self_test()
