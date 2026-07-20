"""
CP-SAT full TD-TSP solver (noTW) — used for the CP-only path.

Owns exact arrival-time propagation under Melgarejo stepwise ``Dij(t)``.
Objective = return time (arrival back at the depot).

Used when ``use_qubo=False`` (p <= max_cluster_size). For the hybrid
cluster-QUBO path see ``cp_master.py``.

Model (circuit + timed arcs):
    * Boolean arc literals + ``AddCircuit`` for a single Hamiltonian tour
    * Integer departure / arrival times per visit
    * Departure bucket ``s = min(floor(depart / d), m-1)``
    * Travel time via ``AddElement`` on the stepwise profile
    * ``return_time`` on the arc that re-enters the depot
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from ortools.sat.python import cp_model

try:
    from .td_data import TravelTimeMatrix, TDTSPInstance, TDCostModel
except ImportError:  # `python cp_base.py`
    from td_data import TravelTimeMatrix, TDTSPInstance, TDCostModel


def solve_tdtsp_cp(
    instance: TDTSPInstance,
    matrix: TravelTimeMatrix,
    *,
    time_limit_seconds: float = 60.0,
    num_workers: int = 8,
    log_search_progress: bool = False,
) -> Dict:
    """
    Solve a Melgarejo noTW TD-TSP with CP-SAT.

    Returns
    -------
    dict with keys
        ``status``, ``tour_open``, ``return_time``, ``solve_time_seconds``,
        ``objective_bound``, plus a ``timing`` block from ``TDCostModel``
        when a feasible tour is found (re-evaluated with the FIFO oracle).
    """
    p = instance.p
    if p < 2:
        raise ValueError("TD-TSP requires at least 2 visits")

    verts = instance.vertices
    services = [int(v.service) for v in instance.visits]
    start_time = int(instance.start_time)
    step = int(matrix.d)
    n_steps = int(matrix.m)
    # Allow a little slack beyond the published horizon for long tours
    horizon = int(matrix.horizon + matrix.tensor.max() * p + sum(services))

    # Integer travel profiles (Melgarejo files are integer-valued)
    profiles: List[List[List[int]]] = [
        [
            [int(matrix.tensor[verts[i], verts[j], s]) for s in range(n_steps)]
            for j in range(p)
        ]
        for i in range(p)
    ]
    max_tt = max(max(max(row) for row in dest) for dest in profiles)

    model = cp_model.CpModel()

    # --- circuit on visit indices ----------------------------------------
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

    # --- times -----------------------------------------------------------
    arrive = [model.NewIntVar(0, horizon, f"arr_{i}") for i in range(p)]
    depart = [model.NewIntVar(0, horizon, f"dep_{i}") for i in range(p)]
    return_time = model.NewIntVar(0, horizon, "return_time")

    model.Add(arrive[0] == 0)
    model.Add(depart[0] == start_time)
    for i in range(1, p):
        model.Add(depart[i] == arrive[i] + services[i])

    # Departure bucket s = min(floor(depart / step), n_steps - 1)
    bucket = [model.NewIntVar(0, n_steps - 1, f"bucket_{i}") for i in range(p)]
    for i in range(p):
        raw = model.NewIntVar(0, horizon // step + 1, f"raw_bucket_{i}")
        model.AddDivisionEquality(raw, depart[i], step)
        model.AddMinEquality(bucket[i], [raw, model.NewConstant(n_steps - 1)])

    # Position along the tour (depot at 0) — also encodes precedences
    pos = [model.NewIntVar(0, p - 1, f"pos_{i}") for i in range(p)]
    model.Add(pos[0] == 0)
    for i in range(p):
        for j in range(1, p):  # arcs into non-depot
            if i == j:
                continue
            model.Add(pos[j] == pos[i] + 1).OnlyEnforceIf(arc[i, j])

    for pred, succ in instance.precedences:
        model.Add(pos[pred] < pos[succ])

    # Timed arc linking
    for i in range(p):
        for j in range(p):
            if i == j:
                continue
            lit = arc[i, j]
            tt = model.NewIntVar(0, max_tt, f"tt_{i}_{j}")
            model.AddElement(bucket[i], profiles[i][j], tt)
            if j == 0:
                # Return to depot
                model.Add(return_time == depart[i] + tt).OnlyEnforceIf(lit)
            else:
                model.Add(arrive[j] == depart[i] + tt).OnlyEnforceIf(lit)

    model.Minimize(return_time)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit_seconds)
    solver.parameters.num_search_workers = int(num_workers)
    solver.parameters.log_search_progress = bool(log_search_progress)

    status = solver.Solve(model)
    status_name = solver.StatusName(status)

    out: Dict = {
        "status": status_name,
        "solver": "CP-SAT (OR-Tools)",
        "n": p,
        "instance": instance.name,
        "return_time": None,
        "tour_open": None,
        "solve_time_seconds": solver.WallTime(),
        "objective_bound": solver.BestObjectiveBound(),
        "time_limit_seconds": float(time_limit_seconds),
    }

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return out

    # Reconstruct tour by following arc literals from the depot
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

    cp_return = int(solver.Value(return_time))
    out["tour_open"] = tour
    out["return_time"] = cp_return

    # Re-evaluate with the FIFO oracle for a ground-truth schedule
    cost = TDCostModel(matrix, instance)
    timing = cost.evaluate_tour(tour)
    out["timing"] = timing
    out["return_time_oracle"] = timing["return_time"]
    out["arrivals"] = timing["arrivals"]
    out["departures"] = timing["departures"]

    return out


def _self_test() -> None:
    print("Self-test: CP-SAT full TD-TSP (cp_base, noTW)")
    print("-" * 60)

    mx = TravelTimeMatrix.load("matrix00")
    inst = TDTSPInstance.load(10, 1)
    print(f"  instance: {inst.name}  p={inst.p}  start={inst.start_time}s")
    print("  solving with CP-SAT (time_limit=60s)...")

    result = solve_tdtsp_cp(
        inst, mx,
        time_limit_seconds=60.0,
        num_workers=8,
        log_search_progress=False,
    )
    print(f"  status          : {result['status']}")
    print(f"  wall time       : {result['solve_time_seconds']:.2f}s")
    print(f"  tour            : {result['tour_open']}")
    print(f"  return_time (CP): {result['return_time']}")
    print(f"  return_time (oracle): {result.get('return_time_oracle')}")
    print(f"  bound           : {result['objective_bound']}")
    print("  ref. optimum (results_noTW): 7605*")

    assert result["status"] in ("OPTIMAL", "FEASIBLE")
    assert result["tour_open"] is not None
    assert result["return_time"] is not None
    # Oracle must agree with the CP return time (integer stepwise model)
    assert abs(result["return_time"] - result["return_time_oracle"]) < 1e-6
    # Should beat the trivial tour (10480) and approach the published opt
    assert result["return_time"] <= 10480
    print("  OK")


if __name__ == "__main__":
    _self_test()
