"""
Hybrid CP master with quantum QUBO subproblem solvers.

Package layout (repo-root ``hybrid_cp_qubo/``):

    code/   - Python modules (this package)
    data/   - Melgarejo Matrices + Instances_noTW
    docs/   - developer notes
    results/- JSON run outputs

Modules:

    td_data.py         - Melgarejo FIFO Dij(t) matrices + no-TW instance loader
    clustering.py      - cluster-first decomposition
    qubo_tsp.py        - per-cluster static TSP-QUBO builder (BQM)
    subsolver_SA.py    - local SimulatedAnnealingSampler backend (BQM)
    subsolver_dwave.py - D-Wave Leap Hybrid CQM backend (matches repo)
    cp_base.py         - OR-Tools CP-SAT full TD-TSP (small / CP-only)
    cp_master.py       - cluster QUBO + CP-SAT timed stitching (large)
    solver.py          - end-to-end orchestrator
    experiments.py     - hybrid path with refined/solvers cluster backends

Scope: Melgarejo Instances_noTW only. Objective = return time.

Policy: if p <= max_cluster_size → CP-only via cp_base (use_qubo=False).
Otherwise → static QUBO/CQM per cluster + cp_master stitch (use_qubo=True).
"""
from .td_data import TravelTimeMatrix, TDTSPInstance, TDCostModel
from .clustering import ClusterResult, cluster_visits
from .qubo_tsp import TSPQUBO, build_tsp_qubo, decode_tour, tour_cost
from .subsolver_SA import solve_tsp_sa
from .subsolver_dwave import build_tsp_cqm, solve_tsp_dwave
from .cp_base import solve_tdtsp_cp
from .cp_master import solve_cluster_qubo_cp, stitch_clusters_cp
from .solver import solve

__all__ = [
    "TravelTimeMatrix",
    "TDTSPInstance",
    "TDCostModel",
    "ClusterResult",
    "cluster_visits",
    "TSPQUBO",
    "build_tsp_qubo",
    "decode_tour",
    "tour_cost",
    "solve_tsp_sa",
    "build_tsp_cqm",
    "solve_tsp_dwave",
    "solve_tdtsp_cp",
    "solve_cluster_qubo_cp",
    "stitch_clusters_cp",
    "solve",
]
