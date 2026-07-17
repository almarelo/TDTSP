# Hybrid CP + QUBO TD-TSP — Developer Notes

## Goal

Optimize Time-Dependent TSP (TD-TSP) against **true arrival-time-dependent
travel costs** \(D_{ij}(t)\), instead of solving a static TSP on a traffic
multiplier and only scheduling the tour afterward.

## Why this design

Urban travel times change with departure time. A single multiplier per time
slot scales every edge the same way, so the optimizer never “sees” cascading
timing effects.

This package splits the work:

- A **constraint-programming (CP-SAT) master** owns the global schedule and
  propagates FIFO arrival times with the real stepwise \(D_{ij}(t)\).
- **Small static TSP subproblems** (QUBO / Hybrid CQM) are used only when the
  instance is large enough to need clustering. Intra-cluster order is then
  fixed (up to rotation); the CP master stitches clusters and times the tour.

Objective (Melgarejo convention): minimize **return time** (arrival back at
the depot). Current scope: **no time windows** (`Instances_noTW` only).

## Implementation

```
code/hybrid_cp_qubo/
├── td_data.py          # Melgarejo matrix + noTW instance loader, FIFO oracle
├── clustering.py       # MDS + K-means; sets use_qubo
├── qubo_tsp.py         # One-hot TSP → dimod BQM
├── subsolver_SA.py     # Local simulated annealing (BQM)
├── subsolver_dwave.py  # D-Wave Leap Hybrid CQM (same style as repo D-Wave)
├── cp_base.py          # Full CP-SAT TD-TSP (small instances)
├── cp_master.py        # Cluster QUBO + CP timed stitching (large instances)
└── solver.py           # Orchestrator + CLI
```

**Data:** Lyon Melgarejo benchmark under `Benchmarking_data/Melgarejo/` —
travel-time tensors `matrix{00,10,20}.txt` (255 nodes × 130 steps × 360 s)
and visit lists in `Instances/Instances_noTW/`.

**Two paths** (`max_cluster_size` default 10; depot included in clustering):

| Condition | Path |
|-----------|------|
| \(p \le 10\) | `use_qubo=False` → `cp_base` solves the full timed tour |
| \(p > 10\) | `use_qubo=True` → static SA/D-Wave per cluster → `cp_master` stitches with \(D_{ij}(t)\) |

Small instances stay CP-only so a frozen QUBO does not erase time dependency.
Large instances freeze each cluster at an entry time, solve a static TSP, then
let CP choose cluster rotations/connections and propagate true travel times
(optional refine: re-freeze at discovered entry times and repeat).

**Credentials:** `DWAVE_API_TOKEN` or `DWAVE_API_KEY` in `.env` / `env` for
the D-Wave path. Prefer `.env` (gitignored).

```bash
source venv/bin/activate
python code/hybrid_cp_qubo/solver.py --size 10 --index 1 --matrix matrix00
python code/hybrid_cp_qubo/solver.py --size 20 --index 1 --matrix matrix00 \
  --qubo-backend sa
```

Outputs go to `results/hybrid_cp_qubo/`.

## Instances tested

| Instance | Matrix | Path | Result | Melgarejo reference |
|----------|--------|------|--------|---------------------|
| `inst_10_1` | `matrix00` | CP-only (`cp_base`) | return_time **7605** (`FEASIBLE` in 60 s) | **7605\*** |
| `inst_20_1` | `matrix00` | Cluster SA + `cp_master` | return_time **14844** (`OPTIMAL` under QUBO block constraints) | **12836\*** |

On n=10 the CP path matched the published optimum. On n=20 the hybrid is
feasible and timed correctly (oracle matches CP), but quality is below the
full TD-TSP optimum because cluster orders are restricted and intra-cluster
costs are static freezes—the expected tradeoff of the decomposition.
