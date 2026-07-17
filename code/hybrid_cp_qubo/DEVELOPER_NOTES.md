Developer Notes

Hybrid **CP master + quantum/Ising subproblem solvers** for Time-Dependent TSP  
(report: `docs/TDTSP_potential_approaches/TDTSP_approaches_report.pdf`, §3).

These notes describe what was implemented in `code/hybrid_cp_qubo/`, why key
design choices were made, and how to run / extend the code.

---

## 1. Goal

Close the gap in the existing SuperQ TD-TSP paper pipeline, where tours are
optimized on a **slot-scaled static** matrix \(D_t = m_t D_{\text{base}}\) and
only afterward scheduled in time.

Approach 2 instead:

- Uses a **true time-dependent** cost \(D_{ij}(t)\) (Melgarejo / Lyon benchmark).
- Lets a **CP-SAT master** own arrival-time propagation (FIFO).
- Delegates only **small static TSP subproblems** to QUBO / Hybrid CQM when
  clustering is active.

**Objective (Melgarejo convention):** minimize **return time** — arrival back
at the depot — not “duration after subtracting start service.”

**Scope (current):** `Instances_noTW` only (no forbidden time windows).

---

## 2. Package layout

```
code/hybrid_cp_qubo/
├── td_data.py          # Melgarejo matrix + noTW instance loader, FIFO oracle
├── clustering.py       # MDS embedding + K-means; use_qubo flag
├── qubo_tsp.py         # One-hot TSP → dimod BQM (Lucas / repo QAOA style)
├── subsolver_SA.py     # Local SimulatedAnnealingSampler (BQM)
├── subsolver_dwave.py  # LeapHybridCQMSampler (matches repo D-Wave path)
├── cp_base.py          # CP-SAT full TD-TSP (CP-only path)
├── cp_master.py        # Cluster QUBO + CP-SAT timed stitching (hybrid path)
├── solver.py           # End-to-end orchestrator + CLI
├── __init__.py
└── DEVELOPER_NOTES.md  # this file
```

Results are written under:

```
results/hybrid_cp_qubo/
```

Data lives in:

```
Benchmarking_data/Melgarejo/
  Matrices/matrix{00,10,20}.txt
  Instances/Instances_noTW/{10,20,30,50,100}/inst_*.txt
  results_noTW.txt          # reference objectives (* = proven optimal)
```

---

## 3. Two execution paths

Controlled by `cluster_visits(..., max_cluster_size=10)`:

| Condition | `use_qubo` | Path | Module |
|-----------|------------|------|--------|
| `p ≤ max_cluster_size` | `False` | CP solves the **entire** TD-TSP | `cp_base.solve_tdtsp_cp` |
| `p > max_cluster_size` | `True` | Static QUBO/CQM per cluster → CP stitches | `cp_master.solve_cluster_qubo_cp` |

**Rationale for CP-only on small instances:** a single-cluster QUBO would freeze
costs at one entry time and throw away the time dependency the CP master is
meant to own. QUBO is used only when decomposition is necessary for scale.

```text
solver.py
   │
   ├─ use_qubo=False ──► cp_base (full circuit + Dij(t))
   │
   └─ use_qubo=True  ──► per-cluster SA/D-Wave (static)
                              │
                              └─► cp_master (block constraints + Dij(t))
```

---

## 4. Module notes

### 4.1 `td_data.py` — time-dependent data layer

- **`TravelTimeMatrix`:** loads Melgarejo `matrixNN.txt`  
  header `n m d` → tensor shape `(n, n, m)`; `d` seconds per step.  
  Lyon: `n=255`, `m=130`, `d=360` (horizon 13 h).
- **`TDTSPInstance`:** parses noTW visit lines  
  `v_i  d_i  f_i` with **`f_i` must be 0** (TW files are rejected).  
  `v_i` is a vertex index into the matrix (not lat/lng).  
  Depot = visit 0; `start_time = service[0]` (Melgarejo convention).
- **`TDCostModel`:** walks a tour with stepwise `Dij(t)`, returns  
  `return_time`, arrivals/departures, edge travel times.  
  Also builds `static_cluster_matrix(visits, entry_time)` for QUBO freezes.

Format reference:  
https://perso.liris.cnrs.fr/christine.solnon/TDTSP.html

### 4.2 `clustering.py` — cluster-first decomposition

- Melgarejo has no coordinates → static distance slice at `t0` via `slice_at`,
  symmetrize, **MDS → 2-D**, then **K-means**.
- Depot **is included** in clustering (not kept as a singleton).
- Default `max_cluster_size=10`.
- Returns `ClusterResult(clusters, use_qubo, coords, t0, ...)`.

### 4.3 `qubo_tsp.py` — static TSP BQM

- Same one-hot encoding as `code/refined/solvers/tdtsp_cluster_qaoa.py`
  (`_tsp_to_qubo`): \(x[i,t]=1\) iff city \(i\) at position \(t\).
- Penalty default: `2 * max(dm) * n`.
- Returns `TSPQUBO(bqm, n, penalty, var_index)` plus `decode_tour` / `tour_cost`.
- Used by the **SA** backend only.

### 4.4 `subsolver_SA.py` / `subsolver_dwave.py`

| Backend | Model | Sampler | Credentials |
|---------|--------|---------|-------------|
| SA | BQM from `qubo_tsp` | `SimulatedAnnealingSampler` | none |
| D-Wave | **CQM** (hard one-hot constraints) | `LeapHybridCQMSampler` | token |

D-Wave path intentionally matches `tdtsp_cluster_dwave.py` / autonomous D-Wave
(Hybrid **CQM**, not BQM/QPU embedding).

Token loading (either name):

- `DWAVE_API_TOKEN` (repo standard in `.env`)
- `DWAVE_API_KEY` (also accepted; present in root `env` file)

Loads from repo-root `.env` and `env`.

**Security:** `.env` is gitignored via `*.env`; plain `env` is **not** — prefer
`.env` or add `/env` to `.gitignore` before committing.

### 4.5 `cp_base.py` — full CP-SAT TD-TSP

Formerly named `cp_master.py` (renamed so “master” means the hybrid orchestrating
CP layer).

Model:

- `AddCircuit` on visit indices  
- Integer arrive / depart; depart bucket `min(⌊depart/d⌋, m-1)`  
- `AddElement` for stepwise travel profiles  
- Minimize `return_time` on the arc back to the depot  
- Precedences supported via tour positions (`pos[pred] < pos[succ]`)

Validated: `inst_10_1` + `matrix00` → **return_time = 7605** (matches
`results_noTW.txt` optimum `7605*`) within a 60s limit (`FEASIBLE`, not always
proven `OPTIMAL`).

### 4.6 `cp_master.py` — hybrid stitching master

Formerly `stitch.py`.

1. **`solve_cluster_tours`:** freeze each cluster at an entry time (initially
   `start_time`); solve static TSP with SA or D-Wave; map local tours → visit
   cycles.
2. **`stitch_clusters_cp`:** same timed circuit as `cp_base`, plus **block
   constraints**: each QUBO cycle may only be rotated (contiguous path arcs).
   Depot cluster must use the rotation that starts at visit 0.
3. **`solve_cluster_qubo_cp`:** optional **refine passes** — update cluster
   entry times from the stitched tour, re-QUBO, re-stitch.

Validated: `inst_20_1` + `matrix00` + SA → `OPTIMAL` under block constraints,
return_time **14844** (Melgarejo full-optimum reference **12836\***). Gap is
expected (static cluster freezes + restricted intra-cluster orders).

### 4.7 `solver.py` — orchestrator / CLI

```bash
# CP-only (n ≤ 10 with default max_cluster_size)
python code/hybrid_cp_qubo/solver.py --size 10 --index 1 --matrix matrix00

# Hybrid (n > 10)
python code/hybrid_cp_qubo/solver.py --size 20 --index 1 --matrix matrix00 \
  --qubo-backend sa --refine-passes 1

# Hybrid with D-Wave Leap
python code/hybrid_cp_qubo/solver.py --size 20 --index 1 --matrix matrix00 \
  --qubo-backend dwave
```

Useful flags: `--max-cluster-size`, `--cp-time-limit`, `--refine-passes`,
`--results-dir`, `--no-write`.

JSON naming:

- CP-only: `inst_{n}_{k}_{matrix}_cp.json`
- Hybrid: `inst_{n}_{k}_{matrix}_qubo_{sa|dwave}.json`

---

## 5. Design decisions (summary)

1. **Melgarejo benchmark** for \(D_{ij}(t)\), not NYC slot multipliers.  
2. **noTW only** for the first implementation.  
3. **Return time** as the sole objective (aligned with `results_noTW.txt`).  
4. **Depot in K-means**; `max_cluster_size=10`.  
5. **CP-only when `p ≤ 10`**; QUBO only when clustering is active.  
6. **SA = BQM**; **D-Wave = Hybrid CQM** (repo parity).  
7. Package name **`hybrid_cp_qubo`** (avoids invalid `+` in import paths).  
8. Naming: **`cp_base`** = full CP solver; **`cp_master`** = hybrid stitch master.

---

## 6. Dependencies

Beyond the repo `requirements.txt`:

- `ortools` (CP-SAT) — installed in the project venv (cp314 wheel).  
- D-Wave Ocean SDK (already used elsewhere) for SA + Leap CQM.  
- `scikit-learn` for MDS / K-means.  
- `python-dotenv` for credentials.

Activate venv before running:

```bash
source venv/bin/activate
```

---

## 7. Self-tests

Each module exposes `_self_test()` / `python <module>.py`:

| Module | What it checks |
|--------|----------------|
| `td_data.py` | Load matrix00 + inst_10_1; trivial tour timing |
| `clustering.py` | n=10 → `use_qubo=False`; n=20 → 2 clusters |
| `qubo_tsp.py` | Known 4-city tour; BQM energy = static cost |
| `subsolver_SA.py` | SA recovers known optimum |
| `subsolver_dwave.py` | Builds CQM; live Leap if token present |
| `cp_base.py` | inst_10_1 → 7605 |
| `cp_master.py` | inst_20_1 hybrid path |
| `solver.py` | Full CLI orchestrator |

---

## 8. Known limitations / next work

- **TW / precedences:** TW rejected; precedences are in the CP models but
  Melgarejo noTW instances used so far usually have `q=0`.  
- **Horizon clamp:** departure past the last Melgarejo bucket stays on the
  final speed profile.  
- **Hybrid quality gap:** cluster + static QUBO cannot match full TD-TSP
  optima (see n=20: 14844 vs 12836*).  
- **K-means sizes:** `ceil(p/max)` does not hard-cap cluster cardinality;
  uneven splits are possible.  
- **D-Wave cost/latency:** each cluster is a Leap Hybrid CQM call.  
- **No 2-opt** on the hybrid global tour (unlike refined cluster solvers).  
- Possible follow-ups: TW support, more refine iterations, shared timed-circuit
  helper between `cp_base` and `cp_master`, benchmark harness over all
  Melgarejo sizes, gitignore for root `env`.

---

## 9. End-to-end smoke results (reference)

| Run | Path | Status | return_time | Melgarejo ref |
|-----|------|--------|-------------|----------------|
| inst_10_1 / matrix00 | `cp_only` | FEASIBLE | **7605** | 7605* |
| inst_20_1 / matrix00 / SA | `cluster_qubo_cp` | OPTIMAL† | **14844** | 12836* |

† Optimal under QUBO block constraints, not the unrestricted TD-TSP.

---

## 10. Mental model 

> The quantum / Ising device solves only small, well-isolated **static** TSP
> subproblems. A classical **CP master** guarantees the global route is timed
> against real, arrival-dependent travel costs.

- Small \(n\): CP *is* the solver (`cp_base`).  
- Large \(n\): CP *is* the master (`cp_master`); QUBO fills in the clusters.
