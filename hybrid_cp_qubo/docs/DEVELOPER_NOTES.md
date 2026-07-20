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
- **Small static TSP subproblems** are used only when the instance is large
  enough to need clustering. Intra-cluster order is then fixed (up to
  rotation); the CP master stitches clusters and times the tour.

Objective (Melgarejo convention): minimize **return time** (arrival back at
the depot). Current scope: **no time windows** (`Instances_noTW` only).

## Layout

```
hybrid_cp_qubo/
├── code/
│   ├── td_data.py               # Melgarejo matrix + noTW instance loader, FIFO oracle
│   ├── json_instances.py        # Adapter: data/instances JSON → hybrid types
│   ├── clustering.py            # MDS + K-means; hard-caps cluster size
│   ├── qubo_tsp.py              # One-hot TSP → dimod BQM
│   ├── subsolver_SA.py          # Local simulated annealing (BQM)
│   ├── subsolver_dwave.py       # D-Wave Leap Hybrid CQM
│   ├── cp_base.py               # Full CP-SAT TD-TSP (small instances)
│   ├── cp_master.py             # Cluster cycles + CP timed stitching
│   ├── solver.py                # Single-run Melgarejo orchestrator + CLI
│   ├── experiments.py           # Melgarejo suite (max cluster 5)
│   └── experiments_original.py  # NYC JSON suite (max cluster 15)
├── data/
│   └── Melgarejo/               # Matrices + Instances_noTW (+ results_noTW.txt)
├── docs/
│   └── DEVELOPER_NOTES.md
└── results/
    ├── suite_summary_matrix00.json   # Melgarejo suite
    └── original/                     # NYC JSON suite
```

## Experiment protocol (`experiments.py`)

| Condition | Path |
|-----------|------|
| \(p \le 10\) | **CP-SAT on the full instance** (`cp_base`) — no clustering |
| \(p > 10\) | Cluster with **`max_cluster_size = 5`**, solve each cluster’s static TSP with one backend, then **CP-SAT timed stitch** |

Cluster backends for \(p > 10\):

1. `cp` — OR-Tools CP-SAT static TSP  
2. `gurobi` — Gurobi MTZ (`refined/solvers`)  
3. `dwave_sa` — D-Wave `SimulatedAnnealingSampler` on BQM (local)  
4. `dwave_leap` — Leap `LeapHybridCQMSampler` (`refined/solvers`)  
5. `qaoa` — AWS Braket QAOA (`refined/solvers`, SV1 by default)  
6. `quanfluence` — Quanfluence Ising (`refined/solvers`)

```bash
source venv/bin/activate
# Default suite: inst_10_1, inst_10_2, inst_20_1, inst_20_2, inst_30_1
python hybrid_cp_qubo/code/experiments.py

# Local cluster backends only (skip cloud)
python hybrid_cp_qubo/code/experiments.py --local-only

# Custom instances
python hybrid_cp_qubo/code/experiments.py --instances 20:1 30:1 --backends cp gurobi
```

Outputs: `hybrid_cp_qubo/results/` (per-run JSON + `suite_summary_matrix00.json`).

**Credentials** (repo-root `.env` / `env`): D-Wave (`DWAVE_API_TOKEN` or
`DWAVE_API_KEY`); QAOA (`AWS_*`, `BRAKET_S3_*`); Quanfluence
(`QUANFLUENCE_USERNAME`, `QUANFLUENCE_PASSWORD`). Empty placeholders in
`.env` cause those backends to error.

---

## Suite results (matrix00, refine_passes=0, CP stitch limit 60 s)

Five Melgarejo noTW instances. Published references from `results_noTW.txt`
(`*` = proven optimal).

### \(p \le 10\): full CP only

| Instance | Backend | Return time | Reference | Gap |
|----------|---------|-------------|-----------|-----|
| `inst_10_1` | `cp_full` | **7605** | 7605\* | **0.0%** |
| `inst_10_2` | `cp_full` | **6854** | 6854\* | **0.0%** |

### \(p > 10\): cluster (max 5) + stitch

| Instance | Backend | Return time | Reference | Gap | Clusters |
|----------|---------|-------------|-----------|-----|----------|
| `inst_20_1` | cp / gurobi / dwave_sa / dwave_leap | **13475** | 12836\* | 5.0% | [3,2,2,4,2,4,3] |
| `inst_20_1` | qaoa / quanfluence | — | — | ERROR (empty AWS / Quanfluence creds in `.env`) | — |
| `inst_20_2` | cp / gurobi / dwave_sa | 14287 | 13176\* | 8.4% | [2,4,2,5,1,2,4] |
| `inst_20_2` | **dwave_leap** | **14263** | 13176\* | **8.2%** | same |
| `inst_20_2` | qaoa / quanfluence | — | — | ERROR (creds) | — |
| `inst_30_1` | **cp** | **18532** | 17167 | **8.0%** | [5,5,4,2,5,4,2,3] |
| `inst_30_1` | dwave_sa | 19009 | 17167 | 10.7% | same |
| `inst_30_1` | dwave_leap | 19100 | 17167 | 11.3% | same |
| `inst_30_1` | gurobi | 19235 | 17167 | 12.0% | same |
| `inst_30_1` | qaoa / quanfluence | — | — | ERROR (creds) | — |

Wall time is dominated by the **60 s CP stitch** budget (~60–112 s per hybrid
run). Cluster subproblems of size ≤ 5 are cheap for classical / Leap backends.

---

## Conclusions (Melgarejo)

1. **Small instances (\(p \le 10\)) should stay full CP.** On `inst_10_1` and
   `inst_10_2`, CP-SAT matched the Melgarejo optima exactly (gap 0). Clustering
   is unnecessary and would only restrict the search.

2. **With max cluster size 5, cluster TSP is easy — backends largely agree.**
   On `inst_20_1`, all four successful backends produced the same return time
   (13475). Closed static cluster costs matched across solvers; differences were
   mostly rotations of the same cycles. At this scale, CP / Gurobi / SA / Leap
   are interchangeable for the *static* subproblem.

3. **Hybrid quality is limited by decomposition, not by the cluster solver.**
   Gaps vs published TD-TSP optima are about **5–12%** on n=20–30. Freezing
   intra-cluster order at a single entry time (and restricting the master to
   those cycles) leaves a systematic gap even when every cluster TSP is solved
   optimally for the static freeze.

4. **Timed CP stitching dominates both runtime and residual variance.** The
   stitch is capped at 60 s (`FEASIBLE`, not always `OPTIMAL` under block
   constraints). On `inst_30_1`, CP and Gurobi returned *identical* cluster
   cycles and static costs but different stitched return times (18532 vs
   19235) — evidence that incomplete CP search / tie-breaking in the master,
   not the cluster backend, drove the spread. Prefer a longer stitch limit (or
   refine passes) before concluding that one cluster solver is better.

5. **Leap Hybrid CQM is competitive and occasionally slightly better.** On
   `inst_20_2`, `dwave_leap` edged out the others (14263 vs 14287). On
   `inst_30_1` it was mid-pack. Local `dwave_sa` is a good free baseline;
   Leap adds cloud latency (~80–110 s total here including stitch).

6. **QAOA and Quanfluence were not evaluated in this suite run.** Repo `.env`
   has empty `AWS_*` / `QUANFLUENCE_*` placeholders (only `DWAVE_API_KEY` in
   `env` was populated). After filling credentials, re-run:

   ```bash
   python hybrid_cp_qubo/code/experiments.py \
     --instances 20:1 20:2 30:1 --backends qaoa quanfluence
   ```

7. **Practical recommendation.** Use **full CP for \(p \le 10\)**. For larger
   instances, cluster at size ≤ 5 and prefer a **reliable classical cluster
   solver** (`cp` or `gurobi`) or **local SA** for development; use
   **`dwave_leap`** when comparing quantum/hybrid hardware. Invest budget in
   the **CP stitch / refine** stage — that is where quality and variance
   currently come from.

---

## Original NYC JSON instances (`data/instances/`)

Suite driver: `experiments_original.py`. Adapter: `json_instances.py`.

### Protocol

| Setting | Value |
|---------|-------|
| Instances | `tdtsp_n{5,10,25,50,100}.json` (NYC Google Maps durations) |
| Time slots | `morning_peak`, `midday`, `evening_peak`, `night` |
| Cluster backends | `cp`, `gurobi`, `dwave_sa`, `dwave_leap` |
| `max_cluster_size` | **15** → full CP when \(n \le 15\); hybrid when \(n > 15\) |
| TD model | Hourly stepwise \(D_{ij}(t) = \mathrm{round}(\mathrm{base}_{ij} \cdot m_{\mathrm{slot}(h)})\) |
| Reported metric | **Tour duration** (seconds) = return_time − start_time |
| CP stitch limit | 60 s; refine_passes = 0 |

```bash
python hybrid_cp_qubo/code/experiments_original.py
# subset example:
python hybrid_cp_qubo/code/experiments_original.py --sizes 25 50 --slots morning_peak
```

Outputs: `hybrid_cp_qubo/results/original/` (56 runs, all successful;
`suite_summary_original.json`).

### Results — tour duration (seconds)

#### \(n \le 15\): full CP (OPTIMAL)

| n | morning_peak | midday | evening_peak | night |
|---|-------------:|-------:|-------------:|------:|
| 5 | 3829 | 3837 | 4012 | **3271** |
| 10 | 6737 | 6616 | 6788 | **5374** |

Night is shortest; evening peak is longest (as expected from traffic multipliers).

#### \(n = 25, 50\): hybrid (OPTIMAL under block constraints)

CP = Gurobi = **Leap** on every slot; local SA is clearly worse (~20–25%).

| n | Slot | cp / gurobi / leap | dwave_sa |
|---|------|-------------------:|---------:|
| 25 | morning_peak | **20770** | 24754 |
| 25 | midday | **18789** | 23238 |
| 25 | evening_peak | **18268** | 22533 |
| 25 | night | **14849** | 19006 |
| 50 | morning_peak | **30610** | 37201 |
| 50 | midday | **28522** | 36203 |
| 50 | evening_peak | **25868** | 32499 |
| 50 | night | **22220** | 29284 |

#### \(n = 100\): hybrid (FEASIBLE stitch in 60 s)

Backends diverge; Leap wins 2 slots, Gurobi wins 2; SA remains worst.

| Slot | cp | gurobi | dwave_sa | dwave_leap | Best |
|------|---:|-------:|---------:|-----------:|------|
| morning_peak | 50581 | 51016 | 60389 | **50270** | leap |
| midday | 46117 | 45933 | 52104 | **45731** | leap |
| evening_peak | 40382 | **39681** | 46973 | 41087 | gurobi |
| night | 37495 | **36842** | 44642 | 38247 | gurobi |

### Conclusions (original JSON)

1. **Full CP for \(n \le 15\)** is appropriate and returns proven optima on
   these instances; time-of-day effects are clear (night ≪ peaks).

2. **At max cluster size 15, Leap matches classical cluster optima on
   \(n=25\) and \(n=50\)** (exact ties with CP/Gurobi). Local SA is not
   competitive at this cluster size — unlike Melgarejo with max size 5,
   where SA often matched the others.

3. **On \(n=100\), Leap is useful:** best duration on morning and midday;
   Gurobi best on evening and night. Mean durations are close
   (Gurobi ~43368 s, Leap ~43834 s, CP ~43644 s; SA ~51027 s). Stitch
   status is only `FEASIBLE` within 60 s, so rankings can still move with
   a longer master budget.

4. **D-Wave benefit is scale-dependent.** With tiny Melgarejo clusters
   (≤5) Leap rarely helped; with NYC clusters up to 15 it ties classical
   solvers on medium sizes and wins half the slots at \(n=100\). Prefer
   Leap when cluster subproblems are large enough that heuristic/quantum
   hybrid search can matter; use CP/Gurobi as the classical reference.
