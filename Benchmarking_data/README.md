# Time-Dependent TSP Benchmarks

This document describes the datasets used for the **Time-Dependent Traveling Salesman Problem (TD-TSP)** cointained in this directory. It is based on the benchmarks used and provided in the experimentation done by the cited  publications.

In the TD-TSP the travel time between two locations depends on the **departure time** (to model traffic congestion), so the cost of a tour depends not only on the order of visits but also on *when* each leg is traversed. This breaks the symmetry, triangle-inequality, and cyclic-invariance assumptions of the classical TSP, so methods must be evaluated on data that genuinely exhibits these temporal dynamics rather than on synthetic or static-equivalent instances.

### Why time-dependent datasets matter

Benchmarking on real time-dependent travel-time data (rather than static distances or randomly perturbed weights) is essential because only such data reveals whether a solver actually exploits *when* to travel. Yang & Fan [2] analyze time-dependent travel-time data from 12 cities and observe that the travel time saved by accounting for time-dependent edge weights follows a **Pareto distribution**, consistently across all cities:

- Roughly **50% of instances show no change in the optimal tour** relative to their static (time-independent) counterpart — time-dependency is irrelevant for them.
- About **20% of instances account for over 80% of the total achievable time savings**, so the meaningful, hard cases are rare but disproportionately important.

This has two practical consequences the paper highlights, and which motivate careful dataset choice here:

- **Average metrics are misleading.** Because most instances are effectively static, reporting only the mean optimality gap or mean travel time over *all* instances blurs the line between general routing skill and the specific ability to exploit temporal patterns. Yang & Fan therefore advocate also evaluating on the *filtered* subset of instances where time-dependency changes the optimal solution.
- **Meaningful instances are scarce.** The small fraction of genuinely time-dependent instances complicates learning and evaluation (exact labels are expensive given the NP-hard nature of TD-TSP), which is exactly why curated, real-world benchmarks like the two datasets below are valuable.

### Publications

**[1] Melgarejo et al. (CPAIOR 2015)** — source of the `Melgarejo/` dataset (travel-time matrices, instances, and reference results).

> Pénélope Aguiar Melgarejo, Philippe Laborie, and Christine Solnon. **A Time-Dependent No-Overlap Constraint: Application to Urban Delivery Problems.** CPAIOR 2015, LNCS 9075, pp. 1–17, Springer. <https://perso.citi-lab.fr/csolnon/TDTSP.html>

```bibtex
@inproceedings{aguiarmelgarejo2015tdtsp,
  title     = {A Time-Dependent No-Overlap Constraint: Application to Urban Delivery Problems},
  author    = {Aguiar Melgarejo, P{\'e}n{\'e}lope and Laborie, Philippe and Solnon, Christine},
  booktitle = {Integration of AI and OR Techniques in Constraint Programming (CPAIOR)},
  series    = {LNCS},
  volume    = {9075},
  pages     = {1--17},
  publisher = {Springer},
  year      = {2015}
}
```

**[2] Yang & Fan (NeurIPS 2025)** — source of the `YangFan/` raw travel-time data (Beijing, Lyon, and the 10 cities) and rl4co environment code.

> Ruixiao Yang and Chuchu Fan. **Neural Combinatorial Optimization for Time Dependent Traveling Salesman Problem.** NeurIPS 2025. <https://openreview.net/pdf?id=UXTR6ZYV1x> · <https://github.com/Brelliothe/NCO4TDTSP>

```bibtex
@inproceedings{yang2025neural,
  title     = {Neural Combinatorial Optimization for Time Dependent Traveling Salesman Problem},
  author    = {Yang, Ruixiao and Fan, Chuchu},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2025},
  url       = {https://openreview.net/pdf?id=UXTR6ZYV1x}
}
```

The raw data sources compiled by the Yang & Fan work (for the `YangFan/data/tdtsp/` tensors):

- **Beijing** — Zhang et al. (2021), *Solving Dynamic Traveling Salesman Problems With Deep Reinforcement Learning*: <https://ieeexplore.ieee.org/document/9537638>
- **Lyon** — Melgarejo et al. (2015) (same as [1] above): <https://perso.citi-lab.fr/csolnon/TDTSP.html>
- **10 cities** — Blauth et al. (2022), *Vehicle Routing with Time-Dependent Travel Times: Theory, Practice, and Benchmarks*: <https://arxiv.org/abs/2205.00889v2>

### The two datasets

This directory is organized into one folder per publication, plus this README:

```
Benchmarking_data/
├── README.md
├── Melgarejo/                   # Dataset 1 - Melgarejo et al. [1]
│   ├── Matrices/
│   ├── Instances/
│   ├── results_noTW.txt
│   └── results_TW.txt
└── YangFan/                     # Dataset 2 - Yang & Fan [2]
    ├── fetch_data.sh                          # downloads the large binaries below
    ├── data/tdtsp/*.npy                       # 12 city travel-time tensors
    ├── testcases/beijing_20_dataset_10000.pt  # 10k 20-node Beijing test set
    └── routing/tdtsp/                         # rl4co TD-TSP environment code
        ├── env.py
        ├── generator.py
        ├── matrix.py
        └── render.py
```

The two per-publication folders keep the internal structure and file names of the
original sources: `Melgarejo/` mirrors the Solnon TD-TSP archive (`Matrices/`,
`Instances/`, `results_*`), and `YangFan/` mirrors the NCO4TDTSP repository
(`data/tdtsp/` tensors, top-level `testcases/`, and the environment code under
`routing/tdtsp/`).

- **`Melgarejo/`** — the benchmark from the **Melgarejo et al.** paper [1]. It bundles three components: the **travel-time matrices** (the time-dependent cost of every edge at every time step), the **instances** (which subset of vertices to visit, plus service durations, precedence constraints, and optional time windows), and the **reference results** (the objective value of the best/optimal tour for each instance–matrix pair). Together these give both the problems *and* known-good answers to validate a solver against.
- **`YangFan/`** — from the **Yang & Fan (Chuchu Fan)** paper [2]. It provides the **raw time-dependent travel-time data** for 12 cities (Beijing, Lyon, and 10 others) as `[nodes, nodes, horizon]` tensors — essentially full city road networks with per-time-bucket travel times — plus a ready-made **test set** (`testcases/`) of 10,000 sampled 20-node Beijing instances for evaluation, and the rl4co environment code that turns the raw tensors into solvable instances. This dataset is the source of large, realistic instances rather than pre-solved reference answers.

> The large binaries in `YangFan/` (the `data/tdtsp/*.npy` tensors and the `testcases/*.pt` test set) are **not tracked in git**. After cloning, run `bash Benchmarking_data/YangFan/fetch_data.sh` to download them.

### How we use these datasets

We will use both datasets to test our TD-TSP solvers under **genuine time-dependency conditions** — that is, on data where the travel time of an edge actually varies with departure time, rather than on static distances or randomly perturbed weights. `Melgarejo/` lets us check correctness and solution quality against **known optimal/reference objective values**, while `YangFan/` supplies **larger, real-world city instances** (and a fixed test set) to probe how well the solvers scale and whether they truly exploit *when* to travel. Evaluating on these ensures our results reflect real temporal routing behavior and are directly comparable to the published benchmarks.

---

## Dataset 1 — `Melgarejo/` (Melgarejo et al., CPAIOR 2015)

The classic TD-TSP benchmark: three time-dependent travel-time matrices, a set of instances (with and without time windows), and reference objective values.

### Directory layout

```
Melgarejo/
├── Matrices/
│   ├── matrix00.txt          # time-dependent travel-time matrix, congestion scenario 0
│   ├── matrix10.txt          # congestion scenario 1
│   └── matrix20.txt          # congestion scenario 2
├── Instances/
│   ├── Instances_noTW/       # instances WITHOUT (forbidden) time windows
│   │   ├── 10/   (60 instances)   inst_10_<id>.txt
│   │   ├── 20/   (60 instances)
│   │   ├── 30/   (60 instances)
│   │   ├── 50/   (20 instances)
│   │   └── 100/  (20 instances)
│   └── Instances_TW/         # instances WITH time windows
│       ├── 10/   inst_10_<id>_TW.txt
│       ├── 20/
│       ├── 30/
│       ├── 50/
│       └── 100/
├── results_noTW.txt          # reference objective values (no time windows)
└── results_TW.txt            # reference objective values (with time windows)
```

Every instance may be combined with **any** of the three travel-time matrices, so each instance file effectively yields three problems (one per congestion scenario).

### Travel-time matrices (`Matrices/matrixNN.txt`)

For all three matrices the number of locations is **n = 255**, the number of time steps is **m = 130**, and the duration of a time step is **d = 360 seconds** (so the modelled horizon is `m × d = 46 800 s = 13 h`).

A matrix file is a whitespace-separated list of `3 + n·n·m` positive integers:

- The **first 3 values** are `n`, `m`, and `d`.
- The remaining `n·n·m` values are the travel times `x[0], x[1], …`.

The travel time from location `i` to location `j` when **departing at time `t`** (`0 ≤ t < m·d`) is:

```
TT(i, j, t) = x[(j + i·n)·m + s]
```

where `s` is the time step containing `t`, i.e. `s = floor(t / d)`.

The three files `matrix00.txt`, `matrix10.txt`, `matrix20.txt` correspond to different congestion scenarios and are reported separately in the results files.

### Instance files (`Instances/…/inst_<p>_<id>[_TW].txt`)

`p` in the filename is the number of vertices to visit; `<id>` is the instance number. Files under `Instances_TW/` carry the `_TW` suffix and add forbidden time windows to some vertices.

Let `p` be the number of visits and `q` the number of precedence constraints. Each file has `1 + p + q` lines:

1. **First line:** `p  q`.
2. **Next `p` lines** — one per vertex to visit. Each line contains:
   - `vi` — index of this vertex in the travel-time matrix (`0 ≤ vi < n`, i.e. `< 255`).
   - `di` — the (visit/service) duration of this vertex.
   - `fi` — the number of forbidden time windows for this vertex.
   - `2·fi` integers `a1 b1 a2 b2 … afi bfi` describing the forbidden windows `[aj, bj)`. (In the `noTW` instances `fi = 0`, so no window values follow.)
3. **Last `q` lines** — precedence constraints, each a pair `pi si` meaning vertex `pi` must be visited before vertex `si`.

The tour must **start and end on the first vertex `v1`**, and the starting time from `v1` equals its visit duration `d1`.

> Note: the large sentinel value `1073741822` (= 2³⁰ − 2) is used as a practical "+∞" upper bound for an open-ended forbidden window.

**Example — `Instances_noTW/10/inst_10_1.txt`:**

```
10 0            # 10 vertices to visit, 0 precedence constraints
222 279 0       # visit vertex 222, duration 279, 0 forbidden windows
190 106 0
81 146 0
32 244 0
9 252 0
132 237 0
240 205 0
74 147 0
127 285 0
150 202 0
```

**Example — `Instances_TW/10/inst_10_1_TW.txt`** (same vertices, two now carry forbidden windows):

```
10 0
222 279 0
190 106 0
81 146 0
32 244 2 0 1375 7920 1073741822    # forbidden: [0,1375) and [7920, +inf)
9 252 0
132 237 0
240 205 2 0 44 4944 1073741822     # forbidden: [0,44) and [4944, +inf)
74 147 0
127 285 0
150 202 0
```

### Results (`results_noTW.txt`, `results_TW.txt`)

Each results file lists, per line, an instance, the travel-time matrix it was solved with, and the objective value of the best tour found:

```
inst_10_1.txt matrix00.txt 7605*
```

Conventions (stated in the header of each file):

- A trailing **`*`** marks a **proven optimal** solution.
- An objective of **`-1`** means **no solution was found**.

Results are grouped by instance size, and each instance appears once per matrix (`matrix00`, `matrix10`, `matrix20`). Small sizes (10–30) are generally solved to optimality; the largest (100-vertex) instances often report best-known values without the optimality mark.

---

## Dataset 2 — `YangFan/` (Yang & Fan, NeurIPS 2025)

This folder holds the raw time-dependent travel-time data used in the Yang & Fan experimentation, plus the rl4co TD-TSP environment code that consumes it. The layout follows the [NCO4TDTSP](https://github.com/Brelliothe/NCO4TDTSP) repository: tensors live in `data/tdtsp/`, test cases in a top-level `testcases/`, and the environment code under `routing/tdtsp/`:

```
YangFan/
├── fetch_data.sh                            # re-downloads the gitignored binaries
├── data/
│   └── tdtsp/
│       └── *.npy                            # 12 city travel-time tensors (below)
├── testcases/
│   └── beijing_20_dataset_10000.pt          # 10k 20-node Beijing test instances
└── routing/
    └── tdtsp/
        ├── env.py                           # rl4co TD-TSP RL environment
        ├── generator.py                     # instance generator (samples node subsets from a city tensor)
        ├── matrix.py                        # time-dependent matrix / interpolation helpers
        └── render.py                        # tour visualisation
```

The tensors come from three raw sources: **Beijing**, **Lyon**, and a set of **10 cities**. Each city is a NumPy array of shape `[nodes, nodes, horizon]`, where `A[i, j, t]` is the travel time from node `i` to node `j` when **departing in time bucket `t`**:

| City | File | Shape `(nodes, nodes, horizon)` | Raw source |
|------|------|---------------------------------|------------|
| Beijing | `beijing.npy` | `(100, 100, 12)` | Zhang et al. (2021) |
| Lyon | `lyon.npy` | `(255, 255, 130)` | Melgarejo et al. (2015) |
| Berlin | `berlin.npy` | `(501, 501, 43)` | Blauth et al. (2022) |
| Cincinnati | `cincinnati.npy` | `(501, 501, 43)` | Blauth et al. (2022) |
| Kyiv | `kyiv.npy` | `(501, 501, 43)` | Blauth et al. (2022) |
| London | `london.npy` | `(501, 501, 43)` | Blauth et al. (2022) |
| Madrid | `madrid.npy` | `(501, 501, 43)` | Blauth et al. (2022) |
| Nairobi | `nairobi.npy` | `(501, 501, 43)` | Blauth et al. (2022) |
| New York | `new_york.npy` | `(501, 501, 43)` | Blauth et al. (2022) |
| San Francisco | `san_francisco.npy` | `(501, 501, 43)` | Blauth et al. (2022) |
| São Paulo | `sao_paulo.npy` | `(501, 501, 43)` | Blauth et al. (2022) |
| Seattle | `seattle.npy` | `(501, 501, 43)` | Blauth et al. (2022) |

This mirrors the three data references given in the NCO4TDTSP repository:

- **Beijing** — from Zhang et al. (2021), who study deep-RL solvers for dynamic TSP.
- **Lyon** — from Melgarejo et al. (2015); this is the same Lyon road-network data as Dataset 1 above. Its `(255, 255, 130)` shape coincides exactly with the `n = 255`, `m = 130` parameters of the Melgarejo benchmark matrices — it is that same data in tensor form.
- **10 cities** (Berlin, Cincinnati, Kyiv, London, Madrid, Nairobi, New York, San Francisco, São Paulo, Seattle) — from the Blauth et al. (2022) time-dependent vehicle-routing benchmark.

All arrays match the customized-dataset format expected by the NCO4TDTSP code (a numpy array of shape `[nodes, nodes, horizon]`).

Loading example:

```python
import numpy as np

tt = np.load("Benchmarking_data/YangFan/data/tdtsp/new_york.npy")  # shape (501, 501, 43)
travel_time = tt[i, j, t]   # from i to j departing in time bucket t
```

