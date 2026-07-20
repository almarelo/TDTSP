"""
Step 1 — Time-dependent data layer (no time windows).

Loads the Melgarejo / Solnon Lyon TDTSP benchmark
(``hybrid_cp_qubo/data/Melgarejo/``) and exposes a FIFO travel-time model
``Dij(t)`` with arrival-time propagation.

Current scope: ``Instances_noTW`` only. The Melgarejo objective used here
is the **return time** — the arrival time back at the depot.

Benchmark format (from https://perso.liris.cnrs.fr/christine.solnon/TDTSP.html):

Matrices
--------
Header line: ``n m d``  (nodes, time steps, seconds per step).
Then ``n * n`` lines of ``m`` integers each (blank lines between origin
blocks). Travel time from ``i`` to ``j`` when departing in time step ``s``
is ``matrix[i, j, s]`` (seconds). Horizon = ``m * d`` seconds.

Instances (noTW)
----------------
Header: ``p q``  (number of visits, number of precedence constraints).
Each of the next ``p`` lines:
    ``v_i  d_i  f_i``
where ``v_i`` is the vertex index into the matrix (0 <= v_i < n),
``d_i`` is the service duration (seconds), and ``f_i`` is the number of
forbidden windows (always 0 for noTW). The remaining ``q`` lines are
precedence pairs ``(pred, succ)`` (visit indices, 0-based).

The tour must start and end on the first visit; the starting departure
time from the depot equals the depot service duration ``d_0``.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
MELGAREJO_ROOT = PACKAGE_ROOT / "data" / "Melgarejo"
DEFAULT_MATRIX_DIR = MELGAREJO_ROOT / "Matrices"
DEFAULT_INSTANCE_DIR = MELGAREJO_ROOT / "Instances" / "Instances_noTW"


# ---------------------------------------------------------------------------
# Travel-time matrix
# ---------------------------------------------------------------------------

class TravelTimeMatrix:
    """
    Stepwise time-dependent travel-time tensor for the Lyon network.

    Attributes
    ----------
    n : int
        Number of locations in the matrix (255 for Melgarejo).
    m : int
        Number of time steps (130).
    d : int
        Duration of one time step in seconds (360).
    horizon : int
        Total planning horizon ``m * d`` (seconds).
    tensor : ndarray, shape (n, n, m)
        ``tensor[i, j, s]`` = travel time (seconds) from i to j when
        departing in time step ``s``.
    """

    def __init__(self, tensor: np.ndarray, step_seconds: int):
        if tensor.ndim != 3 or tensor.shape[0] != tensor.shape[1]:
            raise ValueError(
                f"Expected (n, n, m) tensor; got shape {tensor.shape}"
            )
        self.tensor = np.asarray(tensor, dtype=np.float64)
        self.n, _, self.m = self.tensor.shape
        self.d = int(step_seconds)
        self.horizon = self.m * self.d

    @classmethod
    def from_file(cls, path: Union[str, Path]) -> "TravelTimeMatrix":
        """Parse a Melgarejo ``matrixNN.txt`` file into a 3-D tensor."""
        path = Path(path)
        with path.open() as f:
            header = f.readline().split()
            if len(header) != 3:
                raise ValueError(f"Bad matrix header in {path}: {header}")
            n, m, d = (int(x) for x in header)

            data = np.zeros((n, n, m), dtype=np.float64)
            i = j = 0
            for line in f:
                parts = line.split()
                if not parts:
                    continue  # blank separator between origin blocks
                if len(parts) != m:
                    raise ValueError(
                        f"Expected {m} values per row in {path.name}, "
                        f"got {len(parts)} at origin={i} dest={j}"
                    )
                data[i, j, :] = np.asarray(parts, dtype=np.float64)
                j += 1
                if j == n:
                    j = 0
                    i += 1
            if i != n:
                raise ValueError(
                    f"Incomplete matrix in {path.name}: "
                    f"read {i * n + j} of {n * n} OD pairs"
                )
        return cls(data, step_seconds=d)

    @classmethod
    def load(cls, name: str = "matrix00",
             matrix_dir: Optional[Path] = None) -> "TravelTimeMatrix":
        """Load by short name, e.g. ``'matrix00'`` or ``'00'``."""
        matrix_dir = Path(matrix_dir) if matrix_dir else DEFAULT_MATRIX_DIR
        if not name.startswith("matrix"):
            name = f"matrix{name}"
        path = matrix_dir / f"{name}.txt"
        if not path.exists():
            raise FileNotFoundError(path)
        return cls.from_file(path)

    def time_step(self, t: float) -> int:
        """Bucket index ``s = floor(t / d)``, clamped into ``[0, m-1]``."""
        if t < 0:
            return 0
        s = int(t // self.d)
        return min(s, self.m - 1)

    def travel_time(self, i: int, j: int, t: float) -> float:
        """
        ``Dij(t)`` — travel time from ``i`` to ``j`` departing at time ``t``.

        Uses the stepwise Melgarejo model: the cost is constant within a
        time step. (FIFO is guaranteed by the published matrices.)
        """
        return float(self.tensor[i, j, self.time_step(t)])

    def slice_at(self, t: float,
                 nodes: Optional[Sequence[int]] = None) -> np.ndarray:
        """
        Static ``|nodes| x |nodes|`` cost matrix at departure time ``t``.

        Used by the quantum subproblems: within a cluster the master fixes
        an entry time, and the QUBO sees a frozen cost matrix.
        """
        s = self.time_step(t)
        if nodes is None:
            return self.tensor[:, :, s].copy()
        idx = np.asarray(list(nodes), dtype=int)
        return self.tensor[np.ix_(idx, idx, [s])][:, :, 0].copy()

    def __repr__(self) -> str:
        return (
            f"TravelTimeMatrix(n={self.n}, m={self.m}, "
            f"d={self.d}s, horizon={self.horizon}s)"
        )


# ---------------------------------------------------------------------------
# Instance (no time windows)
# ---------------------------------------------------------------------------

@dataclass
class Visit:
    """One stop in a Melgarejo noTW TDTSP instance."""
    vertex: int   # index into TravelTimeMatrix
    service: int  # service duration (seconds)


@dataclass
class TDTSPInstance:
    """Parsed Melgarejo noTW instance."""
    name: str
    visits: List[Visit]
    precedences: List[Tuple[int, int]]  # (pred_visit_idx, succ_visit_idx)
    path: Optional[Path] = None

    @property
    def p(self) -> int:
        return len(self.visits)

    @property
    def vertices(self) -> List[int]:
        return [v.vertex for v in self.visits]

    @property
    def depot(self) -> int:
        """Visit index of the depot (always 0 in this benchmark)."""
        return 0

    @property
    def start_time(self) -> float:
        """Departure time from the depot = depot service duration."""
        return float(self.visits[0].service)

    @classmethod
    def from_file(cls, path: Union[str, Path]) -> "TDTSPInstance":
        """
        Parse a Melgarejo noTW instance file.

        Forbidden-window fields must be zero (``f_i == 0``); TW instances
        are out of scope for now.
        """
        path = Path(path)
        lines = [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]
        if not lines:
            raise ValueError(f"Empty instance file: {path}")

        header = lines[0].split()
        if len(header) != 2:
            raise ValueError(f"Bad instance header in {path}: {header}")
        p, q = int(header[0]), int(header[1])
        if len(lines) < 1 + p + q:
            raise ValueError(
                f"{path.name}: expected {1 + p + q} non-empty lines, "
                f"got {len(lines)}"
            )

        visits: List[Visit] = []
        for k in range(1, 1 + p):
            toks = [int(x) for x in lines[k].split()]
            if len(toks) < 3:
                raise ValueError(f"Bad visit line {k} in {path.name}: {toks}")
            v_i, d_i, f_i = toks[0], toks[1], toks[2]
            if f_i != 0:
                raise ValueError(
                    f"{path.name} visit {k - 1}: f_i={f_i} (time windows). "
                    f"Only Instances_noTW are supported for now."
                )
            if len(toks) != 3:
                raise ValueError(
                    f"{path.name} visit {k - 1}: expected "
                    f"'v_i d_i 0', got {toks}"
                )
            visits.append(Visit(vertex=v_i, service=d_i))

        precedences: List[Tuple[int, int]] = []
        for k in range(1 + p, 1 + p + q):
            a, b = (int(x) for x in lines[k].split())
            precedences.append((a, b))

        return cls(
            name=path.name,
            visits=visits,
            precedences=precedences,
            path=path,
        )

    @classmethod
    def load(cls,
             size: int,
             index: int,
             instance_dir: Optional[Path] = None) -> "TDTSPInstance":
        """
        Convenience loader for noTW instances, e.g.
        ``TDTSPInstance.load(10, 1)`` → ``Instances_noTW/10/inst_10_1.txt``.
        """
        instance_dir = Path(instance_dir) if instance_dir else DEFAULT_INSTANCE_DIR
        path = instance_dir / str(size) / f"inst_{size}_{index}.txt"
        if not path.exists():
            raise FileNotFoundError(path)
        return cls.from_file(path)

    def __repr__(self) -> str:
        return (
            f"TDTSPInstance({self.name!r}, p={self.p}, "
            f"precedences={len(self.precedences)})"
        )


# ---------------------------------------------------------------------------
# Cost model: arrival-time propagation
# ---------------------------------------------------------------------------

class TDCostModel:
    """
    True time-dependent cost evaluator on top of a ``TravelTimeMatrix``.

    Given an ordered sequence of *visit indices* (into a ``TDTSPInstance``),
    propagates arrival / departure times using ``Dij(t)``.

    Objective
    ---------
    ``return_time`` — arrival time back at the depot (Melgarejo convention).
    This is the value reported in ``results_noTW.txt``.
    """

    def __init__(self, matrix: TravelTimeMatrix, instance: TDTSPInstance):
        self.matrix = matrix
        self.instance = instance
        for v in instance.visits:
            if not (0 <= v.vertex < matrix.n):
                raise ValueError(
                    f"Visit vertex {v.vertex} out of range "
                    f"[0, {matrix.n})"
                )

    def travel(self, from_visit: int, to_visit: int, depart_t: float) -> float:
        """Travel time between two *visits* departing at ``depart_t``."""
        i = self.instance.visits[from_visit].vertex
        j = self.instance.visits[to_visit].vertex
        return self.matrix.travel_time(i, j, depart_t)

    def evaluate_tour(
        self,
        tour_open: Sequence[int],
        start_time: Optional[float] = None,
    ) -> dict:
        """
        Walk an open tour (visit indices, no return copy) under FIFO costs.

        Parameters
        ----------
        tour_open :
            Permutation of ``{0, ..., p-1}`` that starts at the depot (0).
            The return edge to the depot is added automatically.
        start_time :
            Departure time from the depot. Defaults to the Melgarejo
            convention ``service[0]``.

        Returns
        -------
        dict with keys
            ``tour_open``, ``tour_with_return``,
            ``arrivals``, ``departures``,
            ``edge_travel_times``,
            ``return_time`` (objective: arrival back at depot),
            ``start_time``.
        """
        p = self.instance.p
        tour = list(tour_open)
        if len(tour) != p or set(tour) != set(range(p)):
            raise ValueError(
                f"tour_open must be a permutation of 0..{p - 1}; got {tour}"
            )
        if tour[0] != self.instance.depot:
            raise ValueError(
                f"Tour must start at depot visit 0; got start={tour[0]}"
            )

        if start_time is None:
            start_time = self.instance.start_time

        tour_closed = tour + [tour[0]]

        arrivals = [0.0] * (p + 1)
        departures = [0.0] * p
        edge_tts: List[float] = []

        # Depot: arrive at 0, depart at start_time (service baked into
        # Melgarejo's start-time convention).
        arrivals[0] = 0.0
        departures[0] = float(start_time)
        t = departures[0]

        for k in range(len(tour_closed) - 1):
            u = tour_closed[k]
            v = tour_closed[k + 1]
            tt = self.travel(u, v, t)
            edge_tts.append(tt)
            t = t + tt
            arrivals[k + 1] = t
            if k + 1 < p:
                t = t + self.instance.visits[v].service
                departures[k + 1] = t

        return_time = arrivals[-1]

        return {
            "tour_open": tour,
            "tour_with_return": tour_closed,
            "arrivals": arrivals,
            "departures": departures,
            "edge_travel_times": edge_tts,
            "return_time": return_time,
            "start_time": float(start_time),
        }

    def static_cluster_matrix(
        self,
        visit_indices: Sequence[int],
        entry_time: float,
    ) -> np.ndarray:
        """
        Freeze ``Dij(entry_time)`` for the vertices of a visit cluster.

        Returns a dense ``|cluster| x |cluster|`` matrix in *cluster-local*
        order (same order as ``visit_indices``). This is what the quantum
        TSP-QUBO subproblem receives.
        """
        vertices = [self.instance.visits[k].vertex for k in visit_indices]
        return self.matrix.slice_at(entry_time, vertices)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _self_test() -> None:
    """Load matrix00 + inst_10_1 and evaluate a trivial depot-start tour."""
    print("Step 1 self-test: Melgarejo data layer (noTW, return_time)")
    print("-" * 60)

    mx = TravelTimeMatrix.load("matrix00")
    print(f"  matrix          : {mx}")

    inst = TDTSPInstance.load(size=10, index=1)
    print(f"  instance        : {inst}")
    print(f"  vertices        : {inst.vertices}")
    print(f"  start_time      : {inst.start_time}s "
          f"(= depot service {inst.visits[0].service}s)")

    cost = TDCostModel(mx, inst)

    tour = list(range(inst.p))
    result = cost.evaluate_tour(tour)
    print(f"  trivial tour    : {result['tour_with_return']}")
    print(f"  return_time     : {result['return_time']:.1f}s "
          f"(objective; {result['return_time'] / 60:.1f} min)")
    print(f"  edge travel (s) : "
          f"{[round(x, 1) for x in result['edge_travel_times']]}")

    static = cost.static_cluster_matrix(tour, inst.start_time)
    print(f"  static slice @ t0 shape: {static.shape}, "
          f"mean edge = {static[static > 0].mean():.1f}s")

    # results_noTW.txt: inst_10_1.txt matrix00.txt 7605*
    print(f"  ref. optimum (results_noTW): 7605*  (return_time)")
    assert result["return_time"] > 0
    assert "total_duration" not in result
    assert len(result["edge_travel_times"]) == inst.p
    print("  OK")


if __name__ == "__main__":
    _self_test()
