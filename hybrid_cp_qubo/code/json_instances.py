"""
Adapter: original NYC JSON TD-TSP instances → hybrid Melgarejo-style types.

Reads ``data/instances/tdtsp_n{N}.json`` and builds:

* ``TravelTimeMatrix`` — hourly stepwise tensor
  ``Dij(t) = round(base[i,j] * multiplier[slot(hour(t))])``
* ``TDTSPInstance`` — n visits, depot = 0, start clock = slot hour
  (encoded as depot service, matching Melgarejo's start_time convention)

Slot mapping (local clock hour → instance slot name):

    08–11  morning_peak
    12–17  midday
    18–21  evening_peak
    else   night
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from .td_data import TravelTimeMatrix, TDTSPInstance, Visit
except ImportError:
    from td_data import TravelTimeMatrix, TDTSPInstance, Visit

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INSTANCES_DIR = REPO_ROOT / "data" / "instances"

HOUR_STEP_SECONDS = 3600
N_HOURS = 24


def hour_to_slot_name(hour: int) -> str:
    h = int(hour) % 24
    if 8 <= h < 12:
        return "morning_peak"
    if 12 <= h < 18:
        return "midday"
    if 18 <= h < 22:
        return "evening_peak"
    return "night"


def load_json_instance(
    path: Path | str,
) -> Dict[str, Any]:
    path = Path(path)
    with path.open() as f:
        return json.load(f)


def list_json_instances(
    instances_dir: Optional[Path] = None,
    sizes: Optional[List[int]] = None,
) -> List[Path]:
    instances_dir = Path(instances_dir) if instances_dir else DEFAULT_INSTANCES_DIR
    sizes = sizes or [5, 10, 25, 50, 100]
    out: List[Path] = []
    for n in sizes:
        p = instances_dir / f"tdtsp_n{n}.json"
        if p.is_file():
            out.append(p)
    return out


def slot_multipliers(raw: Dict[str, Any]) -> Dict[str, float]:
    """Prefer per-slot computed multipliers from ``time_slots``."""
    out: Dict[str, float] = {}
    for slot in raw.get("time_slots", []):
        name = slot["name"]
        if "multiplier" in slot:
            out[name] = float(slot["multiplier"])
        elif name in (raw.get("multipliers") or {}).get("computed", {}):
            out[name] = float(raw["multipliers"]["computed"][name])
        else:
            out[name] = float(
                slot.get(
                    "nominal_multiplier",
                    (raw.get("multipliers") or {})
                    .get("nominal", {})
                    .get(name, 1.0),
                )
            )
    return out


def build_hourly_tensor(
    base: np.ndarray,
    multipliers: Dict[str, float],
) -> np.ndarray:
    """Shape ``(n, n, 24)`` integer travel times (seconds)."""
    n = base.shape[0]
    tensor = np.zeros((n, n, N_HOURS), dtype=np.float64)
    for h in range(N_HOURS):
        m = float(multipliers[hour_to_slot_name(h)])
        tensor[:, :, h] = np.rint(base * m)
    # Keep zeros on diagonal
    for h in range(N_HOURS):
        np.fill_diagonal(tensor[:, :, h], 0.0)
    return tensor


def to_hybrid_types(
    raw: Dict[str, Any],
    *,
    slot_name: str,
    source_path: Optional[Path] = None,
) -> Tuple[TravelTimeMatrix, TDTSPInstance, Dict[str, Any]]:
    """
    Convert one JSON instance + chosen start slot into hybrid types.

    Depot departure is the slot's representative hour (local).
    """
    slots = {s["name"]: s for s in raw["time_slots"]}
    if slot_name not in slots:
        raise KeyError(f"Unknown slot {slot_name!r}; have {list(slots)}")

    slot = slots[slot_name]
    base = np.asarray(raw["base_distance_matrix"], dtype=np.float64)
    if base.ndim != 2 or base.shape[0] != base.shape[1]:
        raise ValueError(f"Bad base_distance_matrix shape {base.shape}")
    n = int(base.shape[0])
    if int(raw.get("n", n)) != n:
        raise ValueError("Instance n does not match matrix size")

    mults = slot_multipliers(raw)
    tensor = build_hourly_tensor(base, mults)
    matrix = TravelTimeMatrix(tensor, step_seconds=HOUR_STEP_SECONDS)

    start_hour = int(slot["hour"])
    start_seconds = start_hour * HOUR_STEP_SECONDS
    # Melgarejo convention: start_time == depot service
    visits = [Visit(vertex=0, service=start_seconds)]
    visits.extend(Visit(vertex=i, service=0) for i in range(1, n))

    name = source_path.name if source_path else f"tdtsp_n{n}.json"
    instance = TDTSPInstance(
        name=f"{Path(name).stem}_{slot_name}",
        visits=visits,
        precedences=[],
        path=source_path,
    )

    meta = {
        "source_file": str(source_path) if source_path else None,
        "n": n,
        "city": raw.get("city"),
        "tz": raw.get("tz"),
        "metric": raw.get("metric", "driving_duration_seconds"),
        "slot_name": slot_name,
        "slot_label": slot.get("label"),
        "slot_hour": start_hour,
        "slot_multiplier": float(mults[slot_name]),
        "locations": list(raw.get("locations") or []),
        "multipliers": mults,
    }
    return matrix, instance, meta


def load_hybrid(
    size: int,
    slot_name: str,
    instances_dir: Optional[Path] = None,
) -> Tuple[TravelTimeMatrix, TDTSPInstance, Dict[str, Any]]:
    instances_dir = Path(instances_dir) if instances_dir else DEFAULT_INSTANCES_DIR
    path = instances_dir / f"tdtsp_n{size}.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    raw = load_json_instance(path)
    return to_hybrid_types(raw, slot_name=slot_name, source_path=path)
