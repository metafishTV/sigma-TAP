from __future__ import annotations

from typing import Sequence


def pressure_index_cardinality() -> int:
    return 9  # {-4,...,+4}


def aggregate_pressures(negative: Sequence[float], positive: Sequence[float]) -> tuple[float, float, float]:
    A = float(sum(negative))
    O = float(sum(positive))
    return A, O, A - O
