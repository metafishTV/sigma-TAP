from __future__ import annotations

from typing import Iterable


def project_M(features: Iterable[float], epsilon: float) -> int:
    """M = number of active features above threshold."""
    return sum(1 for x in features if x >= epsilon)


def birth_death(prev_features: Iterable[float], next_features: Iterable[float], epsilon: float) -> tuple[int, int]:
    prev = list(prev_features)
    nxt = list(next_features)
    if len(prev) != len(nxt):
        raise ValueError("feature vectors must have same length")
    B = sum(1 for p, n in zip(prev, nxt) if p < epsilon and n >= epsilon)
    D = sum(1 for p, n in zip(prev, nxt) if p >= epsilon and n < epsilon)
    return B, D


def pressure_inverse_inference_allowed(has_identification_assumptions: bool) -> bool:
    """Policy helper: inverse causal pressure inference requires explicit identification assumptions."""
    return bool(has_identification_assumptions)
