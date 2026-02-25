from __future__ import annotations

import math
from math import comb


def innovation_kernel(M: int, alpha: float, a: float) -> float:
    """Exact TAP kernel f(M)=sum_{i=2}^M alpha/a^(i-1) * C(M,i) for integer M."""
    if M < 2:
        return 0.0
    return sum((alpha / (a ** (i - 1))) * comb(M, i) for i in range(2, M + 1))


def innovation_kernel_closed(M: float, alpha: float, a: float) -> float:
    """Overflow-aware closed form interpolation for real-valued M.

    Closed form of sum_{i=2}^M alpha/a^(i-1) * C(M,i), extending M to reals.
    """
    if M <= 1.0:
        return 0.0
    exponent = M * math.log(1.0 + 1.0 / a)
    if exponent > 700:  # float64 overflow vicinity
        return float("inf")
    return max(0.0, alpha * a * (math.exp(exponent) - 1.0 - (M / a)))


def innovation_kernel_two_scale(M: float, alpha: float, a: float, alpha1: float) -> float:
    """Two-scale TAP: adds i=1-like linear innovation to combinatorial term.

    This follows the paper's motivation where single-object evolution can dominate
    early growth before combinatorial terms trigger explosive transitions.
    """
    return max(0.0, alpha1 * M + innovation_kernel_closed(M, alpha, a))


def apply_logistic_constraint(birth_term: float, M: float, carrying_capacity: float | None) -> float:
    """Resource/search limitation proxy inspired by long-run adjacent-possible framing.

    If capacity is provided and positive, scale innovation by max(0, 1 - M/K).
    """
    if carrying_capacity is None or carrying_capacity <= 0:
        return birth_term
    factor = max(0.0, 1.0 - (M / carrying_capacity))
    return birth_term * factor


def compute_birth_term(
    M: float,
    *,
    alpha: float,
    a: float,
    variant: str = "baseline",
    alpha1: float = 0.0,
    carrying_capacity: float | None = None,
) -> float:
    """Compute TAP innovation term B_pre_sigma for a selected variant."""
    variant_norm = (variant or "baseline").strip().lower()

    if variant_norm == "baseline":
        base = innovation_kernel_closed(M, alpha, a)
    elif variant_norm in {"two_scale", "two-scale", "twoscale"}:
        base = innovation_kernel_two_scale(M, alpha, a, alpha1=alpha1)
    elif variant_norm == "logistic":
        base = innovation_kernel_closed(M, alpha, a)
        base = apply_logistic_constraint(base, M, carrying_capacity)
    else:
        raise ValueError(f"Unknown TAP variant '{variant}'. Expected baseline|two_scale|logistic")

    return max(0.0, base)
