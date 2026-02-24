from __future__ import annotations

import math
from math import comb


def innovation_kernel(M: int, alpha: float, a: float) -> float:
    """Exact TAP kernel f(M)=sum_{i=2}^M alpha/a^(i-1) * C(M,i)."""
    if M < 2:
        return 0.0
    return sum((alpha / (a ** (i - 1))) * comb(M, i) for i in range(2, M + 1))


def innovation_kernel_closed(M: float, alpha: float, a: float) -> float:
    """Overflow-aware closed form approximation using real-valued M."""
    exponent = M * math.log(1.0 + 1.0 / a)
    if exponent > 700:  # float64 overflow vicinity
        return float("inf")
    return alpha * a * (math.exp(exponent) - 1.0 - (M / a))
