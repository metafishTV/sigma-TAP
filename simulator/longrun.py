"""Long-run diagnostics for TAP / sigma-TAP trajectories.

Statistical measures inspired by Taalbi (2025):
- Heaps' law: D(k) ~ k^beta for diversification
- Gini coefficient: concentration of innovation across agents
- Top-k share: fraction of total held by top agents
- Diversification rate: dD/dk over time
- Enhanced constraint tag: {adjacency-limited, resource-limited, mixed} + confidence

These diagnostics are independent of the metathetic agent layer and can be
applied to any multi-agent or aggregate trajectory data.
"""
from __future__ import annotations

import math


def heaps_law_fit(D_series: list[float], k_series: list[float]) -> dict:
    """Fit D(k) ~ k^beta via log-log OLS.

    Heaps' law predicts beta < 1: the rate of new type discovery declines
    relative to total innovation as the system matures.

    Returns dict with 'beta', 'intercept', 'r_squared', 'n_points'.
    """
    if len(D_series) < 2 or len(k_series) < 2:
        return {"beta": 1.0, "intercept": 0.0, "r_squared": 0.0, "n_points": 0}

    log_k = []
    log_D = []
    for k, d in zip(k_series, D_series):
        if k > 0 and d > 0:
            log_k.append(math.log(k))
            log_D.append(math.log(d))

    n = len(log_k)
    if n < 2:
        return {"beta": 1.0, "intercept": 0.0, "r_squared": 0.0, "n_points": n}

    mean_x = sum(log_k) / n
    mean_y = sum(log_D) / n
    ss_xy = sum((x - mean_x) * (y - mean_y) for x, y in zip(log_k, log_D))
    ss_xx = sum((x - mean_x) ** 2 for x in log_k)
    ss_yy = sum((y - mean_y) ** 2 for y in log_D)

    if ss_xx < 1e-15:
        return {"beta": 1.0, "intercept": 0.0, "r_squared": 0.0, "n_points": n}

    beta = ss_xy / ss_xx
    intercept = mean_y - beta * mean_x
    r_squared = (ss_xy ** 2) / (ss_xx * ss_yy) if ss_yy > 1e-15 else 0.0

    return {"beta": beta, "intercept": intercept, "r_squared": r_squared, "n_points": n}


def gini_coefficient(values: list[float]) -> float:
    """Standard Gini coefficient.

    0 = perfect equality, 1 = perfect inequality.
    Uses the relative mean absolute difference formula.
    """
    if len(values) <= 1:
        return 0.0

    n = len(values)
    total = sum(values)
    if total <= 0:
        return 0.0

    sorted_vals = sorted(values)
    weighted_sum = 0.0
    for i, v in enumerate(sorted_vals):
        weighted_sum += (2 * (i + 1) - n - 1) * v

    return weighted_sum / (n * total)


def top_k_share(values: list[float], k_frac: float = 0.1) -> float:
    """Fraction of total held by the top k_frac fraction of agents.

    k_frac=0.1 means "what share does the top 10% hold?"
    """
    if not values:
        return 0.0

    total = sum(values)
    if total <= 0:
        return 0.0

    sorted_desc = sorted(values, reverse=True)
    n_top = max(1, int(math.ceil(len(sorted_desc) * k_frac)))
    top_sum = sum(sorted_desc[:n_top])
    return top_sum / total


def diversification_rate(D_series: list[float], k_series: list[float]) -> list[float]:
    """Compute dD/dk at each step — rate of new type discovery per unit innovation.

    Under Heaps' law with beta < 1, this rate should decline over time.
    """
    rates = []
    for i in range(len(D_series) - 1):
        dk = k_series[i + 1] - k_series[i]
        dD = D_series[i + 1] - D_series[i]
        if dk > 0:
            rates.append(dD / dk)
        else:
            rates.append(0.0)
    return rates


def enhanced_constraint_tag(
    sigma: float,
    beta: float,
    gini: float,
    carrying_capacity: float | None,
    m_final: float,
) -> dict:
    """Tag observed dynamics with constraint type and confidence level.

    Decision heuristics grounded in Taalbi (2025):
    - sigma > 1 = super-linear TAP dynamics (adjacency-dominated)
    - sigma ~ 1 = resource-constrained linear dynamics
    - beta < 1 = Heaps' law diversification (sublinear)
    - gini < 0.5 = no winner-take-all

    Returns dict with 'tag', 'confidence', 'reasoning'.
    """
    reasons = []

    # Check if M is near carrying capacity.
    near_capacity = False
    if carrying_capacity is not None and carrying_capacity > 0:
        ratio = m_final / carrying_capacity
        if ratio > 0.8:
            near_capacity = True
            reasons.append(f"M/K={ratio:.2f} (near capacity)")

    # Adjacency-limited indicators.
    adjacency_indicators = 0
    if sigma > 1.3:
        adjacency_indicators += 1
        reasons.append(f"sigma={sigma:.2f} (super-linear)")
    if beta < 0.7:
        adjacency_indicators += 1
        reasons.append(f"beta={beta:.2f} (strong Heaps sublinearity)")
    if carrying_capacity is None:
        adjacency_indicators += 1
        reasons.append("no carrying capacity")

    # Resource-limited indicators.
    resource_indicators = 0
    if 0.8 <= sigma <= 1.2:
        resource_indicators += 1
        reasons.append(f"sigma={sigma:.2f} (near-linear)")
    if near_capacity:
        resource_indicators += 1
    if gini < 0.3:
        resource_indicators += 1
        reasons.append(f"gini={gini:.2f} (low concentration)")

    # Decision.
    if adjacency_indicators >= 2 and not near_capacity:
        tag = "adjacency-limited"
        confidence = "high" if adjacency_indicators >= 3 else "medium"
    elif resource_indicators >= 2 or near_capacity:
        tag = "resource-limited"
        confidence = "high" if resource_indicators >= 3 or near_capacity else "medium"
    else:
        tag = "mixed"
        confidence = "medium" if len(reasons) >= 2 else "low"

    return {
        "tag": tag,
        "confidence": confidence,
        "reasoning": "; ".join(reasons) if reasons else "insufficient indicators",
    }
