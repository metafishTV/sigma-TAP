# simulator/empirical.py
"""Empirical validation metrics — pure functions for comparing
sigma-TAP simulation output against quantitative targets from
the innovation economics literature.

CLAIM POLICY LABEL: exploratory

All functions take arrays/lists as input and return structured
dataclass results. No simulation imports — pure metric computation.
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

import numpy as np
from scipy.stats import linregress


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class YounRatioResult:
    """Exploration/exploitation ratio comparison (Youn et al. 2015)."""
    exploration_count: int
    exploitation_count: int
    exploration_fraction: float
    target: float
    deviation: float
    trajectory_ratios: list[float]


@dataclass
class TaalbiLinearityResult:
    """Innovation rate linearity test (Taalbi 2025)."""
    slope: float
    intercept: float
    r_squared: float
    target_slope: float
    p_value: float
    n_points: int


@dataclass
class HeapsLawResult:
    """Sub-linear diversification test (Taalbi 2025, Heaps' law)."""
    exponent: float
    intercept: float
    r_squared: float
    target: str
    is_sublinear: bool
    n_points: int


@dataclass
class PowerLawResult:
    """Agent innovation distribution test (Taalbi 2025)."""
    exponent: float
    target_exponent: float
    ks_statistic: float
    p_value: float
    n_agents: int
    k_min: int


@dataclass
class EmpiricalValidationResult:
    """Full validation output."""
    youn: YounRatioResult
    linearity: TaalbiLinearityResult
    heaps: HeapsLawResult
    power_law: PowerLawResult
    params_used: dict
    n_steps: int
    n_agents: int


# ---------------------------------------------------------------------------
# Metric functions
# ---------------------------------------------------------------------------

def youn_ratio(
    n_novel_cross: list[int],
    n_absorptive_cross: list[int],
) -> YounRatioResult:
    """Compute exploration/exploitation ratio and compare to Youn et al. 2015.

    Parameters
    ----------
    n_novel_cross : list[int]
        Cumulative novel (exploration) event counts at each snapshot.
    n_absorptive_cross : list[int]
        Cumulative absorptive (exploitation) event counts at each snapshot.

    Returns
    -------
    YounRatioResult
        Exploration fraction compared to 0.6 target.
    """
    if len(n_novel_cross) != len(n_absorptive_cross):
        raise ValueError(
            f"Input lengths must match: n_novel_cross has {len(n_novel_cross)} "
            f"elements, n_absorptive_cross has {len(n_absorptive_cross)}"
        )
    novel = np.asarray(n_novel_cross, dtype=float)
    absorptive = np.asarray(n_absorptive_cross, dtype=float)
    target = 0.6

    # Per-step ratios
    totals = novel + absorptive
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        ratios = np.where(totals > 0, novel / totals, np.nan)
    trajectory_ratios = ratios.tolist()

    # Overall fraction from final values
    final_novel = int(novel[-1]) if len(novel) > 0 else 0
    final_absorptive = int(absorptive[-1]) if len(absorptive) > 0 else 0
    final_total = final_novel + final_absorptive

    if final_total == 0:
        exploration_fraction = float("nan")
        deviation = float("nan")
    else:
        exploration_fraction = final_novel / final_total
        deviation = abs(exploration_fraction - target)

    return YounRatioResult(
        exploration_count=final_novel,
        exploitation_count=final_absorptive,
        exploration_fraction=exploration_fraction,
        target=target,
        deviation=deviation,
        trajectory_ratios=trajectory_ratios,
    )


def taalbi_linearity(
    k_total: list[int],
    dt: float = 1.0,
) -> TaalbiLinearityResult:
    """Test innovation rate linearity (Taalbi 2025).

    Fits log(dk/dt) vs log(k) via OLS; slope ≈ 1.0 expected.

    Parameters
    ----------
    k_total : list[int]
        Cumulative innovation counts at each time step.
    dt : float
        Time interval between snapshots.

    Returns
    -------
    TaalbiLinearityResult
        Slope, intercept, R², and comparison to target slope of 1.0.
    """
    if dt <= 0:
        raise ValueError(f"dt must be positive, got {dt}")
    k_arr = np.asarray(k_total, dtype=float)
    target_slope = 1.0

    # First differences
    dk = np.diff(k_arr) / dt
    k_mid = k_arr[:-1]

    # Filter: need positive dk and positive k_mid for log
    mask = (dk > 0) & (k_mid > 0)
    n_points = int(np.sum(mask))

    if n_points < 2:
        return TaalbiLinearityResult(
            slope=float("nan"),
            intercept=float("nan"),
            r_squared=float("nan"),
            target_slope=target_slope,
            p_value=float("nan"),
            n_points=n_points,
        )

    log_dk = np.log(dk[mask])
    log_k = np.log(k_mid[mask])

    result = linregress(log_k, log_dk)
    return TaalbiLinearityResult(
        slope=result.slope,
        intercept=result.intercept,
        r_squared=result.rvalue ** 2,
        target_slope=target_slope,
        p_value=result.pvalue,
        n_points=n_points,
    )


def heaps_exponent(
    k_total: list[int],
    D_total: list[int],
) -> HeapsLawResult:
    """Test sub-linear diversification (Heaps' law, Taalbi 2025).

    Fits log(D) vs log(k) via OLS; exponent < 1.0 expected.

    Parameters
    ----------
    k_total : list[int]
        Cumulative innovation counts.
    D_total : list[int]
        Cumulative type diversity counts.

    Returns
    -------
    HeapsLawResult
        Exponent, fit quality, and sub-linearity check.
    """
    if len(k_total) != len(D_total):
        raise ValueError(
            f"Input lengths must match: k_total has {len(k_total)} "
            f"elements, D_total has {len(D_total)}"
        )
    k_arr = np.asarray(k_total, dtype=float)
    d_arr = np.asarray(D_total, dtype=float)

    mask = (k_arr > 0) & (d_arr > 0)
    n_points = int(np.sum(mask))

    if n_points < 2:
        return HeapsLawResult(
            exponent=float("nan"),
            intercept=float("nan"),
            r_squared=float("nan"),
            target="< 1.0",
            is_sublinear=False,
            n_points=n_points,
        )

    log_k = np.log(k_arr[mask])
    log_d = np.log(d_arr[mask])

    result = linregress(log_k, log_d)
    return HeapsLawResult(
        exponent=result.slope,
        intercept=result.intercept,
        r_squared=result.rvalue ** 2,
        target="< 1.0",
        is_sublinear=bool(result.slope < 1.0),
        n_points=n_points,
    )


def power_law_fit(
    agent_k_list: list[int],
    k_min: int = 1,
) -> PowerLawResult:
    """Fit power-law to per-agent innovation counts (Taalbi 2025).

    Uses the Hill MLE estimator and KS goodness-of-fit.

    Parameters
    ----------
    agent_k_list : list[int]
        Per-agent innovation counts.
    k_min : int
        Minimum count threshold for inclusion.

    Returns
    -------
    PowerLawResult
        Estimated exponent, KS statistic, and p-value.
    """
    if k_min < 1:
        raise ValueError(f"k_min must be >= 1, got {k_min}")
    k_arr = np.asarray(agent_k_list, dtype=float)
    target_exponent = 2.0

    # Filter to values >= k_min
    k_filtered = k_arr[k_arr >= k_min]
    n = len(k_filtered)

    if n < 2:
        return PowerLawResult(
            exponent=float("nan"),
            target_exponent=target_exponent,
            ks_statistic=float("nan"),
            p_value=float("nan"),
            n_agents=n,
            k_min=k_min,
        )

    # Hill MLE estimator: alpha = 1 + n / sum(ln(k_i / k_min))
    log_sum = np.sum(np.log(k_filtered / k_min))

    if log_sum <= 0:
        # All values equal to k_min — can't estimate
        return PowerLawResult(
            exponent=float("nan"),
            target_exponent=target_exponent,
            ks_statistic=float("nan"),
            p_value=float("nan"),
            n_agents=n,
            k_min=k_min,
        )

    alpha = 1.0 + n / log_sum

    # KS goodness-of-fit
    k_sorted = np.sort(k_filtered)
    empirical_cdf = np.arange(1, n + 1) / n
    theoretical_cdf = 1.0 - (k_sorted / k_min) ** (-(alpha - 1.0))
    ks_stat = float(np.max(np.abs(empirical_cdf - theoretical_cdf)))

    # Approximate p-value via asymptotic formula
    lambda_ks = ks_stat * math.sqrt(n)
    p_value = 2.0 * math.exp(-2.0 * lambda_ks ** 2)
    p_value = min(p_value, 1.0)  # clamp to [0, 1]

    return PowerLawResult(
        exponent=alpha,
        target_exponent=target_exponent,
        ks_statistic=ks_stat,
        p_value=p_value,
        n_agents=n,
        k_min=k_min,
    )
