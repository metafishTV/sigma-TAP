"""Run sigma-TAP simulation and compare against empirical targets.

CLAIM POLICY LABEL: exploratory

Compares simulation output against four quantitative targets from
Youn et al. (2015) and Taalbi (2025).

Usage:
  python scripts/empirical_validation.py
  python scripts/empirical_validation.py --n-agents 20 --steps 300
"""
from __future__ import annotations

import argparse
import math
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from simulator.metathetic import MetatheticEnsemble
from simulator.empirical import (
    EmpiricalValidationResult,
    youn_ratio,
    taalbi_linearity,
    heaps_exponent,
    power_law_fit,
)


# ---------------------------------------------------------------------------
# Status classification
# ---------------------------------------------------------------------------

def classify_status(deviation: float, threshold_match: float = 0.10) -> str:
    """Classify a deviation value as MATCH, CLOSE, or DIVERGENT.

    Parameters
    ----------
    deviation : float
        Absolute deviation from the target value.
    threshold_match : float
        Maximum deviation for a MATCH classification.

    Returns
    -------
    str
        "MATCH" if deviation <= threshold_match,
        "CLOSE" if deviation <= threshold_match * 2.5,
        "DIVERGENT" otherwise.
    """
    if math.isnan(deviation):
        return "N/A"
    if deviation <= threshold_match:
        return "MATCH"
    if deviation <= threshold_match * 2.5:
        return "CLOSE"
    return "DIVERGENT"


# ---------------------------------------------------------------------------
# Validation runner
# ---------------------------------------------------------------------------

def run_validation(
    n_agents: int = 10,
    steps: int = 150,
    alpha: float = 5e-3,
    a: float = 3.0,
    mu: float = 0.005,
    variant: str = "logistic",
    seed: int = 42,
) -> EmpiricalValidationResult:
    """Run a simulation and compute all four empirical validation metrics.

    Parameters
    ----------
    n_agents : int
        Number of agents in the ensemble.
    steps : int
        Number of simulation steps.
    alpha, a, mu : float
        TAP growth parameters.
    variant : str
        TAP variant (e.g. "logistic", "baseline").
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    EmpiricalValidationResult
        Full validation output with all four metric results.
    """
    carrying_capacity = 2e5 if variant == "logistic" else None

    ens = MetatheticEnsemble(
        n_agents=n_agents,
        initial_M=10.0,
        alpha=alpha,
        a=a,
        mu=mu,
        variant=variant,
        carrying_capacity=carrying_capacity,
        seed=seed,
    )
    trajectory = ens.run(steps=steps)

    # Extract trajectory fields
    n_novel = [s["n_novel_cross"] for s in trajectory]
    n_absorptive = [s["n_absorptive_cross"] for s in trajectory]
    k_total_list = [s["k_total"] for s in trajectory]
    D_total_list = [s["D_total"] for s in trajectory]
    agent_k_list = trajectory[-1]["agent_k_list"]

    # Compute all four metrics
    yr = youn_ratio(n_novel, n_absorptive)
    tl = taalbi_linearity(k_total_list)
    he = heaps_exponent(k_total_list, D_total_list)
    pl = power_law_fit(agent_k_list)

    return EmpiricalValidationResult(
        youn=yr,
        linearity=tl,
        heaps=he,
        power_law=pl,
        params_used={
            "alpha": alpha,
            "a": a,
            "mu": mu,
            "variant": variant,
            "seed": seed,
        },
        n_steps=steps,
        n_agents=n_agents,
    )


# ---------------------------------------------------------------------------
# Pretty-print
# ---------------------------------------------------------------------------

def print_validation(result: EmpiricalValidationResult) -> None:
    """Print a formatted summary table of validation results.

    Parameters
    ----------
    result : EmpiricalValidationResult
        Output from run_validation().
    """
    yr = result.youn
    tl = result.linearity
    he = result.heaps
    pl = result.power_law

    # Youn status
    if math.isnan(yr.deviation):
        youn_status = "N/A (no events)"
        youn_measured = "N/A"
        youn_detail = ""
    else:
        youn_status_label = classify_status(yr.deviation)
        youn_measured = f"{yr.exploration_fraction:.3f}"
        youn_detail = f" (delta={yr.deviation:.3f})"
        youn_status = f"{youn_status_label}{youn_detail}"

    # Taalbi status
    if math.isnan(tl.slope):
        taalbi_status = "N/A (insufficient data)"
        taalbi_measured = "N/A"
    else:
        slope_dev = abs(tl.slope - tl.target_slope)
        taalbi_status_label = classify_status(slope_dev, threshold_match=0.20)
        taalbi_measured = f"{tl.slope:.2f}"
        taalbi_status = f"{taalbi_status_label} (r2={tl.r_squared:.2f})"

    # Heaps status
    if math.isnan(he.exponent):
        heaps_status = "N/A (insufficient data)"
        heaps_measured = "N/A"
    else:
        heaps_measured = f"{he.exponent:.2f}"
        if he.is_sublinear:
            heaps_status = "MATCH (sub-linear)"
        else:
            heaps_status = "DIVERGENT (not sub-linear)"

    # Power-law status
    if math.isnan(pl.exponent):
        pl_status = "N/A (insufficient data)"
        pl_measured = "N/A"
    else:
        pl_dev = abs(pl.exponent - pl.target_exponent)
        pl_status_label = classify_status(pl_dev, threshold_match=0.30)
        pl_measured = f"{pl.exponent:.2f}"
        pl_status = f"{pl_status_label} (KS={pl.ks_statistic:.2f})"

    params = result.params_used
    header = f"Empirical Validation Summary ({result.n_agents} agents, {result.n_steps} steps)"

    print(header)
    print("=" * 70)
    print(f"  {'Metric':<22} {'Target':<16} {'Measured':<14} {'Status'}")
    print("-" * 70)
    print(f"  {'Youn exploration':<22} {'0.600':<16} {youn_measured:<14} {youn_status}")
    print(f"  {'Taalbi linearity':<22} {'slope~=1.0':<16} {taalbi_measured:<14} {taalbi_status}")
    print(f"  {'Heaps exponent':<22} {'< 1.0':<16} {heaps_measured:<14} {heaps_status}")
    print(f"  {'Power-law exponent':<22} {'~= 2.0':<16} {pl_measured:<14} {pl_status}")
    print("=" * 70)
    print(f"  Params: alpha={params['alpha']}, a={params['a']}, "
          f"mu={params['mu']}, variant={params['variant']}, seed={params['seed']}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Parse arguments and run the validation pipeline."""
    parser = argparse.ArgumentParser(
        description="Run sigma-TAP empirical validation against literature targets."
    )
    parser.add_argument("--n-agents", type=int, default=10,
                        help="Number of agents (default: 10)")
    parser.add_argument("--steps", type=int, default=150,
                        help="Simulation steps (default: 150)")
    parser.add_argument("--alpha", type=float, default=5e-3,
                        help="TAP alpha parameter (default: 5e-3)")
    parser.add_argument("-a", type=float, default=3.0,
                        help="TAP a parameter (default: 3.0)")
    parser.add_argument("--mu", type=float, default=0.005,
                        help="TAP mu parameter (default: 0.005)")
    parser.add_argument("--variant", type=str, default="logistic",
                        help="TAP variant (default: logistic)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")

    args = parser.parse_args()

    result = run_validation(
        n_agents=args.n_agents,
        steps=args.steps,
        alpha=args.alpha,
        a=args.a,
        mu=args.mu,
        variant=args.variant,
        seed=args.seed,
    )
    print_validation(result)


if __name__ == "__main__":
    main()
