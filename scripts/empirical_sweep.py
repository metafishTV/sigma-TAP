"""Parameter sweep for sigma-TAP empirical validation.

CLAIM POLICY LABEL: exploratory

Runs empirical validation metrics across a grid of parameters to find
best-fit regimes and map parameter sensitivity.

Usage:
  python scripts/empirical_sweep.py
  python scripts/empirical_sweep.py --quick
  python scripts/empirical_sweep.py --steps 500 --top 10
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
import warnings
from dataclasses import dataclass, field
from itertools import product as grid_product

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from simulator.metathetic import MetatheticEnsemble
from simulator.empirical import (
    EmpiricalValidationResult,
    youn_ratio,
    taalbi_linearity,
    heaps_exponent,
    power_law_fit,
)

sys.path.insert(0, os.path.dirname(__file__))
from empirical_validation import classify_status


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_GRID = {
    "alpha": [1e-3, 3e-3, 5e-3, 1e-2],
    "a": [2.0, 3.0, 5.0, 8.0],
    "mu": [0.002, 0.005, 0.01, 0.02],
    "n_agents": [10, 15, 20],
}
QUICK_GRID = {
    "alpha": [1e-3, 5e-3],
    "a": [3.0, 8.0],
    "mu": [0.005, 0.02],
    "n_agents": [10],
}
DEFAULT_SEEDS = [42, 123, 456]
QUICK_SEEDS = [42]
DEFAULT_STEPS = 200
QUICK_STEPS = 50
NAN_PENALTY = 1.0


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class SweepPoint:
    """Single parameter combination with aggregated metric results."""
    params: dict
    results: list[EmpiricalValidationResult]
    mean_youn_deviation: float
    mean_linearity_deviation: float
    mean_heaps_exponent: float
    heaps_match: bool
    mean_powerlaw_deviation: float
    composite_score: float


@dataclass
class SweepResult:
    """Full sweep output: all points, best subset, and sensitivity analysis."""
    points: list[SweepPoint]
    best: list[SweepPoint]
    sensitivity: dict[str, dict[str, float]]
    grid: dict
    seeds: list[int]
    steps: int
    total_sims: int


# ---------------------------------------------------------------------------
# Scoring functions
# ---------------------------------------------------------------------------

def compute_sweep_point(
    params: dict,
    results: list[EmpiricalValidationResult],
) -> SweepPoint:
    """Compute aggregate scores for a single parameter combination.

    Parameters
    ----------
    params : dict
        Parameter values used for this sweep point.
    results : list[EmpiricalValidationResult]
        One result per seed (may be fewer if some seeds failed).

    Returns
    -------
    SweepPoint
        Aggregated metrics and composite score.
    """
    if not results:
        return SweepPoint(
            params=params,
            results=results,
            mean_youn_deviation=NAN_PENALTY,
            mean_linearity_deviation=NAN_PENALTY,
            mean_heaps_exponent=NAN_PENALTY,
            heaps_match=False,
            mean_powerlaw_deviation=NAN_PENALTY,
            composite_score=4 * NAN_PENALTY,
        )

    # Youn deviation: |fraction - 0.6|, NaN -> NAN_PENALTY
    youn_devs = []
    for r in results:
        frac = r.youn.exploration_fraction
        if math.isnan(frac):
            youn_devs.append(NAN_PENALTY)
        else:
            youn_devs.append(abs(frac - 0.6))
    mean_youn_dev = sum(youn_devs) / len(youn_devs)

    # Linearity deviation: |slope - 1.0|, NaN -> NAN_PENALTY
    lin_devs = []
    for r in results:
        slope = r.linearity.slope
        if math.isnan(slope):
            lin_devs.append(NAN_PENALTY)
        else:
            lin_devs.append(abs(slope - 1.0))
    mean_lin_dev = sum(lin_devs) / len(lin_devs)

    # Heaps exponent: average, NaN -> NAN_PENALTY
    heaps_vals = []
    for r in results:
        exp = r.heaps.exponent
        if math.isnan(exp):
            heaps_vals.append(NAN_PENALTY)
        else:
            heaps_vals.append(exp)
    mean_heaps_exp = sum(heaps_vals) / len(heaps_vals)
    heaps_match = mean_heaps_exp < 1.0

    # Heaps deviation for composite: 0.0 if match, else (mean_exponent - 1.0)
    heaps_dev = 0.0 if heaps_match else (mean_heaps_exp - 1.0)

    # Power-law deviation: |exponent - 2.0|, NaN -> NAN_PENALTY
    pl_devs = []
    for r in results:
        exp = r.power_law.exponent
        if math.isnan(exp):
            pl_devs.append(NAN_PENALTY)
        else:
            pl_devs.append(abs(exp - 2.0))
    mean_pl_dev = sum(pl_devs) / len(pl_devs)

    composite = mean_youn_dev + mean_lin_dev + heaps_dev + mean_pl_dev

    return SweepPoint(
        params=params,
        results=results,
        mean_youn_deviation=mean_youn_dev,
        mean_linearity_deviation=mean_lin_dev,
        mean_heaps_exponent=mean_heaps_exp,
        heaps_match=heaps_match,
        mean_powerlaw_deviation=mean_pl_dev,
        composite_score=composite,
    )


def rank_sweep_results(
    points: list[SweepPoint],
    top_n: int = 5,
) -> list[SweepPoint]:
    """Sort sweep points by composite score and return the best top_n.

    Parameters
    ----------
    points : list[SweepPoint]
        All sweep points to rank.
    top_n : int
        Number of top results to return.

    Returns
    -------
    list[SweepPoint]
        Best top_n points sorted by ascending composite score.
    """
    sorted_points = sorted(points, key=lambda p: p.composite_score)
    return sorted_points[:top_n]


def compute_sensitivity(
    points: list[SweepPoint],
    grid: dict,
) -> dict[str, dict[str, float]]:
    """Compute parameter sensitivity as range of mean deviations.

    For each parameter, groups sweep points by parameter value and
    computes the range (max - min) of group-mean deviations.

    Parameters
    ----------
    points : list[SweepPoint]
        All sweep points.
    grid : dict
        Parameter grid used for the sweep.

    Returns
    -------
    dict[str, dict[str, float]]
        Sensitivity per parameter per metric.
    """
    sensitivity: dict[str, dict[str, float]] = {}

    metrics = ["youn", "linearity", "heaps", "powerlaw"]

    for param_name, param_values in grid.items():
        unique_vals = sorted(set(param_values))
        metric_group_means: dict[str, list[float]] = {m: [] for m in metrics}

        for val in unique_vals:
            group = [p for p in points if p.params.get(param_name) == val]
            if not group:
                continue

            metric_group_means["youn"].append(
                sum(p.mean_youn_deviation for p in group) / len(group)
            )
            metric_group_means["linearity"].append(
                sum(p.mean_linearity_deviation for p in group) / len(group)
            )
            metric_group_means["heaps"].append(
                sum(p.mean_heaps_exponent for p in group) / len(group)
            )
            metric_group_means["powerlaw"].append(
                sum(p.mean_powerlaw_deviation for p in group) / len(group)
            )

        param_sens: dict[str, float] = {}
        for m in metrics:
            vals = metric_group_means[m]
            if len(vals) < 2:
                param_sens[m] = 0.0
            else:
                param_sens[m] = max(vals) - min(vals)

        sensitivity[param_name] = param_sens

    return sensitivity


# ---------------------------------------------------------------------------
# Simulation runner
# ---------------------------------------------------------------------------

def run_single_point(
    params: dict,
    seeds: list[int],
    steps: int,
    variant: str = "logistic",
) -> SweepPoint:
    """Run simulation for one parameter combination across all seeds.

    Parameters
    ----------
    params : dict
        Must contain keys: alpha, a, mu, n_agents.
    seeds : list[int]
        Random seeds to average over.
    steps : int
        Number of simulation steps.
    variant : str
        TAP variant (e.g. "logistic", "baseline").

    Returns
    -------
    SweepPoint
        Aggregated results for this parameter combination.
    """
    results: list[EmpiricalValidationResult] = []

    for seed in seeds:
        try:
            carrying_capacity = 2e5 if variant == "logistic" else None
            ens = MetatheticEnsemble(
                n_agents=int(params["n_agents"]),
                initial_M=10.0,
                alpha=params["alpha"],
                a=params["a"],
                mu=params["mu"],
                variant=variant,
                carrying_capacity=carrying_capacity,
                seed=seed,
            )
            trajectory = ens.run(steps=steps)

            n_novel = [s["n_novel_cross"] for s in trajectory]
            n_absorptive = [s["n_absorptive_cross"] for s in trajectory]
            k_total_list = [s["k_total"] for s in trajectory]
            D_total_list = [s["D_total"] for s in trajectory]
            agent_k_list = trajectory[-1]["agent_k_list"]

            yr = youn_ratio(n_novel, n_absorptive)
            tl = taalbi_linearity(k_total_list)
            he = heaps_exponent(k_total_list, D_total_list)
            pl = power_law_fit(agent_k_list)

            result = EmpiricalValidationResult(
                youn=yr,
                linearity=tl,
                heaps=he,
                power_law=pl,
                params_used={
                    "alpha": params["alpha"],
                    "a": params["a"],
                    "mu": params["mu"],
                    "variant": variant,
                    "seed": seed,
                },
                n_steps=steps,
                n_agents=int(params["n_agents"]),
            )
            results.append(result)

        except Exception as exc:
            warnings.warn(
                f"Seed {seed} failed for params {params}: {exc}",
                stacklevel=2,
            )

    return compute_sweep_point(params, results)


def run_sweep(
    grid: dict,
    seeds: list[int],
    steps: int,
    variant: str = "logistic",
    top_n: int = 5,
) -> SweepResult:
    """Run parameter sweep across the full grid.

    Parameters
    ----------
    grid : dict
        Parameter grid: {param_name: [values]}.
    seeds : list[int]
        Random seeds to average over per combo.
    steps : int
        Simulation steps per run.
    variant : str
        TAP variant.
    top_n : int
        Number of best results to highlight.

    Returns
    -------
    SweepResult
        Full sweep output with ranked results and sensitivity.
    """
    # Generate all parameter combinations
    param_names = list(grid.keys())
    param_value_lists = [grid[k] for k in param_names]
    combos = list(grid_product(*param_value_lists))
    total = len(combos)

    points: list[SweepPoint] = []

    for i, combo in enumerate(combos):
        params = {name: val for name, val in zip(param_names, combo)}
        point = run_single_point(params, seeds, steps, variant)
        points.append(point)

        # Progress output
        print(
            f"\r  [{i + 1}/{total}] "
            f"alpha={params.get('alpha', '?'):.0e} "
            f"a={params.get('a', '?'):.1f} "
            f"mu={params.get('mu', '?'):.3f} "
            f"agents={params.get('n_agents', '?')} "
            f"score={point.composite_score:.3f}",
            end="",
            flush=True,
        )

    print()  # newline after progress

    best = rank_sweep_results(points, top_n)
    sensitivity = compute_sensitivity(points, grid)

    return SweepResult(
        points=points,
        best=best,
        sensitivity=sensitivity,
        grid=grid,
        seeds=seeds,
        steps=steps,
        total_sims=total,
    )


# ---------------------------------------------------------------------------
# Pretty-print
# ---------------------------------------------------------------------------

def print_sweep_results(result: SweepResult, top_n: int = 5) -> None:
    """Print formatted tables of best-fit regimes and parameter sensitivity.

    Parameters
    ----------
    result : SweepResult
        Output from run_sweep().
    top_n : int
        Number of best results to display.
    """
    best = result.best[:top_n]

    # ── Table 1: Best-fit regimes ──
    print()
    header = (
        f"Top {len(best)} Parameter Regimes "
        f"(by composite score, {result.total_sims} simulations)"
    )
    print(header)
    print("=" * 78)
    print(
        f"  {'Rank':<6}"
        f"{'alpha':<10}"
        f"{'a':<7}"
        f"{'mu':<9}"
        f"{'agents':<8}"
        f"{'Score':<8}"
        f"{'Youn':<8}"
        f"{'Slope':<8}"
        f"{'Heaps':<7}"
        f"{'PowLaw':<7}"
    )
    print("-" * 78)

    for rank, point in enumerate(best, start=1):
        p = point.params
        alpha_str = f"{p.get('alpha', 0):.3f}"
        a_str = f"{p.get('a', 0):.1f}"
        mu_str = f"{p.get('mu', 0):.3f}"
        agents_str = f"{p.get('n_agents', 0)}"

        # Youn: show average exploration fraction
        if point.results:
            youn_fracs = [
                r.youn.exploration_fraction for r in point.results
                if not math.isnan(r.youn.exploration_fraction)
            ]
            youn_str = f"{sum(youn_fracs) / len(youn_fracs):.2f}" if youn_fracs else "N/A"
        else:
            youn_str = "N/A"

        # Slope: show average linearity slope
        if point.results:
            slopes = [
                r.linearity.slope for r in point.results
                if not math.isnan(r.linearity.slope)
            ]
            slope_str = f"{sum(slopes) / len(slopes):.2f}" if slopes else "N/A"
        else:
            slope_str = "N/A"

        # Heaps: show average exponent
        heaps_str = f"{point.mean_heaps_exponent:.2f}"

        # Power-law: show average exponent
        if point.results:
            pl_exps = [
                r.power_law.exponent for r in point.results
                if not math.isnan(r.power_law.exponent)
            ]
            pl_str = f"{sum(pl_exps) / len(pl_exps):.2f}" if pl_exps else "N/A"
        else:
            pl_str = "N/A"

        print(
            f"  {rank:<6}"
            f"{alpha_str:<10}"
            f"{a_str:<7}"
            f"{mu_str:<9}"
            f"{agents_str:<8}"
            f"{point.composite_score:<8.2f}"
            f"{youn_str:<8}"
            f"{slope_str:<8}"
            f"{heaps_str:<7}"
            f"{pl_str:<7}"
        )

    print("=" * 78)

    # ── Table 2: Sensitivity ──
    print()
    print("Parameter Sensitivity (range of mean deviation across values)")
    print("=" * 68)
    print(
        f"  {'Parameter':<12}"
        f"{'Youn':<12}"
        f"{'Linearity':<12}"
        f"{'Heaps':<12}"
        f"{'Power-law':<12}"
    )
    print("-" * 68)

    metrics = ["youn", "linearity", "heaps", "powerlaw"]
    metric_labels = ["Youn", "Linearity", "Heaps", "Power-law"]

    # Find max sensitivity per metric column for * marking
    max_sens: dict[str, tuple[float, str]] = {}
    for m in metrics:
        best_val = -1.0
        best_param = ""
        for param_name in result.sensitivity:
            val = result.sensitivity[param_name].get(m, 0.0)
            if val > best_val:
                best_val = val
                best_param = param_name
        max_sens[m] = (best_val, best_param)

    for param_name in result.sensitivity:
        sens = result.sensitivity[param_name]
        cells = []
        for m in metrics:
            val = sens.get(m, 0.0)
            marker = "*" if max_sens[m][1] == param_name and max_sens[m][0] > 0 else ""
            cells.append(f"{val:.2f}{marker}")

        print(
            f"  {param_name:<12}"
            f"{cells[0]:<12}"
            f"{cells[1]:<12}"
            f"{cells[2]:<12}"
            f"{cells[3]:<12}"
        )

    print("=" * 68)
    print("  * = most sensitive parameter for this metric")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Parse arguments and run the parameter sweep."""
    parser = argparse.ArgumentParser(
        description="Parameter sweep for sigma-TAP empirical validation."
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Use reduced grid and single seed for fast iteration",
    )
    parser.add_argument(
        "--steps", type=int, default=None,
        help=f"Simulation steps (default: {DEFAULT_STEPS}, quick: {QUICK_STEPS})",
    )
    parser.add_argument(
        "--top", type=int, default=5,
        help="Number of best results to show (default: 5)",
    )
    parser.add_argument(
        "--variant", type=str, default="logistic",
        help="TAP variant (default: logistic)",
    )

    args = parser.parse_args()

    if args.quick:
        grid = QUICK_GRID
        seeds = QUICK_SEEDS
        steps = args.steps if args.steps is not None else QUICK_STEPS
    else:
        grid = DEFAULT_GRID
        seeds = DEFAULT_SEEDS
        steps = args.steps if args.steps is not None else DEFAULT_STEPS

    # Compute total combos
    n_combos = 1
    for vals in grid.values():
        n_combos *= len(vals)
    total_runs = n_combos * len(seeds)

    print("=" * 68)
    print("sigma-TAP Empirical Parameter Sweep")
    print("=" * 68)
    print(f"  Grid:    {n_combos} parameter combinations")
    print(f"  Seeds:   {seeds}")
    print(f"  Steps:   {steps}")
    print(f"  Variant: {args.variant}")
    print(f"  Total:   {total_runs} simulation runs")
    print("-" * 68)
    print()

    t0 = time.time()
    result = run_sweep(
        grid=grid,
        seeds=seeds,
        steps=steps,
        variant=args.variant,
        top_n=args.top,
    )
    elapsed = time.time() - t0

    print_sweep_results(result, top_n=args.top)
    print(f"\n  Completed in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
