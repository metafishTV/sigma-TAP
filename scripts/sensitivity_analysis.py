"""Unified sensitivity analysis: extinction (mu) and adjacency (a) sweeps.

Runs baseline, two_scale, and logistic variants across parameter ranges
using the discrete simulator. Fast — completes in <10 seconds.

Usage:
  python scripts/sensitivity_analysis.py
  python scripts/sensitivity_analysis.py --n-mu 50 --steps 200
"""
from __future__ import annotations

import csv
import math
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from simulator.analysis import adaptive_xi_plateau_threshold, classify_regime
from simulator.hfuncs import h_compression
from simulator.simulate import run_sigma_tap
from simulator.state import ModelParams

ROOT = Path(__file__).resolve().parents[1]

VARIANTS = ["baseline", "two_scale", "logistic"]


def _run_one(params: ModelParams, m0: float, steps: int) -> dict:
    """Run a single simulation and extract summary statistics."""
    rows = run_sigma_tap(
        initial_M=m0, steps=steps, params=params,
        sigma0=1.0, gamma=0.0,
        h_func=lambda s: h_compression(s, decay=0.02),
        append_terminal_state=True,
    )
    tr = [r for r in rows if "M_t1" in r]
    if not tr:
        return {"final_M": m0, "final_Xi": 0.0, "regime": "plateau",
                "blowup_step": None, "transition_step": None}

    xi = [tr[0]["Xi"]] + [r["Xi_t1"] for r in tr]
    m = [tr[0]["M"]] + [r["M_t1"] for r in tr]
    thr = adaptive_xi_plateau_threshold(xi)
    regime = classify_regime(xi, m, thr)
    blowup = next(
        (r.get("blowup_step") for r in tr if r.get("overflow_detected")),
        None,
    )

    # Find transition step: first step where regime changes from plateau.
    transition_step = None
    for i in range(3, len(m)):
        sub_regime = classify_regime(xi[:i+1], m[:i+1], thr)
        if sub_regime != "plateau":
            transition_step = i
            break

    return {
        "final_M": m[-1],
        "final_Xi": xi[-1],
        "regime": regime,
        "blowup_step": blowup,
        "transition_step": transition_step,
    }


def run_extinction_sweep(
    n_mu: int = 30,
    steps: int = 120,
    alpha: float = 1e-3,
    a: float = 8.0,
    m0: float = 10.0,
    mu_range: tuple[float, float] = (1e-4, 5e-1),
) -> list[dict]:
    """Sweep extinction rate mu across all variants."""
    results = []
    mus = np.logspace(np.log10(mu_range[0]), np.log10(mu_range[1]), n_mu).tolist()

    for mu in mus:
        for variant in VARIANTS:
            alpha1 = 10.0 * alpha if variant == "two_scale" else 0.0
            K = 2e5 if variant == "logistic" else None
            params = ModelParams(
                alpha=alpha, a=a, mu=mu,
                beta=0.05, eta=0.02,
                tap_variant=variant,
                alpha1=alpha1,
                carrying_capacity=K,
            )
            summary = _run_one(params, m0, steps)
            summary.update({"mu": mu, "alpha": alpha, "a": a, "m0": m0,
                            "variant": variant, "steps": steps})
            results.append(summary)

    return results


def run_adjacency_sweep(
    a_values: list[float] | None = None,
    steps: int = 120,
    alpha: float = 1e-3,
    mu: float = 0.01,
    m0: float = 10.0,
) -> list[dict]:
    """Sweep adjacency parameter a across all variants."""
    if a_values is None:
        a_values = [2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 32.0]

    results = []
    for a in a_values:
        for variant in VARIANTS:
            alpha1 = 10.0 * alpha if variant == "two_scale" else 0.0
            K = 2e5 if variant == "logistic" else None
            params = ModelParams(
                alpha=alpha, a=a, mu=mu,
                beta=0.05, eta=0.02,
                tap_variant=variant,
                alpha1=alpha1,
                carrying_capacity=K,
            )
            summary = _run_one(params, m0, steps)
            summary.update({"mu": mu, "alpha": alpha, "a": a, "m0": m0,
                            "variant": variant, "steps": steps})
            results.append(summary)

    return results


def _write_csv(rows: list[dict], path: Path) -> None:
    """Write list of dicts to CSV."""
    if not rows:
        return
    path.parent.mkdir(exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {path} ({len(rows)} rows)")


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Sensitivity analysis sweeps")
    parser.add_argument("--n-mu", type=int, default=30)
    parser.add_argument("--steps", type=int, default=120)
    args = parser.parse_args()

    ext_rows = run_extinction_sweep(n_mu=args.n_mu, steps=args.steps)
    _write_csv(ext_rows, ROOT / "outputs" / "extinction_sensitivity.csv")

    adj_rows = run_adjacency_sweep(steps=args.steps)
    _write_csv(adj_rows, ROOT / "outputs" / "adjacency_sensitivity.csv")

    print("\nExtinction sweep summary:")
    for variant in VARIANTS:
        vrows = [r for r in ext_rows if r["variant"] == variant]
        regimes = {}
        for r in vrows:
            regimes[r["regime"]] = regimes.get(r["regime"], 0) + 1
        print(f"  {variant}: {regimes}")

    print("\nAdjacency sweep summary:")
    for variant in VARIANTS:
        vrows = [r for r in adj_rows if r["variant"] == variant]
        for r in vrows:
            print(f"  {variant} a={r['a']}: final_M={r['final_M']:.4g} regime={r['regime']}")


if __name__ == "__main__":
    main()
