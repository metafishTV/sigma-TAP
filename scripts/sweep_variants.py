"""Unified cross-variant comparison sweep.

Runs baseline, two_scale, and logistic variants over the same parameter
grid and emits a single comparison CSV.

Usage:
  python scripts/sweep_variants.py > outputs/variant_comparison.csv
"""
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from simulator.analysis import adaptive_xi_plateau_threshold, classify_regime
from simulator.hfuncs import h_compression
from simulator.simulate import run_sigma_tap
from simulator.state import ModelParams


def main() -> None:
    alphas = np.logspace(np.log10(1e-5), np.log10(1e-2), 8).tolist()
    mus = np.logspace(np.log10(1e-3), np.log10(1e-1), 8).tolist()
    m0_values = [10.0, 20.0, 50.0]
    variants = ["baseline", "two_scale", "logistic"]

    writer = csv.DictWriter(
        sys.stdout,
        fieldnames=[
            "variant", "alpha", "mu", "a", "m0",
            "steps", "final_M", "final_Xi",
            "regime", "blowup_step",
        ],
    )
    writer.writeheader()

    for variant in variants:
        for m0 in m0_values:
            for alpha in alphas:
                for mu in mus:
                    alpha1 = 10.0 * alpha if variant == "two_scale" else 0.0
                    K = 2e5 if variant == "logistic" else None
                    params = ModelParams(
                        alpha=alpha, a=8.0, mu=mu,
                        beta=0.05, eta=0.02,
                        tap_variant=variant,
                        alpha1=alpha1,
                        carrying_capacity=K,
                    )
                    steps = 120 if (alpha >= 1e-3 and mu <= 1e-2) else 40
                    rows = run_sigma_tap(
                        initial_M=m0, steps=steps, params=params,
                        sigma0=1.0, gamma=0.0,
                        h_func=lambda s: h_compression(s, decay=0.02),
                        append_terminal_state=True,
                    )
                    tr = [r for r in rows if "M_t1" in r]
                    if not tr:
                        xi = [0.0]
                        m = [m0]
                    else:
                        xi = [tr[0]["Xi"]] + [r["Xi_t1"] for r in tr]
                        m = [tr[0]["M"]] + [r["M_t1"] for r in tr]
                    thr = adaptive_xi_plateau_threshold(xi)
                    regime = classify_regime(xi, m, thr)
                    blowup = next(
                        (r.get("blowup_step") for r in tr if r.get("overflow_detected")),
                        None,
                    )
                    writer.writerow({
                        "variant": variant,
                        "alpha": f"{alpha:.8g}",
                        "mu": f"{mu:.8g}",
                        "a": "8",
                        "m0": f"{m0:.8g}",
                        "steps": steps,
                        "final_M": f"{m[-1]:.8g}",
                        "final_Xi": f"{xi[-1]:.8g}",
                        "regime": regime,
                        "blowup_step": "" if blowup is None else int(blowup),
                    })


if __name__ == "__main__":
    main()
