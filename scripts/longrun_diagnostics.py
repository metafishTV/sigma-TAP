"""Run metathetic ensemble and compute long-run diagnostics.

CLAIM POLICY LABEL: exploratory
This script produces results from the metathetic multi-agent extension,
which does not derive from the source TAP/biocosmology literature.

Produces:
  outputs/longrun_diagnostics.csv          — per-step ensemble data
  outputs/longrun_diagnostics_summary.json — summary statistics
  outputs/figures/heaps_law.png            — D(k) log-log plot
  outputs/figures/concentration_gini.png   — Gini + top-10% over time

Usage:
  python scripts/longrun_diagnostics.py
  python scripts/longrun_diagnostics.py --n-agents 20 --steps 200
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from simulator.metathetic import MetatheticEnsemble
from simulator.longrun import (
    heaps_law_fit,
    gini_coefficient,
    top_k_share,
    enhanced_constraint_tag,
)
from simulator.analysis import innovation_rate_scaling

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
FIG_OUT = OUT / "figures"


def run_and_diagnose(
    n_agents: int = 10,
    initial_M: float = 10.0,
    alpha: float = 5e-3,
    a: float = 3.0,
    mu: float = 0.005,
    steps: int = 150,
    seed: int = 42,
    variant: str = "logistic",
    self_meta_threshold: float = 0.15,
) -> tuple[list[dict], dict]:
    """Run ensemble and compute all diagnostics.

    Default parameters use logistic TAP variant with growth params,
    producing resource-constrained dynamics that should exhibit
    Heaps' law (beta < 1) per Taalbi (2025).
    """
    ensemble = MetatheticEnsemble(
        n_agents=n_agents,
        initial_M=initial_M,
        alpha=alpha, a=a, mu=mu,
        variant=variant,
        carrying_capacity=2e5 if variant == "logistic" else None,
        self_meta_threshold=self_meta_threshold,
        seed=seed,
    )
    trajectory = ensemble.run(steps=steps)

    # Extract series for diagnostics.
    D_series = [s["D_total"] for s in trajectory]
    k_series = [s["k_total"] for s in trajectory]
    M_series = [s["total_M"] for s in trajectory]

    # Heaps' law.
    heaps = heaps_law_fit(D_series, k_series)

    # Innovation rate scaling on aggregate M.
    scaling = innovation_rate_scaling(M_series)

    # Gini at final step.
    final_k_list = trajectory[-1]["agent_k_list"]
    gini_final = gini_coefficient(final_k_list)
    top10_final = top_k_share(final_k_list, k_frac=0.1)

    # Enhanced constraint tag.
    K = ensemble.carrying_capacity
    tag_result = enhanced_constraint_tag(
        sigma=scaling["exponent"],
        beta=heaps["beta"],
        gini=gini_final,
        carrying_capacity=K,
        m_final=M_series[-1] if M_series else 0.0,
    )

    last = trajectory[-1]
    summary = {
        "heaps_beta": round(heaps["beta"], 4),
        "heaps_intercept": round(heaps["intercept"], 6),
        "heaps_r_squared": round(heaps["r_squared"], 4),
        "innovation_sigma": round(scaling["exponent"], 4),
        "innovation_sigma_r2": round(scaling["r_squared"], 4),
        "gini_final": round(gini_final, 4),
        "top10_share_final": round(top10_final, 4),
        "constraint_tag": tag_result["tag"],
        "constraint_confidence": tag_result["confidence"],
        "constraint_reasoning": tag_result["reasoning"],
        "n_agents_final": last["n_active"],
        "n_dormant_final": last["n_dormant"],
        "n_self_metatheses": last["n_self_metatheses"],
        "n_absorptive_cross": last["n_absorptive_cross"],
        "n_novel_cross": last["n_novel_cross"],
        "n_env_transitions": last["n_env_transitions"],
        "temporal_state_counts_final": last.get("temporal_state_counts", {}),
        "texture_type_final": last["texture_type"],
        "D_total_final": last["D_total"],
        "k_total_final": round(last["k_total"], 4),
        "total_M_final": round(last["total_M"], 4),
        "steps": steps,
        "n_agents_initial": n_agents,
        "claim_policy_label": "exploratory",
        "disclaimer": (
            "This result is exploratory and does not derive from the source "
            "TAP/biocosmology literature. It uses sigma-TAP computational "
            "infrastructure but represents an independent analytical direction "
            "that requires further validation."
        ),
    }

    return trajectory, summary


def write_csv(trajectory: list[dict], path: Path) -> None:
    """Write per-step CSV (excluding agent_k_list which is variable-length)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [k for k in trajectory[0].keys() if k != "agent_k_list"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(trajectory)
    print(f"Wrote {path} ({len(trajectory)} rows)")


def write_summary(summary: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {path}")


def fig_heaps_law(trajectory: list[dict], summary: dict, save_path: Path) -> None:
    """D(k) log-log plot with fitted Heaps exponent."""
    plt.rcParams.update({
        "font.size": 11, "savefig.dpi": 300, "savefig.bbox": "tight",
        "axes.grid": True, "grid.alpha": 0.3,
    })

    D_series = [s["D_total"] for s in trajectory]
    k_series = [s["k_total"] for s in trajectory]

    # Filter to positive values.
    k_pos = [k for k, d in zip(k_series, D_series) if k > 0 and d > 0]
    D_pos = [d for k, d in zip(k_series, D_series) if k > 0 and d > 0]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(k_pos, D_pos, s=8, alpha=0.5, color="#2196F3", label="Ensemble data")

    # Fitted line using OLS intercept (not first-point anchor).
    beta = summary["heaps_beta"]
    r2 = summary["heaps_r_squared"]
    if k_pos and D_pos:
        k_fit = np.linspace(min(k_pos), max(k_pos), 100)
        intercept = summary["heaps_intercept"]  # OLS fitted intercept in log-space
        D_fit = np.exp(intercept) * k_fit ** beta
        ax.plot(k_fit, D_fit, "r--", linewidth=2,
                label=f"Heaps fit: beta={beta:.3f} (R2={r2:.3f})")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Cumulative innovations k")
    ax.set_ylabel("Diversity D (unique types)")
    ax.set_title("Heaps' Law: D(k) ~ k^beta  [exploratory]")
    ax.legend()

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(save_path))
    plt.close(fig)
    print(f"Wrote {save_path}")


def fig_concentration(trajectory: list[dict], save_path: Path) -> None:
    """Gini and top-10% share over time."""
    plt.rcParams.update({
        "font.size": 11, "savefig.dpi": 300, "savefig.bbox": "tight",
        "axes.grid": True, "grid.alpha": 0.3,
    })

    steps_arr = [s["step"] for s in trajectory]
    gini_arr = [gini_coefficient(s["agent_k_list"]) for s in trajectory]
    top10_arr = [top_k_share(s["agent_k_list"], k_frac=0.1) for s in trajectory]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)

    ax1.plot(steps_arr, gini_arr, color="#FF9800", linewidth=1.5)
    ax1.set_ylabel("Gini coefficient")
    ax1.set_title("Innovation Concentration Over Time  [exploratory]")
    ax1.set_ylim(-0.05, 1.05)
    ax1.axhline(y=0.5, color="red", linestyle="--", alpha=0.5,
                label="Gini=0.5 (moderate concentration)")
    ax1.legend(fontsize=9)

    ax2.plot(steps_arr, top10_arr, color="#4CAF50", linewidth=1.5)
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Top-10% share")
    ax2.set_title("Top-10% Innovation Share Over Time")
    ax2.set_ylim(-0.05, 1.05)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(save_path))
    plt.close(fig)
    print(f"Wrote {save_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Metathetic ensemble long-run diagnostics")
    parser.add_argument("--n-agents", type=int, default=10)
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--alpha", type=float, default=5e-3)
    parser.add_argument("--a", type=float, default=3.0)
    parser.add_argument("--mu", type=float, default=0.005)
    parser.add_argument("--variant", type=str, default="logistic",
                        choices=["baseline", "two_scale", "logistic"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"Running metathetic ensemble: {args.n_agents} agents, {args.steps} steps")
    trajectory, summary = run_and_diagnose(
        n_agents=args.n_agents,
        initial_M=10.0,
        alpha=args.alpha,
        a=args.a,
        mu=args.mu,
        steps=args.steps,
        seed=args.seed,
        variant=args.variant,
    )

    write_csv(trajectory, OUT / "longrun_diagnostics.csv")
    write_summary(summary, OUT / "longrun_diagnostics_summary.json")
    fig_heaps_law(trajectory, summary, FIG_OUT / "heaps_law.png")
    fig_concentration(trajectory, FIG_OUT / "concentration_gini.png")

    print(f"\nSummary:")
    print(f"  Heaps beta:       {summary['heaps_beta']:.4f} (R2={summary['heaps_r_squared']:.4f})")
    print(f"  Innovation sigma: {summary['innovation_sigma']:.4f}")
    print(f"  Gini final:       {summary['gini_final']:.4f}")
    print(f"  Top-10% share:    {summary['top10_share_final']:.4f}")
    print(f"  Constraint:       {summary['constraint_tag']} ({summary['constraint_confidence']})")
    print(f"  Self-metatheses:  {summary['n_self_metatheses']}")
    print(f"  Absorptive cross: {summary['n_absorptive_cross']}")
    print(f"  Novel cross:      {summary['n_novel_cross']}")
    print(f"  Env texture:      Type {summary['texture_type_final']}")
    tc = summary.get("temporal_state_counts_final", {})
    state_names = {0: "annihilated", 1: "inertial", 2: "situated", 3: "desituated", 4: "established"}
    tc_str = ", ".join(f"{state_names.get(int(k), '?')}={v}" for k, v in sorted(tc.items()) if v > 0)
    if tc_str:
        print(f"  Temporal:         {tc_str}")
    print(f"\n  [exploratory — see CLAIM_POLICY.md]")


if __name__ == "__main__":
    main()
