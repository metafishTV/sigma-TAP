"""Run metathetic ensemble with TAPS/RIP diagnostic overlay.

CLAIM POLICY LABEL: exploratory
Computes TAPS mode scores post-hoc from trajectory data and generates
diagnostic figures for mode independence analysis.

Produces:
  outputs/figures/taps_correlation.png  — mode correlation heatmap
  outputs/figures/taps_texture.png      — pressure cascade + RIP over time

Usage:
  python scripts/taps_diagnostics.py
  python scripts/taps_diagnostics.py --compare
  python scripts/taps_diagnostics.py --n-agents 15 --steps 200
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from simulator.metathetic import MetatheticEnsemble
from simulator.taps import (
    compute_all_scores,
    compute_anopression,
    compute_rip,
    correlation_matrix,
    pressure_ratio,
)

ROOT = Path(__file__).resolve().parents[1]
FIG_OUT = ROOT / "outputs" / "figures"


def run_ensemble(
    n_agents: int = 10,
    steps: int = 150,
    alpha: float = 5e-3,
    a: float = 3.0,
    mu: float = 0.005,
    variant: str = "logistic",
    seed: int = 42,
    affordance_min_cluster: int = 2,
    skip_disintegration: bool = False,
) -> tuple[list[dict], MetatheticEnsemble]:
    """Run ensemble and return trajectory + ensemble object."""
    ens = MetatheticEnsemble(
        n_agents=n_agents,
        initial_M=10.0,
        alpha=alpha, a=a, mu=mu,
        variant=variant,
        carrying_capacity=2e5 if variant == "logistic" else None,
        seed=seed,
        affordance_min_cluster=affordance_min_cluster,
    )
    trajectory = ens.run(steps=steps)
    return trajectory, ens


def fig_correlation(scores: dict, save_path: Path) -> None:
    """Correlation heatmap of TAPS mode scores."""
    plt.rcParams.update({
        "font.size": 9, "savefig.dpi": 300, "savefig.bbox": "tight",
    })

    result = correlation_matrix(scores)
    labels = result["labels"]
    matrix = np.array(result["matrix"])
    n = len(labels)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)

    # Annotate cells
    for i in range(n):
        for j in range(n):
            val = matrix[i, j]
            color = "white" if abs(val) > 0.6 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=6, color=color)

    plt.colorbar(im, ax=ax, label="Pearson r")
    ax.set_title("TAPS Mode Correlation Matrix  [exploratory]")

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(save_path))
    plt.close(fig)
    print(f"Wrote {save_path}")


def fig_texture(
    trajectory: list[dict],
    ano_scores: dict,
    rip_result: dict,
    corr_result: dict,
    save_path: Path,
) -> None:
    """Pressure cascade texture map + RIP dominance band."""
    plt.rcParams.update({
        "font.size": 9, "savefig.dpi": 300, "savefig.bbox": "tight",
    })

    steps = [s["step"] for s in trajectory]
    n = len(steps)

    # Build synchronous mode pairs for hatching (|r| > 0.7)
    sync_pairs = set()
    for a, b, r in corr_result.get("highly_correlated", []):
        if abs(r) > 0.7:
            sync_pairs.add((a, b))
            sync_pairs.add((b, a))

    fig, (ax_main, ax_pressure, ax_rip) = plt.subplots(
        3, 1, figsize=(12, 8),
        gridspec_kw={"height_ratios": [6, 1, 1]},
        sharex=True,
    )

    # --- Top panel: Stacked pressure cascade ---
    # Anapressive (warm, bottom): oppression -> suppression -> depression -> compression
    ana_layers = ["oppression", "suppression", "depression", "compression"]
    ana_colors = ["#D32F2F", "#E64A19", "#F57C00", "#FFA726"]
    # Anopressive (cool, top): adpression -> impression -> expression
    ano_layers = ["adpression", "impression", "expression"]
    ano_colors = ["#00897B", "#0288D1", "#1565C0"]

    all_layers = ana_layers + ano_layers
    all_colors = ana_colors + ano_colors

    # Stack arrays
    y_stack = np.zeros(n)
    for layer_name, color in zip(all_layers, all_colors):
        y_values = np.array(ano_scores[layer_name])
        hatch = ""
        # Check if this layer is synchronous with any other
        for other in all_layers:
            if other != layer_name and (layer_name, other) in sync_pairs:
                hatch = "//"
                break
        ax_main.fill_between(steps, y_stack, y_stack + y_values,
                             color=color, alpha=0.8, label=layer_name,
                             hatch=hatch, edgecolor="white", linewidth=0.3)
        y_stack = y_stack + y_values

    ax_main.set_ylabel("Pressure Score (stacked)")
    ax_main.set_title("TAPS Pressure Cascade Over Time  [exploratory]")
    ax_main.legend(loc="upper right", fontsize=7, ncol=2)

    # --- Middle band: Pressure spectrum (mode dominance widths) ---
    # For each step, show the dominant mode as proportional width
    mode_names = all_layers
    mode_colors_map = dict(zip(all_layers, all_colors))
    for t_idx in range(n):
        mode_vals = {m: ano_scores[m][t_idx] for m in mode_names}
        total = sum(mode_vals.values())
        if total == 0:
            continue
        y_bottom = 0.0
        for m in mode_names:
            frac = mode_vals[m] / total
            hatch = ""
            for other in mode_names:
                if other != m and (m, other) in sync_pairs:
                    hatch = "//"
                    break
            ax_pressure.bar(steps[t_idx], frac, bottom=y_bottom, width=1.0,
                            color=mode_colors_map[m], hatch=hatch,
                            edgecolor="none")
            y_bottom += frac
    ax_pressure.set_ylabel("Mode", fontsize=7)
    ax_pressure.set_ylim(0, 1)
    ax_pressure.set_yticks([])

    # --- Bottom band: RIP dominance ---
    rip_colors = {"recursion": "#9E9E9E", "iteration": "#42A5F5", "praxis": "#EF5350"}
    for t_idx in range(n):
        dom = rip_result["dominance"][t_idx]
        ax_rip.bar(steps[t_idx], 1.0, width=1.0,
                   color=rip_colors.get(dom, "#9E9E9E"), edgecolor="none")
    ax_rip.set_ylabel("RIP", fontsize=7)
    ax_rip.set_xlabel("Step")
    ax_rip.set_ylim(0, 1)
    ax_rip.set_yticks([])

    # RIP legend
    from matplotlib.patches import Patch
    rip_patches = [Patch(color=c, label=l) for l, c in rip_colors.items()]
    ax_rip.legend(handles=rip_patches, loc="upper right", fontsize=7, ncol=3)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(save_path))
    plt.close(fig)
    print(f"Wrote {save_path}")


def print_summary(scores: dict, ratios: list[float], rip: dict, corr: dict) -> None:
    """Print TAPS diagnostic summary to console."""
    n = len(ratios)
    last = n - 1

    print("\n  TAPS Diagnostic Summary:")
    print(f"  {'='*50}")

    # Pressure ratio
    pr_final = ratios[last]
    pr_label = "entropy" if pr_final > 1.0 else ("extropy" if pr_final < 1.0 else "equilibrium")
    print(f"  Pressure ratio:   {pr_final:.4f} (net {pr_label})")

    # Mode scores at final step
    print(f"\n  Anopressive (normalized):")
    print(f"    Expression:     {scores['expression'][last]:.4f}")
    print(f"    Impression:     {scores['impression'][last]:.4f}")
    print(f"    Adpression:     {scores['adpression'][last]:.4f}")
    print(f"  Anapressive (raw):")
    print(f"    Oppression:     {scores['oppression'][last]:.4f}")
    print(f"    Suppression:    {scores['suppression'][last]:.4f}")
    print(f"    Depression:     {scores['depression'][last]:.4f}")
    print(f"    Compression:    {scores['compression'][last]:.4f}")

    # Transvolution
    print(f"\n  Transvolution:")
    print(f"    Involution:     {scores['involution'][last]:.4f}")
    print(f"    Evolution:      {scores['evolution'][last]:.4f}")
    print(f"    Condensation:   {scores['condensation'][last]:.4f}")

    # RIP
    rip_counts = {"recursion": 0, "iteration": 0, "praxis": 0}
    for d in rip["dominance"]:
        rip_counts[d] = rip_counts.get(d, 0) + 1
    print(f"\n  RIP dominance counts:")
    for mode, count in rip_counts.items():
        print(f"    {mode:12s}:   {count}/{n} steps ({100*count/n:.1f}%)")

    # Correlation
    print(f"\n  Mode independence:")
    print(f"    Independent modes: {corr['independent_count']}/{len(corr['labels'])}")
    if corr["highly_correlated"]:
        print(f"    Correlated pairs:")
        for a, b, r in corr["highly_correlated"]:
            print(f"      {a} <-> {b}: r={r:.4f}")

    print(f"\n  [exploratory — see CLAIM_POLICY.md]")


def main() -> None:
    parser = argparse.ArgumentParser(description="TAPS/RIP diagnostic overlay")
    parser.add_argument("--n-agents", type=int, default=10)
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--alpha", type=float, default=5e-3)
    parser.add_argument("--a", type=float, default=3.0)
    parser.add_argument("--mu", type=float, default=0.005)
    parser.add_argument("--variant", type=str, default="logistic")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--compare", action="store_true",
                        help="Run gated vs ungated comparison")
    args = parser.parse_args()

    print(f"Running TAPS diagnostics: {args.n_agents} agents, {args.steps} steps")
    trajectory, ens = run_ensemble(
        n_agents=args.n_agents, steps=args.steps,
        alpha=args.alpha, a=args.a, mu=args.mu,
        variant=args.variant, seed=args.seed,
    )

    # Compute all TAPS scores
    scores = compute_all_scores(trajectory, mu=args.mu)
    ano_scores = compute_anopression(trajectory, mu=args.mu)
    ratios = pressure_ratio(ano_scores)
    rip = compute_rip(trajectory)
    corr = correlation_matrix(scores)

    # Print summary
    print_summary(scores, ratios, rip, corr)

    # Generate figures
    fig_correlation(scores, FIG_OUT / "taps_correlation.png")
    fig_texture(trajectory, ano_scores, rip, corr, FIG_OUT / "taps_texture.png")

    # Optional: gated vs ungated comparison
    if args.compare:
        print(f"\n  {'='*50}")
        print(f"  Gated vs Ungated Comparison (same seed={args.seed}):")
        print(f"  {'='*50}")

        traj_ungated, _ = run_ensemble(
            n_agents=args.n_agents, steps=args.steps,
            alpha=args.alpha, a=args.a, mu=args.mu,
            variant=args.variant, seed=args.seed,
            affordance_min_cluster=0,  # disable gate
        )
        scores_ungated = compute_all_scores(traj_ungated, mu=args.mu)
        ano_ungated = compute_anopression(traj_ungated, mu=args.mu)
        ratios_ungated = pressure_ratio(ano_ungated)

        last_g = len(ratios) - 1
        last_u = len(ratios_ungated) - 1
        print(f"\n  {'Metric':<25s} {'Gated':>10s} {'Ungated':>10s}")
        print(f"  {'-'*47}")
        print(f"  {'Pressure ratio':<25s} {ratios[last_g]:>10.4f} {ratios_ungated[last_u]:>10.4f}")
        print(f"  {'Involution':<25s} {scores['involution'][last_g]:>10.4f} {scores_ungated['involution'][last_u]:>10.4f}")
        print(f"  {'Evolution':<25s} {scores['evolution'][last_g]:>10.4f} {scores_ungated['evolution'][last_u]:>10.4f}")
        print(f"  {'Condensation':<25s} {scores['condensation'][last_g]:>10.4f} {scores_ungated['condensation'][last_u]:>10.4f}")
        print(f"  {'Preservation':<25s} {scores['preservation'][last_g]:>10.4f} {scores_ungated['preservation'][last_u]:>10.4f}")

        # Final trajectory metrics
        g_last = trajectory[-1]
        u_last = traj_ungated[-1]
        print(f"\n  {'D_total':<25s} {g_last['D_total']:>10d} {u_last['D_total']:>10d}")
        print(f"  {'k_total':<25s} {g_last['k_total']:>10.1f} {u_last['k_total']:>10.1f}")
        print(f"  {'n_self_metatheses':<25s} {g_last['n_self_metatheses']:>10d} {u_last['n_self_metatheses']:>10d}")
        print(f"  {'n_disinteg_redist':<25s} {g_last['n_disintegration_redistributions']:>10d} {u_last['n_disintegration_redistributions']:>10d}")

    print("\nDone.")


if __name__ == "__main__":
    main()
