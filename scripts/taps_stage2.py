"""TAPS Stage 2: comparative and sensitivity analysis.

CLAIM POLICY LABEL: exploratory
Runs parameter sensitivity sweep, mode transition maps, gated vs ungated
divergence analysis, texture validation, and correlation stability.

Produces:
  outputs/figures/taps_sensitivity_heatmap.png
  outputs/figures/taps_transition_maps.png
  outputs/figures/taps_divergence_map.png
  outputs/figures/taps_texture_validation.png
  outputs/figures/taps_correlation_stability.png

Usage:
  python scripts/taps_stage2.py
  python scripts/taps_stage2.py --quick
  python scripts/taps_stage2.py --steps 200
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from simulator.taps import (
    compute_all_scores,
    compute_anopression,
    compute_rip,
    pressure_ratio,
)
from simulator.taps_sensitivity import (
    sweep_taps_modes,
    compute_divergence,
    correlation_stability,
    validate_textures,
    build_transition_map,
    transition_summary,
)
from simulator.metathetic import MetatheticEnsemble

ROOT = Path(__file__).resolve().parents[1]
FIG_OUT = ROOT / "outputs" / "figures"

# Shared rcParams for all figures
_RC = {"font.size": 9, "savefig.dpi": 300, "savefig.bbox": "tight"}


# ---------------------------------------------------------------------------
# Figure 1: Sensitivity heatmap
# ---------------------------------------------------------------------------

def fig_sensitivity_heatmap(sweep_result: dict, save_path: Path) -> None:
    """Mode x Parameter heatmap of normalized sensitivity ranges."""
    plt.rcParams.update(_RC)

    sensitivity = sweep_result.get("sensitivity", {})
    if not sensitivity:
        print("  (no sensitivity data — skipping heatmap)")
        return

    # Build rows (modes) and columns (params)
    mode_names = sorted(sensitivity.keys())
    if not mode_names:
        print("  (no modes in sensitivity — skipping heatmap)")
        return

    param_names = sorted(sensitivity[mode_names[0]].keys())
    if not param_names:
        print("  (no swept parameters — skipping heatmap)")
        return

    # Build matrix
    matrix = np.zeros((len(mode_names), len(param_names)))
    for i, mode in enumerate(mode_names):
        for j, param in enumerate(param_names):
            matrix[i, j] = sensitivity[mode].get(param, 0.0)

    fig, ax = plt.subplots(figsize=(max(6, len(param_names) * 1.5), max(6, len(mode_names) * 0.4)))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")

    ax.set_xticks(range(len(param_names)))
    ax.set_yticks(range(len(mode_names)))
    ax.set_xticklabels(param_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(mode_names, fontsize=7)

    # Annotate cells
    for i in range(len(mode_names)):
        for j in range(len(param_names)):
            val = matrix[i, j]
            color = "white" if val > (matrix.max() * 0.6) else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=6, color=color)

    plt.colorbar(im, ax=ax, label="Normalized sensitivity range")
    ax.set_title("TAPS Sensitivity: Mode x Parameter  [exploratory]")
    ax.set_xlabel("Swept parameter")
    ax.set_ylabel("Mode")

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(save_path))
    plt.close(fig)
    print(f"  Wrote {save_path}")


# ---------------------------------------------------------------------------
# Figure 2: Transition maps (2x3 subplot grid)
# ---------------------------------------------------------------------------

def fig_transition_maps(sweep_result: dict, save_path: Path) -> None:
    """2x3 subplot grid of transition count matrices for 6 classification axes."""
    plt.rcParams.update(_RC)

    transition_maps_list = sweep_result.get("transition_maps", [])
    if not transition_maps_list:
        print("  (no transition maps — skipping)")
        return

    # Use the first grid point's transition map
    tmap = transition_maps_list[0]
    if not tmap:
        print("  (empty transition map — skipping)")
        return

    axes_names = sorted(tmap.keys())
    n_axes = len(axes_names)
    if n_axes == 0:
        print("  (no axes in transition map — skipping)")
        return

    # Pad to 6 if fewer
    n_rows, n_cols = 2, 3
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(14, 8))
    axs_flat = axs.flatten()

    for idx in range(n_rows * n_cols):
        ax = axs_flat[idx]
        if idx >= n_axes:
            ax.set_visible(False)
            continue

        axis_name = axes_names[idx]
        data = tmap[axis_name]
        states = data["states"]
        counts = data["counts"]
        n = len(states)

        im = ax.imshow(counts, cmap="Blues", aspect="auto")
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(states, rotation=45, ha="right", fontsize=6)
        ax.set_yticklabels(states, fontsize=6)
        ax.set_title(axis_name, fontsize=8)

        # Annotate non-zero cells
        for i in range(n):
            for j in range(n):
                val = counts[i, j]
                if val > 0:
                    color = "white" if val > counts.max() * 0.6 else "black"
                    ax.text(j, i, str(val), ha="center", va="center",
                            fontsize=6, color=color)

    fig.suptitle("TAPS Mode Transition Maps (first grid point)  [exploratory]",
                 fontsize=10)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(save_path))
    plt.close(fig)
    print(f"  Wrote {save_path}")


# ---------------------------------------------------------------------------
# Figure 3: Divergence map (gated vs ungated)
# ---------------------------------------------------------------------------

def fig_divergence_map(div_result: dict, save_path: Path) -> None:
    """Line plot of top 10 modes by max divergence across grid points."""
    plt.rcParams.update(_RC)

    mode_divergence = div_result.get("mode_divergence", {})
    if not mode_divergence:
        print("  (no divergence data — skipping)")
        return

    # Rank modes by max divergence
    mode_max = {m: max(vals) for m, vals in mode_divergence.items() if vals}
    if not mode_max:
        print("  (no divergence values — skipping)")
        return

    top_modes = sorted(mode_max.keys(), key=lambda m: mode_max[m], reverse=True)[:10]

    fig, ax = plt.subplots(figsize=(10, 6))

    for mode in top_modes:
        vals = mode_divergence[mode]
        ax.plot(range(len(vals)), vals, label=mode, linewidth=1.0, marker=".", markersize=3)

    # Threshold line
    ax.axhline(y=0.1, color="red", linestyle="--", linewidth=1.0, label="threshold=0.1")

    ax.set_xlabel("Grid point index")
    ax.set_ylabel("Mean |gated - ungated|")
    ax.set_title("TAPS Gated vs Ungated Divergence (top 10 modes)  [exploratory]")
    ax.legend(fontsize=6, ncol=2, loc="upper right")

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(save_path))
    plt.close(fig)
    print(f"  Wrote {save_path}")


# ---------------------------------------------------------------------------
# Figure 4: Texture validation
# ---------------------------------------------------------------------------

def fig_texture_validation(trajectory: list[dict], tex_result: dict, save_path: Path) -> None:
    """Two panels: confusion matrix and dM trajectory with texture bands."""
    plt.rcParams.update(_RC)

    confusion = tex_result.get("confusion_matrix")
    dM_texture = tex_result.get("dM_texture", [])

    if confusion is None or len(dM_texture) == 0:
        print("  (no texture validation data — skipping)")
        return

    fig, (ax_cm, ax_traj) = plt.subplots(1, 2, figsize=(14, 5))

    # --- Left panel: 5x5 confusion matrix ---
    labels = ["unclass.", "placid-rand", "placid-clust", "disturb-react", "turbulent"]
    im = ax_cm.imshow(confusion, cmap="Blues", aspect="auto")
    n = confusion.shape[0]
    ax_cm.set_xticks(range(n))
    ax_cm.set_yticks(range(n))
    ax_cm.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax_cm.set_yticklabels(labels, fontsize=7)
    ax_cm.set_xlabel("dM-derived texture")
    ax_cm.set_ylabel("Environment texture")
    ax_cm.set_title("Texture Confusion Matrix")

    # Annotate counts
    for i in range(n):
        for j in range(n):
            val = confusion[i, j]
            if val > 0:
                color = "white" if val > confusion.max() * 0.6 else "black"
                ax_cm.text(j, i, str(val), ha="center", va="center",
                           fontsize=7, color=color)

    plt.colorbar(im, ax=ax_cm, label="Count")

    # --- Right panel: dM trajectory with background color by texture ---
    steps = [s["step"] for s in trajectory]
    n_steps = len(steps)

    # Compute dM series
    total_M = [s["total_M"] for s in trajectory]
    dM = [0.0] + [total_M[i] - total_M[i - 1] for i in range(1, n_steps)]

    ax_traj.plot(steps, dM, color="black", linewidth=0.8, label="dM")

    # Background color by texture type
    texture_colors = {0: "#EEEEEE", 1: "#C8E6C9", 2: "#FFF9C4",
                      3: "#FFCCBC", 4: "#EF9A9A"}
    texture_labels_map = {0: "unclassified", 1: "placid-rand", 2: "placid-clust",
                          3: "disturbed", 4: "turbulent"}

    for t_idx in range(n_steps):
        tex_type = dM_texture[t_idx] if t_idx < len(dM_texture) else 0
        ax_traj.axvspan(
            steps[t_idx] - 0.5,
            steps[t_idx] + 0.5,
            color=texture_colors.get(tex_type, "#EEEEEE"),
            alpha=0.4,
        )

    ax_traj.set_xlabel("Step")
    ax_traj.set_ylabel("dM")
    ax_traj.set_title("dM Trajectory + dM-derived Texture Type")

    # Legend for texture colors
    from matplotlib.patches import Patch
    tex_patches = [Patch(color=texture_colors[k], label=texture_labels_map[k])
                   for k in sorted(texture_colors.keys())]
    ax_traj.legend(handles=tex_patches, fontsize=6, loc="upper right", ncol=2)

    fig.suptitle("TAPS Texture Validation  [exploratory]", fontsize=10)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(save_path))
    plt.close(fig)
    print(f"  Wrote {save_path}")


# ---------------------------------------------------------------------------
# Figure 5: Correlation stability
# ---------------------------------------------------------------------------

def fig_correlation_stability(corr_result: dict, save_path: Path) -> None:
    """Bar chart of pair stability fractions with threshold lines."""
    plt.rcParams.update(_RC)

    stability_map = corr_result.get("stability_map", {})
    if not stability_map:
        print("  (no stability data — skipping)")
        return

    # Sort pairs by stability fraction descending
    pairs_sorted = sorted(stability_map.items(), key=lambda x: x[1], reverse=True)
    labels = [f"{a}-{b}" for (a, b), _ in pairs_sorted]
    values = [v for _, v in pairs_sorted]

    # Color by threshold
    colors = []
    for v in values:
        if v > 0.8:
            colors.append("#4CAF50")  # green
        elif v < 0.5:
            colors.append("#F44336")  # red
        else:
            colors.append("#FFC107")  # yellow

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.3), 6))
    x = np.arange(len(labels))
    ax.bar(x, values, color=colors, edgecolor="none", width=0.8)

    # Threshold lines
    ax.axhline(y=0.8, color="green", linestyle="--", linewidth=1.0, label="stable (>0.8)")
    ax.axhline(y=0.5, color="red", linestyle="--", linewidth=1.0, label="unstable (<0.5)")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90, ha="center", fontsize=5)
    ax.set_ylabel("Stability fraction")
    ax.set_xlabel("Mode pair")
    ax.set_ylim(0, 1.05)
    ax.set_title("TAPS Correlation Stability Across Parameter Grid  [exploratory]")
    ax.legend(fontsize=7, loc="upper right")

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(save_path))
    plt.close(fig)
    print(f"  Wrote {save_path}")


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------

def print_stage2_summary(sweep: dict, div: dict, corr_stab: dict) -> None:
    """Print Stage 2 diagnostic summary to console."""
    print(f"\n  TAPS Stage 2 Summary:")
    print(f"  {'='*55}")

    # --- Top 5 most sensitive modes ---
    sensitivity = sweep.get("sensitivity", {})
    if sensitivity:
        # Compute total sensitivity per mode
        mode_total = {}
        for mode, params in sensitivity.items():
            mode_total[mode] = sum(params.values())
        top5 = sorted(mode_total.items(), key=lambda x: x[1], reverse=True)[:5]
        print(f"\n  Top 5 most sensitive modes:")
        for mode, total in top5:
            print(f"    {mode:<25s}  total_range={total:.3f}")
    else:
        print(f"\n  (no sensitivity data)")

    # --- Divergence summary ---
    sig = div.get("significant_regimes", [])
    n_grid = len(div.get("grid", []))
    print(f"\n  Divergence summary:")
    print(f"    Grid points:              {n_grid}")
    print(f"    Significant regimes:      {len(sig)} (max_div > 0.1)")

    # --- Transition summary ---
    tmaps = sweep.get("transition_maps", [])
    if tmaps and tmaps[0]:
        t_sum = transition_summary(tmaps[0])
        absorbing = t_sum.get("absorbing_states", {})
        path_ent = t_sum.get("path_entropy", {})
        print(f"\n  Transition summary (first grid point):")
        for axis in sorted(absorbing.keys()):
            abs_list = absorbing[axis]
            ent = path_ent.get(axis, 0.0)
            abs_str = ", ".join(abs_list) if abs_list else "none"
            print(f"    {axis:<22s}  absorbing=[{abs_str}]  H={ent:.2f} bits")

    # --- Correlation stability ---
    stable = corr_stab.get("stable_pairs", [])
    unstable = corr_stab.get("unstable_pairs", [])
    stability_map = corr_stab.get("stability_map", {})
    print(f"\n  Correlation stability:")
    print(f"    Pairs tracked:            {len(stability_map)}")
    print(f"    Stable (>0.8):            {len(stable)}")
    print(f"    Unstable (<0.5):          {len(unstable)}")

    print(f"\n  [exploratory — see CLAIM_POLICY.md]")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="TAPS Stage 2: comparative and sensitivity analysis"
    )
    parser.add_argument("--quick", action="store_true",
                        help="Use reduced grid (~18 points)")
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--n-agents", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Build parameter grid
    if args.quick:
        param_grid = {
            "mu": [1e-3, 1e-2, 1e-1],
            "alpha": [1e-4, 1e-3, 5e-3],
            "a": [4.0, 8.0],
        }
        label = "quick"
    else:
        param_grid = {
            "mu": [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1],
            "alpha": [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3],
            "a": [2.0, 4.0, 8.0, 16.0, 32.0],
        }
        label = "full"

    n_points = len(param_grid["mu"]) * len(param_grid["alpha"]) * len(param_grid["a"])
    print(f"TAPS Stage 2 ({label}): {n_points} grid points, "
          f"{args.steps} steps, {args.n_agents} agents, seed={args.seed}")

    # ---------------------------------------------------------------
    # Phase 1: Parameter sensitivity sweep
    # ---------------------------------------------------------------
    print(f"\n[1/4] Running parameter sensitivity sweep...")
    sweep = sweep_taps_modes(
        param_grid=param_grid,
        n_agents=args.n_agents,
        steps=args.steps,
        seed=args.seed,
    )
    fig_sensitivity_heatmap(sweep, FIG_OUT / "taps_sensitivity_heatmap.png")
    fig_transition_maps(sweep, FIG_OUT / "taps_transition_maps.png")

    # ---------------------------------------------------------------
    # Phase 2: Gated vs ungated divergence
    # ---------------------------------------------------------------
    print(f"\n[2/4] Running gated vs ungated divergence...")
    div = compute_divergence(
        param_grid=param_grid,
        n_agents=args.n_agents,
        steps=args.steps,
        seed=args.seed,
    )
    fig_divergence_map(div, FIG_OUT / "taps_divergence_map.png")

    # ---------------------------------------------------------------
    # Phase 3: Texture validation
    # ---------------------------------------------------------------
    print(f"\n[3/4] Running texture validation...")
    ens = MetatheticEnsemble(
        n_agents=args.n_agents,
        initial_M=10.0,
        alpha=5e-3,
        a=3.0,
        mu=0.005,
        variant="logistic",
        carrying_capacity=2e5,
        seed=args.seed,
    )
    trajectory = ens.run(steps=args.steps)
    tex_result = validate_textures(trajectory)
    fig_texture_validation(trajectory, tex_result, FIG_OUT / "taps_texture_validation.png")

    # ---------------------------------------------------------------
    # Phase 4: Correlation stability
    # ---------------------------------------------------------------
    print(f"\n[4/4] Running correlation stability...")
    corr_stab = correlation_stability(sweep)
    fig_correlation_stability(corr_stab, FIG_OUT / "taps_correlation_stability.png")

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------
    print_stage2_summary(sweep, div, corr_stab)

    print("\nDone.")


if __name__ == "__main__":
    main()
