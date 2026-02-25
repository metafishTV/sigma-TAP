"""Generate publication-ready figures for sigma-TAP analysis.

All figures are self-contained matplotlib — no external tools required.
Output: outputs/figures/*.png at 300 DPI.

Usage:
  python scripts/generate_figures.py
  python scripts/generate_figures.py --only trajectory_variants,realworld_fits
"""
from __future__ import annotations

import csv
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from simulator.continuous import run_continuous
from simulator.state import ModelParams
from simulator.turbulence import compute_turbulence_diagnostics
from simulator.analysis import innovation_rate_scaling

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "figures"

# Consistent color scheme.
COLORS = {
    "baseline": "#2196F3",
    "two_scale": "#FF9800",
    "logistic": "#4CAF50",
    "data": "#333333",
    "null": "#999999",
}

VARIANT_LABELS = {
    "baseline": "Baseline TAP",
    "two_scale": "Two-Scale TAP",
    "logistic": "Logistic TAP",
}


def _setup_style():
    """Apply consistent plot style."""
    plt.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "legend.fontsize": 10,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.alpha": 0.3,
    })


def _make_params(variant: str, alpha=1e-3, a=8.0, mu=0.02) -> ModelParams:
    return ModelParams(
        alpha=alpha, a=a, mu=mu,
        beta=0.05, eta=0.02,
        tap_variant=variant,
        alpha1=10 * alpha if variant == "two_scale" else 0.0,
        carrying_capacity=2e5 if variant == "logistic" else None,
    )


# ── Figure 1: Trajectory comparison ──────────────────────────────────

def fig_trajectory_variants(save_path: str | None = None) -> None:
    """M(t) trajectories for all three variants overlaid."""
    _setup_style()
    fig, ax = plt.subplots(figsize=(9, 5))

    styles = {"baseline": "-", "two_scale": "--", "logistic": "-."}

    for variant in ["baseline", "two_scale", "logistic"]:
        # Use growth-producing params: a=3 gives strong combinatorial coupling,
        # alpha=5e-3 ensures birth > death at M=10.
        K = 80.0 if variant == "logistic" else None
        params = ModelParams(
            alpha=5e-3, a=3.0, mu=0.005, beta=0.05, eta=0.02,
            tap_variant=variant,
            alpha1=0.05 if variant == "two_scale" else 0.0,
            carrying_capacity=K,
        )
        result = run_continuous(
            initial_M=10.0, t_span=(0, 20), params=params,
            sigma0=1.0, gamma=0.0, max_step=0.5, m_cap=1e4,
        )
        ax.plot(result.t, result.M, styles[variant],
                color=COLORS[variant], linewidth=2.5,
                label=VARIANT_LABELS[variant])

    ax.set_xlabel("Time")
    ax.set_ylabel("M(t) — realized objects")
    ax.set_title("TAP Variant Trajectories")
    ax.legend()
    ax.set_yscale("log")

    plt.tight_layout()
    path = save_path or str(OUT / "trajectory_variants.png")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    print(f"  Wrote {path}")


# ── Figure 2: Phase diagram ──────────────────────────────────────────

def fig_phase_diagram(save_path: str | None = None) -> None:
    """Regime map in alpha-mu space, one panel per variant."""
    _setup_style()

    csv_path = ROOT / "outputs" / "variant_comparison.csv"
    if not csv_path.exists():
        print(f"  SKIP phase_diagram (need {csv_path})")
        return

    with open(csv_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    regime_colors = {
        "plateau": "#BBDEFB",
        "exponential": "#FFF9C4",
        "precursor-active": "#FFE0B2",
        "explosive": "#FFCDD2",
        "extinction": "#E0E0E0",
    }

    variants = ["baseline", "two_scale", "logistic"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

    for i, variant in enumerate(variants):
        ax = axes[i]
        vrows = [r for r in rows if r["variant"] == variant]
        for r in vrows:
            alpha = float(r["alpha"])
            mu = float(r["mu"])
            regime = r["regime"]
            color = regime_colors.get(regime, "#FFFFFF")
            ax.scatter(
                math.log10(alpha), math.log10(mu),
                c=color, edgecolors=COLORS[variant], s=40, linewidths=0.8,
            )
        ax.set_xlabel("log10(alpha)")
        ax.set_title(VARIANT_LABELS[variant])
        if i == 0:
            ax.set_ylabel("log10(mu)")

    from matplotlib.patches import Patch
    handles = [Patch(facecolor=c, label=r) for r, c in regime_colors.items()]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False,
               bbox_to_anchor=(0.5, -0.05))

    fig.suptitle("Regime Phase Diagram (alpha-mu space)", fontsize=14, y=1.02)
    plt.tight_layout()
    path = save_path or str(OUT / "phase_diagram_alpha_mu.png")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    print(f"  Wrote {path}")


# ── Figure 3: Extinction sensitivity ─────────────────────────────────

def fig_extinction_sensitivity(
    rows: list[dict] | None = None,
    save_path: str | None = None,
) -> None:
    """Transition timing vs mu, one line per variant."""
    _setup_style()

    if rows is None:
        csv_path = ROOT / "outputs" / "extinction_sensitivity.csv"
        if not csv_path.exists():
            print(f"  SKIP extinction_sensitivity (need {csv_path})")
            return
        with open(csv_path, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
            for r in rows:
                for k in ["mu", "final_M"]:
                    r[k] = float(r[k])
                r["transition_step"] = (
                    int(r["transition_step"]) if r["transition_step"] not in ("", "None", None)
                    else None
                )

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

    line_styles = {"baseline": ("o", "-"), "two_scale": ("s", "--"), "logistic": ("^", "-.")}

    for variant in ["baseline", "two_scale", "logistic"]:
        vrows = sorted(
            [r for r in rows if r["variant"] == variant],
            key=lambda r: r["mu"],
        )
        mus = [r["mu"] for r in vrows]
        final_Ms = [r["final_M"] for r in vrows]
        trans_steps = [r["transition_step"] for r in vrows]
        marker, ls = line_styles[variant]

        ax1.plot(mus, final_Ms, marker=marker, linestyle=ls, color=COLORS[variant],
                 label=VARIANT_LABELS[variant], markersize=5, linewidth=1.8)

        valid_mu = [m for m, t in zip(mus, trans_steps) if t is not None]
        valid_ts = [t for t in trans_steps if t is not None]
        if valid_mu:
            ax2.plot(valid_mu, valid_ts, marker=marker, linestyle=ls,
                     color=COLORS[variant],
                     label=VARIANT_LABELS[variant], markersize=5, linewidth=1.8)

    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_ylabel("Final M")
    ax1.set_title("Extinction Sensitivity: Final State")
    ax1.legend()

    ax2.set_xscale("log")
    ax2.set_xlabel("Extinction rate (mu)")
    ax2.set_ylabel("Transition step")
    ax2.set_title("Extinction Sensitivity: Transition Timing")
    if ax2.get_legend_handles_labels()[1]:
        ax2.legend()

    plt.tight_layout()
    path = save_path or str(OUT / "extinction_sensitivity.png")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    print(f"  Wrote {path}")


# ── Figure 4: Adjacency sensitivity ──────────────────────────────────

def fig_adjacency_sensitivity(
    rows: list[dict] | None = None,
    save_path: str | None = None,
) -> None:
    """Final M and blowup step vs a, one line per variant."""
    _setup_style()

    if rows is None:
        csv_path = ROOT / "outputs" / "adjacency_sensitivity.csv"
        if not csv_path.exists():
            print(f"  SKIP adjacency_sensitivity (need {csv_path})")
            return
        with open(csv_path, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
            for r in rows:
                for k in ["a", "final_M"]:
                    r[k] = float(r[k])
                r["blowup_step"] = (
                    int(r["blowup_step"]) if r["blowup_step"] not in ("", "None", None)
                    else None
                )

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

    line_styles = {"baseline": ("o", "-"), "two_scale": ("s", "--"), "logistic": ("^", "-.")}

    for variant in ["baseline", "two_scale", "logistic"]:
        vrows = sorted(
            [r for r in rows if r["variant"] == variant],
            key=lambda r: r["a"],
        )
        a_vals = [r["a"] for r in vrows]
        final_Ms = [r["final_M"] for r in vrows]
        regimes = [r["regime"] for r in vrows]
        marker, ls = line_styles[variant]

        ax1.plot(a_vals, final_Ms, marker=marker, linestyle=ls,
                 color=COLORS[variant],
                 label=VARIANT_LABELS[variant], markersize=5, linewidth=1.8)

        for a_v, fm, reg in zip(a_vals, final_Ms, regimes):
            if reg == "explosive":
                ax1.scatter([a_v], [fm], s=80, facecolors="none",
                           edgecolors="red", linewidths=2, zorder=5)

        blowups = [(r["a"], r["blowup_step"]) for r in vrows
                    if r["blowup_step"] is not None]
        if blowups:
            ax2.plot([b[0] for b in blowups], [b[1] for b in blowups],
                     "s-", color=COLORS[variant],
                     label=VARIANT_LABELS[variant], markersize=5, linewidth=1.5)

    ax1.set_yscale("log")
    ax1.set_ylabel("Final M")
    ax1.set_title("Adjacency Sensitivity: Final State")
    ax1.legend()

    ax2.set_xlabel("Adjacency parameter (a)")
    ax2.set_ylabel("Blowup step")
    ax2.set_title("Adjacency Sensitivity: Blowup Timing")
    if ax2.get_legend_handles_labels()[1]:
        ax2.legend()

    plt.tight_layout()
    path = save_path or str(OUT / "adjacency_sensitivity.png")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    print(f"  Wrote {path}")


# ── Figure 5: Real-world fits ────────────────────────────────────────

def fig_realworld_fits(save_path: str | None = None) -> None:
    """Data + TAP fit + null model curves for all 3 datasets."""
    _setup_style()

    from scripts.fit_realworld import (
        load_datasets, fit_single_variant, fit_null_models, _euler_tap,
    )

    datasets = load_datasets()
    ds_names = list(datasets.keys())
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for i, ds_name in enumerate(ds_names):
        ax = axes[i]
        ds = datasets[ds_name]
        years = ds["years"]
        counts = ds["counts"]
        t = np.array(years, dtype=float)

        ax.scatter(t, counts, color=COLORS["data"], s=30, zorder=5, label="Data")

        # TAP baseline fit.
        result = fit_single_variant(years, counts, variant="baseline", grid_size=5)
        p = result["params"]
        if p:
            t_data = t - t[0]
            pred = _euler_tap(t_data, counts[0], p["s"], p["p"], p["mu"])
            if pred is not None:
                ax.plot(t, pred, color=COLORS["baseline"], linewidth=2,
                        label=f"TAP (RMSE={result['rmse']:.3f})")

        # Null: logistic growth.
        nulls = fit_null_models(years, counts)
        if "logistic_growth" in nulls:
            ng = nulls["logistic_growth"]
            obs = np.array(counts, dtype=float)
            tt = t - t[0]
            try:
                from scipy.optimize import minimize as _min
                def _lc(x):
                    r, K = x[0], 10 ** x[1]
                    if K <= obs[0]:
                        return 1e6
                    pred_l = K / (1 + ((K - obs[0]) / obs[0]) * np.exp(-r * tt))
                    return float(np.mean((np.log10(np.clip(pred_l, 1, None)) - np.log10(np.clip(obs, 1, None))) ** 2))
                res = _min(_lc, [0.1, math.log10(max(counts) * 2)], method="Nelder-Mead")
                K_fit = 10 ** res.x[1]
                pred_logistic = K_fit / (1 + ((K_fit - obs[0]) / obs[0]) * np.exp(-res.x[0] * tt))
                ax.plot(t, pred_logistic, "--", color=COLORS["null"], linewidth=1.5,
                        label=f"Logistic (RMSE={ng['rmse']:.3f})")
            except Exception:
                pass

        ax.set_xlabel("Year")
        if i == 0:
            ax.set_ylabel("Count")
        ax.set_title(ds["description"].split("(")[0].strip())
        ax.legend(fontsize=8)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    fig.suptitle("Real-World Data: TAP vs Null Model Fits", fontsize=14, y=1.02)
    plt.tight_layout()
    path = save_path or str(OUT / "realworld_fits.png")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    print(f"  Wrote {path}")


# ── Figure 6: Turbulence bandwidth ───────────────────────────────────

def fig_turbulence_bandwidth(save_path: str | None = None) -> None:
    """B(t) and Re_prax with laminar/turbulent shading."""
    _setup_style()

    params = _make_params("baseline", alpha=5e-3, a=3.0, mu=0.005)
    result = run_continuous(
        initial_M=10.0, t_span=(0, 25), params=params,
        sigma0=1.0, gamma=0.0, max_step=0.5, m_cap=1e4,
    )
    diag = compute_turbulence_diagnostics(result, params, sigma0=1.0, gamma=0.0)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    B_plot = np.clip(diag.B_decision, 1e-10, 50)
    ax1.plot(diag.t, B_plot, color=COLORS["baseline"], linewidth=1.5)
    ax1.axhline(y=1.0, color="red", linestyle="--", linewidth=1, alpha=0.7,
                label="B=1 (laminar/turbulent boundary)")
    ax1.fill_between(diag.t, 1e-10, B_plot, where=(B_plot > 1),
                     color=COLORS["baseline"], alpha=0.15, label="Laminar (B>1)")
    ax1.fill_between(diag.t, 1e-10, B_plot, where=(B_plot <= 1),
                     color="#F44336", alpha=0.15, label="Turbulent (B<1)")
    ax1.set_ylabel("Decision Bandwidth B(t)")
    ax1.set_yscale("log")
    ax1.set_title("Turbulence Diagnostics (Baseline TAP)")
    ax1.legend(fontsize=9)

    Re_plot = np.clip(diag.Re_prax, 1e-10, 1e12)
    ax2.plot(diag.t, Re_plot, color="#9C27B0", linewidth=1.5)
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Re_prax")
    ax2.set_yscale("log")
    ax2.set_title("Praxiological Reynolds Number")

    if diag.transition_time is not None:
        for ax in (ax1, ax2):
            ax.axvline(x=diag.transition_time, color="red", linestyle=":",
                       alpha=0.5)

    plt.tight_layout()
    path = save_path or str(OUT / "turbulence_bandwidth.png")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    print(f"  Wrote {path}")


# ── Figure 7: Scaling exponents ──────────────────────────────────────

def fig_scaling_exponents(save_path: str | None = None) -> None:
    """Grouped bar chart of scaling exponents across datasets."""
    _setup_style()

    from scripts.fit_realworld import load_datasets, fit_single_variant, _euler_tap

    datasets = load_datasets()
    ds_names = list(datasets.keys())
    short_names = [n.replace("_", " ").title() for n in ds_names]

    variants = ["baseline", "logistic"]
    exponents = {v: [] for v in variants}

    for ds_name in ds_names:
        ds = datasets[ds_name]
        for variant in variants:
            result = fit_single_variant(ds["years"], ds["counts"],
                                        variant=variant, grid_size=5)
            p = result["params"]
            if p:
                t_data = np.array(ds["years"], dtype=float) - ds["years"][0]
                pred = _euler_tap(t_data, ds["counts"][0],
                                  p["s"], p["p"], p["mu"], K=p.get("K"))
                if pred is not None:
                    # Use growth phase only (first 70%) to avoid saturating
                    # tail pulling the exponent negative on S-shaped curves.
                    n_growth = max(3, int(0.7 * len(pred)))
                    scaling = innovation_rate_scaling(list(pred[:n_growth]))
                    exponents[variant].append(scaling["exponent"])
                else:
                    exponents[variant].append(0)
            else:
                exponents[variant].append(0)

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(ds_names))
    width = 0.35

    for j, variant in enumerate(variants):
        offset = (j - 0.5) * width
        ax.bar(x + offset, exponents[variant], width,
               color=COLORS[variant], label=VARIANT_LABELS[variant], alpha=0.85)

    ax.axhline(y=1.0, color="red", linestyle="--", linewidth=1, alpha=0.5,
               label="sigma=1 (linear scaling)")
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Scaling Exponent (sigma)")
    ax.set_title("Innovation Rate Scaling: dk/dt ~ k^sigma")
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, fontsize=10)
    ax.legend()

    plt.tight_layout()
    path = save_path or str(OUT / "scaling_exponents.png")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    print(f"  Wrote {path}")


# ── Figure 8: Variant regime summary ─────────────────────────────────

def fig_variant_regime_summary(save_path: str | None = None) -> None:
    """Stacked bar chart of regime counts by variant."""
    _setup_style()

    csv_path = ROOT / "outputs" / "variant_comparison.csv"
    if not csv_path.exists():
        print(f"  SKIP variant_regime_summary (need {csv_path})")
        return

    with open(csv_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    regimes_order = ["plateau", "exponential", "precursor-active", "explosive", "extinction"]
    regime_colors = ["#BBDEFB", "#FFF9C4", "#FFE0B2", "#FFCDD2", "#E0E0E0"]
    variants = ["baseline", "two_scale", "logistic"]

    counts = {}
    for variant in variants:
        vrows = [r for r in rows if r["variant"] == variant]
        counts[variant] = {}
        for reg in regimes_order:
            counts[variant][reg] = sum(1 for r in vrows if r["regime"] == reg)

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(variants))
    bottoms = np.zeros(len(variants))

    for reg, color in zip(regimes_order, regime_colors):
        vals = [counts[v][reg] for v in variants]
        ax.bar(x, vals, bottom=bottoms, color=color, label=reg, edgecolor="white",
               linewidth=0.5)
        bottoms += vals

    ax.set_xticks(x)
    ax.set_xticklabels([VARIANT_LABELS[v] for v in variants])
    ax.set_ylabel("Number of parameter combinations")
    ax.set_title("Regime Distribution by TAP Variant")
    ax.legend(loc="upper right")

    plt.tight_layout()
    path = save_path or str(OUT / "variant_regime_summary.png")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    print(f"  Wrote {path}")


# ── Main ─────────────────────────────────────────────────────────────

ALL_FIGURES = {
    "trajectory_variants": fig_trajectory_variants,
    "phase_diagram": fig_phase_diagram,
    "extinction_sensitivity": lambda p=None: fig_extinction_sensitivity(save_path=p),
    "adjacency_sensitivity": lambda p=None: fig_adjacency_sensitivity(save_path=p),
    "realworld_fits": fig_realworld_fits,
    "turbulence_bandwidth": fig_turbulence_bandwidth,
    "scaling_exponents": fig_scaling_exponents,
    "variant_regime_summary": fig_variant_regime_summary,
}


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Generate publication figures")
    parser.add_argument("--only", type=str, default=None,
                        help="Comma-separated figure names to generate")
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)

    if args.only:
        names = [n.strip() for n in args.only.split(",")]
    else:
        names = list(ALL_FIGURES.keys())

    print(f"Generating {len(names)} figures...")
    for name in names:
        if name in ALL_FIGURES:
            print(f"\n[{name}]")
            try:
                ALL_FIGURES[name]()
            except Exception as e:
                print(f"  ERROR: {e}")
        else:
            print(f"  Unknown figure: {name}")

    print("\nDone.")


if __name__ == "__main__":
    main()
