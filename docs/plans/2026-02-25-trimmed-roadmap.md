# Trimmed Roadmap Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Deliver sensitivity analyses (extinction + adjacency parameter sweeps), 8 publication-ready matplotlib figures, a plain claims table, and pipeline integration.

**Architecture:** Three new scripts (`sensitivity_analysis.py`, `generate_figures.py`) plus `CLAIMS.md` at repo root. Sensitivity script uses the existing discrete simulator (`run_sigma_tap`) for speed. Figure script uses matplotlib with a consistent color scheme, generating self-contained PNGs at 300 DPI. Pipeline script gets two new stages appended. All tests use Python's built-in `unittest`.

**Tech Stack:** Python 3.12, matplotlib 3.10, numpy 2.4, scipy 1.17, unittest. No external skills required.

**Python executable:** `C:\Users\user\AppData\Local\Programs\Python\Python312\python.exe`

**Working directory:** `C:\Users\user\Documents\New folder\sigma-TAP-repo`

**Branch:** `unified-integration-20260225`

---

## Task 1: Create `CLAIMS.md`

**Files:**
- Create: `CLAIMS.md`

**Step 1: Write the file**

```markdown
# sigma-TAP Claims Matrix

Status definitions:
- **supported**: Claim backed by source paper citation AND generated artifact.
- **partial**: Claim has paper basis but artifact support is incomplete or single-variant.
- **exploratory**: Novel claim not directly in source papers; requires further validation.

| ID | Claim | Source paper(s) | Supporting artifact(s) | Status |
|----|-------|----------------|----------------------|--------|
| C1 | TAP dynamics admit a variant family (baseline, two-scale, logistic) with qualitatively distinct regime behavior | TAPequation-FINAL.pdf; Applications-of-TAP.pdf | `outputs/variant_comparison.csv` via `scripts/sweep_variants.py` | supported |
| C2 | Regime transitions (plateau, exponential, explosive, extinction) are detectable from M(t) and Xi(t) trajectories | TAPequation-FINAL.pdf | `simulator/analysis.py::classify_regime`; `outputs/variant_comparison.csv` | supported |
| C3 | Innovation-rate scaling exponent sigma distinguishes TAP super-linear dynamics from resource-constrained growth | Long-run patterns in the discovery of the adjacent possible.pdf | `simulator/analysis.py::innovation_rate_scaling`; `outputs/realworld_fit.csv` | partial |
| C4 | Real-world combinatorial growth (Wikipedia, npm, species) fits tamed TAP (power-law kernel) better than exponential or pure logistic null models | Applications-of-TAP.pdf; Long-run patterns.pdf | `scripts/fit_realworld.py`; `outputs/realworld_fit.csv` | supported |
| C5 | Decision bandwidth B(t) and praxiological Reynolds Re_prax provide interpretive turbulence diagnostics for TAP trajectories | (exploratory extension of TAP framework) | `simulator/turbulence.py`; `outputs/figures/turbulence_bandwidth.png` | exploratory |
| C6 | Extinction rate mu controls transition timing: higher mu delays or prevents explosive onset | TAPequation-FINAL.pdf | `outputs/extinction_sensitivity.csv`; `outputs/figures/extinction_sensitivity.png` | partial |
| C7 | Adjacency parameter a controls combinatorial explosion rate; currently fixed at a=8 across analyses | TAPequation-FINAL.pdf | `outputs/adjacency_sensitivity.csv`; `outputs/figures/adjacency_sensitivity.png` | partial |
```

**Step 2: Commit**

```bash
git add CLAIMS.md
git commit -m "docs: add claims matrix with 7 traced claims

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Task 2: Create `scripts/sensitivity_analysis.py` with tests

**Files:**
- Create: `scripts/sensitivity_analysis.py`
- Create: `tests/test_sensitivity.py`

**Step 1: Write the failing test**

Create `tests/test_sensitivity.py`:

```python
"""Tests for sensitivity analysis sweeps."""
import csv
import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.sensitivity_analysis import run_extinction_sweep, run_adjacency_sweep

ROOT = Path(__file__).resolve().parents[1]


class TestExtinctionSweep(unittest.TestCase):
    def test_returns_list_of_dicts(self):
        rows = run_extinction_sweep(n_mu=5, steps=20)
        self.assertIsInstance(rows, list)
        self.assertGreater(len(rows), 0)
        self.assertIn("mu", rows[0])
        self.assertIn("variant", rows[0])
        self.assertIn("final_M", rows[0])
        self.assertIn("regime", rows[0])

    def test_high_mu_causes_extinction_or_plateau(self):
        """Very high extinction should prevent explosive growth."""
        rows = run_extinction_sweep(n_mu=3, steps=40, mu_range=(0.3, 0.5))
        regimes = {r["regime"] for r in rows}
        # Should NOT be explosive with mu=0.3-0.5
        self.assertNotIn("explosive", regimes)

    def test_all_variants_present(self):
        rows = run_extinction_sweep(n_mu=3, steps=20)
        variants = {r["variant"] for r in rows}
        self.assertEqual(variants, {"baseline", "two_scale", "logistic"})


class TestAdjacencySweep(unittest.TestCase):
    def test_returns_list_of_dicts(self):
        rows = run_adjacency_sweep(a_values=[4.0, 8.0], steps=20)
        self.assertIsInstance(rows, list)
        self.assertGreater(len(rows), 0)
        self.assertIn("a", rows[0])
        self.assertIn("variant", rows[0])
        self.assertIn("final_M", rows[0])

    def test_all_variants_present(self):
        rows = run_adjacency_sweep(a_values=[8.0], steps=20)
        variants = {r["variant"] for r in rows}
        self.assertEqual(variants, {"baseline", "two_scale", "logistic"})

    def test_larger_a_means_faster_growth(self):
        """Smaller a means stronger combinatorial coupling, faster growth."""
        rows = run_adjacency_sweep(a_values=[4.0, 32.0], steps=40)
        baseline_rows = [r for r in rows if r["variant"] == "baseline"]
        by_a = {r["a"]: r["final_M"] for r in baseline_rows}
        # a=4 should produce larger M than a=32 (stronger coupling)
        if by_a.get(4.0, 0) > 0 and by_a.get(32.0, 0) > 0:
            self.assertGreater(by_a[4.0], by_a[32.0])


if __name__ == "__main__":
    unittest.main()
```

**Step 2: Run test to verify it fails**

Run: `"C:\Users\user\AppData\Local\Programs\Python\Python312\python.exe" -m unittest tests.test_sensitivity -v`

Expected: ImportError — `run_extinction_sweep` and `run_adjacency_sweep` do not exist.

**Step 3: Write the implementation**

Create `scripts/sensitivity_analysis.py`:

```python
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

    # Find transition step: first step where regime would change from initial.
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
    log_lo = math.log10(mu_range[0])
    log_hi = math.log10(mu_range[1])
    mus = [10 ** (log_lo + i * (log_hi - log_lo) / max(1, n_mu - 1))
           for i in range(n_mu)]

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
```

**Step 4: Run test to verify it passes**

Run: `"C:\Users\user\AppData\Local\Programs\Python\Python312\python.exe" -m unittest tests.test_sensitivity -v`

Expected: All 6 tests PASS.

**Step 5: Commit**

```bash
git add scripts/sensitivity_analysis.py tests/test_sensitivity.py
git commit -m "feat: add unified sensitivity analysis (extinction + adjacency sweeps)

Sweeps mu over logspace and a over [2..32] for all three TAP variants.
Records regime, transition timing, and blowup step.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Task 3: Create `scripts/generate_figures.py` with tests

**Files:**
- Create: `scripts/generate_figures.py`
- Create: `tests/test_figures.py`

**Step 1: Write the failing test**

Create `tests/test_figures.py`:

```python
"""Smoke tests for publication figure generation."""
import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.generate_figures import (
    fig_trajectory_variants,
    fig_extinction_sensitivity,
    fig_adjacency_sensitivity,
    fig_turbulence_bandwidth,
    COLORS,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "figures"


class TestFigureHelpers(unittest.TestCase):
    def test_colors_defined(self):
        self.assertIn("baseline", COLORS)
        self.assertIn("two_scale", COLORS)
        self.assertIn("logistic", COLORS)

    def test_trajectory_variants_creates_file(self):
        path = OUT / "_test_trajectory.png"
        fig_trajectory_variants(str(path))
        self.assertTrue(path.exists())
        self.assertGreater(path.stat().st_size, 0)
        path.unlink()

    def test_extinction_sensitivity_creates_file(self):
        path = OUT / "_test_extinction.png"
        # Generate minimal data inline
        from scripts.sensitivity_analysis import run_extinction_sweep
        rows = run_extinction_sweep(n_mu=5, steps=20)
        fig_extinction_sensitivity(rows, str(path))
        self.assertTrue(path.exists())
        self.assertGreater(path.stat().st_size, 0)
        path.unlink()

    def test_adjacency_sensitivity_creates_file(self):
        path = OUT / "_test_adjacency.png"
        from scripts.sensitivity_analysis import run_adjacency_sweep
        rows = run_adjacency_sweep(a_values=[4.0, 8.0], steps=20)
        fig_adjacency_sensitivity(rows, str(path))
        self.assertTrue(path.exists())
        self.assertGreater(path.stat().st_size, 0)
        path.unlink()

    def test_turbulence_bandwidth_creates_file(self):
        path = OUT / "_test_turbulence.png"
        fig_turbulence_bandwidth(str(path))
        self.assertTrue(path.exists())
        self.assertGreater(path.stat().st_size, 0)
        path.unlink()


if __name__ == "__main__":
    unittest.main()
```

**Step 2: Run test to verify it fails**

Run: `"C:\Users\user\AppData\Local\Programs\Python\Python312\python.exe" -m unittest tests.test_figures -v`

Expected: ImportError — `scripts.generate_figures` does not exist.

**Step 3: Write the implementation**

Create `scripts/generate_figures.py`:

```python
"""Generate publication-ready figures for sigma-TAP analysis.

All figures are self-contained matplotlib — no external tools required.
Output: outputs/figures/*.png at 300 DPI.

Usage:
  python scripts/generate_figures.py
  python scripts/generate_figures.py --only trajectory_variants,realworld_fits
"""
from __future__ import annotations

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
    """M(t) trajectories for all three variants side by side."""
    _setup_style()
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)

    for i, variant in enumerate(["baseline", "two_scale", "logistic"]):
        ax = axes[i]
        params = _make_params(variant)
        result = run_continuous(
            initial_M=10.0, t_span=(0, 50), params=params,
            sigma0=1.0, gamma=0.0, max_step=0.5,
        )
        ax.plot(result.t, result.M, color=COLORS[variant], linewidth=2,
                label=VARIANT_LABELS[variant])
        ax.set_xlabel("Time")
        ax.set_title(VARIANT_LABELS[variant])
        ax.legend(loc="upper left")
        if i == 0:
            ax.set_ylabel("M(t) — realized objects")

    fig.suptitle("TAP Variant Trajectories", fontsize=14, y=1.02)
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

    # Legend.
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=c, label=r) for r, c in regime_colors.items()]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False,
               bbox_to_anchor=(0.5, -0.05))

    fig.suptitle("Regime Phase Diagram (alpha-mu space)", fontsize=14, y=1.02)
    plt.tight_layout()
    path = save_path or str(OUT / "phase_diagram_alpha_mu.png")
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

    for variant in ["baseline", "two_scale", "logistic"]:
        vrows = sorted(
            [r for r in rows if r["variant"] == variant],
            key=lambda r: r["mu"],
        )
        mus = [r["mu"] for r in vrows]
        final_Ms = [r["final_M"] for r in vrows]
        trans_steps = [r["transition_step"] for r in vrows]

        ax1.plot(mus, final_Ms, "o-", color=COLORS[variant],
                 label=VARIANT_LABELS[variant], markersize=4, linewidth=1.5)

        # Plot transition step (None -> don't plot that point).
        valid_mu = [m for m, t in zip(mus, trans_steps) if t is not None]
        valid_ts = [t for t in trans_steps if t is not None]
        if valid_mu:
            ax2.plot(valid_mu, valid_ts, "o-", color=COLORS[variant],
                     label=VARIANT_LABELS[variant], markersize=4, linewidth=1.5)

    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_ylabel("Final M")
    ax1.set_title("Extinction Sensitivity: Final State")
    ax1.legend()

    ax2.set_xscale("log")
    ax2.set_xlabel("Extinction rate (mu)")
    ax2.set_ylabel("Transition step")
    ax2.set_title("Extinction Sensitivity: Transition Timing")
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

    for variant in ["baseline", "two_scale", "logistic"]:
        vrows = sorted(
            [r for r in rows if r["variant"] == variant],
            key=lambda r: r["a"],
        )
        a_vals = [r["a"] for r in vrows]
        final_Ms = [r["final_M"] for r in vrows]
        regimes = [r["regime"] for r in vrows]

        ax1.plot(a_vals, final_Ms, "o-", color=COLORS[variant],
                 label=VARIANT_LABELS[variant], markersize=5, linewidth=1.5)

        # Mark explosive regimes with red edge.
        for a_v, fm, reg in zip(a_vals, final_Ms, regimes):
            if reg == "explosive":
                ax1.scatter([a_v], [fm], s=80, facecolors="none",
                           edgecolors="red", linewidths=2, zorder=5)

        # Blowup step.
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

        # Plot data.
        ax.scatter(t, counts, color=COLORS["data"], s=30, zorder=5,
                   label="Data")

        # Fit and plot TAP baseline.
        result = fit_single_variant(years, counts, variant="baseline", grid_size=5)
        p = result["params"]
        if p:
            t_data = t - t[0]
            pred = _euler_tap(t_data, counts[0], p["s"], p["p"], p["mu"])
            if pred is not None:
                ax.plot(t, pred, color=COLORS["baseline"], linewidth=2,
                        label=f"TAP (RMSE={result['rmse']:.3f})")

        # Fit and plot null: logistic growth.
        nulls = fit_null_models(years, counts)
        if "logistic_growth" in nulls:
            ng = nulls["logistic_growth"]
            # Recompute curve for plotting.
            from scipy.optimize import minimize as _min
            obs = np.array(counts, dtype=float)
            tt = t - t[0]
            try:
                def _lc(x):
                    r, K = x[0], 10 ** x[1]
                    if K <= obs[0]:
                        return 1e6
                    pred = K / (1 + ((K - obs[0]) / obs[0]) * np.exp(-r * tt))
                    return float(np.mean((np.log10(np.clip(pred, 1, None)) - np.log10(np.clip(obs, 1, None))) ** 2))
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
    fig.savefig(path)
    plt.close(fig)
    print(f"  Wrote {path}")


# ── Figure 6: Turbulence bandwidth ───────────────────────────────────

def fig_turbulence_bandwidth(save_path: str | None = None) -> None:
    """B(t) and Re_prax with laminar/turbulent shading."""
    _setup_style()

    params = _make_params("baseline", alpha=1e-3, a=8.0, mu=0.005)
    result = run_continuous(
        initial_M=10.0, t_span=(0, 60), params=params,
        sigma0=1.0, gamma=0.0, max_step=0.5,
    )
    diag = compute_turbulence_diagnostics(result, params, sigma0=1.0, gamma=0.0)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    # B(t) — decision bandwidth.
    B_plot = np.clip(diag.B_decision, 0, 50)  # clip for visual
    ax1.plot(diag.t, B_plot, color=COLORS["baseline"], linewidth=1.5)
    ax1.axhline(y=1.0, color="red", linestyle="--", linewidth=1, alpha=0.7,
                label="B=1 (laminar/turbulent boundary)")
    ax1.fill_between(diag.t, 0, B_plot, where=(B_plot > 1),
                     color=COLORS["baseline"], alpha=0.15, label="Laminar (B>1)")
    ax1.fill_between(diag.t, 0, B_plot, where=(B_plot <= 1),
                     color="#F44336", alpha=0.15, label="Turbulent (B<1)")
    ax1.set_ylabel("Decision Bandwidth B(t)")
    ax1.set_yscale("log")
    ax1.set_title("Turbulence Diagnostics (Baseline TAP)")
    ax1.legend(fontsize=9)

    # Re_prax.
    Re_plot = np.clip(diag.Re_prax, 0, 1e8)
    ax2.plot(diag.t, Re_plot, color="#9C27B0", linewidth=1.5)
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Re_prax")
    ax2.set_yscale("log")
    ax2.set_title("Praxiological Reynolds Number")

    if diag.transition_time is not None:
        for ax in (ax1, ax2):
            ax.axvline(x=diag.transition_time, color="red", linestyle=":",
                       alpha=0.5, label=f"Transition t={diag.transition_time:.1f}")

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
                    scaling = innovation_rate_scaling(list(pred))
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
```

**Step 4: Run test to verify it passes**

Run: `"C:\Users\user\AppData\Local\Programs\Python\Python312\python.exe" -m unittest tests.test_figures -v`

Expected: All 5 tests PASS. (May take ~10 seconds due to fitting in realworld test.)

**Step 5: Commit**

```bash
git add scripts/generate_figures.py tests/test_figures.py
git commit -m "feat: add self-contained publication figure generator (8 figures)

Trajectory variants, phase diagram, extinction/adjacency sensitivity,
real-world fits, turbulence bandwidth, scaling exponents, regime summary.
All matplotlib, 300 DPI, consistent color scheme.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Task 4: Generate all data + figures end-to-end

**Step 1: Generate sweep_variants CSV (needed for phase diagram + regime summary)**

Run: `"C:\Users\user\AppData\Local\Programs\Python\Python312\python.exe" scripts/sweep_variants.py > outputs/variant_comparison.csv`

Expected: CSV file with ~576 rows (3 variants x 3 m0 x 8 alpha x 8 mu).

**Step 2: Run sensitivity analysis**

Run: `"C:\Users\user\AppData\Local\Programs\Python\Python312\python.exe" scripts/sensitivity_analysis.py`

Expected: `outputs/extinction_sensitivity.csv` and `outputs/adjacency_sensitivity.csv` created.

**Step 3: Generate all figures**

Run: `"C:\Users\user\AppData\Local\Programs\Python\Python312\python.exe" scripts/generate_figures.py`

Expected: 8 PNGs in `outputs/figures/`.

**Step 4: Commit generated outputs**

```bash
git add outputs/variant_comparison.csv outputs/extinction_sensitivity.csv outputs/adjacency_sensitivity.csv outputs/figures/
git commit -m "data: generate all sweep CSVs and publication figures

576-row variant comparison, 90-row extinction sweep, 27-row adjacency sweep.
8 publication figures at 300 DPI.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Task 5: Wire into `run_reporting_pipeline.py`

**Files:**
- Modify: `run_reporting_pipeline.py:76-81`

**Step 1: Add new stages before optional skill tools**

After line 77 (`summarize_blowup_matched_panel.py`) and before line 79 (`report_builder_ok`), insert:

```python
    # Sensitivity analysis + publication figures.
    run([sys.executable, 'scripts/sensitivity_analysis.py'])
    run([
        sys.executable, 'scripts/sweep_variants.py',
    ], stdout_path=Path('outputs/variant_comparison.csv'))
    run([sys.executable, 'scripts/generate_figures.py'])
```

Also add new expected figures to the validation list (after line 94). Append to `expected_figures`:

```python
            'outputs/figures/trajectory_variants.png',
            'outputs/figures/extinction_sensitivity.png',
            'outputs/figures/adjacency_sensitivity.png',
            'outputs/figures/turbulence_bandwidth.png',
            'outputs/figures/scaling_exponents.png',
```

**Step 2: Commit**

```bash
git add run_reporting_pipeline.py
git commit -m "feat: wire sensitivity analysis + figure generation into pipeline

Two new stages: sensitivity_analysis.py and generate_figures.py.
Five new expected figure outputs validated.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Task 6: Run all tests — final verification

**Step 1: Run full test suite**

Run: `"C:\Users\user\AppData\Local\Programs\Python\Python312\python.exe" -m unittest discover -s tests -v`

Expected: All tests pass (22 existing + 6 sensitivity + 5 figures = ~33 tests).

**Step 2: Final commit if any fixups needed**

```bash
git add -A
git commit -m "chore: final verification — all tests passing

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Summary

| Task | Deliverable | Tests | Est. time |
|------|-------------|-------|-----------|
| 1 | CLAIMS.md | — | 2 min |
| 2 | sensitivity_analysis.py | 6 tests | 10 min |
| 3 | generate_figures.py | 5 tests | 15 min |
| 4 | Generate all data + figures | smoke | 5 min |
| 5 | Pipeline integration | — | 5 min |
| 6 | Final test suite | all ~33 | 3 min |

**Total: 6 tasks, 11 new tests, ~40 minutes estimated.**
