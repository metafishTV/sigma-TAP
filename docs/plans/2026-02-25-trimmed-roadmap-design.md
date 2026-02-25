# Trimmed Roadmap Design: Sensitivity Analysis + Publication Figures

**Date:** 2025-02-25
**Branch:** `unified-integration-20260225`
**Goal:** Complete the science (sensitivity analyses) and make it visible (publication figures), with minimal governance overhead.

## Deliverables

| # | Deliverable | Produces | Depends on |
|---|-------------|----------|------------|
| 1 | CLAIMS.md | Markdown claims table at repo root | Nothing |
| 2 | scripts/sensitivity_analysis.py | CSVs: extinction sweep, a-parameter sweep | Existing simulator |
| 3 | scripts/generate_figures.py | ~8 publication-ready PNGs in outputs/figures/ | #2 + existing outputs |
| 4 | Pipeline integration | Wire new modules into run_reporting_pipeline.py | #2, #3 |
| 5 | Tests | tests/test_sensitivity.py, tests/test_figures.py | #2, #3 |

## Design Decisions

- **Figures**: Self-contained matplotlib. No external skill dependency. Anyone cloning the repo can produce all plots.
- **Claims**: Standalone CLAIMS.md at repo root. Simple markdown table. No build automation/gates.
- **Sensitivity**: Single unified script sweeping both mu (extinction) and a (adjacency).
- **Dropped**: Claim auditor, report template enforcement, evidence ladder automation, variant interface refactor.

## 1. CLAIMS.md

Simple markdown table with columns: Claim ID, Claim, Source, Artifact, Status.
Populated from paper_iteration_report.json (5 requirements) plus key findings from real-world fitting.
Status values: `supported`, `partial`, `exploratory`. No build enforcement.

## 2. Sensitivity Analysis Script

**File:** `scripts/sensitivity_analysis.py`

Two independent sweeps using the discrete simulator (fast, reliable):

**Extinction sweep (mu):**
- Fix alpha=1e-3, a=8.0, M0=10
- Sweep mu over logspace(1e-4, 5e-1, 30)
- For each mu: run baseline + two_scale + logistic variants
- Record: final_M, regime, transition_step (first step where regime changes from plateau)
- Output: `outputs/extinction_sensitivity.csv`

**Adjacency sweep (a):**
- Fix alpha=1e-3, mu=0.01, M0=10
- Sweep a over [2, 3, 4, 6, 8, 12, 16, 24, 32]
- For each a: run baseline + two_scale + logistic variants
- Record: final_M, regime, transition_step, blowup_step
- Output: `outputs/adjacency_sensitivity.csv`

Both use `run_sigma_tap` with 120 steps. Fast enough to complete in <10 seconds total.

## 3. Publication Figures

**File:** `scripts/generate_figures.py`
**Output dir:** `outputs/figures/`
**Style:** Clean, publication-ready. Consistent color scheme across all figures. Savefig at 300 DPI.

### Figure catalog

| Figure | Data source | Layout |
|--------|-----------|--------|
| `trajectory_variants.png` | run_continuous (3 variants) | 1x3 subplots, M(t) curves, shared y-axis |
| `phase_diagram_alpha_mu.png` | sweep_variants.py CSV | Scatter plot colored by regime in log(alpha)-log(mu) space, one panel per variant |
| `extinction_sensitivity.png` | extinction_sensitivity.csv | Line plot: transition timing vs mu, one line per variant |
| `adjacency_sensitivity.png` | adjacency_sensitivity.csv | Line plot: final M and blowup step vs a, one line per variant |
| `realworld_fits.png` | fit_realworld.py rerun | 1x3 subplots (one per dataset): data points + TAP fit + null model curves |
| `turbulence_bandwidth.png` | run_continuous + turbulence.py | 2-row plot: B(t) top, Re_prax bottom, with laminar/turbulent shading |
| `scaling_exponents.png` | Computed from fit trajectories | Grouped bar chart: scaling exponent per dataset per model |
| `variant_regime_summary.png` | sweep_variants.py CSV | Stacked bar chart: regime counts by variant |

### Color scheme
- Baseline: `#2196F3` (blue)
- Two-scale: `#FF9800` (orange)
- Logistic: `#4CAF50` (green)
- Data points: `#333333` (dark gray)
- Null models: dashed gray lines

## 4. Pipeline Integration

Add two new stages to `run_reporting_pipeline.py`:
- Stage: run sensitivity_analysis.py (produces CSVs)
- Stage: run generate_figures.py (produces PNGs)
- Update expected figure list to include new outputs

## 5. Tests

**tests/test_sensitivity.py:**
- Test extinction sweep produces CSV with expected columns
- Test adjacency sweep produces CSV with expected columns
- Test regime labels are valid strings
- Test that extreme mu=0.5 produces extinction regime

**tests/test_figures.py:**
- Test generate_figures.py runs without error on minimal data
- Test output PNGs exist after run
- (Not pixel-testing — just existence + non-zero file size)
