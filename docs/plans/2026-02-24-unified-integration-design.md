# Unified sigma-TAP Integration Design

**Date:** 2026-02-24
**Goal:** Integrate three frameworks into the repo's existing architecture:
1. Original TAP equation family (Cortês et al.) — already present as variant system
2. Taalbi's resource-constrained empirical framing — logistic variant + long-run diagnostics
3. Our sigma-TAP extensions — continuous-time solver, turbulence diagnostics, real-world fitting

**Design constraints:**
- Build into repo (`paper-first-model-fixes-20260224` branch), not alongside it
- Float64 with overflow tracking (no mpmath)
- Turbulence as optional post-hoc diagnostics (Interpretive layer)
- Real-world fitting uses repo's variant family, not our ad-hoc s·M^p proxy
- Claim auditor extended to cover new outputs

---

## Architecture: New and Modified Files

```
simulator/
  tap.py              # MODIFY: add innovation_kernel_derivative()
  continuous.py        # NEW: scipy ODE solver for all variants
  turbulence.py        # NEW: post-hoc B/Re_prax diagnostics
  analysis.py          # MODIFY: add long-run scaling diagnostics
  state.py             # MODIFY: add ContinuousResult dataclass
  simulate.py          # UNCHANGED
  sigma_tap.py         # UNCHANGED
  hfuncs.py            # UNCHANGED
  pressure.py          # UNCHANGED
  projection.py        # UNCHANGED
  __init__.py          # MODIFY: export new modules

scripts/
  fit_realworld.py     # NEW: hierarchical fitting to empirical data
  sweep_variants.py    # NEW: unified cross-variant comparison

config/
  realworld_datasets.json  # NEW: dataset definitions

skills/
  claim-to-artifact-auditor/  # CARRY OVER from snapshot + extend

tests/
  test_continuous.py   # NEW: continuous vs discrete agreement
  test_turbulence.py   # NEW: turbulence diagnostic sanity
  test_fitting.py      # NEW: fitting smoke test
```

---

## Module 1: `simulator/continuous.py`

Continuous-time ODE integration for all TAP variants.

### Interface

```python
@dataclass
class ContinuousResult:
    t: np.ndarray
    M: np.ndarray
    Xi: np.ndarray
    sigma: np.ndarray
    f: np.ndarray
    terminated_by_overflow: bool
    blowup_time: float | None

def run_continuous(
    initial_M: float,
    t_span: tuple[float, float],
    params: ModelParams,
    sigma0: float = 1.0,
    gamma: float = 0.0,
    t_eval: np.ndarray | None = None,
    method: str = "RK45",
    max_step: float = 1.0,
    m_cap: float = 1e9,
) -> ContinuousResult:
```

### ODE system

```
dM/dt = sigma(Xi) * f(M; alpha, a, variant, alpha1, K) - mu * M
dXi/dt = beta * sigma(Xi) * f(M; ...) + eta * H(Xi)
```

Where `f` dispatches through `compute_birth_term()` and `sigma` through `sigma_linear()`.

### Overflow handling

scipy event function terminates integration when M >= m_cap. Records blowup_time.

### Key design choice

The continuous solver reuses all existing repo functions (compute_birth_term, sigma_linear) rather than reimplementing. This guarantees variant consistency between discrete and continuous solvers.

---

## Module 2: `simulator/turbulence.py`

Post-hoc diagnostic module. Interpretive layer — clearly labeled, not wired into dynamics.

### Interface

```python
@dataclass
class TurbulenceDiagnostics:
    t: np.ndarray
    B_decision: np.ndarray        # sigma(Xi)*tau / f'(M)
    Re_prax: np.ndarray           # f'(M)*M / (sigma(Xi) + alpha)
    laminar_fraction: float       # fraction of t where B > 1
    transition_time: float | None # first t where B drops below 1

def compute_turbulence_diagnostics(
    result: ContinuousResult,
    params: ModelParams,
    sigma0: float,
    gamma: float,
    tau_decision: float = 1.0,
) -> TurbulenceDiagnostics:
```

### Dependency

Requires `innovation_kernel_derivative()` added to `tap.py`:

```python
def innovation_kernel_derivative(M: float, alpha: float, a: float) -> float:
    """df/dM for the closed-form TAP kernel."""
    k = math.log(1.0 + 1.0 / a)
    exponent = M * k
    if exponent > 700:
        return float("inf")
    return max(0.0, alpha * a * (k * math.exp(exponent) - 1.0 / a))
```

---

## Module 3: Long-run diagnostics in `analysis.py`

Taalbi-inspired empirical scaling diagnostics.

### New functions

```python
def innovation_rate_scaling(
    m_traj: list[float],
    dt: float = 1.0,
) -> dict:
    """Fit dk/dt ~ k^sigma. Returns exponent sigma, R^2, residuals.
    sigma ~ 1 = linear (Taalbi resource-constrained).
    sigma > 1 = super-linear (unconstrained TAP)."""

def constraint_tag(
    m_traj: list[float],
    carrying_capacity: float | None,
    dt: float = 1.0,
) -> str:
    """Tag observed dynamics as adjacency-limited, resource-limited, or mixed.
    Uses Taalbi's framing: compare actual rate to theoretical ceiling."""
```

---

## Module 4: `scripts/fit_realworld.py`

Hierarchical fitting of TAP variants to empirical time-series data.

### Strategy

For each variant (baseline, two_scale, logistic) x each dataset:

**Pass 1 — Core physics (4 params):**
- Grid search over (alpha, a, mu, K) where K only applies to logistic
- Refine best grid point with Nelder-Mead
- Cost: log-space MSE

**Pass 2 — Sigma feedback (2 params):**
- Lock best core params from Pass 1
- Fit (gamma, beta) via grid + Nelder-Mead
- Compare sigma-TAP fit vs gamma=0 baseline

**Pass 3 — Null model comparison:**
- Fit exponential, logistic-growth, and power-law models
- Report AIC/BIC or equivalent for model selection

### Datasets

Stored in `config/realworld_datasets.json`:
- Wikipedia English articles (2001-2024)
- npm registry packages (2010-2025)
- Cumulative described species (1750-2020)

### Outputs

- `outputs/realworld_fit.csv` — per-variant per-dataset results
- `outputs/realworld_fit_summary.json` — best variant per dataset, model comparison

---

## Module 5: `scripts/sweep_variants.py`

Unified cross-variant comparison sweep.

Runs all three variants (baseline, two_scale, logistic) over the same parameter grid and emits a single comparison table. Replaces the need to set TAP_VARIANTS env var.

### Output

`outputs/variant_comparison.csv` with columns:
variant, alpha, mu, a, m0, gamma, regime, blowup_step, final_M, final_Xi

---

## Module 6: Claim auditor extension

Carry the `claim-to-artifact-auditor` skill from the snapshot branch. Extend its check registry to cover:

- Fitting results: best variant per dataset, RMSE bounds
- Variant comparison: regime count consistency
- Continuous vs discrete agreement: max relative error bound

### Skill file

Install as `.claude/skills/claim-to-artifact-auditor/SKILL.md` or keep in repo `skills/` directory per existing convention.

---

## Testing strategy

### test_continuous.py
- Run discrete and continuous solvers with same params for all 3 variants
- Assert M trajectories agree within 5% at integer timesteps (continuous uses dt<<1 so it's more accurate; discrete is the reference for consistency)
- Assert overflow detection triggers at same approximate time

### test_turbulence.py
- Verify B decreases monotonically as M grows (known theoretical property)
- Verify Re_prax increases monotonically as M grows
- Verify laminar_fraction = 1.0 when M stays small

### test_fitting.py
- Smoke test: run fitting on one dataset with reduced grid
- Verify output CSV is well-formed
- Verify logistic variant produces finite (non-overflow) fits

---

## What is NOT included

- mpmath arbitrary-precision engine (float64 + overflow cap is sufficient)
- Our standalone monolithic scripts (capabilities decomposed into modules above)
- Our ad-hoc f(M)=s·M^p taming proxy (replaced by logistic variant)
- Figure-spec-enforcer skill (follow-up)
- Manuscript-report-builder skill (follow-up)
- Colab notebook update (follow-up, after integration proven)

---

## Implementation order

1. `simulator/continuous.py` + `tap.py` derivative function
2. `tests/test_continuous.py` — verify before building on top
3. `simulator/turbulence.py` + `tests/test_turbulence.py`
4. Long-run diagnostics in `analysis.py`
5. `config/realworld_datasets.json` + `scripts/fit_realworld.py` + `tests/test_fitting.py`
6. `scripts/sweep_variants.py`
7. Claim auditor extension
8. Integration smoke test: full pipeline run
