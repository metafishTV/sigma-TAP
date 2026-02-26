# TAPS Stage 2 Design: Comparative & Sensitivity Analysis

**CLAIM POLICY LABEL: exploratory**

> All Stage 2 work is exploratory — analytical infrastructure for the TAPS
> framework, which is the project author's original unpublished work. No new
> claims about TAP source literature are introduced.

**Date**: 2026-02-26
**Depends on**: TAPS Stage 1 (complete — 187 tests passing)
**Branch**: unified-integration-20260225

---

## Motivation

Stage 1 revealed three findings that demand investigation:

1. **Gated vs ungated are identical at default params.** The affordance gate
   produces no distinguishable effect. We need to find WHERE in parameter space
   it matters.

2. **6 of 17 modes are highly correlated (|r| > 0.85).** Are these structurally
   linked or artifacts of a narrow parameter regime?

3. **Pressure ratio = 2.15 (net entropy).** Is this universal or
   parameter-dependent?

Additionally, the project author identified the need for structural trajectory
information — pure movement through mode/state space, stripped of quantitative
magnitudes — to complement the parametric sensitivity analysis.

---

## Architecture Decision

**Approach C: New focused module + integration script.**

Create `simulator/taps_sensitivity.py` as a computation module (pure functions,
testable) and `scripts/taps_stage2.py` as the CLI driver. This mirrors Stage 1
architecture (taps.py -> taps_diagnostics.py) and builds on existing sweep
infrastructure without modifying it.

Rationale: keeps existing scripts stable, follows established patterns, is
independently testable.

---

## Work Items

### WI-1a: TAPS Parameter Sensitivity

**Purpose**: Determine which parameters most affect which TAPS modes.

**Module**: `simulator/taps_sensitivity.py`

Core function:

```python
def sweep_taps_modes(
    param_grid: dict,          # {param_name: array_of_values}
    n_agents: int = 20,
    steps: int = 120,
    seed: int = 42,
) -> dict:
    """Run ensemble at each grid point, compute TAPS scores, return summaries.

    Returns dict with:
      'grid': the parameter grid used
      'mode_summaries': {mode_name: ndarray over grid} for mean/std/final
      'sensitivity': {mode_name: {param_name: normalized_range}}
    """
```

**Parameter axes** (building on existing sensitivity_analysis.py ranges):
- `mu` (extinction): 8 values log-spaced, 1e-4 to 5e-1
- `alpha` (growth): 6 values log-spaced, 1e-5 to 1e-2
- `a` (adjacency): [2, 4, 8, 16, 32]

Total grid points: ~240 runs. Each run -> trajectory -> TAPS scores -> summary
statistics (mean, std, final value per mode).

**Sensitivity metric**: Normalized range = (max - min) / max(eps, mean) for each
mode across each parameter axis. Simple, interpretable, no external dependencies.

**Output**: Structured dict + CSV persistence.

### WI-1b: Mode Transition Map

**Purpose**: Track structural movement through mode space, stripped of
quantitative values. Pure categorical state transitions — a directed graph of
pathways.

**Concept**: At each simulation step, classify the system state along several
categorical axes. When a classification changes between consecutive steps,
increment a transition counter. Over a run, this builds a transition matrix
where cell (i,j) = "number of times the system transitioned from state i to
state j."

**Classification axes** (each produces a categorical label per step):

| Axis | States | Source |
|------|--------|--------|
| RIP dominance | recursion, iteration, praxis | `compute_rip()` dominance field |
| Pressure regime | entropy (ratio > 1.2), equilibrium (0.8-1.2), extropy (< 0.8) | `pressure_ratio()` |
| Texture type | type 1-4 | snapshot `texture_type` or dM-variance classification |
| Anopressive dominant | expression, impression, adpression | argmax of normalized triad |
| Syntegration phase | disintegration, preservation, integration, synthesis | argmax of S modes |
| Transvolution direction | involution-dominant, evolution-dominant, balanced | comparison of T modes |

**Implementation**:

```python
def classify_step(taps_scores: dict, step: int) -> dict[str, str]:
    """Classify system state at one step across all categorical axes."""

def build_transition_map(
    taps_scores: dict,
    trajectory: list[dict],
) -> dict[str, np.ndarray]:
    """Build transition count matrices for each classification axis.

    Returns {axis_name: square_ndarray} where entry (i,j) counts transitions
    from state i to state j.
    """

def transition_summary(transition_maps: dict) -> dict:
    """Extract structural features from transition maps.

    Returns:
      absorbing_states: states with high self-transition counts
      common_pathways: most frequent transition pairs
      cycle_signatures: repeated transition sequences
      path_entropy: Shannon entropy of transition distribution per axis
    """
```

**Key deliverable**: Transition maps that can be compared between gated vs
ungated runs — revealing different structural routes even when quantitative
scores appear identical.

**Visualization**: Directed graph (nodes = states, edge width = transition
count) for each classification axis. Small enough to be readable (3-6 nodes
per axis).

### WI-2: Gated vs Ungated Divergence Finder

**Purpose**: Identify parameter regimes where the affordance gate produces
measurably different behavior.

**Approach**: At each grid point from WI-1a, run BOTH gated
(affordance_min_cluster=2) and ungated (affordance_min_cluster=0) with same
seed. Compute per-mode absolute differences.

```python
def divergence_map(
    param_grid: dict,
    seed: int = 42,
    n_agents: int = 20,
    steps: int = 120,
) -> dict:
    """Compute gated-vs-ungated TAPS mode differences across parameter space.

    Returns:
      'mode_divergence': {mode_name: ndarray of |gated - ungated| over grid}
      'transition_divergence': {axis: transition_map_gated vs transition_map_ungated}
      'significant_regimes': list of grid points where divergence > threshold
    """
```

**Divergence metrics**:
- Per-mode: mean |mode_gated[t] - mode_ungated[t]| over time steps
- Per-transition-map: Frobenius norm of (map_gated - map_ungated)
- Threshold for "significant": divergence > 0.1 (tunable)

**Hypothesis**: The gate matters most at intermediate alpha (growth neither
trivially slow nor explosive) and low-to-medium mu (agents survive long enough
for the gate to be tested repeatedly).

### WI-3: Texture Type Validation

**Purpose**: Verify that the dM-variance texture classification (from
taps_mapping.md Section 6) agrees with the environment-derived texture_type
from snapshots.

```python
def classify_dM_texture(
    trajectory: list[dict],
    window: int = 10,
) -> list[int]:
    """Classify texture type from rolling dM variance and mean.

    Returns list of texture type (1-4) per step (first `window` steps = NaN).
    """

def validate_textures(
    trajectory: list[dict],
    window: int = 10,
) -> dict:
    """Compare environment-derived and dM-derived texture classifications.

    Returns:
      env_texture: list[int] from snapshots
      dM_texture: list[int] from rolling variance
      agreement_rate: float
      confusion_matrix: 4x4 ndarray
    """
```

**Key question**: Does the dM variance method produce all 4 texture types, or
does it collapse to 1-2 types at most parameter regimes?

Runs across the same parameter grid as WI-1a to check agreement at each regime.

### WI-4: Correlation Stability Analysis

**Purpose**: Determine whether the 6 correlated mode pairs found in Stage 1 are
structurally linked (stable across parameter space) or parameter-dependent
artifacts.

```python
def correlation_stability(
    sweep_results: dict,    # from sweep_taps_modes
    threshold: float = 0.85,
) -> dict:
    """Track which mode pairs remain correlated across the parameter grid.

    Returns:
      stable_pairs: pairs correlated at >80% of grid points
      unstable_pairs: pairs correlated at <50% of grid points
      stability_map: {pair: fraction_of_grid_points_where_correlated}
      param_dependent: {pair: which_params_break_correlation}
    """
```

**Interpretation**:
- Stable pair (>80% correlated everywhere) -> structurally linked, possibly
  redundant. Consider whether they measure the same underlying phenomenon.
- Unstable pair (<50%) -> Stage 1 finding was an artifact of default params.
  The modes are genuinely independent in most regimes.

### WI-5: Empirical Targets Research Document

**Purpose**: Document real-world datasets that could validate sigma-TAP, even if
we don't collect data at this stage. Establish empirical grounding for future
work.

**Output**: `docs/empirical_targets.md`

**Domains to survey**:

1. **Organizational ecology** — Emery/Trist texture transitions in real
   organizations. Sources: environment survey data, organizational change
   datasets, industry disruption timelines. Mapping: texture types ->
   organizational environment types.

2. **Innovation economics** — Combinatorial innovation rates. Sources: patent
   combination data (Taalbi 2025 reference), technology S-curves, Kauffman's
   NK landscape empirics. Mapping: TAP equation -> patent/innovation counts.

3. **Evolutionary biology** — Speciation/extinction dynamics. Sources: fossil
   record data, phylogenetic branching rates, niche construction evidence
   (Odling-Smee et al. 2003). Mapping: metathesis events -> speciation events,
   deep stasis -> Lazarus taxa.

4. **Chemistry / nuclear physics** — The Bohr periodic table parallel noted by
   the project author. Element synthesis and nuclear cross-sections as
   cross-metathesis analogues. Promethium's isolation as deep stasis. Sources:
   nuclear reaction databases, element abundance data.

5. **Assessment per domain**: data availability, mapping difficulty, priority
   level, specific datasets identified.

Labeled **exploratory** per claim policy. This is research documentation, not
implementation.

---

## Figures to Generate

| Figure | Content | Work Item |
|--------|---------|-----------|
| `taps_sensitivity_heatmap.png` | Parameter x Mode normalized-range matrix | WI-1a |
| `taps_transition_maps.png` | Directed graphs of mode transitions per axis | WI-1b |
| `taps_divergence_map.png` | Gated vs ungated divergence across param space | WI-2 |
| `taps_texture_validation.png` | Confusion matrix + dM variance distributions | WI-3 |
| `taps_correlation_stability.png` | Pair stability fraction across param space | WI-4 |

---

## Testing Strategy

New file: `tests/test_taps_sensitivity.py`

| Test | Validates |
|------|-----------|
| `test_sweep_shape` | sweep_taps_modes with 2x2 grid returns correct array shapes |
| `test_sensitivity_metric` | normalized range computation on known data |
| `test_classify_step` | categorical classification on synthetic scores |
| `test_transition_map_counts` | transition counts match known state sequence |
| `test_transition_self_loops` | constant state -> only self-transitions |
| `test_divergence_identical` | same seed, same params -> zero divergence |
| `test_texture_classification` | synthetic dM series -> correct texture types |
| `test_correlation_stability_all_correlated` | identical trajectories -> 100% stable |

Target: 8-10 new tests.

---

## Available Tooling

**Existing infrastructure** (used directly):
- `simulator/taps.py` — all 17+4 mode computations
- `simulator/sigma_tap.py` + `simulator/metathetic.py` — ensemble runner
- `scripts/sensitivity_analysis.py` — parameter range references
- `scripts/taps_diagnostics.py` — visualization patterns

**Available plugins** (reference as needed):
- **Claude Scientific Skills** (K-Dense) — 147 scientific skills including
  statistical analysis, EDA workflows, scientific visualization. Useful for
  methodology reference if we add advanced sensitivity indices (Sobol, Morris)
  in a later stage.
- **Claude Scientific Writer** (K-Dense) — Publication-ready document
  generation with literature search. Candidate for a future "write the paper"
  phase after Stage 2 results are validated.

---

## Claim Policy Compliance

All outputs labeled **exploratory**. Mandatory disclaimer on any generated
documents. No claims attributed to TAP source literature. The TAPS framework,
mode transition maps, and all analytical infrastructure are the project author's
original unpublished work.

---

## Scope Boundaries

**In scope**: Parameter sensitivity, transition maps, divergence analysis,
texture validation, correlation stability, empirical targets research.

**Out of scope** (deferred to Stage 3+):
- Per-agent topology tracking (agent-agent adjacency matrix) — needed for
  Lizier et al. synchronizability and Mangan & Alon FFL analysis
- Sobol/Morris sensitivity indices — would require SALib dependency; normalized
  range is sufficient for Stage 2
- Trajectory embedding/clustering — UMAP/PCA of TAPS mode vectors; interesting
  but premature before we understand parameter sensitivity
- Automated claim-to-metric auditing — useful but orthogonal to TAPS analysis
