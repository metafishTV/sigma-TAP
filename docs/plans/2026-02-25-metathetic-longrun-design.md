# Design: Metathetic Multi-Agent Layer + Long-Run Diagnostics

**Date:** 2026-02-25
**Branch:** `unified-integration-20260225`
**TASKS.md coverage:** T2.1 (Long-run diagnostics module), T2.2 (Constraint decomposition)
**Claim policy label:** `exploratory` (metathetic mechanics), `paper-consistent extension` (Heaps/Gini diagnostics)

---

## 1. Motivation

Taalbi (2025) makes three testable predictions about long-run innovation dynamics:

1. **Innovation rate scales with cumulative innovations**: dk/dt ~ k^sigma
2. **Diversification follows Heaps' law**: D(k) ~ k^beta, beta < 1
3. **Distribution across organizations is NOT winner-take-all** under resource constraints

Our simulator currently handles prediction 1 (`innovation_rate_scaling()`) but lacks the
multi-agent structure needed for predictions 2 and 3. These require agents with individual
portfolios whose diversity and concentration can be measured.

The metathetic multi-agent layer addresses this by introducing agents that can undergo
identity transformations — metathesis — while preserving accumulated knowledge, producing
ensemble trajectories from which Heaps' law, concentration, and diversification metrics
emerge naturally.

## 2. Theoretical grounding

### 2.1 Metathesis concept

Metathesis (from chemistry/linguistics): a transformation where bonds switch or elements
transpose without rupturing the underlying structure. Applied here: agents can change
their combinatorial identity (product-type portfolio) while preserving their accumulated
knowledge k. This models how organizations diversify, merge, or spawn novel entities
without the accumulated innovation history being lost.

### 2.2 Emery & Trist causal texture (Mode 4 basis)

Emery & Trist (1965), "The Causal Texture of Organizational Environments," established
that environments have their own internal causal processes (L22) that affect organizations
at a different timescale than agent-level dynamics. Their four types of causal texture
map onto TAP regime structure:

| Emery & Trist Type | TAP Regime | L22 Dynamics |
|---|---|---|
| Type I: Placid, randomized | Plateau | Environment static, random |
| Type II: Placid, clustered | Exponential | Environment structured but stable |
| Type III: Disturbed-reactive | Precursor-active | Inter-agent competition dominates |
| Type IV: Turbulent fields | Explosive | Environment itself in flux |

This provides theoretical grounding for Mode 4 (environmental metathesis): the containing
structure is not passive but has its own (very slow) dynamics that constrain and are
affected by agent behavior.

### 2.3 Connection to TAP/biocosmology papers

- TAP equation paper (Cortes et al.): variant family, regime transitions, blow-up dynamics
- Biocosmology paper (Paper2): Type III non-ergodic systems require mixed explanatory modes
- Taalbi (2025): resource-constrained recombinant search, Heaps' law, innovation scaling

The metathetic layer is an *exploratory extension* that aims to bridge aggregate TAP
dynamics with the multi-agent empirical structure Taalbi examines. It is NOT claimed
as a finding of the source TAP literature.

## 3. Architecture

### 3.1 New modules

```
simulator/
  metathetic.py     # MetatheticAgent, MetatheticEnsemble, EnvironmentState
  longrun.py        # heaps_law_fit, gini_coefficient, top_k_share,
                    #   diversification_rate, enhanced_constraint_tag
scripts/
  longrun_diagnostics.py   # CLI: run ensemble, emit CSV + summary JSON + figures
tests/
  test_metathetic.py       # ~8 tests
  test_longrun.py          # existing + ~5 new tests
```

### 3.2 Data flow

```
Parameters + N_agents
       |
       v
MetatheticEnsemble.run(steps)
       |
       |-- each step:
       |     1. Each active agent runs local TAP step (compute_birth_term)
       |     2. Check self-metathesis conditions
       |     3. Check pairwise cross-metathesis conditions
       |     4. Update EnvironmentState (slow drift)
       |     5. Record ensemble snapshot
       |
       v
EnsembleTrajectory (per-step: D_total, k_total, agent_k_list, convergence, env_state)
       |
       v
longrun.py diagnostics
       |
       +-- heaps_law_fit(D_series, k_series) -> {beta, r_squared}
       +-- gini_coefficient(agent_k_list) -> float
       +-- top_k_share(agent_k_list, k=0.1) -> float
       +-- diversification_rate(D_series, k_series) -> array
       +-- enhanced_constraint_tag(sigma, beta, gini) -> {tag, confidence}
       |
       v
outputs/longrun_diagnostics.csv
outputs/longrun_diagnostics_summary.json
outputs/figures/heaps_law.png
outputs/figures/concentration_gini.png
```

## 4. Metathetic mechanics — four modes

### 4.1 Mode 1: Self-metathesis

An agent transforms its own type-identity based on internal innovation pressure.

**Trigger:** Agent's local innovation rate dM/dt exceeds a self-metathesis threshold
proportional to its current portfolio size |type_set|.

**Effect:** Agent gains a new type drawn from the adjacent type-space (type_counter + 1).
Accumulated k is preserved. type_set grows.

**Analog:** Company pivoting into new product line. Single-object mutation in biology.
Maps to TAP paper's i=1 (single-object evolution) channel.

### 4.2 Mode 2: Absorptive cross-metathesis

Two agents whose convergence exceeds their distinctiveness merge into one composite.

**Trigger:** For agents i, j compute:
- Likeness L_ij = Jaccard(type_set_i, type_set_j)
- Goal G_ij = correlation(dM/dt history_i, dM/dt history_j)
- Agent weight W_i = 1 - (|type_set_i intersect global_common_types| / |type_set_i|)
  (distinctiveness = fraction of types not shared widely)

Cross-metathesis eligible when: L_ij + G_ij > W_i + W_j
Absorptive when: L_ij > G_ij (more alike than aligned)

**Effect:** Agent with lower M_local is absorbed into the other. Composite gets union of
type_sets, sum of k values. Absorbed agent goes dormant (state preserved).

**Analog:** Corporate merger/acquisition. Symbiogenesis in biology.

### 4.3 Mode 3: Novel cross-metathesis

Two converging agents produce a fundamentally new third entity.

**Trigger:** Same eligibility as Mode 2, but fires when: G_ij > L_ij (more aligned in
trajectory than alike in identity). Both agents are different enough that their
combination produces novelty rather than consolidation.

**Effect:** Both parent agents go dormant. A new agent spawns with:
- type_set = recombination of parent type_sets (new types generated from the
  combinatorial product of parent types, not just union)
- k = sum of parent k values
- M_local = sum of parent M_local values (energetically additive)

**Analog:** Cross-species speciation. Chemical cross-metathesis producing novel compounds.
This is the TAP i>=2 combinatorial channel operating at the agent level.

### 4.4 Mode 4: Environmental/systemic metathesis

The containing structure has its own slow dynamics that constrain agent metathesis.

**State:** EnvironmentState holds:
- a_env: global adjacency parameter (how rich the combinatorial space is)
- K_env: global carrying capacity (resource ceiling)
- texture_type: current Emery-Trist causal texture (I through IV)
- innovation_potential: scalar field summarizing how much room for novelty remains

**Dynamics:** Updated every env_update_interval steps (default: 10x agent timescale):
- a_env drifts based on total agent diversity: more diverse ensemble -> richer
  adjacency (lower a_env -> stronger combinatorial coupling)
- K_env drifts based on aggregate innovation: more total k -> higher carrying capacity
  (innovation creates resources)
- texture_type transitions based on regime classification of aggregate trajectory
- innovation_potential = K_env - sum(agent.M_local) (remaining room)

**Constraints on agents:**
- Self-metathesis only possible when innovation_potential > 0
- Cross-metathesis threshold scaled by texture_type
  (Type I: high threshold = hard to cross-metathesize in placid environment;
   Type IV: low threshold = turbulent environment facilitates radical transformation)
- Novel cross-metathesis can only produce types up to current a_env limit

**Analog:** Physical constants as emergent from deeper dynamics. The biosphere creating
niches into which life expands (Kauffman). Emery & Trist L22 processes.

## 5. Long-run diagnostics (simulator/longrun.py)

### 5.1 Heaps' law fit

```python
def heaps_law_fit(D_series, k_series) -> dict:
    """Fit D(k) ~ k^beta via log-log OLS.
    Returns: {beta, intercept, r_squared, n_points}
    """
```

Taalbi prediction: beta < 1 (sublinear diversification).

### 5.2 Gini coefficient

```python
def gini_coefficient(values: list[float]) -> float:
    """Standard Gini on agent k-values. 0 = perfect equality, 1 = perfect inequality."""
```

Taalbi prediction: under resource constraints, Gini should NOT approach 1
(no winner-take-all).

### 5.3 Top-k share

```python
def top_k_share(values: list[float], k_frac: float = 0.1) -> float:
    """Fraction of total held by top k_frac of agents."""
```

Complementary concentration metric.

### 5.4 Diversification rate

```python
def diversification_rate(D_series, k_series) -> list[float]:
    """dD/dk at each step — rate of new type discovery per unit innovation."""
```

Should decline over time under Heaps' law (beta < 1).

### 5.5 Enhanced constraint tag

```python
def enhanced_constraint_tag(
    sigma: float,        # innovation rate scaling exponent
    beta: float,         # Heaps exponent
    gini: float,         # concentration
    carrying_capacity: float | None,
    m_final: float,
) -> dict:
    """Returns {tag: str, confidence: str, reasoning: str}
    tag in {adjacency-limited, resource-limited, mixed}
    confidence in {high, medium, low}
    """
```

Decision heuristics (from Taalbi):
- sigma > 1.3 AND beta < 0.7 AND gini < 0.5 -> adjacency-limited, high confidence
- sigma ~ 1.0 AND gini < 0.5 -> resource-limited, high confidence
- Otherwise -> mixed, medium confidence
- Insufficient data points -> any tag, low confidence

## 6. Outputs

### 6.1 CSV: outputs/longrun_diagnostics.csv

Per-timestep ensemble snapshot:
```
step, D_total, k_total, n_active_agents, n_dormant_agents,
gini, top10_share, convergence_measure, texture_type,
a_env, K_env, innovation_potential
```

### 6.2 JSON: outputs/longrun_diagnostics_summary.json

```json
{
  "heaps_beta": 0.72,
  "heaps_r_squared": 0.94,
  "innovation_sigma": 1.05,
  "gini_final": 0.38,
  "top10_share_final": 0.31,
  "constraint_tag": "resource-limited",
  "constraint_confidence": "high",
  "n_agents_final": 15,
  "n_self_metatheses": 42,
  "n_absorptive_cross": 3,
  "n_novel_cross": 1,
  "n_env_transitions": 2,
  "texture_type_final": "Type III",
  "claim_policy_label": "exploratory",
  "disclaimer": "This result is exploratory and does not derive from the source TAP/biocosmology literature."
}
```

### 6.3 Figures

- `outputs/figures/heaps_law.png` — D(k) log-log with fitted beta line
- `outputs/figures/concentration_gini.png` — Gini + top-10% share over time

## 7. Updated claims

### C3 upgrade (partial -> supported)

Add Heaps' law diagnostics and concentration metrics to artifact support.
Now has: innovation_rate_scaling + heaps_law_fit + gini_coefficient.

### New C8 (exploratory)

"Metathetic multi-agent TAP dynamics produce Heaps' law (beta < 1) and
non-winner-take-all concentration, consistent with Taalbi (2025) predictions
under resource-constrained recombinant search."

Status: exploratory. Requires disclaimer.

## 8. Testing strategy

### test_metathetic.py (~8 tests)

1. Agent creation with correct initial state
2. Self-metathesis preserves k, expands type_set
3. Absorptive cross-metathesis: composite has union of types, sum of k
4. Novel cross-metathesis: new agent has recombined types, parents go dormant
5. Dormant agent state is preserved (k, type_set unchanged)
6. Environment state drifts correctly over time
7. Texture type transitions match regime classification
8. Full ensemble runs without error for 100 steps with N=10 agents

### test_longrun.py additions (~5 tests)

1. heaps_law_fit on known power-law data returns correct beta
2. gini_coefficient returns 0 for equal distribution, ~1 for monopoly
3. top_k_share correct for known distribution
4. diversification_rate decreasing for Heaps beta < 1
5. enhanced_constraint_tag returns valid tag + confidence

## 9. Implementation order

1. simulator/longrun.py (pure diagnostic functions — no dependencies on metathetic)
2. simulator/metathetic.py (core agent model)
3. tests/test_metathetic.py + tests/test_longrun.py additions
4. scripts/longrun_diagnostics.py (CLI runner + figures)
5. Update CLAIMS.md
6. Wire into run_reporting_pipeline.py
7. Final test suite verification

## 10. References

- Taalbi, J. (2025). "Long-run patterns in the discovery of the adjacent possible."
  Industrial and Corporate Change, 35(1), 123-149.
- Cortes, M., Kauffman, S.A., Liddle, A.R. & Smolin, L. (2022/2025). "The TAP equation."
  TAPequation-FINAL.pdf
- Cortes, M., Kauffman, S.A., Liddle, A.R. & Smolin, L. (2022). "Biocosmology: Biology
  from a cosmological perspective." Paper2-FINAL.pdf
- Emery, F.E. & Trist, E.L. (1965). "The Causal Texture of Organizational Environments."
  Human Relations, 18, 21-32.
- Kauffman, S.A. (2000). Reinventing the Sacred. Basic Books.
