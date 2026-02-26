# TAPS Stage 1: Mapping + Computation + Correlation Analysis — Design

**Date:** 2026-02-26
**Claim policy label:** exploratory
**Approach:** B (mapping + computation + correlation + comparison + near-stasis fix)

---

## Theoretical Framework

TAPS (Transanoprasyn) and RIP (Recursively Iterative Praxis/Preservation) are a
dynamic dispositional model developed independently by the project author. TAPS
provides structural dynamics; RIP provides functional dynamics. Together with TAP
(Theory of the Adjacent Possible, Kauffman/Cortes et al.), they form a layered
diagnostic framework:

- **TAP mechanics** tell you WHAT happened (innovation events, growth, decay)
- **TAPS disposition** tells you HOW it happened (which pressure modes dominate)
- **RIP rhythm** tells you WHETHER the system is maintaining, developing, or acting

### TAPS Modes

**T — Transvolution** (involution + evolution = condensation):
- Involution: inward folding, interiorization, condension
- Evolution: outward unfolding, exteriorization, expansion
- Condensation: the coupled product of both

**A — Anopression/Anapression** (pressure spectrum, 3/4 asymmetry):
- ANO(+) anopressive, upward/integrative, extropic:
  - Expression: outward manifestation (always = 1 as reference)
  - Impression: inward reception
  - Adpression: sudden causal release (punctuated events)
- ANA(-) anapressive, downward/disintegrative, entropic:
  - Oppression (base): fundamental constraint of being metathesized by reality
  - Suppression: preventing emergence
  - Depression: gradual reduction from above
  - Compression (apex): refined inward constraint

Cascade ordering: expression (ano apex) -> impression -> adpression ->
[fulcrum] -> compression (ana apex) -> depression -> suppression ->
oppression (base). Both poles return to metathesis; together = syntropy.

Syntropy is primary; entropy and extropy are derivative decompositions.

**P — Praxis** (projection + reflection -> consumptive consummation):
- Projection: feedforward exploration of the adjacent possible
- Reflection: feedback self-assessment of readiness
- Action: the actual doing (consumption + consummation)
  - Consumption: disintegration serving integration (reflective cycle)
  - Consummation: integration serving disintegration/transformation (projective cycle)
  - Pure action: the intersection of consumption and consummation (rare locally)
  - Action balance: consummation / (consumption + consummation); empirical target ≈ 0.60

The four elements (projection, reflection, consumption, consummation) form a
metathetic pair-of-pairs that can swap bonding configurations:
  (projection · consummation) + (reflection · consumption)
      ⇄  (projection · reflection) + (consumption · consummation)

Consumption and consummation are always synthetic (P→S direct conjunction).
Projection and reflection are conditionally syntegrative (preservative,
integrative, or disintegrative, but not necessarily synthetic).

Extended from Aristotle's classical praxis (theory/practice/action) per
the project author's framework. Praxis is universal, not exclusively human.
Consumption/consummation weights calibrated against cross-domain empirical
convergence (Youn et al. 2015, Odum 1969, Bauchop & Elsden 1960).
See docs/empirical_targets.md §6b.

**S — Syntegration** (synthetic disintegration):
- Disintegration: coming apart, redistribution
- Preservation: dormancy, maintenance
- Integration: absorptive incorporation
- Synthesis: creation of new structure

### RIP Modes

Dual-P variant:
- **RIP(praxis)**: process-oriented, non-ergodic, entity-like
- **RIP(preservation)**: progress-oriented, ergodic, system-like

Three levels:
- Recursion: bare cycle, one tick, primarily ergodic
- Iteration: recursion whose conditions change non-ergodically (self or forced)
- Praxis/Preservation: agentic action or structural maintenance

### Dialectical Foundation

Extension of Hegelian dialectics (project author's original work):
- Thesis: thing-in-itself
- Athesis: adjacent thing (every other thesis)
- Synthesis: combinatorial product of thesis-athesis interaction
- Metathesis: ongoing process that synthesizes syntheses (the ground level)

Synchronic metathesis: continuous flow of praxis (turns on/off, never stops).
Diachronic metathesis: serial flow of syntegration (starts and stops, creates
and destroys identities).

### Emery & Trist Bridge

Per Emery, F. (1977), *Futures We Are In*:
- L11 (interdependencies within system) = athetic relations
- L12 (system's actions into environment) = thetic relations
- L21 (goals/noxiants from environment) = synthetic relations
- L22 (causal texture of environment) = metathetic relations

### Key Literature Connections

- Niche construction: Odling-Smee, Laland & Feldman (2003)
- Minimum viable population: Frankham, Bradshaw & Brook (2014)
- Knowledge spillovers: Marshall (1890), Krugman (1991)
- Kauffman's enablement: Longo, Montevil, Kauffman (2012)
- Bohm's implicate/explicate order: Bohm, D. (1980), *Wholeness and the
  Implicate Order* — holomovement correlates to syntropy (the combined
  syntropic process), not to the implicate order alone
- Emery & Trist causal textures: Emery & Trist (1965); Emery, F. (1977)
- TAPS, RIP, metathesis extension from Hegel: project author's unpublished
  framework

---

## Architecture

### New Files

- `simulator/taps.py` — pure-function module, post-hoc computation of TAPS/RIP
  scores from trajectory data. No coupling to simulation engine.
- `scripts/taps_diagnostics.py` — CLI script: run ensemble, compute TAPS overlay,
  print results, generate figures, optionally compare gated vs ungated.
- `tests/test_taps.py` — unit tests (~12 tests).
- `docs/taps_mapping.md` — complete mode-to-observable mapping with formulas.

### Modified Files

- `simulator/metathetic.py` — Jaccard near-stasis fix only.
- `tests/test_metathetic.py` — update disintegration tests for near-stasis.

### Not Touched

`run_reporting_pipeline.py`, `longrun_diagnostics.py`, existing scripts.
TAPS diagnostics is standalone.

### Dependency

One-way: `taps.py` reads trajectory dicts (list of snapshots). Nothing else
imports `taps.py`.

---

## Computational Formulas

All functions take trajectory (list of snapshot dicts), return per-step arrays.
Event deltas computed between consecutive snapshots (counters are cumulative).

### T — Transvolution

```
involution[t] = (delta_self + delta_absorptive) / max(1, total_events)
evolution[t]  = (delta_novel + delta_disintegrations + delta_env) / max(1, total_events)
condensation[t] = involution[t] * evolution[t]
```

Involution includes self-metathesis AND absorptive cross: both alter an agent's
interior (self-restructuring + incorporation of external types).

### A — Anopression/Anapression

Anopressive (normalized to sum = 1.0):
```
expression[t] = agents_with_dM_gt_0 / n_active
impression[t] = delta_absorptive / max(1, total_events)
adpression[t] = (delta_self + delta_disintegrations) / max(1, total_events)
-> normalize: each /= (expression + impression + adpression)
```

Anapressive (NOT normalized, can exceed 1.0):
```
oppression[t]   = 1.0 - (mean_dM / max_observed_dM)   [base: always positive]
suppression[t]  = 1.0 - affordance_mean                [gate blocking fraction]
depression[t]   = mu * mean_M / max(epsilon, mean_dM)  [decay/growth ratio]
compression[t]  = mean_M / K  if K exists, else 0      [capacity pressure]
```

```
pressure_ratio[t] = sum(anapressive) / 1.0
  > 1.0 => net entropy
  < 1.0 => net extropy
```

### P — Praxis

```
projection[t] = innovation_potential from snapshot
reflection[t] = affordance_mean from snapshot
action[t]     = total new metathesis events at step t
```

### S — Syntegration

```
disintegration[t] = delta_disintegrations
preservation[t]   = n_dormant / (n_active + n_dormant)
integration[t]    = delta_absorptive
synthesis[t]      = delta_self + delta_novel
```

### RIP Dominance

```
recursion_score[t] = |total_M_change| when total_events == 0 (pure TAP tick)
iteration_score[t] = |affordance_change| + |dormancy_change|
praxis_score[t]    = total_events (agentic actions)
dominance[t]       = argmax(recursion, iteration, praxis)
```

### dM Variance for Texture Mapping

Rolling variance of dM over window=10:
- Low variance + low |dM| = placid (Type 1)
- Low variance + positive dM = placid-clustered (Type 2)
- Medium variance = disturbed-reactive (Type 3)
- High variance = turbulent (Type 4)

---

## Correlation Analysis

`correlation_matrix(taps_scores) -> dict`:
- `matrix`: pairwise Pearson r for all mode score time series
- `labels`: ordered mode names
- `highly_correlated`: list of (mode_a, mode_b, r) where |r| > 0.85
- `independent_count`: modes with no |r| > 0.85 partner

This answers: are the TAPS modes truly independent, or do some collapse?

---

## Comparison Infrastructure

`scripts/taps_diagnostics.py --compare`:
- Gated run: default params (affordance gate ON, redistribution ON)
- Ungated run: affordance_min_cluster=0, skip redistribution
- Same seed, same params otherwise
- Side-by-side text table: Heaps beta, Gini, pressure ratio, syntropy, modes

---

## Jaccard Near-Stasis Fix

Current behavior when total_w == 0 (no Jaccard neighbor): total loss of types
and knowledge, agent dissolved.

Fixed behavior: agent enters deep stasis instead.
- Types preserved (identity doesn't evaporate)
- Knowledge truncated to 5% residual (near-frozen, not zero)
- `_deep_stasis = True` flag (distinct from dormant, dissolved, active)
- k_lost tracks the truncated portion (95%), not total
- n_types_lost = 0 (types preserved)
- Agent can potentially reactivate if cross-metathesis introduces shared types

Four agent states: active, dormant, deep_stasis, dissolved.

---

## Figure Output

### taps_correlation.png
Heatmap of correlation matrix. Modes on both axes. Blue (-1) to white (0) to
red (+1). Annotated with r values.

### taps_texture.png (two-panel + bands)
1. **Top panel (tall)**: Stacked area chart of pressure cascade over time.
   Warm tones (reds/oranges) for anapressive layers (high pressure).
   Cool tones (blues/teals) for anopressive layers (low pressure).
   Fulcrum visible at the color boundary.
2. **Middle band (narrow)**: Pressure spectrum — each mode's dominance as
   proportional width. Cross-hatch where modes are synchronous (|r| > 0.7).
3. **Bottom band (narrow)**: RIP dominance — grey (recursion), blue (iteration),
   red (praxis).

---

## Test Plan (~12 tests)

- TestTransvolution (3): involution/evolution/condensation from synthetic trajectory
- TestAnopression (2): anopressive normalization sums to 1.0; anapressive can exceed 1.0
- TestPressureRatio (2): net entropy when >1.0; net extropy when <1.0
- TestRIPDominance (2): recursion-dominant when no events; praxis-dominant when events fire
- TestCorrelationMatrix (2): output shape correct; highly_correlated detection works
- TestDeepStasis (1): near-stasis agent retains types and residual knowledge

---

## Attribution

TAPS (Transanoprasyn), RIP (Recursively Iterative Praxis/Preservation), the
metathesis extension from Hegel (thesis/athesis/synthesis/metathesis), the Laws
of Adjacency and Inclusion, and the extended praxis definition (projection +
reflection in consumptive consummation) are the original unpublished work of the
project author.

The convergence of TAPS and TAP naming was independent and uncoordinated.
