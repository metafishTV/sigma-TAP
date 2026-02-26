# TAPS/RIP Mode-to-Observable Mapping

**CLAIM POLICY LABEL: exploratory**

> This document is exploratory and does not derive from the source TAP/biocosmology
> literature. It uses sigma-TAP computational infrastructure but represents an
> independent analytical direction that requires further validation.

TAPS (Transanoprasyn), RIP (Recursively Iterative Praxis/Preservation), the
metathesis extension from Hegel (thesis/athesis/synthesis/metathesis), the Laws
of Adjacency and Inclusion, and the extended praxis definition (projection +
reflection in consumptive consummation) are the original unpublished work of the
project author. The convergence of TAPS and TAP naming was independent and
uncoordinated.

---

## 1. Mode-to-Observable Table

All formulas operate on per-step trajectory snapshots. Event deltas are computed
between consecutive snapshots (the simulator records cumulative counters).

### T -- Transvolution

Involution + evolution = condensation. Transvolution captures the coupled
inward-outward folding dynamic of the system at each tick.

| Mode | Formula | Interpretation |
|------|---------|----------------|
| involution | `(delta_self + delta_absorptive) / max(1, total_events)` | Inward folding: self-restructuring and absorptive incorporation |
| evolution | `(delta_novel + delta_disintegration + delta_env) / max(1, total_events)` | Outward unfolding: novelty, redistribution, and environmental transitions |
| condensation | `involution * evolution` | Coupled product of both directions |

Snapshot fields used: `n_self_metatheses`, `n_absorptive_cross`, `n_novel_cross`,
`n_disintegration_redistributions`, `n_env_transitions`.

### A -- Anopression / Anapression

Pressure spectrum with 3/4 asymmetry. Three anopressive modes (normalized to
sum = 1) and four anapressive modes (not normalized, can exceed 1.0).

#### Anopressive (normalized to sum = 1)

| Mode | Formula | Interpretation |
|------|---------|----------------|
| expression | `agents_with_dM>0 / n_active` (always = 1 as reference) | Outward manifestation |
| impression | `delta_absorptive / total_events` | Inward reception |
| adpression | `(delta_self + delta_disintegration) / total_events` | Sudden causal release (punctuated events) |

After raw computation, all three are divided by their sum so that
`expression + impression + adpression = 1.0`.

#### Anapressive (NOT normalized, can exceed 1.0)

| Mode | Formula | Interpretation |
|------|---------|----------------|
| oppression | `1 - (mean_dM / max_observed_dM)` | Base constraint: being metathesized by reality |
| suppression | `1 - affordance_mean` | Gate blocking fraction |
| depression | `mu * mean_M / max(epsilon, \|mean_dM\|)` | Decay/growth ratio |
| compression | `mean_M / K` | Capacity pressure (apex) |

Snapshot fields used: `total_M`, `n_active`, `affordance_mean`, `K_env`,
plus event deltas. Parameter `mu` defaults to 0.005.

#### Pressure Ratio

```
pressure_ratio = sum(oppression, suppression, depression, compression) / 1.0
```

- `> 1.0` = net entropy (breaking exceeds building capacity)
- `< 1.0` = net extropy (building exceeds breaking)
- `= 1.0` = syntropic equilibrium

### P -- Praxis

Projection + reflection leading to consumptive consummation. Extended from
Aristotle's classical praxis (theory/practice/action) per the project author's
framework. Praxis is universal, not exclusively human.

| Mode | Formula | Interpretation |
|------|---------|----------------|
| projection | `innovation_potential` (from snapshot) | Feedforward exploration of the adjacent possible |
| reflection | `affordance_mean` (from snapshot) | Feedback self-assessment of readiness |
| action | `total_events` (per-step delta) | The actual doing |

### S -- Syntegration

Synthetic disintegration: the four-phase cycle of structural transformation.

| Mode | Formula | Interpretation |
|------|---------|----------------|
| disintegration | `delta_disintegrations` | Coming apart, redistribution |
| preservation | `n_dormant / (n_active + n_dormant)` | Dormancy fraction, structural maintenance |
| integration | `delta_absorptive` | Absorptive incorporation |
| synthesis | `delta_self + delta_novel` | Creation of new structure |

### RIP -- Recursively Iterative Praxis/Preservation

Dual-P functional dynamics: RIP(praxis) is process-oriented and non-ergodic;
RIP(preservation) is progress-oriented and ergodic.

| Mode | Formula | Interpretation |
|------|---------|----------------|
| recursion | `\|dM\|` when `total_events == 0` | Bare cycle, one TAP tick, primarily ergodic |
| iteration | `\|delta_affordance\| + \|delta_dormancy\|` | Non-ergodic condition change (self or forced) |
| praxis | `total_events` | Agentic action |
| dominance | `argmax(recursion, iteration, praxis)` | Which mode dominates each step |

---

## 2. Cascade Ordering

The full pressure cascade runs from anopressive apex to anapressive base, with
a fulcrum separating the two hemispheres. Both poles return to metathesis;
together they constitute syntropy.

```
ANO (extropic, cool tones)          ANA (entropic, warm tones)
--------------------------------    --------------------------------
expression   (ano apex)             compression  (ana apex)
    |                                   |
impression                          depression
    |                                   |
adpression                          suppression
    |                                   |
    +------- [FULCRUM] --------+    oppression   (ana base)
```

Reading order through the cascade:

1. **expression** (ano apex) -- outward manifestation
2. **impression** -- inward reception
3. **adpression** -- sudden causal release
4. [fulcrum]
5. **compression** (ana apex) -- refined inward constraint
6. **depression** -- gradual reduction from above
7. **suppression** -- preventing emergence
8. **oppression** (ana base) -- fundamental constraint

Color scheme in figures:
- Cool tones (blues, teals) for anopressive layers
- Warm tones (reds, oranges) for anapressive layers
- The fulcrum is visible at the color boundary

---

## 3. Emery and Trist L-mapping

From Emery, F. (1977), *Futures We Are In*. The L-mapping connects causal
texture theory to the dialectical structure of TAPS.

| L-cell | Emery/Trist Definition | TAPS Dialectical Equivalent |
|--------|------------------------|----------------------------|
| L11 | Interdependencies within the system | Athetic relations (adjacent theses interacting internally) |
| L12 | System's actions into the environment | Thetic relations (thesis asserting itself outward) |
| L21 | Goals/noxiants from the environment | Synthetic relations (environment acting on the system) |
| L22 | Causal texture of the environment | Metathetic relations (the ongoing ground process) |

L22 (metathetic relations) corresponds to the causal texture that shapes all
other relations. In sigma-TAP, this is the environmental parameter regime
(placid, clustered, disturbed-reactive, turbulent) that contextualizes agent
dynamics.

---

## 4. Dialectical Foundation

Extension of Hegelian dialectics by the project author. Classical Hegel provides
thesis/antithesis/synthesis. The TAPS framework replaces antithesis with athesis
and adds metathesis as the ground level.

| Term | Definition |
|------|-----------|
| **Thesis** | Thing-in-itself; any entity with identity |
| **Athesis** | Adjacent thing; every other thesis (not anti-thesis, but co-thesis) |
| **Synthesis** | Combinatorial product of thesis-athesis interaction |
| **Metathesis** | Ongoing process that synthesizes syntheses (the ground level) |

### Temporal Modes of Metathesis

- **Synchronic metathesis**: Continuous flow of praxis. Turns on and off but
  never stops entirely while the system exists. Corresponds to the TAP tick in
  sigma-TAP (the per-step ODE integration that always runs).

- **Diachronic metathesis**: Serial flow of syntegration. Starts and stops,
  creates and destroys identities. Corresponds to metathesis events in sigma-TAP
  (self-metathesis, cross-metathesis, disintegration/redistribution).

---

## 5. Conservation Principles

1. **Expression = 1 as baseline reference.** The anopressive triad normalizes to
   sum = 1, with expression serving as the constant reference against which
   impression and adpression are measured.

2. **Syntropy is primary.** Entropy and extropy are derivative decompositions of
   syntropy, not independent forces. The pressure cascade captures both
   directions; their sum constitutes the syntropic process.

3. **Conservation of praxitive expenditure.** Minimal praxis to maximal
   syntegration ratio. The system conserves action: the least praxis (events)
   that produces the greatest syntegration (structural transformation) is the
   efficient path.

4. **Bohm's holomovement correlates to syntropy**, not to the implicate order
   alone. The holomovement is the combined syntropic process (the continuous
   enfolding and unfolding), while the implicate order is only the enfolded
   aspect. Syntropy as defined here encompasses both the implicate (involution)
   and explicate (evolution) aspects in their coupled dynamic.

---

## 6. dM Variance Texture Mapping

Rolling variance of dM (change in total knowledge M) maps to Emery and Trist's
four environmental texture types. Window size = 10 steps in the diagnostic
implementation.

| Variance | Mean dM | Texture Type | Emery/Trist Label |
|----------|---------|-------------|-------------------|
| Low | Low `\|dM\|` | Type 1 | Placid-randomized |
| Low | Positive dM | Type 2 | Placid-clustered |
| Medium | Variable | Type 3 | Disturbed-reactive |
| High | Variable | Type 4 | Turbulent |

The texture classification provides a bridge between the statistical properties
of the TAP trajectory and the qualitative environmental typology from
organizational ecology.

---

## 7. Literature References

### TAP / Biocosmology (source framework)

- Cortes, Kauffman, Liddle & Smolin (2022/2025). "The TAP equation."
- Cortes, Kauffman, Liddle & Smolin (2022). "Biocosmology: Towards the birth
  of a new science."
- Cortes, Kauffman, Liddle & Smolin (2022). "Biocosmology: Biology from a
  cosmological perspective."
- Cortes, Kauffman, Liddle & Smolin (2025). "The TAP equation: evaluating
  combinatorial innovation in biocosmology."
- Taalbi (2025). "Long-run patterns in the discovery of the adjacent possible."

### Causal Textures

- Emery, F. E. & Trist, E. L. (1965). "The causal texture of organizational
  environments." *Human Relations*, 18(1), 21-32.
- Emery, F. (1977). *Futures We Are In*. Leiden: Martinus Nijhoff.

### Adjacent Possible and Enablement

- Kauffman, S. A. (1993). *The Origins of Order*. Oxford University Press.
- Longo, G., Montevil, M., & Kauffman, S. (2012). "No entailing laws, but
  enablement in the evolution of the biosphere." *Proceedings of the 14th
  Annual Conference on Genetic and Evolutionary Computation*, 1379-1392.

### Niche Construction

- Odling-Smee, F. J., Laland, K. N., & Feldman, M. W. (2003). *Niche
  Construction: The Neglected Process in Evolution*. Princeton University Press.

### Population Viability

- Frankham, R., Bradshaw, C. J. A., & Brook, B. W. (2014). "Genetics in
  conservation management: Revised recommendations for the 50/500 rules."
  *Biological Conservation*, 170, 56-63.

### Knowledge Spillovers

- Marshall, A. (1890). *Principles of Economics*. Macmillan.
- Krugman, P. (1991). "Increasing returns and economic geography." *Journal of
  Political Economy*, 99(3), 483-499.

### Implicate Order

- Bohm, D. (1980). *Wholeness and the Implicate Order*. Routledge.

### TAPS / RIP / Metathesis Extension

- Project author's unpublished framework. TAPS (Transanoprasyn), RIP
  (Recursively Iterative Praxis/Preservation), the metathesis extension from
  Hegel, the Laws of Adjacency and Inclusion, and the extended praxis definition
  are original unpublished work.

---

## 8. Attribution

TAPS (Transanoprasyn), RIP (Recursively Iterative Praxis/Preservation), the
metathesis extension from Hegel (thesis/athesis/synthesis/metathesis), the Laws
of Adjacency and Inclusion, and the extended praxis definition (projection +
reflection in consumptive consummation) are the original unpublished work of the
project author.

The convergence of TAPS and TAP naming was independent and uncoordinated. TAPS
provides dispositional dynamics (HOW things happen); TAP provides mechanical
dynamics (WHAT happens). Together with RIP (WHETHER the system is maintaining,
developing, or acting), they form a layered diagnostic framework where:

- **TAP mechanics** tell you WHAT happened (innovation events, growth, decay)
- **TAPS disposition** tells you HOW it happened (which pressure modes dominate)
- **RIP rhythm** tells you WHETHER the system is maintaining, developing, or acting

---

## 9. Implementation Reference

The computational implementation lives in `simulator/taps.py`. Key functions:

| Function | Returns |
|----------|---------|
| `compute_transvolution(trajectory)` | involution, evolution, condensation per step |
| `compute_anopression(trajectory, mu)` | 7 pressure modes per step |
| `pressure_ratio(ano_scores)` | net entropy/extropy ratio per step |
| `compute_praxis(trajectory)` | projection, reflection, action per step |
| `compute_syntegration(trajectory)` | disintegration, preservation, integration, synthesis per step |
| `compute_rip(trajectory)` | recursion, iteration, praxis, dominance per step |
| `compute_all_scores(trajectory, mu)` | flat dict of all 17 TAPS mode arrays |
| `correlation_matrix(scores)` | pairwise Pearson r, highly correlated pairs, independence count |

Total mode count: 17 (3 T + 7 A + 3 P + 4 S) plus 4 RIP scores.

Diagnostic script: `scripts/taps_diagnostics.py`
Test suite: `tests/test_taps.py`
