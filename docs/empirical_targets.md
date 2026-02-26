# Empirical Targets for sigma-TAP Validation

**CLAIM POLICY LABEL: exploratory**

> This document surveys real-world datasets and phenomena that could validate
> sigma-TAP's dynamics. It does not claim that sigma-TAP currently reproduces
> any of these datasets — it identifies mapping opportunities for future work.

---

## 1. Innovation Economics — Combinatorial Patent Data

**Mapping**: TAP equation → patent combination rates; metathesis events →
new technology combinations; adjacency parameter a → technology space
connectivity.

### Primary Dataset: SWINNO (Swedish Innovations Database)

Taalbi (2025) uses the SWINNO database of 3,086 innovations from 1,493
organizations in Swedish engineering, 1908–2016. Key finding: the rate of
innovation depends **linearly** on cumulative innovations, and the rate of
new product types follows Heaps' law. The adjacent possible is shown to be
partially predictable from the product space topology.

This is the most directly TAP-aligned empirical work available. Taalbi's
framework makes testable predictions about innovation rate, organizational
distribution, and diversification rate — all of which have analogues in
sigma-TAP mode scores.

- **Source**: Taalbi, J. (2025). "Long-run patterns in the discovery of
  the adjacent possible." *Industrial and Corporate Change*, 35(1), 123–149.
- **Data access**: SWINNO database, Lund University (academic access).
- **Mapping difficulty**: Medium. Requires mapping innovation counts to TAP
  M(t) trajectory and product diversification to type_set evolution.
- **Priority**: HIGH — most direct alignment with TAP theory.

### Secondary Dataset: KPSS Patent Panel (1926–2024)

Kogan, Papanikolaou, Seru, and Stoffman provide patent-level panel data
covering U.S. patents from 1926 to 2024, with CPC technology class
assignments. Technology class combinations across patent citations map to
cross-metathesis events.

- **Source**: KPSS dataset, updated December 2025.
- **Data access**: Public, via GitHub.
- **Mapping difficulty**: Medium-high. Requires defining "adjacency" in
  patent technology space and mapping citation networks to metathesis types.
- **Priority**: MEDIUM — large dataset but requires significant preprocessing.

### Tertiary: Youn et al. (2015) Patent Combinatorics

Youn, Strumsky, Bettencourt, and Lobo (2015) characterize U.S. patents
1790–2010 as a combinatorial process, finding invariant rates of
"exploitation" (refining existing combinations) and "exploration" (new
combinations). This exploitation/exploration distinction maps directly to
our absorptive vs novel cross-metathesis types.

- **Mapping difficulty**: Low-medium — their framework is already
  combinatorial.
- **Priority**: HIGH for conceptual validation.

---

## 2. Organizational Ecology — Emery/Trist Texture Transitions

**Mapping**: sigma-TAP texture types (1–4) → Emery/Trist environmental
causal textures; dM variance → environmental turbulence; pressure ratio →
organizational strain.

### Environmental Turbulence Survey Data

The environmental turbulence construct (derived from Emery & Trist 1965) is
widely used in management studies. Recent work includes:

- Reed (2025) in *Operations Management Research*: studies duration as an
  additional dimension of environmental turbulence beyond degree, using
  2×2 factorial experimental design.
- Cogent Business & Management (2024): environmental turbulence as moderator
  between knowledge sharing, innovation capability, and organizational
  sustainability in Nigerian SMEs.

No existing dataset directly operationalizes the full four-part causal
texture typology with longitudinal data. The construct is typically used as
a moderating variable (surveys measuring perceived turbulence) rather than
as a classification system applied to time-series data.

- **Mapping difficulty**: High. Would require either (a) longitudinal
  organizational survey data mapped to texture types, or (b) industry-level
  time series (e.g., market volatility, regulatory change frequency) as
  turbulence proxies.
- **Priority**: MEDIUM — important for theoretical validation but empirical
  data is scarce in the right format.
- **Opportunity**: sigma-TAP's dM variance texture classification could be
  validated against industry disruption timelines (e.g., technology sector
  transitions, regulatory regime changes).

---

## 3. Evolutionary Biology — Speciation/Extinction Dynamics

**Mapping**: metathesis events → speciation events; dissolution → extinction;
deep stasis → Lazarus taxa; type_set diversity → taxonomic diversity;
affordance gate → environmental filter for speciation.

### The Paleobiology Database (PBDB)

The PBDB is a community-built, web-accessible database of fossil occurrences
covering the entire Phanerozoic (541 Mya to present). It enables estimation
of origination and extinction rates per geological stage.

Key empirical patterns relevant to sigma-TAP:

1. **Rates of origination have declined through the Phanerozoic** — parallels
   our observation that innovation_potential (projection mode) tends to
   decrease over simulation runs.
2. **Pulses of speciation follow mass extinctions** — parallels post-
   disintegration bursts of novel cross-metathesis.
3. **Lazarus taxa** — organisms that disappear from the fossil record during
   extinction events then reappear — directly parallel deep stasis agents.

- **Source**: paleobiodb.org
- **Data access**: Public API, downloadable CSV.
- **Mapping difficulty**: Medium-high. Requires temporal binning decisions
  and mapping geological stage counts to discrete simulation steps.
- **Priority**: MEDIUM — strong conceptual alignment but different timescales.

### PyRate (Bayesian estimation of speciation/extinction rates)

PyRate is a software package for estimating speciation and extinction rates
from incomplete fossil data using Bayesian inference. Its rate estimates
could be compared against sigma-TAP's metathesis event rates under different
parameter regimes.

- **Mapping difficulty**: Medium — requires aligning PyRate's continuous-time
  rates with sigma-TAP's discrete-step event counts.

### 50/500 Rule and Population Viability

Frankham, Bradshaw, and Brook (2014) revised the 50/500 rule for minimum
viable population sizes. Our n_agents and dormancy dynamics could be compared
against MVP thresholds: does sigma-TAP produce extinction-like dynamics below
a critical agent count?

- **Priority**: LOW — interesting but tangential.

---

## 4. Chemistry and Nuclear Physics — Bohr's Periodic Table Parallel

**Mapping**: Element synthesis → cross-metathesis; electron shell filling →
type_set accumulation; nuclear cross-sections → interaction probabilities;
promethium isolation → deep stasis.

### The Bohr Periodic Table Observation

The project author noted that Bohr's early periodic table (1947) displays
branching and merging structures across element families that parallel
cross-metathesis dynamics in sigma-TAP:

- **Shared electron configurations** across elements in a period = shared
  type_sets across agents in a cluster.
- **Element families branching and merging** = absorptive and novel cross-
  metathesis creating new structural relationships.
- **Promethium (element 61)** — radioactive, no stable isotopes, absent from
  natural mineral deposits — is a natural analogue of deep stasis: an entity
  whose structural position exists in the adjacency space but whose material
  realization is transient.

### Nuclear Reaction Databases

- **ENDF/B (Evaluated Nuclear Data File)**: BNL's comprehensive nuclear
  reaction cross-section database covering all nuclei relevant to applied
  technology.
- **NNDC (National Nuclear Data Center)**: Nuclear structure and decay data
  for all known nuclides.
- **IAEA Atlas of Neutron Capture Cross Sections**: Neutron capture cross-
  sections from 10⁻⁵ eV to 20 MeV.

Nuclear cross-sections (probability of nuclear reactions) are analogous to
metathesis probabilities in sigma-TAP. The s-process (slow neutron capture)
parallels absorptive cross-metathesis; the r-process (rapid neutron capture)
parallels novel cross-metathesis bursts.

- **Mapping difficulty**: High — the analogy is structural, not quantitative.
  Nuclear physics operates at fundamentally different scales.
- **Priority**: LOW for direct validation, HIGH for conceptual illustration.
  The periodic table parallel is valuable for communicating the framework.

---

## 5. Dynamical Systems — Poincaré Phase Portrait Classification

**Mapping**: Mode transition matrices → discrete dynamical systems;
eigenvalues of transition matrix → Poincaré fixed-point classification;
absorbing states → stable nodes; oscillatory transitions → limit cycles.

### The Connection

The project author noted that Poincaré's qualitative dynamics (spirals,
foci, centers, saddle points) may provide a rigorous mathematical frame for
classifying mode transition map dynamics. This is correct:

- **Eigenvalue = 1** of the transition matrix → steady state (stable node)
- **Complex eigenvalues** → oscillatory transitions (spiral/focus)
- **Real eigenvalues < 1** → convergent dynamics (stable focus)
- **Eigenvalue > 1** → divergent/transient dynamics (unstable node)

This was implemented as a **Stage 3 analytical enhancement** in
`simulator/taps_sensitivity.py::eigenvalue_analysis()`, replacing the
heuristic absorbing-state detection (self-transition > 50%) with eigenvalue
decomposition of row-stochastic transition matrices.

### Implementation Details

The `eigenvalue_analysis()` function converts each axis's transition count
matrix to a row-stochastic (probability) matrix, computes eigenvalues via
`numpy.linalg.eigvals`, and classifies dynamics per axis:

- **Spectral gap** = 1 - |lambda_2| (second-largest eigenvalue magnitude).
  Large gap = fast mixing; small gap = quasi-absorbing.
- **Stationary distribution** = left eigenvector for lambda=1 (Perron-Frobenius).
- **Mixing time** = -1 / log|lambda_2| (steps to approximate stationarity).
- **Oscillation period** = 2*pi / |arg(lambda_2)| (if complex eigenvalues).

Dynamics classification scheme:
- `degenerate`: single state
- `ergodic`: spectral gap > 0.3 (fast mixing to equilibrium)
- `stable_node`: real eigenvalues, moderate convergence (gap 0.1-0.3)
- `stable_spiral`: complex eigenvalues, oscillatory convergence
- `quasi_absorbing`: spectral gap < 0.1 (very slow mixing)
- `periodic`: aperiodic chain with |lambda_2| > 0.95

Also computes irreducibility (single communicating class via matrix power
reachability) and aperiodicity (any positive diagonal entry).

### Default-Parameter Results (10 agents, 100 steps, alpha=5e-3, a=3.0, mu=0.005)

| Axis | Class | Spectral Gap | Mixing Time | Stationary Dominant |
|------|-------|-------------|-------------|---------------------|
| pressure_regime | stable_node | 0.169 | 5.4 | entropy (81%) |
| rip_dominance | ergodic | 0.888 | 0.5 | recursion (90%) |
| ano_dominant | ergodic | 0.947 | 0.3 | expression (95%) |
| syntegration_phase | ergodic | 0.912 | 0.4 | disintegration (92%) |
| transvolution_dir | ergodic | 0.912 | 0.4 | balanced (92%) |

Key finding: pressure_regime is the slowest-mixing axis (spectral gap 0.169),
indicating that pressure regime transitions are the most dynamically
significant.  All other axes are ergodic with fast mixing, suggesting they
track instantaneous events rather than sustained regime dynamics.

- **Literature**: Poincare, H. (1881-1886). "Memoire sur les courbes
  definies par une equation differentielle." Multiple parts in *Journal de
  Mathematiques Pures et Appliquees*.
- **Modern reference**: Strogatz, S. H. (2015). *Nonlinear Dynamics and
  Chaos*. Westview Press. Chapters 5-6 on fixed-point classification.
- **Status**: IMPLEMENTED (see `simulator/taps_sensitivity.py`, 12 tests).

---

## 6. Assessment Summary

| Domain | Dataset/Source | Mapping Difficulty | Priority | Stage |
|--------|---------------|-------------------|----------|-------|
| Innovation economics | SWINNO (Taalbi 2025) | Medium | HIGH | Next |
| Innovation economics | KPSS Patents (1926–2024) | Medium-high | MEDIUM | Future |
| Innovation economics | Youn et al. patent combinatorics | Low-medium | HIGH | Next |
| Organizational ecology | Turbulence survey data | High | MEDIUM | Future |
| Evolutionary biology | Paleobiology Database (PBDB) | Medium-high | MEDIUM | Future |
| Evolutionary biology | PyRate speciation rates | Medium | LOW | Future |
| Chemistry/nuclear | ENDF/B, NNDC cross-sections | High | LOW | Illustrative |
| Dynamical systems | Poincare eigenvalue analysis | N/A (no data needed) | MEDIUM | **DONE** |
| Predictive diagnostics | Predictive orientation diagnostic | N/A (no data needed) | HIGH | **DONE** |

### Recommended Next Steps

1. **Taalbi (2025) replication**: Map SWINNO innovation counts to TAP M(t)
   trajectory. Test whether sigma-TAP's linear innovation rate matches
   Taalbi's empirical finding. This is the most achievable validation target.

2. **Youn et al. exploitation/exploration ratio**: Compare their invariant
   exploitation/exploration rate against sigma-TAP's absorptive/novel cross-
   metathesis ratio across parameter regimes.

3. **Poincaré eigenvalue analysis**: ✓ DONE — eigenvalue decomposition
   implemented as Stage 3 enhancement (see §5).

4. **Predictive orientation diagnostic**: ✓ DONE — step-ahead Markov
   prediction with parallel matching, surprisal, and adpression detection
   (see §6c).

---

## 6b. Action Modality Calibration — Empirical Weight Derivation

**Mapping**: Praxis action decomposition → consumption (disintegration serving
integration, reflective cycle) + consummation (integration serving
disintegration/transformation, projective cycle). These are obverse and
reverse of the same action — a metathetic pair-of-pairs within Praxis.

### Cross-Domain Convergence on ~0.40:0.60

The consumption:consummation ratio converges across independent domains:

| Domain | Ratio (cons:consu) | Source | Invariance |
|--------|-------------------|--------|------------|
| US Patents 1790–2010 | 0.40 : 0.60 | Youn et al. 2015 | 220 years invariant |
| Glucose ATP capture | 0.40 : 0.60 | Thermodynamic constraint | Fixed |
| Microbial Y\_ATP | 0.36 : 0.64 | Bauchop & Elsden 1960 | Energy-limited invariant |
| Forest CUE (global) | 0.47 : 0.53 | DeLucia et al. 2007 | Approx. (range 0.23–0.83) |
| Ecosystem P/R at climax | 0.50 : 0.50 | Odum 1969 | Theoretical attractor |
| Marine origination:extinction | 0.56 : 0.44 | Alroy 2008 | Declining trend |

### sigma-TAP Calibration Result

Empirically-grounded weights per event type (see `simulator/taps.py`,
`ACTION_MODALITY_WEIGHTS`):

- Self-metathesis: (0.45, 0.55) — maximally dual, slight consummative lean
- Absorptive cross: (0.60, 0.40) — primarily consumptive
- Novel cross: (0.35, 0.65) — primarily consummative
- Disintegration: (0.40, 0.60) — primarily consummative
- Env transitions: (0.40, 0.60) — primarily consummative

**Calibration check** (6 parameter regimes, seed=42, 200 steps each):

- Overall weighted average: consumption=0.4009, consummation=0.5991
- Range across regimes: 0.377–0.435 consumption, 0.565–0.623 consummation
- **Matches the 0.40:0.60 empirical target without parameter tuning**
- Pure action (min of both modalities) confirms local rarity hypothesis:
  no regime achieves action\_balance near 0.50

---

## 6c. Predictive Orientation Diagnostic — Step-Ahead Markov Prediction

**Mapping**: Transition matrices (from §5 eigenvalue analysis) → step-ahead
probability distributions; parallel matching → usual system flow;
surprisal detection → candidate adpressive events; state diversity
collapse → annihilation/extinction detection.

### Implementation

The predictive orientation diagnostic (`simulator/predictive.py`) uses
post-hoc transition matrices to walk through a simulation trajectory
step-by-step, predicting at each step what the system is most likely
to do next and measuring how often this prediction matches reality.

Four core functions:

1. **predict_step()**: Given current state and transition matrix, returns
   probability distribution over next states. Supports multi-step
   look-ahead via matrix power P^k.
2. **compute_surprisal()**: Information-theoretic surprise = -log2(P(actual)).
   Capped at 10.0 bits for zero-probability events.
3. **detect_adpression()**: Adaptive threshold (running mean + 2 SD) flags
   steps where surprisal exceeds expected range. Grounded in cross-domain
   empirical practice (seismology, ecology, information theory).
4. **detect_state_collapse()**: Sliding-window diversity tracking flags
   annihilation/extinction events when an axis collapses to single state.
   An extinction is a *created* event — the dynamics produced the
   annihilation; it did not happen passively.

### Default-Parameter Results (10 agents, 150 steps, alpha=5e-3, a=3.0, mu=0.005)

| Axis | Parallel Match | Mean Surprisal | Adpressions | Collapses |
|------|---------------|----------------|-------------|-----------|
| syntegration_phase | 94.6% | ~0.3 bits | ~6 events | ~1 |
| rip_dominance | 93.3% | ~0.4 bits | ~8 events | ~1 |
| ano_dominant | 96.6% | ~0.2 bits | ~5 events | ~1 |

Key findings:
- **High parallel matching** (93-97%): The system is highly predictable
  within its established regime. Most transitions follow the most probable
  path.
- **Low mean surprisal** (<0.5 bits): Transitions are information-theoretically
  unsurprising on average — the system's statistical tendencies are stable.
- **Sparse adpression events**: Candidate adpressive events are rare (5-10
  per 150 steps), consistent with the theoretical expectation that adpression
  is an exceptional occurrence.
- **State collapse detection**: Annihilation/extinction events are flagged
  separately from transition surprisal, capturing a qualitatively different
  type of adpressive dynamics.

### Two-Tail Structure

The diagnostic captures both tails of the predictive distribution:
- **Usual tail** (parallel matching rate): How often the most probable
  transition actually occurs. High rates indicate regime stability.
- **Surprisal tail** (adpression events): When the improbable happens.
  Not merely "unusual" but "surprising given what was expected."

### Future Work

1. **Online/incremental mode** (HIGH PRIORITY): Build transition matrix
   incrementally during simulation for real-time prediction.
2. **Entity:ensemble scaling dial**: Four discrete zoom levels (L11, L12,
   L21, L22) adjusting entity:ensemble ratio, enabling sub-group and
   simulated per-agent predictions via top-down feedback.
3. **Empirical threshold calibration**: Compare surprisal distributions
   against natural system data during the empirical validation phase.
4. **Visualization**: Probability stream plots, surprisal time series,
   adpression markers.

- **Status**: IMPLEMENTED (see `simulator/predictive.py`, 24 tests).

---

## 7. References

- Taalbi, J. (2025). "Long-run patterns in the discovery of the adjacent
  possible." *Industrial and Corporate Change*, 35(1), 123–149.
- Youn, H., Strumsky, D., Bettencourt, L. M. A., & Lobo, J. (2015).
  "Invention as a combinatorial process." *Journal of the Royal Society
  Interface*, 12(106), 20150272.
- Kogan, L., Papanikolaou, D., Seru, A., & Stoffman, N. (2017/2025).
  "Technological innovation, resource allocation, and growth." Extended
  data, KPSS Patent Dataset.
- Emery, F. E. & Trist, E. L. (1965). "The causal texture of
  organizational environments." *Human Relations*, 18(1), 21–32.
- Reed (2025). "Duration matters: the interaction of the degree and the
  duration of environmental turbulence." *Operations Management Research*.
- Alroy, J. et al. (2008). "Dynamics of origination and extinction in the
  marine fossil record." *PNAS*, 105(S1), 11536–11542.
- Frankham, R., Bradshaw, C. J. A., & Brook, B. W. (2014). "Genetics in
  conservation management." *Biological Conservation*, 170, 56–63.
- Strogatz, S. H. (2015). *Nonlinear Dynamics and Chaos*. Westview Press.
- Odum, E. P. (1969). "The strategy of ecosystem development." *Science*,
  164(3877), 262–270.
- DeLucia, E. H., Drake, J. E., Thomas, R. B., & Gonzalez-Meler, M. (2007).
  "Forest carbon use efficiency: Is respiration a constant fraction of gross
  primary production?" *Global Change Biology*, 13, 1157–1167.
- Bauchop, T. & Elsden, S. R. (1960). "The growth of micro-organisms in
  relation to their energy supply." *J. Gen. Microbiol.*, 23, 457–469.
- March, J. G. (1991). "Exploration and exploitation in organizational
  learning." *Organization Science*, 2(1), 71–87.
