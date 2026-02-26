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

This is a **Stage 3 analytical enhancement**: replace the current heuristic
absorbing-state detection (self-transition > 50%) with eigenvalue
decomposition of the transition matrix for rigorous Poincaré-style
classification.

- **Literature**: Poincaré, H. (1881–1886). "Mémoire sur les courbes
  définies par une équation différentielle." Multiple parts in *Journal de
  Mathématiques Pures et Appliquées*.
- **Modern reference**: Strogatz, S. H. (2015). *Nonlinear Dynamics and
  Chaos*. Westview Press. Chapters 5–6 on fixed-point classification.
- **Priority**: MEDIUM — mathematically elegant upgrade path, no new data
  collection needed.

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
| Dynamical systems | Poincaré eigenvalue analysis | N/A (no data needed) | MEDIUM | Stage 3 |

### Recommended Next Steps

1. **Taalbi (2025) replication**: Map SWINNO innovation counts to TAP M(t)
   trajectory. Test whether sigma-TAP's linear innovation rate matches
   Taalbi's empirical finding. This is the most achievable validation target.

2. **Youn et al. exploitation/exploration ratio**: Compare their invariant
   exploitation/exploration rate against sigma-TAP's absorptive/novel cross-
   metathesis ratio across parameter regimes.

3. **Poincaré eigenvalue analysis**: Implement eigenvalue decomposition of
   transition matrices as a Stage 3 enhancement — no external data needed.

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
