# Taalbi (2025) — Long-Run Patterns in the Discovery of the Adjacent Possible

**Source**: Josef Taalbi, "Long-run patterns in the discovery of the adjacent possible," *Industrial and Corporate Change*, Volume 35, Number 1, pp. 123–149 (2025). doi:10.1093/icc/dtaf028. Department of Economic History, Lund University. 27 pages including appendices.

**Core contribution**: Unifies TAP-style recombinant innovation with Weitzman's resource constraints and Tria et al.'s urn models to produce three testable predictions: (1) innovation rate is linear in cumulative innovations, (2) firm innovation distribution follows a power law with exponent ≈ −2, (3) product diversification follows Heaps' law with exponent ν/ρ ≈ 0.587. Validates all three against 108 years of Swedish engineering innovation data (1908–2016). Demonstrates that resource constraints tame TAP super-exponential growth to exponential at the organization level, preventing winner-take-all dynamics while preserving cumulative advantage.

---

## The Three-Part Adjacent Possible (Section 2.1)

Innovation spaces are partitioned into:
1. **Already discovered** — the set of drawn elements $S$
2. **Currently discoverable** (adjacent possible $U$) — reachable by recombining elements of $S$
3. **Out of reach** — innovations not currently accessible but potentially reachable in future

The adjacent possible $U$ has two components:
- **New product types** $U_{NC}$ (new "colors") — genuinely novel combinations
- **Product improvements** $U_C$ ("copies") — improvements within already-known categories

### Effective Adjacent Possible

The *effective* adjacent possible is the subset of $U$ actually searchable given resource constraints. Three constraints reduce it from the full combinatorial space:

**1. Integration complexity**: Recombination length $\lambda$ limits the number of elements combined. For $\lambda \ll D$ (product diversity), combinations are approximated by $\binom{D^*}{\lambda} \approx \frac{D^{*\lambda}}{\lambda!}$.

**2. Absorptive capacity** (Cohen & Levinthal 1990): Organizations search only a subset $D^* \leq D$ of available knowledge types, determined by their capacity to recognize, assimilate, and apply new knowledge.

**3. Resource constraints** (Weitzman 1998): The ability to test and process materials limits search to a subset $R$ of possible recombinations.

### Size of the Adjacent Possible

Number of adjacent new product types:

$$|U_{NC}| = \nu \frac{D^{*!}}{\lambda!(D^*-\lambda)!} = \nu|R|$$

where $0 \leq \nu \leq 1$ is the fraction of recombinations yielding genuinely new types.

Number of adjacent product improvements:

$$|U_C| = \rho \frac{k}{D} |R|$$

where $\rho \geq 0$ is the probability of finding a new improvement, and $k/D$ captures how cumulative experience per product type enables improvements.

Total effective adjacent possible:

$$|U| = |U_{NC}| + |U_C| = \left(\nu + \rho \frac{k}{D}\right)|R|$$

This is Equation (3) — the central framework equation.

---

## Rate of Innovation (Section 2.2)

The TAP assumption: innovation rate equals the size of the effective adjacent possible:

$$\frac{dk}{dt} = |U|$$

### Two Regimes (Figure 2)

Following Weitzman (1998), two limiting cases:
- **Adjacent-possible-limited**: Innovation rate grows super-linearly (early stage)
- **Resource-constrained**: Innovation rate grows linearly or sub-linearly (mature stage)

### General Rate Equation

Using $D \sim k^\gamma$ (product diversity grows sub-linearly with cumulative innovations) and $D^* \sim D^\beta$ (search space grows sub-linearly with product diversity):

$$\frac{dk}{dt} \propto k^{1-\gamma} D^{*\lambda} \qquad \text{(Equation 4)}$$

This simplifies to:

$$\frac{dk}{dt} \propto k^\sigma \qquad \text{where } \sigma = 1 - \gamma + \gamma\beta\lambda \qquad \text{(Equation 5)}$$

**Recombination length $\lambda$ determines the growth regime**: $\lambda = 1$ gives exponential solutions; $\lambda > 1$ gives super-exponential. If $\lambda$ is unconstrained (grows with $D$), TAP blow-up occurs.

### Resource-Constrained Case: The Linear Result

When resources constrain search space growth to a limiting rate $\eta$ (i.e., $D^*(t) \propto \eta^t$), the differential equation reduces to:

$$\frac{dk}{dt} \propto k \ln \eta \qquad \text{(Equation 7)}$$

**The rate of innovation is linear in cumulative innovations.** This is the Bianconi-Barabási model: $k$ is the "rich-get-richer" mechanism; $\ln \eta$ is organizational "fitness."

Solution: innovation within organizations follows **exponential curves in time** (not super-exponential), because resource constraints tame TAP growth.

### Hypotheses

- **H1a**: Rate of innovation is initially super-linear, eventually linear in cumulative innovations (resource constraints bind)
- **H1b**: Rate depends positively on search space size, with coefficient $\lambda \geq 2$

---

## Distribution Across Organizations (Section 2.3)

From the linear attachment kernel $dk/dt \propto k$:

$$P_k \sim k^{-2} \qquad \text{(Equation 8)}$$

The distribution of innovations across organizations follows a **power law with exponent ≈ −2**. Sub-linear kernels would give stretched exponentials; super-linear would give winner-take-all. The linear result sits precisely at the boundary.

If variation in organizational fitness $\eta$ is non-negligible, the distribution becomes **log-normal** (from the Central Limit Theorem applied to $\log k = \int \eta \, dt$).

- **H1c**: Distribution follows power law with exponent ≈ −2 (or log-normal if fitness varies)

---

## Product Diversification: Heaps' Law (Section 2.4)

The rate of product diversification (new types per total innovation):

$$\frac{dD}{dk} = \frac{\nu D}{\nu D + \rho k} \qquad \text{(Equation 10)}$$

This is identical to the dynamic equation from Tria et al. (2014) for urn models.

### Two Regimes

**Deepening search** ($\nu < \rho$): Explorative search is more difficult/costly than local search. Produces **Heaps' law**:

$$D \sim (\rho - \nu)^{\nu/\rho} k^{\nu/\rho} \qquad \text{(Equation 11)}$$

Product diversity grows sub-linearly with cumulative innovations. The share of genuinely new types declines over time.

**Widening search** ($\nu > \rho$): New types are easier to discover (paradigm shifts, new technological opportunities). Product diversity approaches a fixed share of total innovations.

The literature "unanimously suggests" deepening search is the norm; widening occurs only during paradigm shifts.

- **H2**: Long-run product diversification follows Heaps' law (deepening regime dominates)

---

## Product Space and Predictability (Section 2.5)

### The Unprestatable Question

Kauffman argues the adjacent possible is "unprestatable" — impossible to predict in detail. Fleming argues "any component is at risk of being recombined with any other component" — no constraints on what can combine.

Yet empirical product space studies (Hidalgo et al. 2007, Neffke & Henning 2013) show strong constraints: product diversification is highly structured by existing capabilities.

### Resolution: Timescale Dependence

Both perspectives are correct at different timescales:
- **Short-run**: Technological trajectories constrain recombination → community structure in product space → predictable diversification paths
- **Long-run**: Community structures vanish → product space becomes dense → any firm can potentially reach any product in few steps → unprestatable

- **H3a**: Product space has community structure at shorter time spans
- **H3b**: Firms' positions in product space predict their adjacent possible (future products)

### Proximity Measure

Product proximity $\phi_{ij}$ defined by weighted co-occurrence within firms:

$$\phi_{ij} = \sum_l \frac{k_{il}}{k_i} \cdot \frac{k_{jl}}{k_l} \qquad \text{(Equation 13)}$$

where $k_{il}$ = cumulative innovations by firm $l$ in product group $i$, $k_l$ = total cumulative innovations by firm $l$.

Product "density" $\omega_{jl}$ measures how close firm $l$ is to product $j$:

$$\omega_{jl} = \frac{\sum_i x_{il} \phi_{ij}}{\sum_i \phi_{ij}} \qquad \text{(Equation 14)}$$

---

## Empirical Data and Methods (Section 3)

### LBIO Dataset

**Literature-Based Innovation Output** (LBIO) methodology applied to Swedish engineering industry, 1908–2016:
- Two primary trade journals: *Teknisk tidskrift* (1871–1967) → *Ny Teknik* (1967–present) and *Tidningen Verkstäderna* (1905–present)
- 3086 innovations by 1493 organizations
- Product classification: ISIC Rev. 3 at 3-digit level
- Captures actual innovations (not patents), including commercialization year
- 53% coverage of all innovations in the full 15-journal database

### Organizational Continuity

Mergers and acquisitions tracked through company histories. Organizations collapsed to corporate group level when motivated by documented histories (e.g., Volvo's constituent divisions traced to independent predecessor firms).

### Search Space Measurement

$D^*$ estimated from cumulative unique CPC (patent classification) classes cited by firm. Decomposed into recent change (past 5 years) and lagged level to test knowledge depreciation.

---

## Key Empirical Results (Section 4)

### Result 1: Innovation Rate Linear in Cumulative Innovations (H1a ✓)

Negative binomial regressions confirm:
- **Coefficient on log cumulative innovations: 0.997** (Model 1, pooled) — essentially exactly linear
- Initially super-linear (inset of Figure 4a shows exponent ≈ 1.5 early on)
- Linearity "cannot be rejected" across all model specifications
- Search space (D*) significantly positive: baseline $\lambda = 1.14$ (all firms), $\lambda = 1.80$ (post-1910 entrants)
- Knowledge depreciation detected: recent search space expansion matters more than lagged level

### Result 2: Power-Law Distribution with Exponent ≈ −2 (H1c ✓)

- Estimated exponent: **−2.213** (distribution), **−1.111** (cumulative distribution)
- Kolmogorov-Smirnov goodness-of-fit: cannot reject power law (p = 0.99–1.00 across benchmark periods)
- Log-normal cannot be excluded as equally good fit, consistent with theoretical prediction when fitness variation is non-negligible

### Result 3: Heaps' Law with ν/ρ = 0.587 (H2 ✓)

- Log-log regression of product diversity on cumulative innovations: **exponent = 0.587** (R² = 0.820)
- Confirms deepening search regime ($\nu < \rho$): rate of genuinely new product types declines over time
- Product diversity grows sub-linearly — organizations increasingly exploit existing knowledge rather than exploring new domains

### Result 4: Product Space Structure (H3a, H3b ✓)

**Network characteristics** (Table 7):
- Sparse small-world network (small-worldness > 1 across all periods)
- Moderate community structure in sub-periods (modularity 0.36–0.38) but weak in full period (0.23)
- High transitivity in early periods (37%) declining over time (17%) — constraints loosen
- Average path length 2.5–3.7 — any product reachable in few diversification steps

**Prediction performance** (Table 8):
| Measure | Random | Binary co-occurrence | Weighted co-occurrence | XGBoost |
|---|---|---|---|---|
| Sensitivity | 0.92 | 0.76 | 0.77 | 0.81 |
| Specificity | 0.10 | 0.67 | 0.67 | 0.83 |
| **Balanced accuracy** | 0.51 | 0.71 | 0.72 | **0.82** |

Product space substantially predicts adjacent possible innovations. XGBoost achieves 82% balanced accuracy in predicting which products firms will innovate in next period.

### Summary Table (Table 6)

| Test | Equation | Result |
|---|---|---|
| H1a: Linearity σ | $dk/dt = k^\sigma$ | Linearity cannot be rejected |
| H1b: Recombination length λ | $dk/dt \propto k^{1-\gamma} D^{*\lambda}$ | 1.14 avg, 1.8 for post-1910 firms |
| H1c: Heaps' law ν/ρ | $D \sim (\rho-\nu)^{\nu/\rho} k^{\nu/\rho}$ | 0.587 |
| H2: Power law distribution | $P_k \sim k^{-2}$ | Exponent ≈ 2, or log-normal |

---

## Discussion Conclusions (Section 5)

**Resource constraints are the key moderator**: The unconstrained TAP equation predicts super-exponential growth, but resource-constrained organizations follow exponential growth curves. This prevents winner-take-all dynamics while preserving cumulative advantage.

**Predictability is timescale-dependent**: Short-run product space structure enables prediction; long-run structure vanishes as "virtually any elements can be recombined" (Fleming 2001). Innovation "oscillates between cumulativity and renewal, continuity and disruption."

**Not a contradiction of TAP**: Individual organizations are resource-constrained, but macroeconomic aggregate could still show super-linear patterns if the population of inventors/firms increases enough to explore untapped adjacent possibilities beyond individual firm capacity. However, "checks to growth" likely make such scenarios intermittent rather than permanent.

---

## Mathematical Appendices (Condensed)

### Appendix A: Heaps' Law Derivation

From $dD/dk = \nu D/(\nu D + \rho k)$, substituting $z = D/k$ and solving:
- Deepening ($\nu < \rho$): $D \sim (\rho - \nu)^{\nu/\rho} k^{\nu/\rho}$ — sub-linear (Heaps' law)
- Widening ($\nu > \rho$): $D \sim \frac{\nu - \rho}{\nu} k$ — linear share

### Appendix B: Bianconi-Barabási Derivation

From $dk/dt = (\rho/\lambda!) k^{1-\gamma} \eta^{\lambda t}$ with Heaps' law substitution, one obtains $dk/dt = (\lambda/\gamma) \ln\eta \cdot k$, yielding exponential growth $k = \exp(C + (\lambda/\gamma) \ln\eta \cdot t)$.

### Appendix C: Power-Law Distribution Derivation

Master equation approach with linear attachment kernel ($dk/dt \propto k$) gives $P_k \sim k^{-2}$. Sub-linear kernels give stretched exponentials: $P_k \propto k^{-\sigma} \exp(-\mu m k^{1-\sigma}/(1-\sigma))$. Cross-sectional variation in $\eta$ (fitness) produces log-normal via CLT.

---

## Key Concepts for sigma-TAP Integration

1. **The "Youn ratio" in sigma-TAP**: The project's Youn ratio (target ≈ 0.60, achieved 0.62) measures alignment with empirical innovation patterns. This paper establishes the empirical benchmark: Heaps' law exponent ν/ρ = 0.587, power-law exponent ≈ −2, and linear innovation rate. The Youn ratio likely measures some combination of these regularities in the simulation output.

2. **Resource constraints = the sigma function**: Taalbi's central result — resource constraints tame TAP super-exponential to exponential — is precisely what sigma-TAP's sigma feedback does. The sigma function is the computational implementation of Weitzman's resource constraint, producing the "logistic TAP" that the Cortês et al. paper left as unsolved.

3. **Deepening vs. widening search = self-metathesis vs. cross-metathesis**: Deepening search (ν < ρ, exploiting existing knowledge, Heaps' law) maps to L11/self-metathesis dominance. Widening search (ν > ρ, paradigm shifts, new technological opportunities) maps to L12/cross-metathesis dominance. The ν/ρ ratio is the empirical marker for the balance between exploitation and exploration.

4. **Linear rate with cumulative advantage**: The dk/dt ∝ k result means past innovation compounds — "the rich get richer" — but without winner-take-all. This is the stable intermediate regime that the bare TAP equation cannot produce (extinction instability). The sigma-TAP affordance-exposure accumulation (Xi) is designed to produce exactly this linear cumulative advantage.

5. **Product space = type-set adjacency topology**: The product space network (proximity $\phi_{ij}$ based on co-occurrence within firms) is the empirical analogue of sigma-TAP's type-set similarity. Agents with overlapping type-sets are "proximate" in product space. The community structure at short timescales corresponds to the family groups / topology tracking planned for Stage 3B.

6. **Timescale-dependent predictability**: Short-run structure (community, constraints) vs. long-run dissolution (any element can recombine with any other). This maps directly to Bateson's Table D correction arc: correction possible at interpersonal (short-run, L12) but impossible at cultural (long-run, L22). Also maps to the two-scale TAP: exponential phase (structured, predictable) gives way to blow-up (unstructured, unpredictable).

7. **Absorptive capacity = reception field**: Cohen & Levinthal's absorptive capacity (the ability to recognize, assimilate, and apply new knowledge) maps to the per-agent reception field / fidelity band (§5.57). The search window $D^*$ — the subset of knowledge actually searched — is the firm-level analogue of the agent's aperture function.

8. **Heaps' law exponent as calibration target**: ν/ρ = 0.587 provides a precise empirical calibration point for sigma-TAP. The ratio of novel types discovered to total innovations should decline sub-linearly. In sigma-TAP terms: the ratio of new type acquisitions (cross-metathesis) to total metathetic events (self + cross) should follow Heaps' law scaling.

9. **Power-law exponent ≈ −2 as distributional target**: The distribution of total innovations across organizations (agents) should follow $P_k \sim k^{-2}$ or log-normal. This constrains the sigma-TAP population dynamics — neither too equal (all agents similar) nor too extreme (winner-take-all).
