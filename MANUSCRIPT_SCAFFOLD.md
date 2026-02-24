# σ-TAP Manuscript Scaffold (Paper-Ready)

## 0) Scope and venue presets (decide before drafting)

### Venue preset A — Journal of Theoretical Biology / Complexity (default)
- Lead taxonomy as **methodological framing** (reader guidance) in Section 2 prose.
- Keep defensive role implicit.
- Word budget target: ~8,000–12,000 (JTB-friendly), with concise methods note on initialization policy.

### Venue preset B — Entropy-style formalist audience
- Add an explicit **definitional table** early in Section 2 with four rows:
  - Derived
  - Closure
  - Constitutive
  - Interpretive
- State admissibility/identifiability assumptions adjacent to this table.
- Keep full equation set visible (avoid over-compression).

## 1) Introduction

- **Problem setting:** TAP models innovation as combinatorial expansion with non-ergodic growth structure.
- **Gap:** base TAP lacks explicit path-dependent efficiency modulation from consummation loading.
- **Contributions:**
  1. formal TAP→σ-TAP articulation with explicit layer taxonomy,
  2. executable simulator + phase-structure experiments,
  3. empirical diagnosis of when σ-feedback shifts occupancy vs accelerates timing,
  4. explicit continuous/discrete discrepancy as a falsifiable baseline test.

### 1.1 Methodological note (standalone citable point)

State early (one short paragraph):
> **Initialization policy must be declared** when fitting TAP-like phase diagrams; otherwise coefficient signs can flip under policy-induced covariate coupling (Mode A vs Mode B), making cross-paper boundary coefficients uninterpretable.

(Then reference full diagnostic in Section 3.)

## 2) Model (compress articulation Sections 1–8)

- Combinatorial postulate and TAP derivation.
- Closed-form kernel and extinction coupling.
- σ-TAP extension with consummation state `Xi_t`.
- Two-channel consummation update and minimal executable `H` construction.

### 2.1 Layer taxonomy declaration (mandatory)

- **Derived:** TAP kernel/identities.
- **Closure:** conditional drift laws for `(M_t, Xi_t)`.
- **Constitutive:** pressure/boundary observables and chart choices.
- **Interpretive:** physical/domain mapping claims.

Author note: in JTB/Complexity, present this as methodological orientation; in Entropy, convert to explicit table with symbol-level declarations.

## 3) Phase structure of TAP

- Mode A/B/C separatrix framing in `(alpha, mu, M0)`.
- Empirical boundary fitting and sign-reversal diagnosis.
- Structural `a`-dependence of phase occupancy.
- Initialization sensitivity statement using pilot relation:
  - `log(mu) ~ 1.28*log(alpha) + 2.97*log(M0) + C`.

### 3.1 Sign-reversal subsection (methods-relevant)

Frame as method result, not only empirical curiosity:
- Mode A (`M0` policy-coupled) and Mode B (fixed-seed) can induce opposite coefficient signs.
- Therefore, published phase diagrams must include explicit initialization policy.

## 4) σ-TAP empirics

Opening sentence to include verbatim in draft:
> “We present findings in order of **statistical clarity before mechanistic specificity**.”

1. **Precursor-active resolution (cleanest):** all gamma>0 precursor rows resolve to explosive at long horizon.
2. **`a`-gating of gamma effects (structural):** low-`a` too fast, high-`a` often too slow, intermediate windows expose gamma action.
3. **Matched-panel acceleration (composition-corrected):** at `a=8`, matched cells show monotone blow-up acceleration with gamma.
4. **Recruited cells (`gamma: 0 -> 0.2`, `a=8`):**
   - 3 near-boundary amplification,
   - 2 discrete-threshold compensation.
- Inference bundle (single script, three modes): paired sign-flip test for matched timing, paired occupancy-shift test plus bootstrap CI, and Mode A/B coefficient bootstrap CIs (including side-by-side sign-reversal CI separation).

## 5) Limitations and falsifiability

- **Continuous/discrete discrepancy:** Mode-C `M*` baseline can underpredict effective discrete explosive threshold in some slices.
- Reframe as both limitation **and positive test**:
  - limitation: analytic baseline is not exact for discrete dynamics,
  - positive finding: discrepancy is measurable/localizable, so the baseline is empirically falsifiable.
- **Low-gamma weak occupancy shift:** sigma effect is more kinetic (timing) than thermodynamic (large boundary redraw) on pilot grids.
- **Pilot scale caveat:** current sweeps are structured pilots, not final estimators.

## 6) Discussion and deployment framing

- Application instantiation: same closure architecture, domain-specific observables/projections.
- Domain mapping templates: economics/social/environment/evolutionary via projection choices.
- Identifiability gate required before structural causal claims in observed-only `M_t` settings.

## 7) Venue fit and draft length planning

- **JTB:** target ~8k–12k words; scaffold naturally fits this range at normal density.
- **Complexity:** similar structure, but tighten proofs/results prose for concision.
- **Entropy:** include fuller equation/declaration block and explicit taxonomy table up front.

Practical recommendation:
- Start from JTB-form scaffold,
- keep a toggle appendix for Entropy-style formal expansion,
- maintain one shared results core to avoid divergence across submissions.
