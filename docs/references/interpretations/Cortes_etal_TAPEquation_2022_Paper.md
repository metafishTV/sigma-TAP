# Cortes_etal_TAPEquation_2022_Paper — sigma-TAP Interpretation

> Distillation: [../distilled/Cortes_etal_TAPEquation_2022_Paper.md](../distilled/Cortes_etal_TAPEquation_2022_Paper.md)
> Date interpreted: 2026-03-04
> Project: sigma-TAP

## Project Significance

This is the **foundational equation paper** for the sigma-TAP project. Every parameter, every growth regime, and every structural result maps directly to the simulator's architecture. The re-distillation clarifies the precise formal content and separates project-specific readings from neutral extraction.

| Concept (from distillation) | Project Mapping | Relationship |
|----------------------------|----------------|--------------|
| TAP equation (Eq. 1) | `tap.py` → `compute_birth_term()` — power-law variant | confirms |
| Adjacent Possible | Core ontological claim — TAP operationalizes what TAPS theorizes at the agent level | confirms |
| Blow-up | `m_cap` mechanism in state.py — operational ceiling preventing overflow | confirms |
| $t_\text{add one}$ (add-one time) | Analytic validation target for ODE solver in `continuous.py` | confirms |
| Hockey-stick dynamics | Characteristic output of all sigma-TAP simulations — validated in 8 publication figures | confirms |
| Extinction instability | Maps to TAPS constraint analysis: praxis (P) completion = structural binary (blow-up or collapse) | extends |
| Constant α case | Base case in `tap.py` — used for theoretical validation, not primary simulations | confirms |
| Power-law α case | Primary case in `sigma_tap.py` — $\alpha$, $a$, $\mu$ are the three governing parameters | confirms |
| Tetration | Extreme growth regime requiring `m_cap` (default 1e9, operational 1e4 for explosive params) | confirms |
| Universality (Markov property) | The bare equation is history-independent; sigma-TAP *adds* non-Markovian memory via L-matrix and TAPS signatures | extends |
| Two-scale TAP | Partially mapped: $\alpha_1$ separation not yet implemented as distinct parameter, but family groups (Stage 3B) create analogous two-rate dynamics | extends |
| Differential TAP | Not implemented — conceptual variant only | novel (low priority) |
| Logistic TAP (unsolved) | **σ-TAP's turbulence mechanism serves this role** — scarcity-mediated feedback as the logistic moderator the paper identifies as open | extends |
| Continuous blow-up integral | Could inform `continuous.py` ODE formulation — rigorous blow-up time from $\int dM/F[M]$ | extends |
| Stochastic vs deterministic | sigma-TAP uses deterministic TAP + stochastic agent-level structure (L-matrix events) — a third path not discussed in the paper | extends |

## Integration Points

- **Power-law TAP as the project's core engine**: The simulator implements Eq. (8) directly. Parameters $\alpha = 5 \times 10^{-3}$, $a = 3.0$ or $8.0$, $\mu = 0.005$ are the project's standard values. The analytic blow-up estimate $t_\text{blow-up} \approx a/[\alpha(M_0-1)]$ provides validation targets for numerical output.
  - **Candidate forward note**: Eq. (12) should be checked against the ODE solver's blow-up times across the parameter space (empirical_sweep.py).

- **Extinction instability → metathetic fragility**: The paper's proof that $\mu_\text{critical}$ equilibrium is unstable maps to a deeper claim in sigma-TAP: without the σ feedback, TAP systems have no stable intermediate between extinction and blow-up. The σ-field (affordance-exposure accumulation) is precisely what creates the stable intermediate — it is the project's answer to the logistic TAP open problem.
  - **Concept map entry**: `Cortes_etal:extinction_instability` → maps to σ-field necessity (the structural reason sigma-TAP needs σ).

- **Universality = Markov property, broken by L-matrix**: The paper's universality result means the bare TAP equation is memoryless. sigma-TAP's per-agent L-matrix ledger (Stage 3A) and TAPS signatures (cross-metathesis classification) deliberately break this symmetry — agents carry history that shapes their combinatorial future. This is a *designed departure* from the source equation.
  - **Concept map entry**: `Cortes_etal:universality` → maps to L-matrix non-Markovian extension.

- **Two-scale TAP → family groups**: The paper's two-scale variant ($\alpha_1 M_t$ + combinatoric) describes a regime where single-object evolution dominates initially. In sigma-TAP, family groups (Stage 3B) create analogous dynamics: within-family evolution ($\alpha_1$-like) is faster than between-family combination ($\alpha$-like). The exponential-to-blow-up transition is the regime sigma-TAP must navigate.
  - **Candidate forward note**: Implement explicit $\alpha_1$ parameter for within-family evolution rate, separate from cross-family $\alpha$.

- **Continuous blow-up integral → ODE solver**: The published version's Eq. (16') [$t_\text{blow-up} = \int dM/F[M]$] provides a rigorous blow-up time that could be computed analytically for the power-law case and compared to `continuous.py`'s ODE solver output. This would give an independent validation of the solver's blow-up behavior.
  - **Candidate forward note**: Add analytic blow-up integral computation to `analysis.py` as a validation check.

- **Logistic TAP = σ-TAP's open frontier**: The paper explicitly identifies the logistic variant (resource competition moderating blow-up) as unsolved. sigma-TAP's turbulence mechanism — scarcity-mediated feedback where agents compete for affordance-exposure — is an agent-level implementation of exactly this idea. Whether it truly moderates blow-up (vs. merely delaying it) is a key empirical question for the project.
  - **Concept map entry**: `Cortes_etal:logistic_TAP` → maps to σ-field turbulence, scarcity_operator.

- **Unforeshadowability → practical implications**: The paper's strongest applied claim — that TAP blow-up is essentially unpredictable from pre-blow-up behavior — has direct implications for sigma-TAP's output interpretation. The simulator's hockey-stick curves should be presented with this caveat: the apparent plateau phase gives no information about how much time remains before blow-up.

## Open Questions

1. **Continuous blow-up integral implementation**: Should `continuous.py` compute the analytic integral $\int dM/F[M]$ as a comparison to the ODE solver's numerical blow-up time? This would provide a rigorous validation target.

2. **Explicit two-scale parameter**: Should sigma-TAP implement $\alpha_1$ as a separate parameter (single-object evolution rate) distinct from the combinatoric $\alpha$? Currently, family groups approximate this behavior structurally rather than parametrically.

3. **Stochastic-deterministic relationship**: The paper notes that nonlinearity breaks the equivalence between mean stochastic and deterministic TAP. sigma-TAP uses deterministic TAP with stochastic agent events — is this a distinct third regime? What are its mathematical properties?

4. **Logistic TAP formal status**: sigma-TAP's turbulence mechanism is an *agent-level* answer to the logistic TAP question. But does it actually solve the *equation-level* problem? Could the σ-modulated TAP equation be written in closed form and analyzed for whether it prevents blow-up or merely delays it?

5. **Tetration and computational limits**: The paper's tetration result means that TAP systems produce numbers that exceed any fixed-precision representation in very few steps. The project's `m_cap` mechanism is a practical workaround. Is there a principled way to handle the post-blow-up regime (what does the system look like *after* it TAPs)?
