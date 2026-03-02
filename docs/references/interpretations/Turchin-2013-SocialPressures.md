# Turchin-2013-SocialPressures — sigma-TAP Interpretation

> Distillation: [../distilled/Turchin-2013-SocialPressures.md](../distilled/Turchin-2013-SocialPressures.md)
> Date interpreted: 2026-03-02
> Project: sigma-TAP

## Project Significance

| Concept (from distillation) | Project Mapping | Relationship |
|----------------------------|----------------|--------------|
| PSI (Ψ = MMP × EMP × SFD) | L-channel pressure decomposition: MMP ↔ L11+L21 (population self-org + env pressure), EMP ↔ L12 (elite projection), SFD ↔ L22 (institutional causal texture) | confirms — the three-compartment structure recapitulates L-matrix decomposition at macro scale |
| Relative wage (w = W/g) | Sigma feedback accumulator: w functions as a slow-moving sigma-like variable that modulates agent behavior across the system | extends — provides a concrete empirical operationalization of how sigma(Xi) accumulates as a distributional metric |
| Elite overproduction (ė = μ₀(w₀/w − 1)) | Endogenous mu (§5.22): social mobility μ inversely related to relative wage is precisely the mechanism proposed for counter-thesis generation driven by innovation activity | confirms — Turchin's μ is the structural-demographic instantiation of sigma-TAP's endogenous mu parameter |
| Relative elite income (ε = (1 − wλ)/e) | Scarcity-abundance jana: ε is an inverse measure of intraelite scarcity; declining ε = increasing scarcity for each elite member as the pool grows | confirms — the Janus-type dipole (scarcity/abundance) operating at the elite stratum, with e and w as the two drivers |
| Five-year wage lag (τ) | Praxitive deferral / differential time dilation (§5.25, §5.55): the lag between structural conditions changing and wages adjusting is an instance of inter-register coupling time | extends — provides empirical evidence (τ ≈ 5 years) for the general principle that different L-channels operate on different timescales |
| Non-market forces (C) | L22 environmental trust (τ_context): institutional/cultural factors that modulate outcomes beyond pure market dynamics correspond to the environmental causal texture channel | confirms — C captures exactly what L22 models: the ground conditions that shape but don't directly participate in agent interactions |
| Labor supply/demand (D/S) | Affordance availability in type_set space: when D > S, the affordance landscape is rich (many opportunities per agent); when S > D, it is impoverished | extends — D/S provides a structural-demographic version of the affordance exposure accumulation (Xi) that sigma-TAP models per-agent |
| Structural pressures vs. triggers | Metathetic time (Π) vs. thetic time (π) in the four time-modes framework (§5.45): structural pressures = slow L22 drift, triggers = fast L11 phase transitions | confirms — Turchin's analytic separation maps directly to differential time dilation architecture |
| Secular cycles | Temporal state machine trajectory (§5.4, Stage 3B): growth → stagnation → crisis → depression maps to expanding → established → declining → inertial states with hysteresis at each transition | extends — the secular cycle pattern provides a macro-scale validation target for the 5-state temporal machine's behavior |
| Multiplicative PSI structure | Simultaneity requirement for paradigm shift: all four time-modes (π, α, σ, Π) must align for system-level phase change (§5.45) | challenges — Turchin uses multiplication (any factor near zero suppresses the whole), while sigma-TAP's paradigm-shift concept requires alignment (phase coherence) rather than amplitude multiplication |
| Elite-population coupling via w | Reciprocity gauge (§5.63): declining w while elite surplus grows = failing reciprocity (extraction without reciprocation); the transaction ceases to be biunimodal | extends — provides the structural mechanism by which reciprocity gauge failure manifests at population scale |
| Modeling philosophy (spectrum of simple models) | ODE/ABM duality in sigma-TAP architecture: ODE layer (continuous.py) = dynamically complete with drastic simplification, ABM layer (metathetic.py) = rich agent detail with exogenous env inputs | confirms — Turchin advocates exactly the dual-model architecture sigma-TAP implements |

## Integration Points

- **Endogenous mu (§5.22)**: Turchin's μ = μ₀(w₀/w − 1) is the strongest empirical precedent for sigma-TAP's planned endogenous mu mechanism. The key structural insight: mu should be inversely related to relative wages (or the sigma-TAP equivalent: relative affordance satisfaction). When agents are relatively well-served by existing type_sets (high w), there is little pressure for novel counter-thesis generation (low mu). When agents are relatively deprived (low w), the surplus flowing to already-successful agents creates pressure for new entrants and novel approaches (high mu). Implementation implication: Phase 4 (endogenous mu) of the Stage 3B plan should consider Turchin's functional form as a starting point.
  - **Candidate forward notes**: §5.22 should reference Turchin 2013 as empirical anchor for the mu(w) relationship.

- **L-channel decomposition of pressure**: The PSI = MMP × EMP × SFD structure suggests that sigma-TAP's L-matrix channels can be used to construct analogous pressure indices at the simulation level. MMP corresponds to aggregate L11+L21 metrics (population-level self-organization and environmental pressure on agents). EMP corresponds to L12 metrics (agent-to-agent projection, competition for positions in the type_set space). SFD corresponds to L22 metrics (environmental causal texture degradation — loss of institutional trust, accumulation of systemic debt).
  - **Candidate forward notes**: The §5.67 forward note on L-channel phase space visualization should include PSI decomposition as a target visualization — plotting MMP-analog, EMP-analog, and SFD-analog trajectories from simulation output.

- **Secular cycle as validation target**: The qualitative shape of Turchin's secular cycle (growth → stagnation → crisis → depression, with ~20-year lag between wage decline and elite explosion) provides a macro-scale validation target for sigma-TAP. If the 5-state temporal machine and sigma feedback loop are correctly implemented, multi-agent simulations run over long time horizons should produce analogous cyclical dynamics in aggregate metrics.
  - **Concept map entries**: `Turchin:SecularCycle` maps to `TemporalStateMachine` in concept map.

- **Five-year lag as coupling time**: The empirically measured 5-year lag between structural conditions and wage response provides a calibration target for the differential time dilation mechanism (§5.25). In sigma-TAP terms, this is the coupling time between the Π-register (L22, slowest) and the π-register (L11, fastest), mediated through the α-σ intermediate registers. The lag should emerge naturally from the temporal state machine's hysteresis thresholds rather than being hard-coded.
  - **Candidate forward notes**: §5.25 should note the Turchin five-year lag as an empirical constraint on coupling-time parameters.

- **Trust metrics and SFD**: Turchin's State Fiscal Distress component uses public distrust in government (Pew Research data) as a direct empirical input. This validates the four-level trust architecture (§5.20): τ_context (L22) corresponds to generalized institutional trust. The Pew distrust data showing cyclical escalation (each peak higher than the last) suggests that trust erosion is ratcheted — partial recovery but never to baseline. This has implications for how sigma-TAP models τ_context: it should exhibit asymmetric dynamics (easier to erode than to rebuild).
  - **Candidate forward notes**: Phase 3 (trust metrics, §5.20) should model τ_context with asymmetric erosion/recovery dynamics, citing Turchin's government distrust data as the empirical pattern.

- **Youth bulge and temporal state**: The A₂₀₋₂₉ component of MMP suggests that agent age/temporal-phase distribution affects system-level instability potential. In sigma-TAP terms, the distribution of agents across the 5 temporal states (expanding, established, declining, inertial, recovering) functions like a demographic age structure. A system with many agents in the "expanding" state (analogous to youth bulge) has higher instability potential than one with a balanced distribution.

## Open Questions

1. **Multiplicative vs. phase-alignment**: Turchin's PSI uses multiplication (any factor near zero suppresses the whole). Sigma-TAP's paradigm shift concept uses phase alignment (all four time-modes must be in phase). Are these structurally equivalent? Multiplication implies amplitude matters; phase alignment implies timing matters. The relationship between these two formalizations needs theoretical work.

2. **Spatial disaggregation**: Turchin acknowledges his model treats the US as a single unit and cannot capture regional variation. Sigma-TAP's topology/family groups (Phase 5, Stage 3B) would address this by allowing differential pressure accumulation across spatial clusters. But the mapping between Turchin's national-level PSI and sigma-TAP's per-agent/per-cluster metrics is not straightforward — at what level of aggregation do structural-demographic dynamics become meaningful?

3. **Non-market forces operationalization**: Turchin proxies the complex of non-market forces with a single variable (real minimum wage). Sigma-TAP's L22 channel is meant to capture this same complexity. But L22 in the current implementation is a single scalar (environmental transition counter). How should L22 be enriched to capture the multidimensional institutional/cultural environment that Turchin's C represents?

4. **Elite overproduction endgame**: Turchin's model shows elite numbers growing explosively but does not model the resolution (how elite overproduction eventually self-limits through civil war, institutional reform, or elite decimation). Sigma-TAP needs a mechanism for this: how does demetathesis resolve? The §5.26 forward note on beneficial vs. detrimental apocalypses is relevant but not yet specified.
