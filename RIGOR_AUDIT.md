# Rigor Audit and Application Plan (Grounded in Four PDFs)

## Sources used
1. `TAPequation-FINAL.pdf` — formal TAP behavior: plateau, explosive divergence, blow-up approximations, extinction-innovation balance, and two-scale variant framing.
2. `Paper1-FINAL.pdf` — biocosmology motivation: expanding biological configuration spaces and entropy/state-accounting implications.
3. `Paper2-FINAL.pdf` — methodological layer: non-ergodic Type-III systems, mixed explanatory modes, and fourth-law framing.
4. `Applications-of-TAP.pdf` — application-forward TAP synthesis emphasizing deployable domains and variant families (two-scale/differential/logistic).

---

## A) Rigor audit of `COMPLETE_SYSTEM_ARTICULATION.md`

### A1. What is already rigorous
- Standard TAP derivation and kernel closed form are algebraically consistent.
- Distinction between derived equations, closures, constitutive terms, and interpretive physical mapping is explicitly present.
- Projection-first architecture and micro-event identity are clearly defined.
- New unambiguity statement correctly frames `(M_t, Xi_t)` equations as conditional-drift/mean-field closures of `M_{t+1}-M_t=B-D`.

### A2. Remaining rigor gaps to close
1. Equation-by-equation status tags are still implicit in places.
2. Admissibility assumptions (measurability, bounded moments, regularity of `sigma`, denominator conventions for `Lambda_t`) are not fully formalized.
3. Identifiability/calibration workflow must be declared before layer-wise falsification; otherwise architecture and calibration failures are confounded.
4. Validation metrics against empirical/simulated trajectories are not standardized.
5. Pressure layer remains constitutive (acceptable) but inverse inference must be policy-constrained (forward prediction allowed; inverse causal claims require extra identification assumptions).

---

## B) How to apply this in practice (modular simulator + table datasets)

## B0. Identifiability gate (must run before B3)

Before layered validation, determine what can be identified from observables:
- If only `M_t` is observed: only reduced-form forecasting claims are admissible.
- If `B_t,D_t` are observed (or reliably reconstructed): layered falsification is admissible.
- If `Xi` is latent, require a declared `Xi` proxy before structural claims on `(beta, eta)`.

Minimal gate table:
- **Observed `{M_t}`** → fit possible; architecture-vs-calibration not separable.
- **Observed `{M_t, Xi_proxy}`** → partial separation; test improvement from `Xi` channel.
- **Observed `{M_t, B_t, D_t}`** → strongest testability for closure layers.

### B0.1 Xi plateau threshold policy

`xi_plateau` must be run-normalized (not global). Recommended default:
\[
\xi_{\mathrm{plateau}} = \max\left(c\,\mathrm{median}\left(|\Delta \Xi/\Delta t|\right)_{\mathrm{warmup}},\;\epsilon_{\mathrm{rel}}\,\max(\Xi_{\mathrm{warmup}})\right)
\]
with warmup window from early trajectory and constants `(c, epsilon_rel)` recorded per run.

This avoids meaningless cross-sweep fixed thresholds when `(beta, eta)` rescale the Xi-channel.

## B1. Minimal modular architecture
- `state/` : microstate representation `X=(X_imp, X_exp)` and evolution operator `R`.
- `projection/` : feature map `Psi`, thresholding, projection `Phi`, and event extractors `B,D`.
- `tap/` : `f(M)` and baseline TAP dynamics.
- `sigma_tap/` : closures for `sigma(Xi)`, one-channel/two-channel `Xi` updates.
- `pressure/` : optional `pi_k` observables and aggregators `A,O,Pi`.
- `simulate/` : scenario runners, sweeps, seeds, reproducibility controls.
- `analysis/` : metrics, closure errors, and export to data tables.

## B2. Canonical table schemas
1. `runs`
   - `run_id, seed, model_variant, alpha, a, mu, beta, eta, sigma_form, pressure_form, config_hash`.
2. `timeseries`
   - `run_id, t, M_t, Xi_t, B_t, D_t, f_t, sigma_t, Lambda_t, Delta_cond_t, G_cond_t`.
3. `pressure_timeseries` (optional)
   - `run_id, t, pi_m4,...,pi_p4, A_t, O_t, Pi_t`.
4. `summary`
   - `run_id, plateau_len, growth_regime, blowup_proxy_time, extinction_fraction, closure_rmse, drift_bias`.

## B3. Baseline validation protocol
- Step 0 (**required**): run B0 identifiability gate and record admissible claim class.
- Step 1: Fit/check TAP-only drift against observed `(M_t)`.
- Step 2: Add `Xi_t` closure; evaluate residual reduction **conditional on observability class**.
- Step 3: Add pressure chart for forward prediction; test out-of-sample gain.
- Step 4: Quantify uncertainty using seed ensembles and parameter bootstraps.
- Step 5: If inverse causal pressure claims are attempted, require explicit identification assumptions and sensitivity analysis.

### B3.1 Budget line with required Pass C

If Pass C is mandatory, include it explicitly in compute planning:
\[
N_{\mathrm{PassC}} = k\times s\times n_{\mathrm{LHS}}\times r
\]
For `(s=3, n_LHS=150, r=24)`,
\[
N_{\mathrm{PassC}} = k\times 10{,}800.
\]
At `k=9`, this is `97,200` additional runs.

Interpretation: Pass C can more than double baseline load, so resource budgeting must be committed before sweep launch.

---

## C) Alignment with the four papers

- From `TAPequation-FINAL.pdf`: simulator must reproduce plateau/explosive regimes and extinction-sensitive transitions.
- From `Paper1-FINAL.pdf`: keep non-ergodic growth accounting explicit; report realized-vs-possible growth proxies.
- From `Paper2-FINAL.pdf`: preserve mixed explanatory stack (micro + functional observables), do not collapse everything to a 2D law.
- From `Applications-of-TAP.pdf`: implement domain-ready mappings and compare TAP variants (baseline vs two-scale/differential/logistic) inside one simulation harness.

---

## D) Skills that would be useful to add

1. `math-model-rigor-auditor`
   - Enforces theorem-vs-closure labels, notation collision checks, admissibility assumptions, and identifiability tests.
2. `dynamics-sim-dataset-scaffold`
   - Generates modular simulation skeletons plus standardized tabular schemas and benchmark experiments.
3. `paper-grounding-linker`
   - Produces traceable claim-to-source maps from local PDFs to model sections.

These are not currently available in-session; creating them would materially improve velocity and consistency.


---

## E) Application mapping and dataset design (TAP use-cases)

Grounded in `Applications-of-TAP.pdf` and `TAPequation-FINAL.pdf`, the same simulator can be instantiated across economics, social systems, environmental change, evolutionary biology, and law/physical-law contexts by swapping only the feature map/projection layer (and optionally variant form).

### E1. Domain-to-model mapping
- **Economics/innovation**: realized objects = products/patents/modules; births = newly introduced combinations; deaths = obsolescence/market exit.
- **Social systems**: realized objects = practices/memes/institutions; births = newly stabilized combinations; deaths = abandonment.
- **Environmental change**: realized objects = viable ecological configurations; births/deaths from threshold crossings in resilience indicators.
- **Evolutionary biology**: realized objects = functional proteins/traits/assemblies; births/deaths via emergence/extinction events.

### E2. Standardized experiment outputs across domains
Keep table schemas identical (`runs`, `timeseries`, optional `pressure_timeseries`, `summary`) so cross-domain comparisons are possible.

### E3. Cross-domain KPIs
- plateau duration before acceleration,
- extinction-adjusted innovation rate,
- closure error (mean-field drift mismatch),
- realized-to-possible ratio proxy,
- regime classification frequency across parameter sweeps.

### E4. Reproducibility requirements
For each dataset release include: parameter manifest, seed list, projection definition (`Psi`, threshold `epsilon`), and closure-form declaration (`sigma` form and whether `Lambda` is ratio-defined or constitutive).


---

## F) Repository verification log (re-checked)

To avoid accidental omission, the repository was re-scanned multiple times for PDF sources before updating this audit. The currently detected PDF set is:
- `TAPequation-FINAL.pdf`
- `Paper1-FINAL.pdf`
- `Paper2-FINAL.pdf`
- `Applications-of-TAP.pdf`

`Applications-of-TAP.pdf` is now incorporated as an application-forward source. It appears to overlap TAP-equation material while adding stronger deployment emphasis. The same mapping template in Sections C/E remains the required protocol for any further source additions:
1. claim-to-source extraction,
2. architecture impact assessment (core/closure/constitutive layer),
3. dataset/KPI deltas.

This preserves source-traceability rigor while enabling practical expansion.



## G) Resolved reviewer-style concerns (latest pass)

1. **Constraining \(H(\mathbf X_t)\)**: now addressed by an explicit construction principle in `COMPLETE_SYSTEM_ARTICULATION.md` Section 8.1 using implicate code-length/entropy-proxy decrease with admissibility constraints and a minimal-complexity selection rule.
2. **RIP canonicity**: now addressed by proposing a hybrid information geometry (discrete event layer + Fisher-metric closure manifold) to define orthogonal projectors canonically (Section 12.1).
3. **Nine-pressure motivation**: now addressed as a finite signed basis choice \(9=1+2\times 4\), with three principled generators and an empirical pruning rule (Section 10.1).

These upgrades keep the system non-reductive while reducing underdetermination.


## H) Current simulator implementation state

The repository now contains an initial simulator scaffold:
- `simulator/state.py`
- `simulator/projection.py`
- `simulator/tap.py`
- `simulator/sigma_tap.py`
- `simulator/pressure.py`
- `simulator/simulate.py`
- `simulator/analysis.py`
- `simulator/hfuncs.py`
- `scripts/run_demo.py`

This is an MVP implementation of the design layer (not yet a full empirical pipeline).


### H1. Minimal end-to-end core-loop test (now implemented)

A single-run, non-sweep trajectory test is now implemented in `scripts/run_demo.py`:
- executes `run_sigma_tap(...)` end-to-end,
- instantiates `H` via `h_compression(...)`,
- outputs trajectory data for `(M_t, Xi_t)` plus an explicit `xi_traj` vector,
- includes terminal-state row to avoid off-by-one reconstruction ambiguity.

This is the first compositional proof that the design-layer core equations run with a concrete `H` channel.


### H2. Regime classifier + boundary pilot

Implemented `classify_regime(...)` and `find_fixed_point(alpha, mu, a)` in `simulator/analysis.py`, and a cheap boundary pilot sweep script `scripts/sweep_alpha_mu.py` (10x10 grid over `alpha in [1e-5,1e-2]`, `mu in [1e-3,1e-1]`, fixed `a=8`). The sweep now auto-initializes just above criticality with `M0 = 1.05*M*` (fallback `M0=3` only if no root).

Current pilot output with above-threshold initialization now shows a visible transition structure: `exponential` 34, `explosive` 53, `precursor-active` 3, `plateau` 10, with `under_resolved=True` reduced to 27/100. This confirms the boundary is strongly initialization-sensitive and is best treated as a surface in `(alpha, mu, M0)`.


### H3. Empirical phase-boundary fit

Implemented a lightweight logistic boundary fit on `(log(alpha), log(mu))` in `simulator/analysis.py` via `fit_explosive_logistic_boundary(...)`, with executable entrypoint `scripts/fit_boundary.py`.

Current fit on `outputs/sweep_alpha_mu.csv` (positive class = `{explosive, precursor-active}`):
- train accuracy: `0.96`
- coefficients (standardized feature space):
  - intercept: `1.1011`
  - log_alpha: `-2.1782`
  - log_mu: `5.4427`

This is a first quantitative phase-diagram summary from the simulator and should be treated as exploratory (small-grid pilot, initialization-dependent surface).


### H4. 3D manifold framing and dual-slice policy

Phase boundaries are estimated on the 3D parameter-initialization manifold `(alpha, mu, M0)`. Published 2D diagrams are projections/slices under explicit initialization policies.

- **Mode A (near-critical)**: `M0 = 1.05*M*(alpha,mu)` — physically interpretable for critical-operation hypotheses and separatrix probing.
- **Mode B (fixed-seed)**: `M0 = c` — geometrically cleaner 2D boundary for deployment/forecasting from common starting seeds.
- **Mode C (critical-point trace)**: analytical `M*`-isocurves (`M*(alpha,mu)=c`) from fixed-point equation (no simulation required).

Added 3D logistic fit (`log alpha`, `log mu`, `log M0`) and emitted `outputs/boundary_fit_3d.json` to quantify initialization sensitivity via coefficient on `log M0`. Mode A has intentional covariate coupling because `M0` is policy-defined from `(alpha,mu)`, so Mode B is required for independent identification of `c3`.


### H5. Mode B identifiability run (fixed M0)

To break Mode-A collinearity (`M0` as deterministic function of `(alpha,mu)`), a Mode B sweep was added and executed with fixed seeds `M0 in {10,20,50}` over the same 10x10 `(alpha,mu)` grid.

Artifacts:
- `outputs/sweep_mode_b.csv` (300 rows)
- `outputs/boundary_fit_mode_b_3d.json`

Observed regime counts:
- `plateau`: 214
- `exponential`: 11
- `precursor-active`: 5
- `explosive`: 70

3D logistic fit on Mode B (positive `{explosive, precursor-active}`):
- train accuracy: `0.9733`
- coefficients: `intercept=-4.2176`, `log_alpha=4.1925`, `log_mu=-2.1876`, `log_m0=3.5232`

Interpretation: `log_m0` remains large and positive when varied independently, confirming initialization sensitivity as a genuine effect rather than only Mode-A collinearity.

Sign flip note (Mode A vs Mode B): the coefficient-direction reversal (`log_alpha, log_mu`) is expected and substantive. In Mode A, `M0=1.05*M*(alpha,mu)` induces covariate coupling so `(alpha,mu)` partly proxy initialization scale; in Mode B (fixed seeds), this confound is removed and the direct effect appears (`alpha` promotes explosive outcomes, `mu` suppresses them).

Back-transforming Mode-B standardized coefficients using feature standard deviations in `outputs/boundary_fit_mode_b_3d.json` gives approximate natural-log slopes:
- `log(alpha)` slope: `+4.1925 / 2.216 ~ +1.89`
- `log(mu)` slope: `-2.1876 / 1.477 ~ -1.48`
- `log(M0)` slope: `+3.5232 / 0.801 ~ +4.40`

So an empirical boundary relation is
`log(mu) ~ 1.28*log(alpha) + 2.97*log(M0) + C`,
indicating initialization scale can be at least as consequential as extinction pressure on this pilot grid.

Design caveat: Mode B with global seeds `{10,20,50}` is intentionally decoupled but unevenly populates the manifold (many low-seed cells remain plateau). This does not invalidate Mode B; it means Mode A and Mode B are complementary slices of the same 3D separatrix and should be reported jointly, with Mode C isocurves as analytic baseline.

### H6. σ-feedback activation sweep (`gamma>0`) and `a`-sensitivity slice

To activate the actual σ-TAP channel (rather than TAP-equivalent `gamma=0`), a combined Mode-B sweep was run with:
- `gamma in {0.0, 0.05, 0.2}`
- `a in {2,4,8,16}`
- `M0 in {10,20,50}`
- same 10x10 `(alpha,mu)` grid and existing classifier pipeline.

Artifacts:
- `scripts/sweep_sigma_feedback.py`
- `outputs/sweep_sigma_feedback.csv` (3600 rows)
- `outputs/sweep_sigma_feedback_summary.json`

Pilot findings:
- aggregate regimes: `plateau=2100`, `exponential=73`, `precursor-active=37`, `explosive=1390`
- by `gamma`: explosive rows rise modestly (`460 -> 461 -> 469`) while plateau rows fall (`702 -> 702 -> 696`) as `gamma` increases from `0` to `0.2`.
- by `a`: explosive share decreases strongly with larger `a` (`a=2` highest explosive counts; `a=16` lowest), showing phase geometry is materially `a`-dependent on this grid.

Interpretation: the σ-feedback channel is now empirically activated; on this pilot grid it shifts boundary occupancy modestly but in the expected direction (higher `gamma` increases explosive-side occupancy), while `a` has a large first-order effect.

### H7. Precursor-active longitudinal follow-up

To test whether `precursor-active` is a finite-horizon artifact, all precursor rows from the gamma-activated sweep were re-run with longer horizons (`steps_long=max(600,4*steps_short)`).

Artifacts:
- `scripts/followup_precursor_longitudinal.py`
- `outputs/precursor_longitudinal.csv`
- `outputs/precursor_longitudinal_summary.json`

Result on current pilot:
- short-horizon precursor rows analyzed: `24`
- long-horizon labels: `explosive=24`
- long-horizon under-resolved: `0`

Interpretation: on this configuration, precursor-active behaves as a transitional class that resolves to explosive under longer integration, not a stable terminal regime.

### H8. Overflow-safe observability (cap + blow-up step)

To avoid propagating `inf` through outputs, the core simulator now caps trajectory states at finite ceilings (`m_cap`, `xi_cap`, default `1e9`) and emits overflow observables:
- `overflow_detected` (row-level)
- `blowup_step` (first step where overflow/cap was triggered)

This preserves an interpretable blow-up proxy while keeping downstream CSV/JSON aggregation numerically stable.

### H9. High-`gamma` threshold probe (fixed `a=8`, `M0=20`)

To test whether the weak low-`gamma` response was range-limited, a targeted sweep was run with:
- `gamma in {0.5,1.0,2.0,5.0}`
- fixed `a=8`, `M0=20`
- same 10x10 `(alpha,mu)` grid.

Artifacts:
- `scripts/sweep_gamma_threshold.py`
- `outputs/sweep_gamma_threshold.csv`
- `outputs/sweep_gamma_threshold_summary.json`

Observed counts by `gamma`:
- `0.5`: `plateau=79`, `exponential=1`, `explosive=20`
- `1.0`: `plateau=77`, `exponential=1`, `explosive=22`
- `2.0`: `plateau=77`, `exponential=1`, `explosive=22`
- `5.0`: `plateau=73`, `precursor-active=1`, `explosive=26`

Blow-up timing proxy (mean `blowup_step`) decreases with `gamma`:
- `0.5`: `59.45`
- `1.0`: `55.23`
- `2.0`: `42.14`
- `5.0`: `36.62`

Interpretation: on this fixed-seed slice, higher `gamma` does become regime-shifting at the high end and clearly accelerates blow-up timing even when coarse regime occupancy shifts are moderate.

### H10. Blow-up interaction cross-tab by `(a, gamma)`

To test the proposed three-way interaction mechanism, we computed a cross-tab of blow-up timing from `outputs/sweep_sigma_feedback.csv` using the emitted `blowup_step` field.

Artifacts:
- `scripts/analyze_blowup_interaction.py`
- `outputs/blowup_interaction_by_a_gamma.csv`
- `outputs/blowup_interaction_by_a_gamma.json`

Per-cell means (`rows_with_blowup / 300`, `mean_blowup_step`):
- `a=2`: `208-209/300`, mean step about `6.0-6.23` (very fast blow-up; little room for gamma timing effects)
- `a=4`: `149-150/300`, mean step about `9.97-10.22`
- `a=8`: `73-78/300`, mean step about `20.73-24.0` (slower, broader timing window)
- `a=16`: `30-32/300`, mean step about `42.56-47.33` (few blow-ups, long delays)

Interpretation: the cross-tab supports the conditional-loading picture qualitatively (`a` controls whether there is time for σ-feedback to act), but the gamma-gradient on mean blow-up time is not monotone at every `a` on this pilot grid. The strongest empirical signal remains:
1) `a` sets baseline blow-up propensity/timing,
2) higher `gamma` increases explosive occupancy modestly,
3) high-`gamma` fixed-seed probes show clearer acceleration in blow-up timing.

### H11. Matched-panel check for `a=8` non-monotonicity

To test whether `a=8` non-monotonic mean blow-up timing was a composition artifact, we ran a matched-panel analysis over identical cells `(alpha,mu,m0)` across `gamma in {0,0.05,0.2}`.

Artifacts:
- `scripts/analyze_matched_blowup_panel.py`
- `outputs/blowup_matched_panel_a8.csv`
- `outputs/blowup_matched_panel_a8_summary.json`

Result:
- candidate cells at `a=8`: `300`
- blow-up cells: `73` at `gamma=0`, `78` at `gamma=0.2`
- newly recruited at `gamma=0.2`: `5` (with `73` retained)
- matched-panel size (blow-up present in all three slices): `73`
- matched means (blow-up step):
  - `gamma=0`: `21.14`
  - `gamma=0.05`: `20.73`
  - `gamma=0.2`: `19.56`
  - monotone non-increasing: `True`

Interpretation: the apparent rise in the unmatched `a=8` mean at `gamma=0.2` is a survivor/composition effect from recruiting slower borderline blow-up cells. On the matched panel, gamma shows clean acceleration.

### H12. Recruited-cell extraction (`a=8`, `gamma: 0 -> 0.2`)

To isolate the five border-crossing events, we extracted cells in `blow2 - blow0` (blow up at `gamma=0.2` but not at `gamma=0`) and compared each to the Mode-C isocurve proxy `mu_iso(alpha | M*=M0, a=8)`.

Artifacts:
- `scripts/extract_recruited_cells.py`
- `outputs/recruited_cells_a8.csv`
- `outputs/recruited_cells_a8_summary.json`

Result:
- recruited cells: `5`
- near-boundary by `|log(mu/mu_iso)| < 0.5`: `3/5`
- mechanism split:
  - `near_boundary_amplification`: `3`
  - `discrete_threshold_compensation`: `2`

Near-boundary cases are textbook marginal trajectories (`M0/M* ~ 0.99-1.04`, `mu/mu_iso ~ 0.82-1.06`) tipped by persistent sigma amplification.

The two compensation cases are:
- `(alpha=0.001, mu=0.001, M0=20)` with `M0/M*=1.74`
- `(alpha=0.00464, mu=0.001, M0=10)` with `M0/M*=2.41`

These are above the continuous fixed-point threshold but still fail to escape at `gamma=0`, then blow up at `gamma=0.2` (steps `152`, `158`). We therefore label them **discrete-threshold compensation** rather than seed-deficit events.

Limitation (new): continuous Mode-C (`M*`) underestimates the effective explosive threshold for this discrete implementation in parts of parameter space; this discrepancy should be reported explicitly when using analytic isocurves as boundary surrogates.


## I) Paper-ready empirical synthesis mapped to layer taxonomy

This section compresses H6-H12 into the framework’s layer classes so manuscript claims stay category-correct.

1. **Derived layer (TAP kernel):**
   - Combinatorial kernel structure is treated as the theoretical base; empirical runs do not re-derive it, they test closure consequences under parameterized scenarios.

2. **Closure layer (σ-TAP dynamics):**
   - `gamma>0` activation confirms σ-channel has measurable effects.
   - Dominant effect is kinetic: stronger acceleration of blow-up timing than large occupancy shifts at low/moderate gamma.
   - At high gamma (`0.5..5`), occupancy shifts become clearer while timing acceleration remains the strongest signal.

3. **Constitutive-observational layer (phase boundary as measured object):**
   - Regime maps and logistic separators are empirical summaries on chosen observables/slices (Mode A/B/C), not theorem-level boundaries.
   - Structural parameter `a` strongly deforms boundary occupancy (largest first-order effect in current pilot).

4. **Secondary closure finding (recruitment mechanisms):**
   - Recruited `a=8` cells (`gamma:0 -> 0.2`) split into:
     - near-boundary amplification (`3/5`), and
     - discrete-threshold compensation (`2/5`).

5. **Interpretive/limitations layer:**
   - Precursor-active behaves as transient loading in current runs (`24/24` resolve explosive at long horizon).
   - Continuous Mode-C fixed-point baseline can underestimate effective explosive threshold in discrete implementation for some slices; this is an explicit limitation, not a contradiction.

Suggested manuscript phrasing:
> “Empirically, σ-feedback acts primarily as a kinetic amplifier on the closure dynamics, with modest low-gamma boundary recruitment and stronger timing acceleration. Boundary geometry is chiefly set by `(alpha, mu, a, M0)` under explicit constitutive slicing policy, while discrete-map effects induce measurable deviations from continuous fixed-point surrogates in selected parameter regions.”

### H13. Inferential support bundle (single script, three modes)

To support manuscript p-value / CI statements without over-expanding scope, a single inference script was added with three analysis modes:
- `matched`: paired timing test on matched blow-up panel (`a=8`, `gamma:0 -> 0.2`),
- `occupancy`: paired occupancy shift + bootstrap CI on proportion difference,
- `coefficients`: bootstrap CIs for Mode-A vs Mode-B boundary coefficients (plus Mode-B 3D).

Artifact/script:
- `scripts/inferential_stats.py`
- `outputs/inferential_stats.json`

Current run (`--mode all --n-boot 200 --n-perm 500 --n-boot-coef 40`) reports:
- matched timing: mean blow-up step `21.14 -> 19.56`, paired sign-flip `p ~= 0.002`.
- occupancy shift (full paired 300-cell panel): explosive/precursor fraction `0.260 -> 0.273`, difference `0.0133` with bootstrap 95% CI `[0.0033, 0.0267]`; paired sign-flip `p ~= 0.150`.
- coefficient CIs: Mode-A vs Mode-B 2D `log_alpha` and `log_mu` CIs are non-overlapping (`True/True`), strengthening the sign-reversal methodological claim.

Interpretation: this closes the main inference gap for current manuscript scope. Strongest inferential signal is timing acceleration; occupancy shifts are modest at low gamma and should be described as such.
