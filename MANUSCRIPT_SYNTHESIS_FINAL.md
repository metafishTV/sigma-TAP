## σ-TAP: Path-Dependent Efficiency in the Theory of the Adjacent Possible

---

## 1. Introduction

Complex systems across biology, economics, and social organization share a structural feature: the arrival of new functional components creates opportunities for further combinations that did not previously exist. A technology enables new technologies; a protein fold enables new protein interactions; an institutional form enables new institutional arrangements. The space of what is possible expands as a function of what has been realized, and the expansion is faster than linear. This is the adjacent possible, and it produces a characteristic phenomenology — long periods of near-stasis punctuated by episodes of rapid combinatorial expansion — that recurs across domains too diverse to be explained by domain-specific mechanisms alone.

The Theory of the Adjacent Possible (TAP) formalizes this observation as a discrete dynamical system in which the count of realized objects $M_t$ evolves under a balance between combinatorial innovation and extinction. The innovation kernel grows super-exponentially in $M_t$, producing a sharp transition between plateau and explosive regimes separated by an unstable fixed point $M^*$. TAP has been applied to biological complexity, economic innovation, and the thermodynamics of open-ended systems, and its qualitative phenomenology — plateau, sharp divergence, extinction sensitivity — is well established analytically. What has been less systematically studied is the phase structure of TAP as a surface in full parameter-initialization space, the role of path-dependent efficiency modulation in shifting or accelerating that structure, and the relationship between continuous analytical predictions and discrete simulation behavior.

This paper addresses three gaps. The first is architectural: base TAP has no memory. Innovation rates depend only on the current count $M_t$, not on the path by which that count was reached. This excludes learning effects, capability accumulation, and mechanisms by which prior innovation reshapes the ease of subsequent innovation. We introduce $\sigma$-TAP, a minimal extension that adds a consummation state $\Xi_t$ accumulating through realized births and implicate densification, with a linear efficiency map $\sigma(\Xi_t)$ modulating the innovation kernel. The extension is minimal in that it adds one state variable and one feedback parameter $\gamma$, recovers standard TAP at $\gamma = 0$, and uses a projection-first architecture that makes closure assumptions explicit rather than implicit.

The second gap is methodological. Published TAP-family phase diagrams implicitly depend on initialization policy — the rule by which $M_0$ is chosen — but this dependence is rarely declared and its consequences are under-characterized. We show that two natural initialization policies (near-critical initialization, Mode A, and fixed-seed initialization, Mode B) can yield phase-boundary coefficients with opposite signs for the same system, with non-overlapping bootstrap confidence intervals. The sign reversal is not a numerical artifact but a structural consequence of covariate coupling in the near-critical design. Phase diagrams are constitutive summaries under a named initialization policy, not intrinsic geometric objects, and cross-paper comparisons require that policy to be declared.

The third gap is empirical. We characterize when $\sigma$-feedback shifts the phase boundary versus accelerating trajectories already inside it, identify the gating role of combinatorial difficulty $a$, confirm precursor-active as a genuine transient loading state rather than a classifier artifact, and localize a measurable discrepancy between continuous fixed-point predictions and discrete simulation behavior. Claims are backed by committed simulation artifacts and, where appropriate, formal inferential tests.

The paper is organized as follows. Section 2 presents the model with an explicit four-layer taxonomy separating derived results, closure assumptions, constitutive observables, and interpretive mappings. Section 3 develops phase structure across three initialization modes. Section 4 reports $\sigma$-TAP empirical findings. Section 5 discusses limitations and falsifiability. Section 6 addresses deployment framing and application instantiation.

---

## 2. Model

### 2.1 Combinatorial postulate and TAP derivation

Let $M_t \in \mathbb{N}$ denote the number of realized objects present at discrete time $t$. We adopt three primitive assumptions: (i) realized objects can participate in forming higher-order combinations; (ii) larger combinations are harder to realize, with difficulty growing geometrically in combination order; and (iii) a fraction of realized objects becomes inactive or obsolete at each step. These assumptions are model primitives, not derived statements. Everything that follows is either a consequence of these primitives (derived), a conditional-drift approximation that closes the dynamics on observable quantities (closure), or an additional observational chart layered atop the closed dynamics (constitutive). We make this taxonomy explicit in Section 2.3 and maintain it throughout.

Three dimensionless parameters characterize the postulate: $\alpha > 0$ is the base realization coefficient, $\mu \in [0,1]$ is the extinction or obsolescence fraction per step, and $a > 1$ is the difficulty ratio suppressing higher-order combinability.

For combination order $i$, candidate combinations number $\binom{M_t}{i}$, weighted by realization rate $\alpha/a^{i-1}$. Summing over nontrivial orders and adding surviving objects:

$$M_{t+1} = M_t(1-\mu) + \sum_{i=2}^{M_t} \frac{\alpha}{a^{i-1}} \binom{M_t}{i}. \tag{TAP-1}$$

The second term defines the innovation kernel $f(M) := \sum_{i=2}^{M} \frac{\alpha}{a^{i-1}} \binom{M}{i}$, giving

$$M_{t+1} = M_t(1-\mu) + f(M_t). \tag{TAP-2}$$

Setting $x = 1/a$ and applying the binomial theorem:

$$f(M) = \alpha a \left[ \left(1 + \frac{1}{a}\right)^M - 1 - \frac{M}{a} \right]. \tag{TAP-3}$$

This is an algebraic identity requiring no approximation. For large $M$, $f(M) \sim \alpha a (1+1/a)^M$ (super-exponential). In the continuous mean-field interpolation $dM/dt \approx \alpha a\,e^{\kappa M}$ with $\kappa := \ln(1+1/a)$, integration yields a finite-time singularity; this is a property of the continuous approximation and should not be read as a theorem about the discrete map. The adjacent-possible cardinality $|\mathcal{A}(M)| = 2^M - M - 1$ grows doubly exponential relative to $M$, giving the formal non-ergodicity backbone of TAP.

### 2.2 Projection-first architecture and event identity

Define microstate $\mathbf{X}_t = (\mathbf{X}_t^{\mathrm{imp}}, \mathbf{X}_t^{\mathrm{exp}})$. A feature map $\Psi$ and threshold $\varepsilon$ define active set $\mathcal{S}(\mathbf{X}_t^{\mathrm{exp}})$ and projection $M_t = \Phi(\mathbf{X}_t^{\mathrm{exp}}) := |\mathcal{S}|$. Births $B(\mathbf{X}_t)$ and deaths $D(\mathbf{X}_t)$ are threshold-crossing counts, satisfying the exact identity

$$M_{t+1} - M_t = B(\mathbf{X}_t) - D(\mathbf{X}_t). \tag{P1}$$

TAP and $\sigma$-TAP are mean-field closures of (P1), not direct models of $\mathbf{X}_t$. This framing makes the architecture unambiguous and prevents category errors in the closure declarations that follow.

### 2.3 Layer taxonomy

We distinguish four layers of claim throughout this paper:

**Derived.** Algebraic consequences of the combinatorial postulate: the TAP kernel (TAP-3), its asymptotics, the fixed-point equation $f(M^*) = \mu M^*$, and the adjacency-cardinality non-ergodicity statement. These require no empirical calibration and are stable under changes to simulation implementation.

**Closure.** Conditional-drift approximations that close the dynamics on $(M_t, \Xi_t)$: $\mathbb{E}[B \mid M_t, \Xi_t] \approx \sigma(\Xi_t)f(M_t)$ and $\mathbb{E}[D \mid M_t] \approx \mu M_t$. These are assumptions, not theorems. Their adequacy is an empirical question addressed in Section 4.

**Constitutive.** Observational charts defined on top of the closed dynamics: regime classifiers, empirical phase boundaries, pressure coordinates. Phase diagrams are constitutive summaries under explicit initialization policies, not intrinsic properties of the equations.

**Interpretive.** Domain-mapping claims: identifying $M_t$ with patent counts, species, or institutional configurations; linking $\Xi_t$ to learning accumulation. These are heuristic mappings requiring separate empirical grounding.

Maintaining this taxonomy prevents the most common category error in complex-systems modeling: presenting a fitted phase boundary (constitutive) as if it followed from the equations (derived), or treating a domain mapping (interpretive) as if it were a tested closure assumption. We revisit it explicitly when framing each set of results.

### 2.4 $\sigma$-TAP: path-dependent efficiency extension

Base TAP has no memory: future innovation rates depend only on $M_t$, not on how that count was reached. We introduce consummation state $\Xi_t \geq 0$ and efficiency map $\sigma: \mathbb{R}_{\geq 0} \to \mathbb{R}_{\geq 0}$. The minimal $\sigma$-TAP system is

$$M_{t+1} = M_t(1-\mu) + \sigma(\Xi_t)f(M_t), \tag{S1}$$

$$\Xi_{t+1} = \Xi_t + \beta B(\mathbf{X}_t) + \eta H(\mathbf{X}_t). \tag{S2}$$

Here $\beta > 0$ is the birth-channel loading rate, $\eta \geq 0$ is the implicate-channel rate, and $H(\mathbf{X}_t) \geq 0$ is an implicate densification term. Two-channel update (S2) permits $\Xi_t$ to accumulate even when $M_t$ is stationary, corresponding to organizational enfolding increasing future efficiency without immediate realized-count growth.

Two distinct uses of $B$ appear in the system. In (P1) and (S2), $B(\mathbf{X}_t)$ denotes the event-level birth count — the realized number of threshold crossings at step $t$. In the closure declaration $\mathbb{E}[B \mid M_t, \Xi_t] \approx \sigma(\Xi_t)f(M_t)$, the same symbol appears inside a conditional expectation. The former is a microstate observable; the latter approximates its distribution. Wherever $B$ appears in a forward simulation we use the realized value; wherever it appears in a claim about the law of motion we invoke the closure. The system is non-Markov in $M_t$ alone but Markov in $(M_t, \Xi_t)$.

### 2.5 Efficiency map and $H$-construction

We take $\sigma(\Xi) = 1 + \gamma\Xi$ with $\gamma \geq 0$, so $\gamma = 0$ recovers standard TAP. For the simulation experiments we use the minimal-complexity instantiation

$$H_t = \max(0, \delta\Xi_t), \quad \delta = 0.02,$$

which satisfies non-negativity, bounded second moments, and approximate orthogonality to the birth residual. This is the paper's baseline instantiation rather than the unique admissible one; the admissibility conditions permit any non-negative, bounded-moment functional satisfying the orthogonality soft constraint, and alternative constructions would not alter the closure-layer claims.

Three derived observables serve diagnostic purposes only and do not enter the forward dynamics:

$$\Lambda_t := \frac{B(\mathbf{X}_t)}{\sigma(\Xi_t)f(M_t)}, \quad \Delta_{\mathrm{cond}}(t) := D(\mathbf{X}_t) - \mu M_t, \quad G_{\mathrm{cond}}(t) := H(\mathbf{X}_t).$$

---

## 3. Phase Structure of TAP

Phase structure results reported in this section are constitutive summaries over the closure dynamics established in Section 2 — empirical instruments for reading simulated behavior under named initialization policies, not derived properties of the equations.

### 3.1 The separatrix as a surface in $(\alpha, \mu, M_0)$

The long-run behavior of TAP trajectories is not determined by $(\alpha, \mu)$ alone. The same parameter pair can produce explosive growth or indefinite plateau depending on $M_0$. The primary object is a separatrix surface $\mathcal{S}(\alpha, \mu, M_0) = 0$ in three-dimensional parameter-initialization space. Published two-dimensional phase diagrams are policy-defined slices or projections of this surface, not intrinsic properties of the equations.

The surface is anchored by the unstable fixed point $M^*$ satisfying $f(M^*) = \mu M^*$, or equivalently

$$\alpha a \left[\left(1+\frac{1}{a}\right)^{M^*} - 1 - \frac{M^*}{a}\right] = \mu M^*. \tag{FP}$$

For each $(\alpha, \mu, a)$, equation (FP) typically admits a unique positive solution on the sampled parameter domain, found numerically. The fixed point is unstable — a threshold, not an attractor.

We define three complementary modes for probing the separatrix:

**Mode A (near-critical initialization).** Set $M_0 = 1.05 \cdot M^*(\alpha, \mu, a)$. Natural for critical-transition questions but introduces covariate coupling: $M_0$ is a deterministic function of $(\alpha, \mu, a)$, so the three parameters are not independently varied.

**Mode B (fixed-seed initialization).** Set $M_0 \in \{10, 20, 50\}$ independently of $(\alpha, \mu, a)$. Cleaner for deployment-relevant questions and breaks the Mode A coupling, enabling independent estimation of initialization sensitivity.

**Mode C (analytical isocurves).** Trace $M^* = c$ isocurves from (FP). No simulation required. For fixed $a$ and $M^*$:

$$\mu = \alpha \cdot K(M^*, a), \quad K(M^*, a) := \frac{a\left[\left(1+\frac{1}{a}\right)^{M^*} - 1 - \frac{M^*}{a}\right]}{M^*},$$

linear in $(\alpha, \mu)$ with slope exactly 1 in log-log space. Mode C isocurves serve as analytical baselines.

All three modes answer different questions; neither subsumes the other. Mode B is the primary estimation dataset; Mode C is the analytical reference; Mode A results inform mechanistic interpretation.

### 3.2 Regime classification and sweep design

Trajectories are classified into five regimes: plateau, exponential, explosive, precursor-active, and extinction. Precursor-active is assigned when $\Xi_t$ grows above a run-normalized threshold while $M_t$ remains near its initial value. The threshold is defined adaptively per run to remain invariant across $(\beta, \eta)$ rescaling.

Phase-boundary estimation uses logistic regression on $(\log\alpha, \log\mu, \log M_0)$ with positive class $\{\texttt{explosive}, \texttt{precursor-active}\}$. The resulting classifier is a constitutive summary — an empirical instrument, not a derived result.

All sweeps use a $10 \times 10$ log-spaced grid over $\alpha \in [10^{-5}, 10^{-2}]$ and $\mu \in [10^{-3}, 10^{-1}]$, with $a$ and $M_0$ varied across experiments. Trajectories are capped at $M_{\mathrm{cap}} = \Xi_{\mathrm{cap}} = 10^9$; the step at which either cap is first triggered is recorded as $t_{\mathrm{blowup}}$ and used as a blow-up timing proxy throughout. All sweep outputs include parameter manifests, seed lists, regime labels, and $t_{\mathrm{blowup}}$ fields to support replication.

### 3.3 Initialization sensitivity and the sign-reversal diagnostic

**This subsection reports a methodological finding.** The fitted sign of phase-boundary coefficients depends on initialization policy, and the dependence is large enough to reverse the sign of individual coefficients across policies, with non-overlapping confidence intervals.

Under Mode A, the 2D logistic fit yields a bootstrap mean standardized coefficient of $-0.75$ on $\log\alpha$ (95% CI $[-1.03, -0.41]$) and $+2.15$ on $\log\mu$ (95% CI $[1.96, 2.31]$). The negative sign on $\log\alpha$ is counterintuitive: higher innovation rate predicts plateau. Under Mode B, the 2D fit yields $+1.21$ on $\log\alpha$ (95% CI $[1.03, 1.40]$) and $-0.62$ on $\log\mu$ (95% CI $[-0.84, -0.39]$). Signs on both coefficients are reversed. Bootstrap CIs are non-overlapping across modes for both $\log\alpha$ and $\log\mu$, confirming the reversal is distinguishable from sampling variation rather than a point-estimate artifact.

The reversal mechanism is the Mode A coupling. Setting $M_0 = 1.05 \cdot M^*(\alpha, \mu, a)$ creates a systematic confound: $M^*$ grows with $\alpha$ and shrinks with $\mu$, so cells with high $\alpha$ and low $\mu$ receive large seeds while cells with low $\alpha$ and high $\mu$ receive small seeds. The Mode A fit captures initialization geometry through $(\alpha, \mu)$ as proxies, producing coefficients that reflect initialization scale rather than direct parameter effects. Mode B removes this confound by holding $M_0$ fixed.

The Mode B 3D fit adding $\log M_0$ yields bootstrap means $+1.32$ on $\log\alpha$ (95% CI $[1.14, 1.48]$), $-0.68$ on $\log\mu$ (95% CI $[-0.86, -0.51]$), and $+1.15$ on $\log M_0$ (95% CI $[0.94, 1.30]$). All CIs exclude zero; all bootstrap CIs and coefficient intervals are available in `outputs/inferential_stats.json`. Rearranging the Mode B 3D decision boundary gives the empirical relation

$$\log\mu \approx 1.28 \cdot \log\alpha + 2.97 \cdot \log M_0 + C, \tag{B1}$$

where $C$ depends on $a$. The slope on $\log M_0$ exceeds that on $\log\alpha$: initialization sensitivity is at least comparable to innovation-rate sensitivity as a predictor of explosive-regime membership on this pilot grid.

**The methodological implication** is direct: TAP-family phase diagrams without a declared initialization policy cannot be compared across papers, because the same system under different protocols yields coefficients with opposite signs and non-overlapping confidence intervals. Phase diagrams are constitutive summaries under a named policy, not intrinsic geometric objects.

### 3.4 Structural dependence on the difficulty ratio $a$

The parameter $a$ controls the geometric penalty on higher-order combinability and has a strong first-order effect on phase occupancy independent of $\sigma$-feedback. Across the Mode B sweep at $\gamma = 0$, explosive counts are 626, 448, 224, and 92 for $a = 2, 4, 8, 16$ out of 900 cells — a factor-of-seven reduction as $a$ doubles twice. For large $M$ and large $a$, $\kappa = \ln(1+1/a) \approx 1/a$ becomes small, slowing super-exponential growth and shifting $M^*$ upward, placing higher demands on initialization for explosive escape from any fixed seed. Boundary laws estimated at one $a$-value are slice-specific and should not be extrapolated without re-estimation; we report Mode B boundaries separately for each $a$.

### 3.5 Continuous fixed-point baseline and discrete discrepancy

Mode C isocurves give an analytical boundary estimate requiring no simulation. At $a = 8$, the isocurve gives critical extinction rate $\mu_{\mathrm{iso}}(\alpha \mid M^* = M_0, a)$: trajectories with $\mu < \mu_{\mathrm{iso}}$ should be explosive from seed $M_0$; those with $\mu > \mu_{\mathrm{iso}}$ should plateau.

The empirical results are largely consistent with this prediction but reveal a measurable discrepancy. Two cells with $M_0/M^* = 1.74$ and $M_0/M^* = 2.41$ — seeds well above the continuous fixed point — fail to escape at $\gamma = 0$ and escape at $\gamma = 0.2$. The continuous analysis predicts escape; the discrete map fails to produce it. The observed pattern is consistent with early-period drift in the discrete map: near small $M$ values, the trajectory may fall below an effective discrete threshold before the super-exponential term dominates. A dedicated trajectory-level test is deferred to future work.

This discrepancy is simultaneously a limitation and a positive finding: the deviation is measurable and localizable, so Mode C is falsifiable rather than unfalsifiable. We treat Mode C isocurves as approximate baselines rather than exact discrete predictors throughout, with particular caution in the small-$M$, large-$a$ region.

---

## 4. $\sigma$-TAP Empirics

We present findings in order of statistical clarity before mechanistic specificity. The earliest results are high-consensus transition outcomes; later results are mechanism-resolved boundary effects on a small recruited set. This is an evidence-ordering convention, not an importance ranking.

### 4.1 Precursor-active is a transient loading regime

We tested whether precursor-active is a genuine transitional state or a finite-horizon artifact by re-running all precursor-labeled cells at extended horizon $T_{\mathrm{long}} = \max(600, 4 \cdot T_{\mathrm{short}})$. All 24 short-horizon precursor cases reclassify to explosive at long horizon, with zero long-horizon under-resolved runs. Precursor-active is therefore a transient loading regime, not a terminal state. The two-channel consummation update (S2) is doing what it was designed to do: $\Xi_t$ accumulates while $M_t$ is stationary, and the accumulated consummation subsequently drives escape. This is the most direct empirical confirmation of the $\sigma$-TAP closure architecture available from the current experiments. Blow-up steps at long horizon span a wide range consistent with the loading interpretation; explicit values are in `outputs/precursor_longitudinal.csv`, and the recruited-cell subset ($a = 8$, $\gamma = 0.2$) shows blow-up steps of 17, 58, 59, 152, and 158.

### 4.2 Structural gating by difficulty ratio $a$

Across the $\gamma \in \{0, 0.05, 0.2\}$ Mode B panel, explosive counts are 626, 448, 224, and 92 for $a = 2, 4, 8, 16$ (per 900-cell block over $(\alpha, \mu, M_0)$ combinations), aggregated over the $\gamma$-panel. This confirms that combinatorial difficulty is a first-order structural control on explosive volume.

The gating mechanism is quantitative. The sigma multiplier $\sigma(\Xi) = 1 + \gamma\Xi$ departs meaningfully from unity only when $\gamma\Xi > 1$. As an order-of-magnitude estimate: with $\beta = 0.05$ and typical pre-blowup $f(M)$ values, $\Xi$ accumulates at roughly $0.05 \cdot f(M)$ per step. At $a = 2$, mean blow-up occurs around step 6, leaving $\Xi \lesssim 0.3$ — insufficient for the multiplier to matter at $\gamma = 0.2$. At $a = 16$, mean blow-up near step 47 allows $\Xi$ values of 2–6 for boundary-adjacent cells, where $\gamma\Xi$ reaches 0.4–1.2. These are rough estimates from mean-step figures; run-level $\Xi$ distributions would sharpen them. The mechanistic claim is: $\sigma$-feedback is a kinetic amplifier whose observable strength is gated by the combinatorial-difficulty timescale set by $a$.

### 4.3 Matched-panel acceleration after composition correction

At $a = 8$, unmatched means are approximately 21.1, 20.7, 24.0 for $\gamma = 0, 0.05, 0.2$ — apparently non-monotone because $\gamma = 0.2$ recruits five additional blow-up cells ($73 \to 78$). These recruited cells blow up slowly by construction, raising the $\gamma = 0.2$ unmatched mean even if feedback accelerates every individual trajectory.

On the matched 73-cell panel, mean blow-up step is 21.14, 20.73, 19.56. Mean paired reduction from $\gamma = 0$ to $\gamma = 0.2$ is 1.58 steps (two-sided sign-flip permutation test, $p < 0.001$, $n = 73$). The apparent non-monotonicity is entirely a composition artifact; within matched cells, $\gamma$-driven acceleration is monotone and statistically distinguishable from zero.

Cell-level data confirm the gating prediction: cells blowing up in 3–7 steps at $\gamma = 0$ show zero or one-step change — insufficient loading time. Cells blowing up in 70–126 steps show the largest accelerations; the cell $(\alpha = 0.0021, \mu = 0.0046, M_0 = 20)$ drops from 126 to 107 steps (15% reduction). Acceleration scales with base blow-up time, consistent with the $\Xi$-loading mechanism. This supports a kinetic interpretation of $\sigma$-feedback: for cells already in the blow-up-supporting region, increased $\gamma$ primarily advances time-to-instability.

### 4.4 Recruited-cell mechanisms at $a = 8$, $\gamma: 0 \to 0.2$

The five recruited cells split into three near-boundary amplification cases and two discrete-threshold compensation cases.

Near-boundary cases have $M_0/M^* \in [0.989, 1.042]$ and $|\log(\mu/\mu_{\mathrm{iso}})| < 0.2$ — initialized essentially at the continuous threshold and within 20% of the analytical critical extinction rate. Blow-up steps are 17, 58, and 59, consistent with slow near-boundary dynamics. These are textbook separatrix-tipping events: a small persistent perturbation moves a marginal trajectory across a sharp threshold.

The two compensation cases have $M_0/M^* = 1.74$ and $M_0/M^* = 2.41$ — well above the continuous fixed point — yet fail to escape at $\gamma = 0$, blowing up at steps 152 and 158 at $\gamma = 0.2$. The observed pattern is consistent with early-period drift in the discrete map preventing escape despite initialization above the continuous $M^*$; persistent sigma amplification supplies the missing drift. A dedicated trajectory-level test is deferred to future work.

These are distinct mechanisms with different implications. Near-boundary amplification is boundary tipping in the standard thermodynamic sense. Discrete-threshold compensation reflects a gap between continuous prediction and discrete-map behavior that the analytical baseline does not anticipate, reinforcing the Section 3.5 recommendation to treat Mode C isocurves as approximate rather than exact discrete predictors.

### 4.5 High-$\gamma$ probe: timing versus occupancy

In the targeted high-$\gamma$ panel ($a = 8$, $M_0 = 20$), explosive counts are 20, 22, 22, 26 out of 100 for $\gamma = 0.5, 1, 2, 5$, while mean blow-up step drops from 59.5 to 36.6 (38% reduction). Timing acceleration remains the stronger effect over this tested range. There is no sharp $\gamma^*$ at which boundary expansion becomes dominant; the progression is continuous, consistent with $\sigma$-feedback acting as a smooth kinetic amplifier rather than a bifurcation-inducing perturbation at these parameter values.

For the primary $\gamma = 0$ vs $\gamma = 0.2$ paired 300-cell panel, explosive-or-precursor fraction rises from 0.260 to 0.273 (difference $= 0.013$); bootstrap 95% CI is $[0.003, 0.027]$; paired sign-flip permutation on binary outcomes yields $p = 0.120$. The bootstrap CI and the permutation test answer different questions — the former quantifies plausible effect magnitude, the latter tests the directional null — and both should be read together. The occupancy shift is small in absolute terms and not robustly detectable at this panel size and $\gamma$ range. Inferential statistics were rerun at higher resampling counts for the camera-ready version; all claim directions and significance thresholds are stable across runs.

The combined picture is consistent: $\sigma$-feedback at biologically and economically plausible feedback strengths is primarily a kinetic effect, accelerating approach to instability within the explosive region while producing a small, directionally positive but statistically modest occupancy shift at the separatrix. Both effects are gated by $a$ through the loading-time mechanism established in Section 4.2.

---

## 5. Limitations and Falsifiability

We distinguish three categories of limitation: constraints on the analytical baseline, constraints arising from pilot scale, and constraints on the closure-layer claims themselves. Each is paired with its falsifiability handle.

### 5.1 Continuous baseline is approximate for discrete dynamics

Mode C predicts escape from any seed above $M^*$, but this prediction fails for two $a = 8$ cells ($M_0/M^* = 1.74$ and $2.41$) at $\gamma = 0$. The continuous analysis can underestimate the effective explosive threshold in the small-$M$, large-$a$ region where early discrete drift can push trajectories below an effective threshold before the super-exponential term dominates. The natural follow-up is to characterize the extent of the continuous-to-discrete gap across $(a, M_0/M^*, \alpha, \mu)$ space, which would convert the current qualitative observation into a quantitative correction map enabling Mode C to be used with known error bounds. That investigation is deferred; the conservative recommendation stands: validate Mode C against Mode B simulation in any parameter region with small $M$ or large $a$. The discrepancy is simultaneously a limitation and a falsifiability handle — it is measurable and localizable, so Mode C is empirically falsifiable rather than an unqualified assertion.

### 5.2 Kinetic-dominant conclusion is range-limited

The $\sigma$-feedback experiments find dominant timing acceleration and modest boundary expansion over the tested $\gamma$ range, where the sigma multiplier reaches values of order 1.2–2.0. Whether a qualitative transition to boundary-expansion-dominant behavior exists at much higher $\gamma$ is open. A stress-test sweep at $\gamma \in \{10, 20, 50\}$ — beyond domain-plausible feedback strengths but useful for establishing whether such a transition exists at all — is a direct falsification path. We do not run it here because the current range covers the domain of primary interest for the application layer in Section 6, and because the $a$-gating mechanism implies very large $\gamma$ effects will remain invisible at $a = 2$ regardless.

### 5.3 Pilot-grid generalization limits

All boundary fits and regime counts derive from a $10 \times 10$ log-spaced $(\alpha, \mu)$ grid with $M_0 \in \{10, 20, 50\}$ and $a \in \{2, 4, 8, 16\}$. Three specific generalization limitations follow. First, the logistic boundary fits (equation B1 and analogues by $a$) are pilot estimates on a coarse grid; bootstrap confidence intervals on all reported coefficients are now available in `outputs/inferential_stats.json` via the inferential bundle, so the analysis has moved beyond pure point estimates, but out-of-sample performance on denser grids remains untested. Second, the matched-panel analysis rests on 73 cells at $a = 8$ and three $\gamma$ values; a larger panel would sharpen quantitative acceleration estimates. Third, the precursor-active resolution result is unanimous across 24 cells but from a single sweep configuration; a sweep varying $(\beta, \eta)$ jointly with $\gamma$ would test whether the loading interpretation generalizes across consummation parameter settings. None of these limitations undermines the qualitative findings of Section 4; they bound the precision and scope of quantitative claims and specify the natural expansion path.

### 5.4 Identifiability constraints on closure-layer claims

Closure-layer claims are validated here in simulation where both $M_t$ and $\Xi_t$ are directly observable. In empirical applications, $\Xi_t$ is latent. If only $M_t$ is observed, only reduced-form forecasting claims are admissible — the $\sigma$-TAP extension cannot be distinguished from a time-varying-parameter TAP. If $B_t$ and $D_t$ are observed or reconstructed alongside $M_t$, partial structural testing becomes possible. The gate should be documented as part of the methods section of any empirical application, with the admissible claim class stated before results are reported. Failure to do so conflates two distinct failure modes — the closure architecture being wrong, and the calibration or observability being insufficient — which cannot be disentangled post-hoc.

---

## 6. Discussion and Deployment Framing

### 6.1 Summary of main findings

Four findings emerge from the experiments, ordered by the same evidence hierarchy used in Section 4:

1. **Precursor-active is a genuine transient loading state.** All 24 precursor-active cells in the $\gamma > 0$ panel resolved to explosive at long horizon. $\Xi_t$ loads during plateau while $M_t$ is stationary; accumulated consummation subsequently drives escape. This is the most direct available confirmation of the two-channel closure architecture.

2. **Difficulty ratio $a$ is the dominant structural control on phase occupancy.** Explosive counts drop from 626 to 92 as $a$ increases from 2 to 16, aggregated over the $\gamma$-panel of the Mode B sweep. The gating mechanism — $a$ controls the combinatorial timescale and therefore the window available for $\Xi$-loading feedback — provides a unified explanation for why $\gamma$ effects are strong at intermediate $a$ and invisible at low $a$.

3. **$\sigma$-feedback is primarily kinetic at the parameter values studied.** Matched-panel blow-up acceleration is 1.58 steps on average ($p < 0.001$, $n = 73$), scaling with base blow-up time. The occupancy shift is 1.3 percentage points with bootstrap 95% CI $[0.003, 0.027]$ but permutation $p = 0.120$ — small and directionally positive but not robustly detectable at this panel size.

4. **Initialization policy determines the sign of fitted boundary coefficients.** Bootstrap CIs on standardized coefficients are non-overlapping between Mode A and Mode B for both $\log\alpha$ and $\log\mu$. This is a methodological finding with direct consequences for how TAP-family results should be reported and compared.

### 6.2 The layer taxonomy as a reporting discipline

The derived/closure/constitutive/interpretive taxonomy introduced in Section 2 is not merely an organizational device. It is a discipline for preventing a category error that is common in complex-systems modeling: presenting a fitted phase boundary (constitutive) as if it followed from the equations (derived), or treating an application mapping (interpretive) as if it were a tested closure assumption. The most common failure mode in extending theoretical frameworks is elevation — treating a constitutive observable as a derived quantity after sufficient familiarity has made it feel necessary. Future work extending this framework, whether adding pressure coordinates, testing on empirical data, or introducing stochastic birth-death processes, should maintain these layer distinctions explicitly. The taxonomy is a guard against that drift.

### 6.3 Application instantiation

The $\sigma$-TAP architecture is domain-neutral by design. The core equations for $(M_t, \Xi_t)$ remain fixed across applications; the feature map $\Psi$, threshold $\varepsilon$, $H$-functional construction, and observable $\Xi_t$-proxy are domain-specific choices, while the combinatorial postulate, TAP kernel, closure declarations, and identifiability gate are invariant.

In economics and technology, $M_t$ counts realized innovations, products, or functional modules. The consummation state $\Xi_t$ maps naturally to accumulated capability or organizational learning. The precursor-active signature — $\Xi_t$ loading while $M_t$ is stationary — corresponds to capability accumulation prior to visible product proliferation, recognizable in technology S-curves.

In social systems, $M_t$ counts stabilized practices, institutional forms, or cultural configurations above a viability threshold. The non-ergodicity backbone is particularly relevant: the adjacent possible expands faster than any trajectory can explore it, so social evolution is path-dependent and irreversible in a formal sense.

In evolutionary biology, $M_t$ counts realized functional structures — protein folds, trait assemblages, or ecological guilds — above a viability threshold. The parameter $a$ maps to constraints on higher-order co-evolution: at high $a$, complex multi-component innovations are strongly suppressed. The gating result from Section 4.2 has a direct biological reading: evolutionary systems with high combinatorial difficulty require large existing complexity before super-exponential expansion can sustain itself, consistent with the observation that major evolutionary transitions tend to occur from already-complex precursors.

In environmental systems, $M_t$ counts viable ecological regime-configurations above a resilience threshold. The discrete-threshold compensation finding from Section 4.4 is directly relevant: continuous resilience models may overestimate the robustness of states that are above the continuous threshold but vulnerable to early-period drift in discrete dynamics — a finding with implications for tipping-point analysis.

### 6.4 Identifiability gate for empirical deployment

With only $M_t$ observed, only reduced-form forecasting claims are admissible; the consummation loading dynamics are invisible in the scalar trajectory. With $B_t$, $D_t$, and $M_t$ observed or reliably reconstructed alongside a declared $\Xi$-proxy, partial structural testing becomes possible: the closure assumption $\mathbb{E}[B \mid M_t, \Xi_t] \approx \sigma(\Xi_t)f(M_t)$ can be tested against observed birth rates. Any deployment should pass this gate explicitly before reporting architecture-level claims, with the admissible claim class stated in the methods section before results are reported.

### 6.5 Open directions

Three questions are left open and constitute natural extensions with the existing infrastructure.

The continuous-to-discrete gap identified in Sections 3.5 and 4.4 is real and localized but unmapped in extent. A systematic comparison of Mode B simulation boundaries against Mode C isocurves across $(a, M_0/M^*, \alpha, \mu)$ would convert the current qualitative observation into a correction map, enabling Mode C to be used as an approximate predictor with known error bounds.

The high-$\gamma$ regime is unexplored. Whether a qualitative bifurcation in the kinetic-versus-thermodynamic balance exists above $\gamma = 5$ is an open question. A targeted stress-test sweep at $\gamma \in \{10, 20, 50\}$ with the existing infrastructure would answer it directly.

A stochastic extension replacing deterministic mean-field closures with Poisson or negative-binomial birth processes would produce probability-of-explosion estimates rather than binary regime labels, and would allow the matched-panel analysis to be replaced by survival analysis on the blow-up time distribution — sharpening the kinetic-versus-thermodynamic distinction considerably.

---
