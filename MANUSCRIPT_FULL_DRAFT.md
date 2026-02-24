# σ-TAP Manuscript Draft

## 1. Introduction

Complex systems across biology, economics, and social organization share a structural feature: the arrival of new functional components creates opportunities for further combinations that did not previously exist. A technology enables new technologies; a protein fold enables new protein interactions; an institutional form enables new institutional arrangements. The space of what is possible expands as a function of what has been realized, and the expansion is faster than linear. This is the adjacent possible, and it produces a characteristic phenomenology—long periods of near-stasis punctuated by episodes of rapid combinatorial expansion—that recurs across domains too diverse to be explained by domain-specific mechanisms alone.

The Theory of the Adjacent Possible (TAP) formalizes this observation as a discrete dynamical system in which the count of realized objects $M_t$ evolves under a balance between combinatorial innovation and extinction. The innovation kernel grows super-exponentially in $M_t$, producing a sharp transition between plateau and explosive regimes separated by an unstable fixed point $M^*$. TAP has been applied to biological complexity, economic innovation, and the thermodynamics of open-ended systems, and its qualitative phenomenology—plateau, sharp divergence, extinction sensitivity—is well established analytically. What has been less systematically studied is the phase structure of TAP as a surface in full parameter-initialization space, the role of path-dependent efficiency modulation in shifting or accelerating that structure, and the relationship between continuous analytical predictions and discrete simulation behavior.

This paper addresses three gaps. The first is architectural: base TAP has no memory. Innovation rates depend only on current count $M_t$, not on the path by which that count was reached. This excludes learning effects, capability accumulation, and mechanisms by which prior innovation reshapes the ease of subsequent innovation. We introduce $\sigma$-TAP, a minimal extension that adds a consummation state $\Xi_t$ accumulating through realized births and implicate densification, with a linear efficiency map $\sigma(\Xi_t)$ modulating the innovation kernel. The extension is minimal in that it adds one state variable and one feedback parameter $\gamma$, recovers standard TAP at $\gamma=0$, and uses a projection-first architecture that makes closure assumptions explicit rather than implicit.

The second gap is methodological. Published TAP-family phase diagrams implicitly depend on initialization policy—the rule by which $M_0$ is chosen—but this dependence is rarely declared and its consequences are under-characterized. We show that two natural initialization policies (near-critical initialization, Mode A, and fixed-seed initialization, Mode B) can yield phase-boundary coefficients with opposite signs for the same system, with non-overlapping bootstrap confidence intervals. The sign reversal is not a numerical artifact but a structural consequence of covariate coupling in the near-critical design. Phase diagrams are constitutive summaries under a named initialization policy, not intrinsic geometric objects, and cross-paper comparisons require that policy to be declared.

The third gap is empirical. We characterize when $\sigma$-feedback shifts the phase boundary versus accelerating trajectories already inside it, identify the gating role of combinatorial difficulty $a$, confirm precursor-active as a genuine transient loading state rather than a classifier artifact, and localize a measurable discrepancy between continuous fixed-point predictions and discrete simulation behavior. Claims are backed by committed simulation artifacts and, where appropriate, formal inferential tests.

The paper is organized as follows. Section 2 presents the model with an explicit four-layer taxonomy separating derived results, closure assumptions, constitutive observables, and interpretive mappings. Section 3 develops phase structure across three initialization modes. Section 4 reports $\sigma$-TAP empirical findings. Section 5 discusses limitations and falsifiability. Section 6 addresses deployment framing and application instantiation.

---

## 2. Model

### 2.1 Combinatorial postulate and TAP derivation

Let $M_t \in \mathbb{N}$ denote the number of realized objects present at discrete time $t$. We assume: (i) realized objects can form higher-order combinations, (ii) larger combinations are harder to realize with geometrically increasing difficulty, and (iii) a fraction of realized objects becomes inactive each step. These are primitives.

Parameters: $\alpha > 0$ (base realization coefficient), $\mu \in [0,1]$ (extinction/obsolescence fraction), $a>1$ (difficulty ratio).

For order $i$, candidate combinations are $\binom{M_t}{i}$, weighted by $\alpha/a^{i-1}$. Then:

$$
M_{t+1}=M_t(1-\mu)+\sum_{i=2}^{M_t}\frac{\alpha}{a^{i-1}}\binom{M_t}{i}. \tag{TAP-1}
$$

Define innovation kernel

$$
f(M):=\sum_{i=2}^{M}\frac{\alpha}{a^{i-1}}\binom{M}{i},
$$

so

$$
M_{t+1}=M_t(1-\mu)+f(M_t). \tag{TAP-2}
$$

Using $x=1/a$:

$$
f(M)=\alpha a\sum_{i=2}^{M}\binom{M}{i}x^i
=\alpha a\left[\left(1+\frac1a\right)^M-1-\frac{M}{a}\right]. \tag{TAP-3}
$$

For large $M$, $f(M)\sim \alpha a(1+1/a)^M$ (super-exponential). A continuous interpolation gives finite-time blow-up; this is a property of the interpolation, not a theorem about the discrete map. Adjacent-possible cardinality $|\mathcal{A}(M)|=2^M-M-1$ gives the formal non-ergodicity backbone.

### 2.2 Projection-first architecture and event identity

Let microstate $\mathbf{X}_t=(\mathbf{X}^{\mathrm{imp}}_t,\mathbf{X}^{\mathrm{exp}}_t)$. A feature map $\Psi$ and threshold $\varepsilon$ define active set $\mathcal{S}$, and projection $M_t=\Phi(\mathbf{X}^{\mathrm{exp}}_t)=|\mathcal{S}|$. Define births $B(\mathbf{X}_t)$ and deaths $D(\mathbf{X}_t)$ by threshold crossings. Then exact identity:

$$
M_{t+1}-M_t=B(\mathbf{X}_t)-D(\mathbf{X}_t). \tag{P1}
$$

TAP and $\sigma$-TAP are mean-field closures of (P1), not direct models of $\mathbf{X}_t$.

### 2.3 Layer taxonomy

- **Derived:** algebraic consequences of postulate (kernel, asymptotics, fixed-point equation, non-ergodicity statement).
- **Closure:** conditional-drift assumptions, e.g. $\mathbb{E}[B\mid M_t,\Xi_t]\approx \sigma(\Xi_t)f(M_t)$, $\mathbb{E}[D\mid M_t]\approx \mu M_t$.
- **Constitutive:** observational charts (regime labels, fitted boundaries, pressure coordinates).
- **Interpretive:** domain mappings (e.g., patents/species/institutions; $\Xi_t$ as learning/capability).

### 2.4 $\sigma$-TAP: path-dependent efficiency extension

$$
M_{t+1}=M_t(1-\mu)+\sigma(\Xi_t)f(M_t), \tag{S1}
$$

$$
\Xi_{t+1}=\Xi_t+\beta B(\mathbf{X}_t)+\eta H(\mathbf{X}_t). \tag{S2}
$$

Here $\beta>0$, $\eta\ge0$, $H\ge0$. Two-channel consummation allows $\Xi_t$ growth while $M_t$ is stationary. Process is non-Markov in $M_t$ alone, Markov in $(M_t,\Xi_t)$.

### 2.5 Efficiency map and $H$-construction

Baseline:

$$
\sigma(\Xi)=1+\gamma\Xi,\quad \gamma\ge0,
$$

so $\gamma=0$ recovers TAP.

Simulation instantiation:

$$
H_t=\max(0,\delta\Xi_t),\quad \delta=0.02.
$$

Diagnostics:

$$
\Lambda_t=\frac{B(\mathbf{X}_t)}{\sigma(\Xi_t)f(M_t)},\quad
\Delta_{\mathrm{cond}}(t)=D(\mathbf{X}_t)-\mu M_t,\quad
G_{\mathrm{cond}}(t)=H(\mathbf{X}_t).
$$

---

## 3. Phase Structure of TAP

### 3.1 Separatrix as surface in $(\alpha,\mu,M_0)$

Primary object:

$$
\mathcal{S}(\alpha,\mu,M_0)=0.
$$

Two-dimensional diagrams are policy-defined slices/projections. Fixed point:

$$
\alpha a\left[\left(1+\frac1a\right)^{M^*}-1-\frac{M^*}{a}\right]=\mu M^*. \tag{FP}
$$

Modes:
- **Mode A:** $M_0=1.05\,M^*(\alpha,\mu,a)$ (near-critical, coupled design).
- **Mode B:** $M_0\in\{10,20,50\}$ (fixed-seed, decoupled).
- **Mode C:** analytical isocurves $M^*=c$, with $\mu=\alpha K(M^*,a)$.

### 3.2 Regime classification and sweep design

Regimes: plateau, exponential, explosive, precursor-active, extinction.  
Positive class for boundary fit: $\{\texttt{explosive},\texttt{precursor-active}\}$.  
Logistic features: $(\log\alpha,\log\mu,\log M_0)$.  
Grid: $10\times10$ log-spaced $\alpha\in[10^{-5},10^{-2}],\mu\in[10^{-3},10^{-1}]$.  
Caps: $M_{\text{cap}}=\Xi_{\text{cap}}=10^9$. Blow-up proxy: first cap-hit step $t_{\text{blowup}}$.

### 3.3 Initialization sensitivity and sign-reversal diagnostic

Mode A (2D): $\log\alpha$ coef mean $-0.75$ (95% CI $[-1.03,-0.41]$); $\log\mu$ $+2.15$ (95% CI $[1.96,2.27]$).  
Mode B (2D): $\log\alpha$ $+1.22$ (95% CI $[1.07,1.37]$); $\log\mu$ $-0.61$ (95% CI $[-0.79,-0.47]$).  
CIs are non-overlapping across modes for both coefficients.

Mode B (3D): $\log\alpha=+1.32$ (95% CI $[1.14,1.48]$), $\log\mu=-0.68$ (95% CI $[-0.86,-0.55]$), $\log M_0=+1.15$ (95% CI $[0.94,1.30]$).

Empirical boundary form:

$$
\log\mu \approx 1.28\log\alpha + 2.97\log M_0 + C. \tag{B1}
$$

Implication: phase diagrams are constitutive summaries under named initialization policy.

### 3.4 Structural dependence on $a$

Mode B, $\gamma=0$, explosive counts (out of 900): $626,448,224,92$ for $a=2,4,8,16$.  
Boundary laws are $a$-slice specific and should be estimated per $a$, not pooled.

### 3.5 Continuous baseline and discrete discrepancy

Mode C gives analytical reference isocurves. At $a=8$, two cells with $M_0/M^*=1.74$ and $2.41$ fail to escape at $\gamma=0$, then escape at $\gamma=0.2$. This supports a localized continuous–discrete gap and motivates treating Mode C as approximate baseline, not exact discrete predictor.

---

## 4. $\sigma$-TAP Empirics

We present findings in order of statistical clarity before mechanistic specificity. This is an evidence-ordering convention, not an importance ranking.

### 4.1 Precursor-active is a transient loading regime

Re-running all 24 precursor-labeled cells at

$$
T_{\mathrm{long}}=\max(600,4T_{\mathrm{short}})
$$

yields 24/24 reclassification to explosive, with zero long-horizon under-resolved runs. Precursor-active is therefore a transient loading regime, not a terminal state.

### 4.2 Structural gating by difficulty ratio $a$

Across $\gamma\in\{0,0.05,0.2\}$, explosive counts are $626,448,224,92$ for $a=2,4,8,16$, confirming strong $a$-gating of explosive occupancy.

Mechanistically, $\sigma(\Xi)=1+\gamma\Xi$ matters when $\gamma\Xi$ is nontrivial. Short-time blow-up at low $a$ limits loading time; larger $a$ allows longer pre-blowup windows and stronger feedback accumulation.

### 4.3 Matched-panel acceleration after composition correction

At $a=8$, unmatched means are approximately $21.1,20.7,24.0$ for $\gamma=0,0.05,0.2$, apparently non-monotone due to composition shift (73 blow-up cells at $\gamma=0$, 78 at $\gamma=0.2$).

On matched 73-cell panel, means are $21.14,20.73,19.56$. Mean paired reduction from $\gamma=0\to0.2$ is $1.58$ steps (two-sided sign-flip permutation, $p=0.002$, $n=73$). Within matched cells, acceleration is monotone and distinguishable from zero.

Cell-level pattern supports loading-time gating: very fast blow-up cells change little; slower baseline cells show larger absolute acceleration (e.g., $(\alpha,\mu,M_0)=(0.0021,0.0046,20)$: $126\to107$, 15% reduction).

### 4.4 Recruited-cell mechanisms at $a=8$, $\gamma:0\to0.2$

Five recruited cells split into:
1. **Near-boundary amplification (3):** $M_0/M^*\in[0.989,1.042]$, $|\log(\mu/\mu_{\mathrm{iso}})|<0.2$, blow-up steps $17,58,59$.  
2. **Discrete-threshold compensation (2):** $M_0/M^*=1.74,2.41$, blow-up steps $152,158$ at $\gamma=0.2$, despite non-escape at $\gamma=0$.

These mechanisms are distinct and support treating Mode C as approximate baseline.

### 4.5 High-$\gamma$ probe: timing versus occupancy

At $a=8,M_0=20$, $\gamma\in\{0.5,1,2,5\}$: explosive counts $20,22,22,26$ (of 100), mean blow-up step $59.5\to36.6$ (38% reduction). Timing effect dominates over tested range; no sharp $\gamma^*$ appears.

For paired 300-cell $\gamma=0$ vs $0.2$ panel, explosive-or-precursor fraction rises $0.260\to0.273$ ($+0.013$); bootstrap 95% CI $[0.003,0.027]$; paired sign-flip permutation on binary outcomes gives $p=0.150$. Read jointly: occupancy shift is directionally positive but small and not robustly detectable at this sample size/range.

Overall: $\sigma$-feedback is primarily kinetic in this regime, with modest occupancy movement.

---

## 5. Limitations and Falsifiability

### 5.1 Continuous baseline is approximate for discrete dynamics

Mode C predicts escape above $M^*$, but fails for two $a=8$ cells ($M_0/M^*=1.74,2.41$) at $\gamma=0$. Continuous analysis can underestimate effective discrete threshold in small-$M$, large-$a$ regions. This is also a falsifiability handle: discrepancy is measurable and localized.

### 5.2 Kinetic-dominant conclusion is range-limited

Current $\gamma$-range shows dominant timing acceleration and modest boundary expansion. Whether a transition to boundary-dominant behavior emerges at much higher $\gamma$ remains open. A stress-test sweep ($\gamma\in\{10,20,50\}$) is a direct falsification path.

### 5.3 Pilot-grid generalization limits

Results are from $10\times10$ $(\alpha,\mu)$ grid with $M_0\in\{10,20,50\}$, $a\in\{2,4,8,16\}$. This establishes phase structure but not fine boundary precision or interpolation guarantees. Larger/denser sweeps are natural next steps.

### 5.4 Identifiability constraints on closure-layer claims

Closure-layer statements are validated in simulation where $(M_t,\Xi_t)$ are observed. In empirical settings, $\Xi_t$ is latent. If only $M_t$ is observed, only reduced-form claims are admissible; structural $\sigma$-TAP claims require declared $\Xi$-proxy and observability-class gate.

---

## 6. Discussion and Deployment Framing

### 6.1 Main findings

1. Precursor-active is a genuine transient loading state (24/24 resolve to explosive).  
2. Difficulty ratio $a$ is a dominant structural control (explosive counts $626\to92$).  
3. $\sigma$-feedback is primarily kinetic in tested range (matched acceleration 1.58 steps, $p=0.002$; occupancy shift small).  
4. Initialization policy flips fitted boundary signs with non-overlapping CIs (Mode A vs B).

### 6.2 Taxonomy as reporting discipline

The derived/closure/constitutive/interpretive split is a safeguard against category errors: fitted boundaries are not derived laws; domain mappings are not closure validation.

### 6.3 Application instantiation

Model core is domain-neutral; measurement layer is domain-specific. Natural templates include innovation economics, social-institutional systems, evolutionary biology, and environmental regime dynamics.

### 6.4 Identifiability gate for empirical deployment

With only $M_t$: reduced-form forecasting claims only.  
With $B_t,D_t,M_t$ and a declared $\Xi$-proxy: partial structural tests become admissible.  
Claim class should be declared before inference.

### 6.5 Open directions

1. Map continuous–discrete gap over parameter space.  
2. Extend high-$\gamma$ sweep to test for qualitative transition.  
3. Add stochastic birth-death extension (e.g., Poisson/NB) and move from binary regime labels to survival-style timing analysis.
