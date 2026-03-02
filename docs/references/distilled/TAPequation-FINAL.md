# Cortês, Kauffman, Liddle & Smolin (2022) — The TAP Equation: Evaluating Combinatorial Innovation

**Source**: Marina Cortês, Stuart A. Kauffman, Andrew R. Liddle, and Lee Smolin, "The TAP equation: evaluating combinatorial innovation in biocosmology," arXiv:2204.14115v4 [q-bio.PE], October 8, 2025 (original April 2022). Perimeter Institute / Instituto de Astrofísica / Institute for Systems Biology. 10 pages. Also present in repository as `Applications-of-TAP.pdf` (published version with updated references and Section 4 on continuous blow-up).

**Core contribution**: Provides the mathematical analysis of the TAP equation — analytical solutions, blow-up time estimates, extinction equilibrium, and several equation variants. Establishes the characteristic "hockey-stick" dynamics (extended plateau → sudden super-exponential blow-up) and proves this transition is essentially unpredictable from pre-blow-up behavior.

---

## The TAP Equation (Fundamental Form)

The Theory of the Adjacent Possible (TAP) holds that the near-future outcomes of a developmental process are limited by the objects that already exist. The TAP equation counts ways new objects can be generated from *combinations* of existing objects — a model of combinatorial innovation.

**Equation (1)**:

$$M_{t+1} = M_t(1 - \mu) + \sum_{i=2}^{M_t} \alpha_i \binom{M_t}{i}$$

Where:
- $M_t$ = number of existing objects (types) at time $t$
- $\alpha_i$ = decreasing constants for the difficulty of combining $i$ elements
- $\mu$ = extinction rate of existing objects ($0 \leq \mu < 1$)
- $\binom{M_t}{i}$ = combinatorial combinations of existing elements

The sum starts at $i = 2$ (not $i = 1$) to avoid duplicating pre-existing objects. The equation is not strictly well-defined for non-integer $M_t$ — fixable by analytic interpolation (factorials → Gamma functions) or stochasticity.

The equation produces dramatically explosive behavior, much faster than exponential: not only do the combinatoric terms rapidly become vast, but so does the number of terms in the summation. Typical solutions effectively diverge to infinity in finite time ("blow-up"). Technically discrete-step solutions remain finite (every term is finite), but any continuum approximation necessarily diverges at a finite time.

---

## Case 1: Constant $\alpha_i = \alpha$

When all $\alpha_i$ are equal, the sum reduces analytically (sum of Pascal's triangle row minus first two entries):

$$M_{t+1} = M_t(1 - \mu) + \alpha(2^{M_t} - M_t - 1)$$

**Growth demonstration** ($\alpha = 1$, $\mu = 0$, $M_0 = 2$):
- $t = 0$: $M = 2$
- $t = 1$: $M = 3$
- $t = 2$: $M = 7$
- $t = 3$: $M = 127$
- $t = 4$: $M \approx 10^{38}$
- $t = 5$: $M \approx \exp(10^{38})$

Five steps produces a number larger than the number of particles in the observable universe.

### Blow-Up Mechanism

Blow-up occurs when the combinatoric term dominates the linear term: when $M_t \cdot 2^{-M_t} < \alpha$. Before this threshold, there is a plateau of slow growth.

**Doubling time estimate** (time for cumulated combinatoric effect to overcome the linear term):

$$t_{\text{double}} = \frac{M_0}{\alpha(2^{M_0} - M_0 - 1)}$$

**Tighter bound** — time to add just one new item:

$$t_{\text{add one}} = \frac{1}{\alpha(2^{M_0} - M_0 - 1)}$$

Key insight: adding one new object *at least doubles* the accessible options (it can substitute for any item in anything previously possible). Hence after adding one item, the next takes at most half as many steps, then half again. This geometric series converges, so the result diverges within $2 \cdot t_{\text{add one}}$ steps. Only discretization prevents infinite values.

**The doubling time is a very rapidly decreasing function of time** — at least halving with each new item. By comparison, exponential growth (constant doubling time) is extremely mild.

### Extinction Equilibrium

For any $(M_0, \alpha)$, there exists a critical extinction rate above which extinction dominates:

$$\mu_{\text{critical}} = \frac{\alpha(2^{M_0} - M_0 - 1)}{M_0}$$

If $\mu_{\text{critical}} > 1$: initial growth so rapid extinction cannot overwhelm it. Otherwise: fine-tuning required for extinction to have lasting effect. The evolution is *unstable* about the constant value at $\mu = \mu_{\text{critical}}$ — without fine-tuning (or in presence of stochastic effects), extinction soon either dominates completely or becomes negligible. Hence $\mu$ is not a very crucial parameter.

Even highly-tuned initial extinction rate does not substantially prolong the period before blow-up (Table 1 demonstrates this numerically).

### Universality

The shape of the curve is universal with respect to $M_0$: the evolution onward from a given value of $M_t$ matches what would be obtained if that value were chosen as the initial condition $M_0$.

---

## Case 2: Power-Law $\alpha_i$

A more realistic parametrization where combining more objects is progressively harder:

$$\alpha_i = \frac{\alpha}{a^{i-1}}$$

Each additional element reduces success rate by factor $a$. The summation is still analytically soluble:

$$M_{t+1} = M_t(1 - \mu) + \alpha a \left[\left(1 + \frac{1}{a}\right)^{M_t} - \frac{M_t}{a} - 1\right]$$

This is structurally very similar to Case 1 (recovered for $a = 1$). The factor $a$ appears in the base of the exponential: $(1 + 1/a)^{M_t}$ instead of $2^{M_t}$, slowing growth but not changing the qualitative outcome.

**Blow-up time estimate** (for $a \gg M_0$):

$$t_{\text{blow-up}} \approx \frac{a}{\alpha(M_0 - 1)}$$

The plateau length grows roughly proportional to $a$ for a given $M_0$. Reducing combinatorial efficiency stretches the plateau but does not prevent blow-up.

**Note on Steel et al.**: Their stochastic implementation is not a discretization of the TAP equation itself — it requires small calculational timesteps to keep creation probabilities below unity. It is a stochastic discretization of a *continuum approximation* to the original equation. Their numerical analysis also assumes a fixed upper limit (usually $i_{\max} = 4$) in the summation.

---

## Continuous Blow-Up (Section 4, Published Version)

The discrete TAP equation technically never diverges (every term is finite), but any continuous approximation diverges in finite time. Taking the continuous limit $dt \to 0$:

$$\frac{dM(t)}{dt} = F[M]$$

where $F[M]$ is the analytically-continued evolution function. The time to reach infinity from initial value $M_0$:

$$t_{\text{blow-up}} = \int_{M_0}^{\infty} \frac{dM}{F[M]}$$

For any sufficiently rapidly growing $F[M]$ (which TAP always gives), this integral converges to a finite value — rigorously establishing finite-time blow-up. Applied to the power-law case with $M_0 = 2$, $a = 1000$, $\alpha = 1$: the integral gives $t_{\text{blow-up}} \approx 1381$, very close to the discrete model's computed blow-up time of 1392.

This provides a rigorously-defined estimate that can also be interpreted as the time before which the discrete timestep model must lose viability due to the rate of evolution exceeding the timestep resolution.

### Applied Contexts (Published Version)

The published version explicitly connects TAP dynamics to:
- **GDP per capita growth** and **manufactured goods diversity** (Koppl et al.)
- **US patent family trees** — combinatorial innovation in technology
- **The Great Acceleration** (Steffen et al. 2015) — environmental/climatic indicators showing hockey-stick trajectories
- **Technological Singularity** (Vinge 1993, Kurzweil 2006/2024) — TAP as the mathematical substrate underlying singularity predictions
- **Bellina et al. (2024)** — connection to real-world macroevolutionary singularities
- **Devereaux (2021)** — demonstrates the impossibility of controlling TAP-type instability via economic policy decisions

---

## Late-Time Behavior: Tetration

At late times, the TAP equation with constant $\alpha$ simplifies to:

$$M_{t+1} \approx \alpha \cdot 2^{M_t}$$

At every step the current value is shifted into the exponent, producing an exponential tower (tetration):

$$M_t = 10^{10^{10^{10^{\cdots}}}}$$

The number of 10s in the tower and the top power depend on how many steps were taken before entering this regime and at what value.

---

## TAP Equation Variants

### Variant 1: Two-Scale TAP

Motivated by contexts where single-object evolution dominates (e.g., dog breeding — mutation and breeding dominate over genome-merging with other species):

$$M_{t+1} = M_t(1 - \mu) + \alpha_1 M_t + \sum_{i=2}^{M_t} \alpha \binom{M_t}{i}$$

Where $\alpha_1 \gg \alpha$ is the single-object evolution rate. The evolution rate $\alpha_1$ is perfectly degenerate with extinction $\mu$, so all previous results apply but with $\alpha_1 > \mu$ corresponding to anti-extinction.

**New behavior**: Instead of plateau → explosion, the two-scale TAP produces **exponential growth → explosion**. The exponential phase (driven by the $\alpha_1 M_t$ term) gives way to TAP blow-up when $M_t$ reaches a value where the combinatoric term (suppressed by small $\alpha$) overtakes.

**Critical finding**: The exponential portion of the curve is very accurately exponential, giving essentially *no warning* of the impending sharp transition to TAP blow-up.

### Variant 2: Differential TAP

Corrects for regeneration of the same items at each step by subtracting previous step's creation:

$$M_{t+1} = M_0 + \sum_{i=2}^{M_t} \binom{M_t}{i}$$

Only slightly moderates growth compared to original (sequence for $M_0 = 2$: {2, 3, 6, 59, $6 \times 10^{17}$, ...}). Not tractable for $\alpha_i < 1$ since tracking individual created objects is required.

### Variant 3: Logistic TAP (Proposed, Not Solved)

Add a $-M^2$ suppression term (analogous to logistic mapping in population dynamics) to model resource competition among inventors. Goal: moderate later stages of explosion. Reserved for future investigation.

---

## Key Results and Implications

1. **Hockey-stick dynamics**: The generic TAP behavior is an extended plateau followed by sudden super-exponential blow-up. This shape appears to capture observed form of GDP growth, patent diversity, manufactured goods diversity, and potentially environmental catastrophe indicators.

2. **Blow-up is essentially unpredictable**: The transition to blow-up is sudden, explosive, and *not foreshadowed by any features of the curves until its onset*. There is no pre-blow-up signature that could serve as early warning.

3. **Extinction is either dominant or irrelevant**: Due to the instability at $\mu_{\text{critical}}$, extinction either overwhelms the system completely or becomes negligible. No stable intermediate regime exists without extreme fine-tuning.

4. **Combinatorial suppression stretches but does not prevent**: Increasing $a$ (making combinations harder) lengthens the plateau linearly but does not change the qualitative outcome. The explosion always comes.

5. **Two-scale TAP is more insidious**: When single-object evolution creates exponential pre-blow-up growth, the transition to TAP blow-up is even harder to detect because the exponential phase looks "normal."

6. **Environmental implications**: If TAP underlies economic development, it must also underlie environmental exploitation. The blow-up being unpredictable and unstoppable once underway has implications for species survival on the timescale of decades.

---

## Key Concepts for sigma-TAP Integration

1. **The TAP equation as implemented in sigma-TAP**: The simulator uses the power-law variant (Case 2) with parameters $\alpha$ (base combination rate), $a$ (combinatorial suppression factor), and $\mu$ (extinction rate). The equation $B = \alpha \cdot a \cdot [(1 + 1/a)^M - M/a - 1]$ is computed in `compute_birth_term()`.

2. **Three parameters govern behavior**: $\alpha$ controls the base combination rate (timescale), $a$ controls combinatorial suppression (plateau length), $\mu$ controls extinction. In sigma-TAP: $\alpha = 5 \times 10^{-3}$, $a = 3.0$ or $8.0$, $\mu = 0.005$.

3. **Blow-up as consummation**: The TAP blow-up represents the system reaching its term — praxis completing. In sigma-TAP this maps to agents approaching $m_{\text{cap}}$, requiring capping mechanisms.

4. **Universality of curve shape**: The evolution from any point $M_t$ is identical to starting fresh at $M_0 = M_t$. This means agent history before a given point is irrelevant to subsequent dynamics (Markov property of the TAP equation). The TAPS signature and L-matrix add non-Markovian memory that the bare equation lacks.

5. **Extinction instability = metathetic fragility**: The critical extinction rate creates a knife-edge: systems either grow explosively or collapse. No stable intermediate. In sigma-TAP, the sigma feedback (affordance-exposure accumulation) is intended to modulate this instability — creating the stable intermediate that the bare equation cannot produce.

6. **Two-scale TAP = different growth regimes**: The two-scale variant (exponential → blow-up) maps to environments where single-agent evolution ($\alpha_1$, self-metathesis) dominates before combinatorial innovation ($\alpha$, cross-metathesis) takes over. The transition between these regimes is undetectable from within.

7. **Logistic TAP as open problem**: The question of whether resource competition can moderate blow-up is unsolved in the paper. In sigma-TAP, the sigma function (scarcity-mediated feedback) serves precisely this role — it is an implementation of a logistic-like TAP modulator.
