# Analytic Relationship of Relative Synchronizability to Network Structure and Motifs — Slideshow Distillation

**Source**: Lizier, J.T. "Analytic relationship of relative synchronizability to network structure and motifs." Presentation slides (26 slides). Centre for Complex Systems, School of Computer Science, The University of Sydney. Accompanies Lizier et al., *PNAS* 120(37), e2303332120 (2023).

**Relationship to main paper distillation**: This is a SUPPLEMENTARY distillation. The main paper distillation is at [Lizier-Synchronizability-Motifs.md](Lizier-Synchronizability-Motifs.md). This slideshow provides presentation-level framing, pedagogical scaffolding, and several interpretive insights not foregrounded in the paper itself.

---

## What the Slideshow Adds Beyond the Paper

### 1. Pedagogical Scaffolding (Slides 2-11)

The slideshow builds the problem more explicitly than the paper:

- **Slide 2**: Frames the core problem — "Ideally we need a common framework to understand the dynamic capabilities of various network structures" and to understand "how different dynamic processes relate to each other." This positions synchronizability not as an end in itself but as a **canonical case** for the general structure-dynamics question.

- **Slide 4**: Kuramoto model context — the order parameter o defined as centroid of phases. The population of P oscillators with natural frequency omega_X, phase theta_X, coupled by strength K in network A_XY. Makes explicit that complex network structure has "non-trivial effects" beyond coupling strength.

- **Slides 7-10**: Build up the limitations stack more clearly than the paper:
  - Slide 7: Prior heuristics (lambda_min/lambda_max of L, lambda_max of C) were "limited: heuristic, empirical or symmetric"
  - Slide 9: Even the Korniss (2007) symmetric result is labeled as applying only under "special circumstances"
  - Slide 10: The gap = no general analytic solution connecting structure to synchronizability

- **Slide 11**: Compilation of prior structural findings that the paper explains:
  - Regular/small-world less synchronizable (Atay et al. 2004)
  - Full connectivity increases sync; partitionability decreases it (Atay & Karabacak 2006)
  - Synchronizability of motifs in isolation (Moreno Vega et al. 2004) — with implicit warning about extrapolation
  - Decreases with eigenvalue spread (Korniss 2007)
  - Decreases as network easier to partition (Bauer & Jost 2013)

### 2. The Three-Part Aim (Slides 5, 12)

The slideshow makes the research program's three parts explicit:

1. **Determine whether synchronizable** (sync conditions)
2. **Rank relative quality** (<sigma^2> computation)
3. **Understand how this relates to network structure, including motifs**

Plus a fourth (future work): **Extend to dynamics with coupling delays**

### 3. Key Interpretive Insight: Information Storage Motifs (Slide 22)

**This is the most important addition.** Slide 22 provides an interpretation not prominently featured in the paper:

> "Why are high proportions of closed walk motifs detrimental to sync?"
>
> - "They introduce reinforcement of noise"
> - "Operates differently for discrete vs continuous time"
> - **"They store traces of dynamics in one part of the network only"**
> - "In contrast to other dual-walks, they hinder dissemination or transfer of perturbations"
> - **"These are information storage motifs"**
> - "Sync dynamics as another reason for these motifs' prevalence"

Citing: Lizier, Atay & Jost, "Information storage, loop motifs, and clustered structure in complex networks," Physical Review E 86, 026110 (2012).

**Critical reinterpretation**: Closed dual walk motifs are not merely "bad for sync" — they are **information storage** structures. They **store traces of dynamics locally** rather than disseminating them. This connects synchronizability directly to **information-theoretic** properties of network structure. The reason these motifs are prevalent in real networks (especially biological ones like brain networks) is precisely because they serve a functional role: storing local information at the expense of global coordination.

### 4. Sync Insights Summary (Slides 23-24)

Slide 23 consolidates:
- Explains small-world/regular less synchronizable
- Same for symmetric networks
- Stronger contribution of shorter motifs — **"but all required"** (italicized emphasis not in paper)
- Caveats: about proportions, careful with negative weights
- Better than eigenspectrum heuristics

Slide 24 adds:
- **"Modularity will enhance sync inside module but decrease it across network"** — stated more directly than in the paper
- Auto- vs cross-covariance interpretation reiterated
- Eigenvalue heuristics comparison using the c-varying experiment (most dramatic demonstration)

### 5. Future Direction: Coupling Delays (Slides 12, 25)

The slideshow mentions a second paper in preparation:

> "J.T. Lizier, F. M. Atay, J. Jost, 'Relating synchronizability with delays to network structure and motifs', in preparation."

This is significant for sigma-TAP because the four time-modes (pi, alpha, sigma, Pi) operate at different rates, and coupling between channels involves inherent delays. Time-delayed coupling is the natural regime for multi-scale sigma-TAP dynamics.

---

## Visual Content: Key Diagrams from the Slides

The slideshow contains several diagrams that convey structural information not available from text alone. These are documented here because the main paper distillation was derived from text, and the visual representations add interpretive clarity.

### Slide 5: Network Structure Diagram

A directed weighted network with 5 nodes labeled A, Y, X, Z, B. Key features:
- **Node A** has a **self-loop** (arrow returning to itself) — the only node with one
- Directed edges connect the nodes asymmetrically; a dashed line labeled C_YX shows the coupling weight from node X to node Y
- Two walk paths are highlighted in red/orange: a walk Y→X→Z and a continuation toward B
- This diagram grounds the entire paper's formalism: the connectivity matrix C=[C_YX] encodes exactly this structure, and the question is what ⟨σ²⟩ this structure produces

**sigma-TAP relevance**: The self-loop on node A is the visual prototype of L11 (self-metathesis). The asymmetric directed edges are L12/L21. The question "does the structure of these connections determine synchronizability?" is isomorphic to our question "does the L-matrix wiring pattern determine collective dynamics?"

### Slide 6: Dynamics Comparison Table + Network

A small 4-node colored network (green, red, blue, yellow with directed coupling) illustrates the dynamics. Below it, a comparison table:

| | Discrete (VAR) | Continuous (Ornstein-Uhlenbeck) |
|---|---|---|
| **Dynamics** | x̃(t+1) = x̃(t)C + ε̃(t) | dx̃(t) = -x̃(t)(I-C)dt + dw̃(t) |
| **Sync condition** | \|λ\| < 1 | Re(λ) < 1 |

Both converge to σ²(∞) → 0 without ongoing noise. The 4-node network is color-coded to show individual node states that would synchronize under these dynamics.

### Slide 13: Covariance Projection

Mathematical derivation showing the key step: expressing ⟨σ²⟩ = (1/N) trace(Ω_U), where:
- Ω = lim_{t→∞} ⟨x̃(t)^T x̃(t)⟩ is the network covariance matrix
- x̄U = x̃ - x̄ψ₀ is the state vector minus the synchronized component
- Ω_U = U^T Ω U projects into the space orthogonal to sync
- "If we can write down Ω_U then we can write down ⟨σ²(∞)⟩"

This is the pivotal mathematical move: reducing synchronizability to a covariance computation in the orthogonal complement of the synchronized state.

### Slide 19: Derivation Steps for Continuous-Time

Shows the three-stage expansion of trace(U^T A U):
1. **Expanding the trace**: Separates into diagonal (sum over i) minus off-diagonal (sum over i,j divided by N) terms
2. **Expanding the matrix product**: Makes element-level products C^{m-u}_{ki} C^u_{ki} (same start k, same end i for diagonal) and C^{m-u}_{ki} C^u_{kj} (same start k, different ends i,j for off-diagonal) explicit
3. **Weighted dual-walk counts**: Introduces the notation w^{k→i,m-u}_{k→i,u} = w_{a→b,M₁} w_{a→e,M₂} = (C^{M₁})_{ab} (C^{M₂})_{ae} — making the DUAL WALK structure visible as a product of two walk counts from the same origin

This is the derivation heart: the matrix algebra naturally decomposes into products of walk counts, which ARE the process motifs.

### Slides 20-21: Process Motif Diagrams (CRITICAL VISUAL)

These two slides contain the paper's most important visual contribution — the **process motif diagrams** that make the abstract equations concrete.

**Slide 20 (Continuous-time, Eq. 22 equivalent)**:

The equation ⟨σ²⟩ is shown as a difference of two sums, each illustrated with a colored node diagram:

- **Left term (positive, closed dual walks)**: Two colored nodes — green node k (top/origin) and blue node i (bottom/destination). Two curved arrow paths of lengths (m-u) and u BOTH run from k to i. Both walks start at the same node AND end at the same node. This forms a **closed loop structure**: k→...→i (via path 1) and k→...→i (via path 2). The "closure" is that both paths converge to the same endpoint.

- **Right term (negative, all dual walks, divided by N)**: Same green origin node k, but now two DIFFERENT destination nodes — blue node i and red node j. Walk of length u goes k→i, walk of length (m-u) goes k→j. The endpoints are **open** — they don't converge.

- **Interpretation**: ⟨σ²⟩ measures whether walks from k that have a non-zero walk count to i of length u "preferentially" end at the SAME node i for the complementary length (m-u). If yes → closed walks dominate → BAD for sync.

**Slide 21 (Discrete-time, Eq. 23 equivalent)**:

Identical motif diagram topology to slide 20, but with the additional constraint: both walks must be of the **same length** u (not lengths m-u and u). The "of same length" is given special emphasis. In discrete time, only walks of matching length contribute to the deviation.

**sigma-TAP relevance of these diagrams**: The visual makes clear that the "closure" of dual walks is about **convergence** — two independent processes originating from the same source both returning to the same target. In sigma-TAP terms: if agent k's self-metathesis path (L11) and its cross-metathesis path (L12/L21) both loop back to reinforce the same local state, this is a closed dual walk. It stores information locally (good for identity persistence) but hinders global synchronization (bad for collective coordination). The visual also shows why the SUBTRACTION matters — it's the PROPORTION of closed walks relative to all walks that determines ⟨σ²⟩, not the absolute count.

### Slide 22: Information Storage Motifs (CRITICAL VISUAL)

This slide contains TWO distinct diagrams:

**Top diagram**: Closed walk motif illustrations showing colored node pairs with curved loop arrows. These depict the closed dual walks from slides 20-21 but now explicitly labeled as "information storage" structures. The loops represent "traces of dynamics in one part of the network only."

**Bottom diagram**: A directed network with nodes labeled o, k, i, j, m, l showing **clustered structure with internal loops**. This is from Lizier, Atay & Jost (2012) and illustrates how loop motifs CREATE clustered structure in complex networks. The diagram shows:
- Tight internal loops within clusters (nodes linked by bidirectional or short-cycle edges)
- Weaker between-cluster connections
- The visual argument: networks with high proportions of closed walk motifs will naturally exhibit clustered/modular topology

This is the visual bridge between the synchronizability result and the information-theoretic interpretation: closed walks = loops = information storage = clustered structure = modularity = local sync + global desync.

**sigma-TAP relevance**: The bottom diagram is essentially a visual representation of the family group architecture (§5.4). Tight internal loops within a cluster = family members sharing TAPS signatures (intra-group synchronization via L11+L12/L21). Weak between-cluster links = L22-mediated environmental coupling. The reason real networks (including brain networks) maintain these loop-rich clusters is precisely because they serve a functional purpose: local information storage (practico-inert) at the expense of global homogenization.

---

## Key Concepts for sigma-TAP Integration (Supplementary to Main Distillation)

1. **Closed dual walks = information storage motifs, not just sync-hindering structures**. This reframes the entire relationship to sigma-TAP: L11 (self-metathesis) and reciprocal L12/L21 loops are not merely "bad for global coordination" — they are the mechanism by which agents **store their own history locally**. The TAPS signature is precisely such stored information. Without local storage (closed walks), agents would have no persistent identity. The tension is between storing enough locally to maintain identity (TAPS signature persistence) and disseminating enough globally to participate in cross-metathesis (network synchronization).

2. **"Traces of dynamics in one part of the network only"** = the practico-inert at the agent level. Sedimented past praxis (the practico-inert) is stored locally in the agent's type_set, L-matrix, and TAPS signature. These are "traces of dynamics" that remain in one part of the network. The practico-inert stores potential energy (gas/fuel metaphor from §5.43) — and this storage IS the closed dual walk motif structure.

3. **Modularity enhances local sync, decreases global sync** = family group architecture confirmed. Slide 24 states this directly. In sigma-TAP: family groups (§5.4) should exhibit internal TAPS signature convergence (local synchronization) while maintaining inter-group diversity (global desynchronization). The L22 channel provides weak global coupling to prevent complete fragmentation.

4. **"All [motif orders] required"** — cannot shortcut with low-order approximation alone for clustered networks. For sigma-TAP: when topology is introduced (Stage 3B Phase 5), the system will develop clustered structure. At that point, low-order metrics (direct pairwise similarity) will not suffice to characterize system synchronization — longer-range walk structures will matter. The Youn ratio (0.60 target) may need to be evaluated via full motif analysis, not just pairwise Jaccard.

5. **Coupling delays = four time-modes**. The forthcoming Lizier/Atay/Jost paper on delays is directly relevant to the differential time dilation mechanism (§5.25): pi (fastest, L11), alpha (L12), sigma (L21), Pi (slowest, L22). Coupling delays between channels will create different contributions to synchronizability depending on the relative rates. When available, that paper should be acquired.
