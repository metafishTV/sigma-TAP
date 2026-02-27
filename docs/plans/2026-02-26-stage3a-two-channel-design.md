# Stage 3A: Two-Channel Consummation — Per-Agent TAPS Signatures

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix the Youn ratio (stuck at 1.0) by giving agents dispositional
signatures derived from TAPS sub-modes, then using signature similarity to
classify cross-metathesis events as novel (β·B) or absorptive (η·H).

**Architecture:** Per-agent event ledger (Emery L-matrix channels) → 4-letter
TAPS signature → three-level tension classification at cross-metathesis time.

**Tech Stack:** Python 3.12, existing metathetic engine, existing TAPS scoring.

**Claim policy label:** exploratory

---

## 1. Problem Statement

Across 576 simulations (192 parameter combinations × 3 seeds), the Youn ratio
is stuck at 1.0 — every cross-metathesis event classifies as novel, zero as
absorptive. The empirical target from Youn et al. (2015) is ≈0.60
exploration / 0.40 exploitation, a ratio stable across 220 years of US patent
data and independently observed in metabolic, ecological, and evolutionary
systems.

### 1.1 Root Cause

The current classification rule (metathetic.py line 623):

```python
if L > G:   # absorptive
else:       # novel (including tie L == G)
```

Where L = Jaccard similarity and G = goal alignment (clamped ≥ 0). For
absorptive to fire, L must **strictly exceed** G. But agents that pass the
eligibility gate (L + G > threshold) tend to have L ≈ G due to structural
correlation: agents who share types also grow together. The tie-break always
favors novel. Result: absorptive never fires.

### 1.2 Theoretical Motivation (§3.5 Two-Channel Hypothesis)

The sigma-TAP Xi dynamics track two channels:

    dXi = β·B + η·H

- **β·B** — manifest births: new feature activations (the novel channel)
- **η·H** — implicate densification: experience deepening without new features
  (the absorptive channel)

Currently, H = h_decay · Xi is a damping term with no independent information
source. The two-channel hypothesis: absorptive events are not failed novelty —
they are the η·H channel made concrete. Interaction that deepens existing
repertoire rather than creating new territory is exploitation, and it should
constitute ≈40% of cross-metathesis events.

---

## 2. Design: Per-Agent L-Matrix Ledger

### 2.1 Emery's Four Channels

The per-agent event ledger explicitly tracks Emery & Trist's (1965) four
channels of causal texture, applied at the individual agent level:

| Channel | Meaning | Per-Agent Field |
|---------|---------|-----------------|
| **L11** | System-internal (intrapraxis) | `n_self_metatheses` |
| **L12** | System → environment (outward projection) | `n_novel_cross` + `n_absorptive_given` |
| **L21** | Environment → system (inward reception) | `n_absorptive_received` |
| **L22** | Environment → environment (causal texture) | `n_env_transitions` |

All integer counters. Incremented in existing metathesis code paths.
Net cost: 5 integers per agent.

### 2.2 Non-Ergodicity of Intrapraxis

The agent's L11 channel (self-metathesis, intrapraxis) is non-ergodic:
path-dependent, irreversible, unique to the agent's history. From each
agent's perspective, the interpraxitive channels (L12, L21, L22) are ergodic
in degrees — the degree of ergodicity determined by the strength of
interrelations (bonds, conjunctions) between agents. Strongly bonded agents
are less ergodic to each other; weakly connected agents more interchangeable.

### 2.3 L12 / L22 Conjugacy

High L22 participation (capacity to shape environmental causal texture)
necessarily entails high conditioning by collective L12 from lower strata.
One agent's L12 (system → environment) is another agent's L21
(environment → system). The more channels of environmental texture an agent
shapes, the more channels of constituent action condition it. High L22
capacity and high L12 conditioning are conjugate.

---

## 3. Design: TAPS Signature Derivation

From the L-matrix ledger, each agent derives a 4-letter dispositional
signature reflecting its current operating mode across the four TAPS
dimensions.

### 3.1 T-letter (Transvolution)

Inward-folding vs outward-unfolding, computed from event channel balance:

- Involution weight = L11 + L21 (self-metathesis + absorptive received)
- Evolution weight = L12 + L22 (novel cross + absorptive given + env transitions)

| Condition | Letter | Meaning |
|-----------|--------|---------|
| Involution > evolution × 1.2 | **I** | Involution-dominant |
| Evolution > involution × 1.2 | **E** | Evolution-dominant |
| Neither | **T** | Transvolution (balanced condensation) |

The balanced case uses **T** (not C) for symmetry with the dimension name
and so that a three-way balance across all four dimensions produces the
signature **TIE** (T-dimension balanced, etc.) — extremely rare but
aesthetically satisfying.

### 3.2 A-letter (Anopression)

Dominant creative pressure, derived from agent's recent dynamics:

- **Expression (E):** Agent has positive recent dM AND affordance_score > 0.5
  (actively growing in a connected context — free expression)
- **Impression (I):** L21 events dominate recent history (absorbing, being
  shaped by environment — bounded reception)
- **Adpression (A):** Recent self-metathesis or disintegration event
  (punctuated transformation under pressure)

Assigned by strongest condition. Default **E** if none dominate (expression
as the unmarked case — the agent is simply being).

### 3.3 P-letter (Praxis)

Action modality, derived from channel dominance:

- Consumption weight: L21 events (disintegration serving integration —
  reflective cycle, agent metabolizes)
- Consummation weight: L12 events (integration serving transformation —
  projective cycle, agent produces)

| Condition | Letter | Meaning |
|-----------|--------|---------|
| Consumption clearly dominates | **R** | Reflective (metabolizing) |
| Consummation clearly dominates | **U** | Projective (producing) |
| Balanced | **X** | Pure action (rare — overlap of both limitrophes) |

### 3.4 S-letter (Syntegration)

Structural mode, derived from agent state and recent events:

| Dominant condition | Letter | Meaning |
|--------------------|--------|---------|
| Recent disintegration events | **D** | Disintegration |
| Dormant / high steps_since_metathesis | **P** | Preservation |
| L21 absorptive events dominate | **I** | Integration |
| L11 + novel L12 dominate | **S** | Synthesis |

### 3.5 Windowing

Stage 3A: signatures computed from **cumulative** event counts (lifetime).
The agent's dispositional character is the integral of its history.

Future refinement: windowed (recent-only) signatures, giving both
"character" (lifetime) and "mood" (recent window).

---

## 4. Design: Three-Level Tension Classification

### 4.1 System-Level Tension Spectrum

| System tension | Youn ratio | Character |
|----------------|------------|-----------|
| **Over-tension** | → 1.0 | Pathological overproduction — no consolidation (current state) |
| **High tension** | ≈ 0.6 | Productive — consistently making well (Youn target) |
| **Mid tension** | ≈ 0.5 | Balanced — syntegrative equilibrium |
| **Low tension** | < 0.5 | Absorptive-dominant — densification, possibly stagnating |

High tension (0.6) is the productive operating point, not an extreme. The
current state (1.0) is over-tension. The mechanism should naturally move the
system from over-tension toward high tension.

### 4.2 Per-Event Classification

When two agents pass the eligibility gate (unchanged: L + G > (W1 + W2) ×
threshold), compare their TAPS signatures:

**Signature similarity** = number of matching letters out of 4.

| Match count | Tension level | Classification | Channel |
|-------------|---------------|----------------|---------|
| 3–4 matches | Low tension | **Absorptive** (Mode 2) | η·H — densification within shared dispositional space |
| 2 matches | Mid tension | **Secondary rule*** | Boundary — could go either way |
| 0–1 matches | High tension | **Novel** (Mode 3) | β·B — exploration across dispositional boundaries |

*Mid-tension secondary rule: fall back to existing L vs G comparison as
tiebreaker. If L > G → absorptive; if L ≤ G → novel. The old rule, in a
context where it can actually function.

### 4.3 What's Preserved

- Eligibility gate unchanged — same events fire, same frequency
- Total cross-metathesis count unchanged
- Only the novel/absorptive classification changes
- Heaps' law, Taalbi linearity, power-law exponent unaffected (they don't
  depend on novel/absorptive classification)

### 4.4 What Changes

- Youn ratio moves from 1.0 toward 0.6 target
- TAPS diagnostic scores shift (more absorptive → higher impression scores,
  different involution/evolution balance)
- ACTION_MODALITY_WEIGHTS calibration may need validation against new
  event distribution

---

## 5. Forward Notes (Bookmarked, Not Implemented in 3A)

### 5.1 Shadow Ticks (Anapressive Complement to Affordance Ticks)

Binary ticks of systemic pressure — the constraining face of affordance.
Three decay/generation regimes:

- **Connected/supported:** shadow ticks decay quickly (relational context
  provides processing capacity)
- **Isolated:** shadow ticks linger (slow decay, no relational support)
- **Anomic:** shadow ticks actively increase — agent is embedded in L22 but
  disconnected from normative fabric, so environmental causal texture
  generates pressure that cannot be processed. Internalized, identified
  isolation within a social field (Durkheim).

Decay rate modulated by stranger → friend transition rate (relational
connectivity). Generation rate modulated by L22 turbulence without
corresponding relational support.

### 5.2 Anapressive Transformation by L22 Participation Capacity

Per-agent anapressive load as a function of raw pressure modulated by L22
engagement capacity. Agents with high L22 participation (executives,
presidents — wide collective choice, narrow personal choice) experience
anapressive pressure transformed (rooted, inverted, compressed) by their
participatory role. Agents in isolation experience raw anapressive pressure
at full intensity.

Potentially gauge-invariant: the transformation depends on relative
participation, not absolute scale. Determined by degrees of participation,
adaptation, and interpraxitive affect.

Exact mathematical form (square root, inversion, imaginary mirror, etc.)
to be explored empirically once per-agent profiles generate data.

### 5.3 Durkheim Disintegration from TAPS Profile Collapse

Disintegration as an active outcome of L22 overdetermination, not passive
timeout. Detectable from TAPS signature evolution: rising compression +
falling expression + collapsing action balance → agent's configuration
space "desaturates" until self-maintenance becomes impossible.

Intrapraxitive feedforward loops not simply conditioned but determined by
intrapraxitive feedback loops from L22, which overdetermine and desaturate
the agent's configuration space. The disintegration is *produced* by the
dynamics, not merely suffered.

### 5.4 Family Groups (Stage 3B)

Non-ergodic group = group-in-fusion. Non-ergodic collective = fused
collective. Both conditional upon the individuals composing them, even
under strong L22 conditioning.

Family groups, once established, convert quasi-ergodic interpraxis into
more stable non-ergodic, path-dependent interpraxis, expanding both group
configuration possibilities and individual agent possibilities.

Groups should have learning/density curves tracking configurational
maturity. Max syntegration for min prax coupled with free expression and
bounded involution (bounded to the evolution of the metathesis they're
nested within).

### 5.5 Distance-Based Observation Decay

Events observed by nearby agents cause internalization and response —
"impressed by the event witnessed, even if indirect." Effect diminishes
with distance from event.

Stage 3A proxy: Jaccard similarity as distance metric.
Stage 3B: true topological distance from agent-agent adjacency tracking.

### 5.6 Overconservation Detection

Agents whose local signature diverges from ensemble field not because
they're exploring but because they're stuck in a past pattern — detectable
as high tension with low action balance. The ensemble orientation may have
shifted across generations while current agents misapprehend the situation
and revert to overconservation.

### 5.7 Emery's Four Channels — Consistent Reference

The per-agent ledger, TAPS signature derivation, and cross-metathesis
classification should consistently reference Emery's L-matrix channels
(L11, L12, L21, L22) in code comments, docstrings, and documentation.

### 5.8 Praxiological Time and the Deferral Mechanism

Total syntegrative consummation — the completion of an agent's self-praxis —
equals death. A system with no remaining tension, no unactualized potential,
no gap between what it is and what it could be, is a closed system. The
agent becomes a product, no longer a process.

Therefore praxis must be perpetually deferred. The deferral is not a failure
but a structural condition of continued existence. The distance from
consummation completion IS the agent's own non-ergodic time — not clock
time, not simulation steps, but the measure of remaining possibility space.

- Agent with vast unresolved tension = agent with time
- Agent whose signature has stabilized, action balance converged,
  praxis approaching closure = agent running out of time

This reframes `steps_since_metathesis` as a crude proxy for what it
actually measures: how close the agent is to having nothing left to happen.

The 40% exploitation fraction (Youn target) may be the deferral itself
made quantitative: what must remain unfinished, recycled, absorbed back
as densification rather than expansion. Push past it toward 100%
consummation and you model agents dying without the system knowing it.

Three modes of death, each with distinct causal structure:
1. **Stagnation** (current): insufficient praxis → timeout → disintegration
2. **L22 overdetermination** (Durkheim/anomie, §5.3): external pressure
   desaturates configuration space → disintegration
3. **Consummation completion** (this note): internal completion → no
   tension remains → disintegration. Death by fulfillment.

The deferral mechanism (implementation TBD) prevents pathway 3. Detectable
from TAPS signature: signature stabilization, action balance convergence,
signature diversity within agent's history collapsing.

### 5.9 Embedded Sub-Mode Combinatorics

The four-letter TAPS signature pulls from primary dimension labels. A
deeper combinatorial scheme using embedded sub-mode letters
(T→I/E/T, A→E/I/A, P→R/U/X, S→D/P/I/S) is noted for future
investigation. Could produce richer dispositional names but risks
confusing notation.

### 5.10 Irreducible Alterity and Filial Relations

Per Levinas: fraternity requires individualities whose "singularity
consists in each referring to itself" — a logical status not reducible
to genus differences. An individual "having a common genus with another
individual would not be removed enough from it." Society must be a
fraternal community to be consummative.

The TAPS signature is our attempt at an irreducible alterity account —
a dispositional name unique to each agent's history. But the
_MIN_SIG_EVENTS threshold reveals a gap: before enough events
accumulate, agents ARE reducible to genus (the default "TEXS"). They
have no alterity yet.

Currently, cross-metathesis conflates two distinct relation types:

- **Filial** (novel cross): parent → child lineage. But children start
  with zero L-matrix history — no inherited alterity from parents.
- **Alliance** (absorptive cross): but absorption destroys one party.
  True alliance would preserve both while creating mutual conditioning.

Stage 3B should distinguish these:

1. **Filial inheritance**: children inherit weighted L-matrix history
   from parents, giving them dispositional character from birth (not
   blank-slate "TEXS"). Both parent IDs stored (matrimonial AND
   patrimonial lines — both genetically functional even if one is
   unrecorded). The child's unicity "does and does not coincide"
   with the parents' unicities; the non-coincidence is realized
   concretely in the sibling relation.
2. **Alliance mode**: a third cross-metathesis outcome where both agents
   persist with a bond (L12/L21 coupling), rather than one consuming
   the other. Alliance strengthens non-ergodic interpraxis without
   destroying either party's singularity. Per Deleuze & Guattari:
   alliances are **rhizomatic** (any-point-to-any-point, horizontal,
   forming and dissolving — like slime mold), while filiations are
   **arborescent** (tree-like, vertical, irreversible lineage).
3. **Lineage tracking**: parent IDs stored on child agents, enabling
   genealogical analysis and filial group detection. The filial tree
   intertwines matrimonial and patrimonial lines.
4. **Asymmetric cross-metathesis**: per Levinas, the I-Other relation
   is constitutively asymmetric — "the inevitable orientation of being
   'starting from oneself' toward 'the Other'." The initiating and
   responding agents in cross-metathesis should have distinct roles.

### 5.11 Praxitive Syntegration vs. Syntegrative Praxis (with Agent/Artifact Assignment)

Two complementary modes of agent engagement:

- **Syntegrative praxis** (making internal → skill development): agent
  develops an internal capacity. Woodworking-as-skill is outward-tending
  (practiced on material); love-as-skill is inward-tending (restructures
  relational capacity). More about "how I am." Maps to L11+L21 events.
- **Praxitive syntegration** (making external → production): agent
  externalizes capacity into an artifact or effect. A chair is more
  concrete (directly legible); a painting is more abstract (encodes
  internal states). More about "how I do." Maps to L12 events and
  artifact production (§5.14).

Neither is purely abstract or concrete — each has a predominant tendency.
Less abstract syntegrative praxis → more concrete (outward-tending
internal development). Less concrete praxitive syntegration → more
abstract (inward-tending outward development).

**Key assignment**: every praxis involves both syntegration and
praxis-as-externalization, but each entity type has a characteristic
**focus** (predomination in relation, not operation in exclusion —
breaking this would violate the law of included adjacency):

- **Agents do praxitive syntegration**: their primary work is
  externalizing — integrating outward. Their praxis is projection-
  reflection, externalization-as-internalization-deferred.
- **Artifacts do syntegrative praxis**: their primary work is
  making-internal — they hold and re-present what agents have
  externalized, and that holding IS their praxis.

This captures predomination / focus, not exclusive operation.

### 5.12 Surplus Value vs. Expressed Value

- **Surplus/innovation value** can tend toward infinity. It is
  conditioned by finite actuality (the agent's actual history). This is
  the ever-growing type-space (`_next_type_id`), the possibility frontier.
- **Normal/expressed/alpha value** is finite. It is determined by
  infinite possibility but bounded in expression. This is `k_total`
  and `M_total` at any given step.

The Youn ratio may be the quantitative expression of this tension:
exploitation draws finite value from known possibilities; exploration
reaches into the infinite surplus.

### 5.13 Praxistatic Surplus and Agent Layering

**Praxistatic surplus**: the static inertia of the past within the
present, mediated by the agent's direct relations. The bulk of the past
acts as a weighted layer — indirectly active toward the agent through
memory and deferred praxes whose effects propagate through time.

This suggests a **two-layer agent architecture**:

- **Manifest/conscious layer**: the current TAPS signature, active
  type-set, L-matrix ledger. Future-oriented — what the agent is doing
  and becoming. Used for cross-metathesis classification.
- **Latent/subconscious layer**: the accumulated praxistatic surplus,
  deferred effects, shadow ticks (§5.1). Past-oriented — what the
  agent has been and what presses from behind. Slower-moving, heavier
  structure that modulates the manifest layer.

Both layers are present-situated. The manifest is the agent's engagement
with future possibility; the latent is the agent's accumulated
historical weight. The tension between them constitutes the agent's
temporal experience.

Implementation note: the latent layer could be modeled as exponentially
weighted moving averages of L-matrix ratios, providing a "deep
signature" that changes slowly vs. the manifest signature that responds
to recent events. The windowed "mood" vs. cumulative "character"
distinction (previously bookmarked) fits here.

The latent S-layer in particular captures **habit formation**: praxes
repeated enough to sink from manifest (conscious, effortful) to latent
(subconscious, automatic). An agent whose latent S is "S" (synthesis)
but manifest S is "D" (disintegration) is a constructive agent in
crisis — habitual creativity under destructive pressure.

### 5.14 Artifact Agents

A second agent type: **ArtifactAgent**. Artifacts are produced by
MetatheticAgents and persist independently. They serve to simulate any
form of media, from cave paintings to novels, that preserve active
fragments of innovation.

Key properties:

1. **Always >1**: innovation is always partially externalized into
   multiple artifacts. No singleton artifacts (prevents structural
   equivalence with agents).
2. **Preserve active fragments**: carry a subset of the producing
   agent's type-set and/or knowledge, frozen at moment of production.
   More complex/refined artifacts preserve more information.
3. **Quality depends on three factors**:
   - *Time*: artifacts degrade or become less legible over time
   - *Relativity*: cross-identity function / degree of relatability
     between artifact and encountering agent. Sumerians reading
     cuneiform vs. moderns reading Sumerian.
   - *Interactivity*: clay tablet (low) < notebook (medium) <
     newspaper (medium-high) < cell phone (high). Higher interactivity
     = artifact behaves more like an agent.
4. **Survive extinction events**: when all agents in a lineage go
   dormant, artifacts can persist and be encountered by future agents.
   Manuscripts surviving Bronze Age collapse while populations did not.
   Creates a cultural transmission channel that bypasses filial
   inheritance — a horizontal, time-spanning rhizomatic link.
5. **Future**: artifacts with varying interactivity parameters approach
   agent-like behavior (responsive institutions, AI systems). The
   spectrum from pure trace (fossil) to near-agent (interactive system)
   is the history of media technology.

Artifacts connect praxitive syntegration (§5.11) to the persistence
layer: they are the externalized products that outlast their producers.
Per the unificity framework: artifacts are conserved traces of praxis
in configuration space — "praxis is always conserved at least as an
irreversible change, a trace, in the mesh of configuration space."

**Artifact substrate classes** (future work):

- **Oral**: songs, stories, spoken traditions. No physical substrate;
  persist only through agent-to-agent transmission. Most fragile but
  most intimate (requires direct relational contact). A rhizomatic
  link requiring living hosts.
- **Physical**: tools, buildings, manuscripts. Persist independently
  of agents but degrade over time. The Bronze Age manuscripts that
  survive population collapse.
- **Interactive**: notebooks, cell phones, responsive institutions.
  Bidirectional — the artifact is acted upon and acts back.

**Containment asymmetries and nesting rules**: a song can be
**incarnated** within a play (performed, embodied, materially present
in the play's runtime). A play can only be **disincarnated** within a
song (referenced, alluded to, carried as information without material
embodiment). The asymmetry is ontological: incarnated containment
gives the contained entity a body; disincarnated containment gives it
only a name. This forms a partial order on artifact classes with two
typed relations: **incarnation** (X is materially embodied inside Y)
vs. **disincarnation** (X is alluded to within Y without embodiment).

**Meshability** (term after Arno Gruen): some artifacts invite
participatory engagement (rhizomatic), others enforce spectatorial
distance (arborescent). "Meshability" captures the texture of how
things fit together — compatibility of form, not dominance. Replaces
the earlier "joinability" term.

**Vocality / directionality spectrum**: a systematic vocabulary for
interaction topology, mapping directly onto both the simulation
architecture and the philosophical framework:

| Mode | Vocality | Directionality | Maps to | Implies |
|------|----------|----------------|---------|---------|
| **Mono-** | Monovocal | Monodirectional | Singularity | Irreducible point — one voice, one direction |
| **Biuni-** | Biunivocal | Biunidirectional | Unity | Two-that-are-one, dyadic exchange; **locality** |
| **Poly-** | Polyvocal | Polydirectional | Multiplicity | Many voices, irreducible to pairwise; **nonlocality** |
| **Meso-** | Mesovocal | Mesodirectional / mesocosmic | Unificity | Context — neither one nor many but consummation of both |

**Locality mapping**: biuni- implies locality because two-point
exchange has a definite *here* and *there* (even self↔self is local
to that agent). Poly- implies nonlocality because the interaction
cannot be decomposed into channels — it is everywhere-at-once in the
ensemble. Meso- mediates: neither pinned to a locale nor dissolved
into everywhere, but the contextual stratum where local and nonlocal
become legible together.

**Biunivocality as self-to-self transmission**: the special case
where both endpoints are the same agent (self↔self = L11 channel =
self-metathesis). This IS the making of time — the agent projects
forward and reflects backward in the same act, creating temporal
thickness. Praxitive time (§5.7) is literally the rate of
biunivocal exchange. (Aside: if the Universe is the ultimate
singularity-that-is-also-everything, its time-making would be
polydirectional biunivocality — all channels, but only to itself.)

**Current state**: cross-metathesis is strictly biunidirectional
(two agents meet). Whether irreducible polyvocal mechanics are needed
— multi-agent events that cannot be decomposed into pairwise channels
without losing something (ensemble singing, collective ritual, a
market) — is a deep architectural question for later stages. This
distinction is non-trivial and, as far as we know, not formalized in
a general (physics-level) sense.

**Source/sink typology**: a book affects differently than a toothbrush
or a pocketwatch. Each artifact class has distinct affective (how it
changes the agent's A-letter / mode of being) vs. effective (how it
changes the agent's P-letter / mode of action) profiles.

### 5.15 TAPS as Four Existential Questions

Each signature letter answers a distinct existential question.

**Ontological register** (what the mode IS):

| Letter | Question | Sub-modes |
|--------|----------|-----------|
| **T** | How do I *become*? | I (involutionarily), E (evolutionarily), T (transvolutionarily) |
| **A** | How *am* I? | A (adpressively), I (impressively), E (expressively) |
| **P** | How do I *act*? | R (consumptively), U (consummatively), X (balanced) |
| **S** | How do I *create*? | D (disintegratively), P (preservatively), S (synthetically), I (integratively) |

**Phenomenological register** (what the mode DOES for the agent):

| Letter | Question | Elaboration |
|--------|----------|-------------|
| **T** | How do I *grow*? | What direction do I need to grow towards? |
| **A** | How do I *feel*? | What direction do I need to channel my energy? |
| **P** | How do I *see*? | What directions do I need to face to act appropriately? |
| **S** | How do I *create*? | What types of force provide the most harmonious result? |

The two registers gauge each other: the ontological register names
the structure, the phenomenological register names the lived quality.
T is literally growth direction; A is literally the agent's energetic
state; P is literally where it's looking (observation, partner
selection); S is literally what kind of innovation it produces.

Together: the agent's dispositional name — its answer to the four
fundamental questions of praxitive existence. This framing grounds
the signature system in the unificity framework where every agent is
simultaneously a unity (itself), a multiplicity (its type-set and
relations), and a participant in unificity (the ensemble context,
which is "always the most actualized, being the consummation of the
combinatorial consumption of the unities & the multiplicities").

### 5.16 Innovation Decay / Forgetting

Currently, k (knowledge) only increases or redistributes. Agents
cannot *forget*. But not all knowledge is like riding a bike — some
skills degrade without practice, some knowledge becomes obsolete.

Proposed: a per-agent `k_decay_rate` bleeding k each step, modulated by:

- **Type activity**: types involved in recent events (matched in
  Jaccard comparisons, used in cross-metathesis) decay slower —
  they're being "practiced."
- **Latent layer weight**: deeply habituated knowledge (high
  praxistatic surplus) decays slower — procedural/physical memory.
- **Isolation**: agents with fewer connections (low affordance score)
  lose knowledge faster — no relational reinforcement.
- **Acquisition mode**: types from self-metathesis (syntegrative
  praxis, internally developed) decay slower than types from
  absorptive cross (externally received). What you built yourself
  you remember longer than what was given to you.

Different types of knowledge decay differently: procedural (high
physical memory, slow decay) vs. declarative (low physical memory,
faster decay). This creates a natural selection pressure on types —
actively used and deeply integrated knowledge persists; peripheral
and unreinforced knowledge fades.

**Note on random accidents / catastrophic skill loss**: Considered
and deliberately deferred. A random "accident roll" that removes
skill-sets would model the *cause* of adaptation but not the
*process* — and the process is what we're tracking. Agents who lose
capabilities in real life readapt via mode-shifting and
re-specialization, which is already modeled by existing decay +
relearning mechanics. Adding an accident mechanic would add
complication without enriching what we actually measure.

### 5.17 Agent-Ensemble Primacy (Actualized vs. Actualizing)

The unificity framework establishes that neither agents nor ensemble
are derivative of the other. Both are primary, but in different
registers:

- **Ensemble-level metrics** (Youn ratio, signature distribution,
  Heaps exponent, power-law tail) are the **most actualizing**
  stratum. This is where patterns crystallize into observable form,
  where the process becomes legible. But the ensemble does not "do"
  the consummation.

- **Agents** are the **most actualized** stratum. Information is
  encoded at the singularity — each agent is a concrete instantiation
  where alterity is irreducible. But agents alone are not the full
  picture; they are actualized *in* their actualizing work.

The only non-composite entities are the identities at the central
point of each, which IS their alterity. Each identity is relative to
all others (maintaining relativity throughout) but simultaneously and
irreducibly alteric (no other particle is this particle, no other
agent is this agent). The feeding between strata forms a spiral —
arising from and collapsing into life lived by singularities.

**Modeling implication**: ensemble metrics should NOT be treated as
derivative summaries of agent states, nor agent states as mere
particles of ensemble dynamics. Both strata are primary in
equivalence, as long as agents "matter" (i.e., make matter, exist in
form that can move into sets). Otherwise the information is latent or
passed — missed its chance to survive infinitely qua praxis.

This connects to §5.11: agents' actualizing work IS praxitive
syntegration (externalization), while the ensemble's actualizing
pattern is the context — the unificity — within which that work
becomes legible.

### 5.18 Sartre's Critique and the Metathesis Framework

Reference: Appendix to Critique of Dialectical Reason, Vol. 2
(unfinished notes for Book III).

**Core mapping**: Sartre's "totalization" = a naive (undifferentiated)
description of what we call metathesis. "Naive" in the precise sense
that totalization describes the general form of the dialectical spiral
without modal articulation — no TAPS signature, no L-matrix channels,
no way to specify WHICH TYPE of totalization is occurring (growth,
affective, active, creative).

**Totalization/detotalization = absorptive/novel cross-metathesis**:
Sartre: "totalization is the way in which detotality is totalized;
detotalization is a product of totalization, and totalization is a
product of detotalization." Novel cross de-totalizes (breaks apart,
creates new entities). Absorptive cross re-totalizes (merges,
consolidates). The Youn ratio measures the balance of this spiral.

**Totality as enfoldment = metathesis as involution**: Sartre's
totality (completed, closed state) maps to T-letter "I" (involution).
An agent at Youn=0 (all absorptive) is Sartre's totality — dead
because complete. This connects to §5.8 (consummation-completion as
third death pathway).

**Practico-inert = artifact agent**: Sartre's definition: "government
of man by worked matter proportionate to the government of inanimate
matter by man." Artifacts do syntegrative praxis — they hold and
re-present externalized praxis, conditioning agents who encounter
them. The praxis/practico-inert dialectic IS the agent/artifact
asymmetry given modal specificity.

**Counter-finality**: "By acting upon one element, you make another
more fragile." Unintended consequences of innovation. Partially
captured by L22 (environmental texture changes); more fully by future
polydirectional dynamics.

**Verdi passage = absorptive cross working correctly**: Sartre
describes Verdi preserving singing while integrating Wagner's
orchestral contradiction, producing "increased complexity in the
tension and order." This IS successful absorptive cross — the agent
absorbs external elements, preserves core identity (signature), gains
complexity. The alternative (becoming Wagnerian) = absorptive cross
that destroys identity. Our sig_sim >= 3 threshold gatekeeps this.

**"The tool is forged as it forges"** = biunidirectional interactivity.
But Sartre lacks our biuni/poly/meso spectrum to distinguish when
this becomes irreducibly multi-agent.

**What Sartre couldn't resolve**: The "totalization without a
totalizer" — history totalizes without any single agent totalizing.
Sartre was stuck between constituent dialectic (individual praxis)
and constituted dialectic (collective process) without a third term.
Unificity IS that third term: neither agent (unity) nor ensemble
(multiplicity) but the mesovocal context in which both are primary.
This is why the Critique remained unfinished — the architecture needed
the meso- register that Sartre intuited but couldn't formalize.

---

## 6. Success Criteria

1. **Youn ratio moves toward 0.6** across parameter sweep (currently 1.0)
2. **No regression** in Heaps' law, Taalbi linearity, or power-law exponent
3. **Per-agent signatures are interpretable** — agents in similar ecological
   niches develop similar signatures
4. **273 existing tests pass** with no modifications
5. **New tests** cover: ledger tracking, signature derivation, classification
   logic, Youn ratio improvement

---

## 7. Scope Boundary

**In scope (Stage 3A):**
- Per-agent L-matrix ledger (5 counters)
- TAPS signature derivation (4 letters from local data)
- Three-level tension classification (signature similarity → novel/absorptive)
- Jaccard as distance proxy where needed
- Cumulative (lifetime) signatures

**Out of scope (bookmarked):**
- Shadow ticks / anapressive per-agent tracking (§5.1, §5.2)
- Durkheim disintegration mechanism (§5.3)
- Family groups / topology tracking (§5.4, Stage 3B)
- Distance-based observation decay beyond Jaccard proxy (§5.5, Stage 3B)
- Ensemble TAPS field / full dialectical tension (Approach 3)
- Windowed signatures ("mood" vs "character")
- Overconservation detection (§5.6)
- Praxitive time / deferral mechanism (§5.8)
- Consummation-completion as third disintegration pathway (§5.8)
- Filial inheritance / alliance mode / lineage tracking (§5.10, Stage 3B)
- Asymmetric cross-metathesis roles (§5.10, Stage 3B)
- Praxitive syntegration vs. syntegrative praxis (§5.11)
- Surplus vs. expressed value distinction (§5.12)
- Praxistatic surplus / two-layer agent architecture (§5.13)
- Artifact agents with substrate classes and containment rules (§5.14)
- TAPS as four existential questions mapping (§5.15)
- Innovation decay / forgetting mechanism (§5.16)
- Agent-ensemble primacy and actualized vs. actualizing strata (§5.17)
- Praxis assignment: artifacts do syntegrative praxis, agents do praxitive syntegration (§5.11, §5.17)
- Biunidirectional vs. polydirectional/polyvocal interactivity (§5.14)
- Containment vs. reference nesting rules for artifacts (§5.14)
- Meshability (Gruen) replaces joinability (§5.14)
