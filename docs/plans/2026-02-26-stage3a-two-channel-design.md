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

### 5.18a Term Mapping: Sartre → Metathesis Framework

Full translation table (Sartre on left, our terms on right):

| Sartre (Critique) | Metathesis Framework |
|---|---|
| Incarnation | Incarnation |
| Totalization-of-envelopment | Metathesis-as-enfoldment |
| Totalization of exteriority | Extensive metathesis = explicate order |
| Totalization of interiority | Intensive metathesis = implicate order |
| Anti-labour | Entropy |
| Immanence | Anapression (pressing in/down, reflective) |
| Transcendence | Anopression (pressing out/up, projective) |
| Exteriority of immanence | Explicate order (cf. extensive metathesis) |
| Transcendent exteriority (unthinkable limit) | Implicate order = anopressive extensivity (projective delimit) |
| Transcendence & internal limit of practical freedom | Anopression & reflective delimit of praxitive freedom |
| Unity — Unification | Synthesis — Synthesized |
| Conflict — Contradiction | Tension — Contra-diction |
| Totalization & retotalization | Metathesis & remetathesis |
| Retotalized totalization | Remetathesized metathesis |
| Alteration & alienation | Alterity & disincarnation |
| Drift — deviation | Drift — deviation (unchanged) |
| Anti-dialectic | Syntegrative metathesis |
| Diachronic totalization | Diachronic metathesis |
| Synchronic totalization | Synchronic metathesis |
| Pledge, pledged | Impress, impressed |
| Practico-inert | Praxistatic |
| Milieu | Matrix |
| Ensemble | Network |

**Key architectural implications**:

- Extensive/intensive metathesis maps to L12/L21 channels and to
  manifest/latent layers (§5.13). The explicate order IS the manifest
  layer; the implicate order IS the latent layer.
- Anapression/anopression maps to involution/evolution in the T-letter
  and specifies the directional vector that "immanence"/"transcendence"
  leave implicit.
- "Transcendent exteriority" as "projective delimit" (not "unthinkable
  limit") is where the framework surpasses Sartre: the infinite is not
  unthinkable but projectable.
- Anti-dialectic = syntegrative metathesis confirms §5.11: artifacts
  do syntegrative metathesis (= anti-dialectic); agents do praxitive
  syntegration (= dialectic proper).
- Group-in-fusion = the biuni→poly transition in the vocality spectrum
  — the moment when serial/pairwise becomes collective/polydirectional.
- Praxis-process / process-praxis maps to our phenomenological /
  ontological registers: the same metathesis viewed from within vs.
  from without.

**Glossary terms near-deployable in the framework**:

- **Active passivity**: agent choosing absorptive cross (freely
  consenting to inertia for the common praxis). High L21 signature.
- **Adversity-coefficient** (implying prosperity-coefficient): per-agent
  environmental resistance parameter. Connected to L22 channel.
- **Group-in-fusion (gif)**: biuni→poly transition in vocality.
  Implies institution-in-fusion (iif) and individual-in-fusion.
- **Hexis**: inert stable condition = praxistatic surplus (§5.13).
  Confusion with "exis" (vol. 1) reflects real terminological ambiguity.
- **Passive activity**: what artifacts do — they are passively active.
  Syntegrative praxis IS passive activity.
- **Pledged/impressed group**: group constituted by shared impression.
  L-matrix records impressions.
- **Gathering**: pre-group state = series capable of becoming a group.
  Relevant to §5.4 (family group formation).
- **Negatite**: activity containing negativity integrally. Relevant to
  novel cross-metathesis (creation through dissolution).

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
- Sartre→metathesis term mapping (§5.18, §5.18a)
- Hexis as processual concept / praxis↔hexis cycle (§5.19)
- Scarcity-abundance jana / trust metric τ (§5.20)
- Counter-thesis typology (counter-thesis, -athesis, -synthesis, -metathesis) (§5.21)
- Endogenous mu / counter-thesis generation by innovation activity (§5.22)
- The differential as dividuation / agent-project field (§5.23)
- Practico-inert ↔ praxistatic: ontological ↔ phenomenological tetrad mapping (§5.24)
- Differential time dilation across L-channels with central synthetic-deferral clock (§5.25)
- Beneficial vs. detrimental apocalypses / group fusion typology (§5.26)
- Metathesis/demetathesis dialectic and demetathesis bin (§5.27)
- Metathematization, drift, M(t)/s(t) split, sigma-TAP math gaps (§5.28)
- Actor/artifact naming convention (§5.29)
- RIP flow function: 8 mode configs, 6 orderings, L-matrix mapping, falsification regime (§5.30, Stage 3X-RIP)
- Condensation/condension/evolution clarification for T-letter (§5.30)

---

## §5.19 Hexis as Processual Concept (Praxis ↔ Hexis Cycle)

Reference: Sartre, *Critique of Dialectical Reason* Vol. 1, concept of hexis/exis.

Hexis is not merely "inert, stable condition opposed to praxis" (Vol. 2 glossary). The fuller picture from Vol. 1: **hexis is what praxis becomes when it is absorbed into the practico-inert field** — actively degraded praxis. Not the absence of action but the *sedimentation* of what was once active freedom into passive disposition.

This gives hexis a **temporal structure**: praxis → hexis is a *transition*, not an opposition. An agent whose TAPS signature has stabilized (approaching consummation-completion, §5.8) is precisely an agent whose praxis is becoming hexis — living activity sedimenting into passive disposition.

**Group-level mapping**: A group-in-fusion is a group *recovering praxis from hexis* — dissolving inert serial dispositions under pressure to become active again. A State is a fused institution; a Society (civilization) is a fused collective. The fusion/dissolution cycle maps to the praxis↔hexis cycle at the group level.

**Implementation note**: the praxis→hexis transition could be detected from TAPS signature stabilization rate. An agent whose signature hasn't changed in N steps despite ongoing events is undergoing hexis. The reverse (hexis→praxis, i.e., group-in-fusion) would be detectable from sudden signature diversification under pressure.

## §5.20 Scarcity-Abundance as Jana / Trust Metric τ

Sartre treats scarcity as a foundational given. The unificity framework reformulates:

**Scarcity and abundance are a jana** — a Janus-type dipole (not a spectrum). Where there is scarcity there is always at least an abundance *of* scarcity; where there is abundance there is always at least a scarcity *of* scarcity itself (which makes abundance fragile). The two faces are simultaneous, not endpoints of a gradient. Jana (or janum) rather than spectrum/spectra.

**Trust metric τ**: a per-agent-pair or per-cluster measure (0 to 1) that modulates the effective scarcity experienced by agents:

```
effective_scarcity(agent) = resource_pressure / (1 + τ_mean * sharing_coefficient)
```

A high-trust system (τ → 1) transforms what would be scarcity into effective abundance because agents share and sacrifice. A low-trust system (τ → 0) experiences raw resource pressure. The same M_total *vorhanden* (at hand) looks like abundance or scarcity depending on τ.

**Signification/meaning bridge**: signification is what the metrics say from outside (M_total = 50, Gini = 0.4). Meaning is what those numbers *feel like* from inside the system — and τ is what transforms signification into meaning. Two systems with identical M_total, Gini, and Heaps exponent can have radically different *meanings* for their agents depending on τ. The trust metric would be the first metric closer to meaning than to signification.

**Implementation**: τ accumulates per-agent-pair through successful interactions (cross-metathesis events where both parties benefit or survive), decays in absence of interaction. Feeds into cross-metathesis threshold — high-trust pairs have lower thresholds for cooperation and alliance mode.

**Smooth/striated texture gradients** (Deleuze & Guattari): high trust smooths the field (continuous, flowing, experiential — closer to meaning); low trust striates it (segmented, measured, structural — closer to signification). This could be a correlative bridge metric.

## §5.21 Counter-Thesis Typology

Sartre's "counter-finality" is flat — all unintended consequences are counter-finalities. The metathesis framework gives modal articulation via four types:

| Type | Opposes | Example |
|------|---------|---------|
| **Counter-thesis** | A thesis directly | Head-on resistance; a superseded tool opposing its replacement |
| **Counter-athesis** | A non-thesis (stasis/absence) | Forced activation; something countering a resting state |
| **Counter-synthesis** | An achieved synthesis | Disintegrating force against an integration; a shattered alliance |
| **Counter-metathesis** | The process of metathesis itself | Resistance to transformation as such; the most conservative/inertial force |

**Artifact pinning**: counter-theses are tagged to artifacts. An artifact isn't just "an obstacle" — it is a *specific type* of counter-thesis depending on its constitution. A tradition that prevents exploration is a counter-metathesis. A shattered alliance is a counter-synthesis. A failed experiment is a counter-thesis.

**Agent response modes**: when an agent encounters a counter-thesis, four response modes:
- **Resist** → the counter-thesis becomes a thesis to be overcome
- **Accept** → it merges with existing structure
- **Ignore** → it remains latent until conditions change
- **Dissolve** → it disintegrates and clears

**Key insight**: realizing the source of a limit is simultaneous with realizing how to delimit it. The detection event IS the first step of resolution — no separate "diagnosis" step needed.

Counter-thesis generation could also be an artifact quality/rating function: a master's artifact generates different counter-thesis types than a novice's. A master inspires through craft (counter-thesis to complacency); a novice inspires through audacity (counter-thesis to timidity).

## §5.22 Endogenous Mu / Counter-Thesis Generation by Innovation Activity

Currently mu (extinction rate) is a fixed exogenous parameter. Sartre's analysis of counter-finality as structural (not accidental) suggests mu should be at least **partially endogenous** — generated by agents' own combinatorial activity.

**Mechanism**: innovating produces new sources of decay because becoming better at a given action pushes the boundary of that action to scales affording greater articulation (greater syntegration for less praxis), but the increase of difficulty scales with the increase of counter-theses generated and their type and how those types enter the field.

```
mu_effective(agent) = mu_base + k_counter * agent.recent_innovation_rate * counter_thesis_type_weight
```

**Feedback loop**:
```
innovation_activity → greater reach → more boundary contact →
more counter-theses generated (typed) → higher effective mu →
more disintegration pressure → requires more praxis to survive
```

The *type* of counter-thesis (§5.21) determines the weight and mode of entry. Counter-metatheses generate the heaviest drag (resistance to change itself); counter-theses generate direct but lighter opposition.

This creates a natural self-limiting dynamic: explosive combinatorial growth generates its own drag. Agents cannot innovate without limit because innovation itself produces the conditions that resist further innovation.

## §5.23 The Differential as Dividuation / Agent-Project Field

**Dividuation** (Bohm's rheomodic term): the structural gap between project and realization is better described as *dividuation* rather than *deviation*. Deviation implies departure from a norm; dividuation implies a constitutive splitting — the project and the realization are always already dividuated. The gap is not error but the condition of their existence as distinct entities. Perpetual serial dividuation.

**Agent-project field**: each actor and artifact should have a field parameter representing the space between the project (what is aimed at) and the field (the materiality it passes through). This is a **possibility space with a fidelity gradient**:

- **Low fidelity** (static/noise): the early, consumptive exploration phase — many possibilities, none resolved. Television static metaphor.
- **High fidelity**: the consummated, actualized outcome — a specific product, agent, artifact, skill, or habit.

The process of consumptive exploration resolves fidelity, collapsing possibility into actuality.

**Structural constraint**: a task can't make something the task could never make to begin with. The fidelity resolution is bounded by the nature of the process — no miracles, no discontinuous leaps without *some* participation, even indirect (butterfly effect still requires the butterfly). Non-programmed novelty within pre-programmed entropy.

**Adjacent-element generation**: when an agent combines elements, there is a small probability of producing an *adjacent* element rather than the exact target. The fidelity parameter governs this:

```
fidelity = f(agent.k, agent.trust, artifact.quality, field_distance)
outcome = target_element    with probability fidelity
        = adjacent_element  with probability (1 - fidelity)
```

Where "adjacent" is structurally constrained to the type-space neighborhood of the target. The non-exact target must be *relatable* to the task — something the process could have made, even if it wasn't the explicit goal.

**Mesological gauge**: the differential functions as a mesological gauge constant or gauge function that logs a constant between project and field. The becoming-metathesis is a thetic-athetic-synthetic dialectic moving toward actualization, which is not rigid but a dynamic existent entering into the dynamics of reality upon consummation.

## §5.24 Practico-Inert ↔ Praxistatic: Ontological ↔ Phenomenological Tetrad Mapping

Two complementary field types mapping to the two TAPS registers:

| Field | Forces | Register | Being-mode | Maps to |
|-------|--------|----------|------------|---------|
| **Practico-inert** | Centrifugal/centripetal | Ontological tetrad (become/am/act/create) | Dead-becoming | Multiplicity |
| **Praxistatic** | Conduction | Phenomenological tetrad (grow/feel/see/create) | Living-becoming | Unity |
| **Together** | — | — | Being-becoming | Unificity (belonging) |

**Motor/gas/lubricant metaphor**:
- **Practico-inert** = the motor (structural engine of temporalization). Without it, there would be either frozen totality or nothing.
- **Praxistatic** = the gas/combustion (energetic discharge that makes the motor run). The discharge caused by highly energetic practico-inert fields of mediated adjacency in reciprocal TAPS.
- **Unificity** = the lubricant (ensures laminar flows propagate smooth).

**Key insight**: without static flux (praxistatic), inert concretions (practico-inert) could never crystallize beyond a mold, could never become other to themselves. If deviation could only deviate, there could never be a place or thing that could return or deviate *again*. Deviation and standardization are two sides of the same coin.

**Praxistatic fields interact with practico-inert fields**: the former are living-becoming, the latter dead-becoming, both are being-becoming, the former is unity, the latter is multiplicity, together they are belonging, unificity.

**Smooth/striated** (Deleuze & Guattari): these textural gradients may provide a bridge/metric between signification and meaning. Smooth = continuous, experiential (closer to meaning/praxistatic); striated = segmented, structural (closer to signification/practico-inert).

## §5.25 Differential Time Dilation Across L-Channels

Four historical time scales mapped to Emery L-matrix channels with Einsteinian gravity analogy:

| Time Scale | Sartre | L-channel | Gravity Analogy | Speed |
|------------|--------|-----------|-----------------|-------|
| **Metathetic time** | Time of the system | L22 (env texture) | Highest gravity → slowest time | ~10 steps/tick |
| **Synthetic time** | Time of secondary systems | L21 (env→sys) | | ~7 steps/tick |
| **Athetic time** | Time of general/partial events | L12 (sys→env) | | ~3 steps/tick |
| **Thetic time** | Very swift time of particular actions | L11 (self) | Lowest gravity → fastest time | Every step |

**Central clock**: the time of the constant deferral of synthetic time. Once something is synthesized, it is immediately consummated into the non-synthetic, making room for the next synthesis. This ensures there can never be a final synthesis (which would be a metathesized metathesis — impossible). The deferral creates the ticking.

**Nervous system analogy**: signals arrive from different senses at different rates, staggered, and the brain gates, processes, repackages, and represents them as a unified coherent signal experienced with no delay — but the nervous system has a built-in time delay to accomplish this. The simulation needs the equivalent: a gating mechanism that synchronizes four differential timelines to a central deferral tick.

**Implementation concept**: instead of a single `step` counter, each L-channel has its own clock rate synchronized to the central deferral tick. L11 ticks every step; L22 ticks at env_update_interval. The interaction between strata: fast-time events (L11) accumulate to trigger slow-time transitions (L22); slow-time shifts (L22) suddenly reconfigure conditions for fast-time events (L11).

**Current partial implementation**: env_update_interval=10 in metathetic.py already creates a two-speed system (agent dynamics vs. environmental drift). The expansion would formalize this into four speeds with principled clock ratios.

## §5.26 Beneficial vs. Detrimental Apocalypses / Group Fusion Typology

Apocalypse (Greek: disclosure, revelation) is not inherently destructive. A revelation can be of light or darkness, beginning or end. Every beginning must come from an end; every end must be a beginning.

Two types:

| Type | Pressure source | Orientation | Resource context | Characteristic |
|------|----------------|-------------|------------------|----------------|
| **Beneficial (endogenous)** | Internal | External benefit | High mu, abundant resources, high M_t potential | Group fuses *from within* — shared positive vision |
| **Detrimental (exogenous)** | External | Internal damage | Low mu, scarce resources, low M_t potential | Group fuses *from without* — shared threat response |

**Seriality amplification**:
- Beneficial: amplifies seriality qua *individuality* and group exchange (individuals become more praxistatically charged as individuals)
- Detrimental: amplifies seriality qua *collective* and group exchange (individuals become more praxistatically charged as collective members)

**Dissolution vs. retention asymmetry**:
- After detrimental (external) apocalypse, when pressure eases → groups tend to *dissolve* (common enemy gone)
- After beneficial (internal) apocalypse, when momentum dies → groups tend to *retain* (shared creation binds)

**Surprise cases** (anomalies worth detecting):
- Groups that *retain* after trauma (post-war solidarity persisting beyond threat)
- Groups that *dissolve* after endogenous spiral (team falls apart when project ends)

These would appear as anomalies in signature clustering data — detectable and potentially revealing of deeper dynamics about group durability vs. fragility.

**Unanimous possibility**: unanimous participatory sociality is always an available possibility for a group. The Haudenosaunee (Iroquois Confederacy) attest explicitly to unanimous decision-making over thousands of years.

**Trigger generalization**: the apposite praxis for group-in-fusion need not always be antagonistic. Building projects, clean water initiatives, shared excitement — all can catalyze fusion, particularly when unanimous.

## §5.27 Metathesis / Demetathesis Dialectic and Demetathesis Bin

Four properties of the metathesis/demetathesis dialectic:

1. **Metathesis never completes**, otherwise, annihilation
2. **Demetathesis never happens to itself** — a breaking down can only stop because of a change in praxis-process
3. **Demetathesis is a syntegration of metathesis** — the system destabilizes/desituates/defers itself by its own praxis (not an external product but an internal transformation)
4. **Metathesis is a syntegration of demetathesis** — the system re-stabilizes/resituates/refers in response to instability/desituation/deferral

**Key shift from Sartre**: "product" → "syntegration." Products are external; syntegrations are internal transformations. The system doesn't *produce* instability as output; it *syntegrates* it — makes instability part of its own structure.

**Demetathesis bin**: a reservoir of low-fidelity possibilities that actors cannot access except through a gradient of epiphany events. Two-stage gate:

1. **Virtual possibility** (in demetathesis bin): the possibility exists but the actor doesn't know it can access it
2. **Actual possibility** (in metathesis bin): the actor has realized the path (the epiphany) and can now attempt synthesis
3. **Actualized** (in the world): the synthesis is consummated

The realization from demetathesis → metathesis bin isn't necessarily the synthesis of the thing itself — it can be the synthesis of the *grasp of the way* to synthesize the thing. The actor realizes HOW before they DO.

**Epiphany gradient**: some possibilities emerge gradually through accumulated experience ("gradual epiphany" = mastering a craft); others arrive in a flash ("sudden epiphany" = eureka moment). Both legitimate. The gradient could be governed by:
- Proximity to existing types (closer = more gradual)
- Time spent in the bin (longer = builds toward sudden)
- Actor's current praxistatic charge (higher = more receptive)
- Trust/relational context (supported actors have more epiphanies)

**Anapression of Destiny**: the pressure of the Goal drives the actor to *change how one changes* rather than to overcome a change in the old way. This is a meta-level shift — not "solve this problem differently" but "change the way I approach problems." The demetathesis bin contains precisely these meta-level possibilities.

## §5.28 Metathematization, Drift, M(t)/s(t) Split, Sigma-TAP Math Gaps

**Metathematization**: temporalization = metathematization of praxis, distinct from metathesis. The drift-effect: an actor makes their own history — not a metathesis that's made, not a metathesis that's metathesized, but a metathematization that's synthesized qua metathesis and the deferral of praxis and its syntegration of itself in anopressive transvolutory motion, iterating recursions of preserving metathesis, i.e. of surviving. Metathematization moves in spirals — each circuit passes through the same points but displaced.

**M(t)/s(t) split**: the current codebase uses M(t) as both the possibility space and the actual trajectory. The framework requires distinguishing:

- **M(t)** = the possibility field / axial direction / what-could-be given current parameters. The geodesic. Like light finding the shortest path.
- **s(t)** = the actual trajectory / realized path / what-is. The path that dividuates from M(t) through incarnation.
- **sigma(t)** = the coupling/feedback between M(t) and s(t). Currently sigma(Xi) = sigma0 × (1 + gamma × Xi).

**Drift** = differential between axial direction and actual path:
```
drift(t) = M(t) - s(t)
```

The drift is **textured by events**, not by the agent directly. No one right-minded drifts from the goal on purpose — drift is caused by the field (counter-theses, artifacts with hysteresis, environmental pressure), not by agent intention.

**Volume ratio**: the distance or threshold of the space of the actual that dividuates from or towards the goal is determined by the ratio between the M(t) field's volume and the s(t) field's volume as they change over time. Large M(t) volume relative to s(t) = lots of room to deviate = high drift potential. Small ratio = tightly constrained = low drift.

**The spiral unfolds TAPS through the RIP of TAP through TAPS.** (Full RIP definition pending from user — needed to formalize the recursive structure.)

### Sigma-TAP Math Review (Session 2026-02-28)

**Verified correct**:
1. TAP kernel: `f(M) = alpha * a * (exp(M * ln(1+1/a)) - 1 - M/a)` — algebraically correct via binomial theorem ✓
2. Sigma feedback: `sigma(Xi) = sigma0 * (1 + gamma * Xi)` — mathematically sound ✓
3. ODE coupling in continuous.py: correct positive feedback loop ✓

**Architectural gaps identified**:
1. **Sigma feedback NOT wired into metathetic.py** — agents run raw TAP (sigma=1 effectively). The core sigma-TAP learning loop exists in continuous.py but not in the multi-agent simulation. ⚠️
2. **No per-agent Xi accumulation** in metathetic.py — agents have k (knowledge) but not Xi (affordance exposure). ⚠️
3. **No separate s(t)** — M(t) is both ideal and actual trajectory. ⚠️
4. **No drift metric** — no computation of deviation from ideal trajectory. ⚠️
5. **No counter-thesis generation** — mu is purely exogenous. ⚠️
6. **No trust metric** — effective scarcity is uniform across agents. ⚠️

None of these are *errors* — the math does what it says. But the math is incomplete relative to the framework being built. These gaps represent where theoretical architecture has outgrown implementation.

## §5.29 Actor / Artifact Naming Convention

Renaming for clarity:

| Current Term | New Term | Role |
|---|---|---|
| MetatheticAgent | **Actor** (praxitive agent) | Does praxitive syntegration (externalizing) |
| ArtifactAgent (future) | **Artifact** (syntegrative agent) | Does syntegrative praxis (internalizing-holding) |

Both are *agents* in the general sense (they act in the simulation). The distinction is in their mode of agency:
- **Actors** project outward — their praxis is externalization
- **Artifacts** hold inward — their praxis is passive activity (Sartre's term), internalizing-holding what actors have externalized

This aligns with §5.11 (praxis assignment) and §5.14 (artifact agents), now with cleaner terminology.

## §5.30 RIP Flow Function — Dedicated Stage (Stage 3X-RIP)

> **Dedicated substage**: RIP-to-L-matrix testing, validation, and integration is allocated its own robust dedicated stage (Stage 3X-RIP) rather than being folded into another stage. This isolates the falsification regime and prevents untested RIP assumptions from contaminating other work.

### Definition

**RIP = Recursive/Reflective · Iterative/Integrative · Preservative/Praxitive**

RIP is the **flow function** of TAPS. Where TAPS names the four modal dimensions of a system's dispositional state (the coordinate system), RIP names how the system *moves through* TAPS space (the velocity field). TAP (Kauffman's Theory of the Adjacent Possible) provides the *rate* of movement.

The full architecture: **TAP provides rate, TAPS provides position, RIP provides direction.**

Each RIP position has two modes:
- **R**: Recursive (self-entry, depth) OR Reflective (self-observation, distance)
- **I**: Iterative (sequential stepping, repetition with displacement) OR Integrative (unifying, composition)
- **P**: Preservative (maintaining structure) OR Praxitive (creating structure)

### Combinatorial Space

- **8 mode configurations** (2³ = 8 combinations of which mode each position takes)
- **6 orderings** (3! permutations of which position is primary/secondary/tertiary)
- **Total**: 48 possible RIP states (before empirical reduction)

Ordering notation: X^Y^Z means X is **primary** (drives/initiates), Y is **secondary** (mediates/channels), Z is **tertiary** (the aim/horizon). The primary generates the energy, the secondary gives it form, the tertiary gives it direction.

### The 8 Mode Configurations

Each combination must describe what it signifies by virtue of the terms themselves:

**1. Recursive-Iterative-Preservative (Rc·Ir·Pv)**
The system enters itself, steps through itself, to hold itself together. The homeostatic floor: heartbeat, cell cycle, thermostat. Not death — the *prerequisite* of life. Every living system has this as a substrate. *Homeostasis.*

**2. Recursive-Iterative-Praxitive (Rc·Ir·Px)**
The system enters itself, steps through itself, to act outward. Drilling, training, developing skill through repetitive self-referential work. A pianist practicing scales. A compiler iterating through code to produce an executable. *Practice / Technique development.*

**3. Recursive-Integrative-Preservative (Rc·Ig·Pv)**
The system enters itself, unifies what it finds, to maintain its structure. The immune system: recursively identify foreign bodies, integrate them into recognition patterns, preserve the organism. A tradition incorporating new members to preserve continuity. *Healing / Tradition.*

**4. Recursive-Integrative-Praxitive (Rc·Ig·Px)**
The system enters itself, unifies what it finds, to create something new. The inventor going deep into their own knowledge, synthesizing disparate elements, producing innovation. Depth-first creative search. A dream integrating fragments into new imagery motivating waking action. *Invention / Creative synthesis.*

**5. Reflective-Iterative-Preservative (Rf·Ir·Pv)**
The system stands back to observe itself, processes step by step, to maintain its structure. Quality control. Auditing. A consciousness maintaining coherence through sequential self-monitoring. A culture periodically reflecting on its practices, iterating through them, preserving what works. *Monitoring / Quality assurance.*

**6. Reflective-Iterative-Praxitive (Rf·Ir·Px)**
The system stands back to observe itself, processes step by step, to act. Strategic planning. A general surveying the field, working through options sequentially, executing. Science at its most disciplined: observe, hypothesize step by step, experiment. *Strategy / Scientific method.*

**7. Reflective-Integrative-Preservative (Rf·Ig·Pv)**
The system stands back, sees the whole, brings things together, to maintain something essential. Wisdom. Philosophical contemplation. An elder integrating a lifetime of experience to preserve knowledge for the next generation. Meditation that integrates scattered experience into preserved equanimity. *Wisdom / Contemplative preservation.*

**8. Reflective-Integrative-Praxitive (Rf·Ig·Px)**
The system stands back, sees the whole, brings things together, to create. The painter stepping back from the canvas (reflection), seeing how everything relates (integration), making the next mark (praxis). The full creative cycle at its most articulate. *Full creative cycle / Praxis consummated.*

### The 6 Orderings

**R^I^P — Self-reference drives processing toward structural stance.**
The system begins with *itself*, processes *through* that self-encounter, arrives at a structural orientation. Self-originating action. Generative because the agent is the source — nothing external needed. When preservative: homeostasis from within. When praxitive: creativity from self-encounter. Potentially the most generative ordering because the system is self-starting.

**R^P^I — Self-reference drives structural stance toward processing.**
The system begins with itself, immediately takes a structural stance (preserve or create), and the aim is processing. Self-motivated structuring that seeks method. An artist with a vision (R→P) seeking technique (→I).

**I^R^P — Processing drives self-reference toward structural stance.**
The system begins by *doing*, encounters itself through doing, arrives at a stance. Learning through doing. Apprenticeship: iterate the craft, discover yourself in it, find what to preserve or create.

**I^P^R — Processing drives structural stance toward self-reference.**
The system begins by doing, takes a position, and the aim is self-knowledge. Doing that becomes self-knowledge. Trial and error leading to self-understanding. Praxis that teaches the agent who they are.

**P^R^I — Structural stance drives self-reference toward processing.**
The system begins from commitment (preserve or create), encounters itself in that commitment, aims at processing. Committed self-discovery seeking method. A revolutionary (Px) reflecting on their position (R) seeking means to realize it (I).

**P^I^R — Structural stance drives processing toward self-reference.**
The system begins from commitment, processes iteratively or integratively, aims at self-reference. Action seeking self-knowledge through method. A craftsperson (P) practicing (I) to understand themselves (R). Praxis iterating toward wisdom.

### Death in RIP Terms

No single RIP configuration = death. Death is the **collapse of the RIP flow function itself** — when no configuration is viable, when the system can no longer recurse OR reflect, iterate OR integrate, preserve OR create. Death is the absence of RIP, not a particular value of it.

Alternatively: death is the **loss of the capacity to shift modes**. A system frozen at Rc·Ir·Pv that CANNOT become Rf·Ig·Px is not dead yet, but approaching hexis. The dying process is mode-lock — when the mode-flip frequency approaches zero. Life = the capacity to shift modes. Death = mode-lock completed.

### RIP ↔ L-Matrix (Speculative — Subject to Falsification)

Tentative mapping (to be tested, not assumed):

| RIP Position | Possible L-channel | Rationale |
|---|---|---|
| R (Recursive/Reflective) | L11 (self→self) | Self-referential processing |
| I (Iterative/Integrative) | L21 (env→system) | Processing received input |
| P (Preservative/Praxitive) | L12 (system→env) | Structural output |
| — (context) | L22 (env texture) | Field in which RIP operates |

**WARNING**: This mapping is speculative. The modes within each position (Recursive vs Reflective, etc.) may flip the channel orientation. Reflective might map to L21 (receiving from the field) rather than L11. This is why the falsification regime (below) is non-negotiable.

### Condensation / Condension / Evolution Clarification

The T-letter triad has deeper structure than I/E/T:

| Direction | Term | Maps to | Order (Bohm) |
|---|---|---|---|
| **Involution** | Condension (specific vector) | Unity | Implicate (enfolding, densification) |
| **Evolution** | Diffusion (specific vector) | Multiplicity | Explicate (unfolding, diffusion) |
| **Transvolution** | Condensation (tendency) | Unificity | Actualized structure (consummation of both) |

Transvolution is NOT "balanced between I and E" — it is their *consummation*, a third irreducible thing. This is consistent with unificity being irreducible to unity + multiplicity.

### Stage 3X-RIP: Dedicated Falsification and Integration Stage

**Prerequisites**: Stages 3A (complete), 3B (topology/trust/endogenous mu), enough simulation data to classify agent behavior.

**Scope**: Wire RIP flow function into the simulation, validate against L-matrix data, falsify or confirm each component.

#### Strict Falsification Regime

**F1. Redundancy test**: Can the 8 mode configurations be distinguished empirically from agent behavior, or do some collapse into identical dynamics? **Method**: run paired simulations with agents assigned specific RIP configs; measure whether output distributions differ significantly (p < 0.01). If indistinguishable pairs exist, collapse them and note the reduced effective dimensionality.

**F2. Ordering significance test**: Does ordering (R^I^P vs I^R^P vs P^R^I etc.) produce measurably different dynamics, or only mode configuration matters? **Method**: hold mode configuration constant, permute orderings, measure trajectory divergence. If orderings don't matter, RIP reduces from 48 to 8 states.

**F3. Completeness test**: Are there observed agent dynamics that NO RIP configuration can describe? **Method**: catalog all observed trajectory shapes from prior stages; attempt to fit each with at least one RIP configuration; any unclassifiable trajectory falsifies completeness and requires extending RIP.

**F4. Non-degeneracy test**: Do all configurations actually occur in unconstrained simulation, or do dynamics converge to a small subset? **Method**: run long simulations with diverse initial conditions; catalog emergent RIP states; measure effective dimensionality. If agents settle into only 2-3 states, the others are mathematical artifacts without operational meaning.

**F5. L-matrix correspondence test**: Does the tentative R↔L11, I↔L21, P↔L12 mapping hold? **Method**: track per-agent L-matrix channels alongside RIP-classified behavior; compute mutual information between each R/I/P mode and each L-channel. The mapping must be non-arbitrary.

**F6. TAPS orthogonality test**: Is RIP genuinely independent of TAPS, or derivable from TAPS alone? **Method**: compute mutual information between RIP state and TAPS signature. If MI approaches the entropy of RIP, it is redundant with TAPS and should be eliminated or reformulated.

**F7. Death-is-absence test**: Is death best described as absence of RIP flow (mode-lock) rather than a specific configuration? **Method**: examine agents approaching disintegration; determine whether they occupy a specific RIP state or whether RIP dynamics simply stop (mode-flip frequency → 0).

**F8. Predictive power test**: Does knowing RIP state improve prediction of next TAPS transition beyond TAPS alone? **Method**: build two predictors — TAPS-only and TAPS+RIP. If TAPS+RIP doesn't improve accuracy, RIP is descriptive but not functionally useful and should be deferred.

**Pass criteria**: RIP is retained if it passes F1 (distinguishability), F6 (orthogonality from TAPS), AND F8 (predictive power). Failing any of these three is grounds for reformulation or rejection. F2-F5 and F7 inform the *shape* of RIP but don't determine whether it exists.

### Open Questions (To Be Resolved During Stage 3X-RIP)

1. **Is RIP 3-positional or should it be 4-positional?** TAPS has 4 letters. A flow vector in 4D space needs only 3 independent components to specify direction — this could explain the asymmetry structurally.

2. **Discrete states or continuous manifold?** The 48 states may be vertices of a continuous space rather than discrete bins. The "varying degrees of control & articulation" language suggests continuous weighting.

3. **How does RIP relate to the demetathesis bin (§5.27)?** The transition from virtual to actual possibility (epiphany) might be a specific RIP transition. Gradual epiphany = smooth RIP rotation; sudden epiphany = RIP phase flip.

4. **Can RIP be detected from existing simulation data?** Or does it require new state variables? If detectable from L-matrix + TAPS alone, it may be an emergent property rather than an assigned one.

5. **Power-law nesting (deferred)**: The 8 mode configurations and 6 orderings documented above are **base combinations**. The full RIP combinatorial includes power-law nesting: R^(I^P) ≠ (R^I)^P — the power structure allows functions to be governed by nested functions. A primary RIP function (e.g., R) can have the others as power laws, and powers can have powers. This creates recursive depth: RIP flows that describe their own flow, enabling self-referential configurations consistent with metathematization (§5.28). The base combinations determine the powered combinations — mapping the base level first, then exploring nesting, is the correct staging.

---

## §5.31 Hexis Refined — Autonomic / Protonomic, Neutral Valence

Hexis is NOT degraded praxis (§5.19's framing was preliminary). Refined understanding:

- **Hexis** = the **autonomic** register. It is a *reaction* (automatic, non-deliberative).
- **Praxis** = the **protonomic** register. It is a *response* (deliberative, intentional).

Hexis is **neutral** — it can be beneficial (a well-practiced skill becoming automatic, an institution that runs smoothly) or detrimental (a habit loop that resists change, sedimented oppression). The valence comes from context, not from hexis itself.

**Fusion↔diffusion cycle** (replaces fusion↔dissolution for term symmetry):
- At the group level, **fusion** = convergence toward shared hexitive ground (common practices, shared assumptions). Maps to hexis — automatic cohesion.
- **Diffusion** = divergence into protonomic exploration, agents breaking from shared ground. Maps to praxis — deliberate differentiation.
- The cycle: diffusion → exploration → new affordances → some become hexitive (automatic) → fusion → new shared ground → pressure for diffusion again.

**Abundance → hexis**: When abundance is practiced long enough, it becomes automatic (hexitive), conditioning the protonomic space of agents. What was once a deliberate praxitive achievement becomes part of the autonomic background. This is how innovation sedimentation works.

> **Implementation note**: The praxis↔hexis transition could be modeled as a per-agent fluency metric. High-fluency behaviors are hexitive (low cost, automatic). Novel behaviors are praxitive (high cost, deliberative). The transition rate from praxitive → hexitive is a learning curve parameter.

## §5.32 Self-to-Self Trust (τ_self)

Extends §5.20's trust metric τ with an intra-agent component:

- **τ_pair** (from §5.20): inter-agent trust modulating effective scarcity between agents.
- **τ_self**: intra-agent trust — an agent's confidence in its own praxitive capacity.

Real-world referents: deer in headlights, stage fright, imposter syndrome. The agent *can* act but doesn't believe it can. Low τ_self → the agent perceives abundance as scarcity (it sees the affordances but can't move toward them). High τ_self → the agent can exploit its actual affordance space.

τ_self is the **self-to-self channel** of the trust metric — it lives in L11 (self→self). This connects to §5.7 (praxitive time / deferral): an agent's subjective time-experience may be modulated by τ_self. Low self-trust = time seems to speed up (not enough time to decide) = effective compression of decision horizon.

**Trust↔dividuation connection**: High trust (both τ_pair and τ_self) minimizes *detrimental* dividuation (the gap between project and realization becomes smaller — the agent can close the gap). Low trust maximizes detrimental dividuation (the gap widens — the agent projects but can't realize). Trust is the **fidelity gauge** — it determines where on the lo-fi → hi-fi gradient (§5.23) the agent's adjacent-element generation sits.

## §5.33 Counter-Thesis Participatory Nature

Corrects §5.21's framing. Counter-theses do NOT "oppose" — they are **participatory**.

| Counter-type | Relation | Character |
|---|---|---|
| Counter-thesis | Participatory at best, attempting participatory at worst | Engages *with* the thesis |
| Counter-athesis | **Apposite** (= antithetical) | Genuine opposition, the actual "anti" |
| Counter-synthesis | Participatory dissolution of synthesis | Unravels integration |
| Counter-metathesis | Participatory dissolution of meta-level | Unravels the frame itself |

The key distinction: **counter-thesis ≠ antithesis**. The counter-thesis is ALWAYS attempting to participate in what it counters. Only the counter-athesis is genuinely oppositional (and even then, "opposition" is apposition — standing beside, not standing against). This aligns with the connection fallacy (§5.37): nothing is truly dis-connected; counter-theses are always already juncted with what they counter.

## §5.34 Discontinuous Leaps and Participation

If **participation** is present — even indirect, even minimal — discontinuous leaps CAN occur. The annular distribution (§5.35) allows for jumps across the empty center: an agent doesn't need to traverse the full type-space continuously if participation provides a bridge.

This resolves the apparent contradiction between gradual epiphany and sudden insight (§5.27's demetathesis bin). Both are possible:
- Gradual: high participation, continuous traversal through the annular region.
- Sudden: minimal participation (even just observing from afar), leap across the empty center.

The probability of discontinuous leaps is a function of participation degree, not spatial adjacency in type-space.

## §5.35 Annular / Deadzone Distribution in Type-Space

The adjacent-element generation distribution (§5.23's agent-project field) should NOT be uniform or Gaussian. It should be **annular** (donut-shaped):

- **Empty at exact center**: Zero probability of generating an element identical to the current state. The system cannot step into itself — Heraclitus's river (extended: you can't step into the same river even once, see §5.44).
- **Peaked near a characteristic radius**: Maximum probability at some displacement from center — the "sweet spot" of adjacent possibility, neither too close (redundant) nor too far (inaccessible).
- **Fading at periphery**: Probability decays at large displacement — truly distant possibilities are vanishingly rare under normal conditions (but see §5.34: participation can bridge this).

This constrains adjacent-element generation to a **topological annulus** rather than a filled disk. The inner radius represents the minimum novelty threshold. The outer radius represents the maximum reach of the agent's current affordance horizon.

> **Implementation note**: Replace the Gaussian assumption in adjacent-element generation with an annular kernel: `p(r) ∝ r * exp(-(r - r_peak)² / (2 * w²))` where r_peak is the sweet-spot radius and w is the width. The r-prefactor enforces the deadzone at center.

---

## §5.36 Seed Entropy (Parameter Offset)

Rather than starting simulations from mathematically perfect initial conditions and relying on stochastic processes to break symmetry, **embed roughness a priori**:

Offset all parameter values by ~10⁻⁷ at simulation start. Each agent gets slightly different effective α, μ, etc. This encodes the principle that **no two agents are ontologically identical** — even "identical" initial conditions carry irreducible micro-differences.

This is how nature works: no physical system starts from perfect homogeneity. The 10⁻⁷ offset is below the threshold of meaningful parameter difference but above floating-point epsilon, providing a natural symmetry-breaking seed without distorting dynamics.

> **Implementation note**: In `MetatheticSimulator.__init__`, add a small per-agent perturbation: `agent.alpha_effective = params.alpha * (1 + rng.uniform(-1e-7, 1e-7))` for each relevant parameter. Use the simulation's RNG for reproducibility.

## §5.37 Connection Fallacy — Junction Terminology

**Everything is always already connected.** The language of "connection / disconnection / reconnection" commits a fallacy: it presupposes a prior state of separateness. The correct terms:

| Fallacious term | Correct term | Meaning |
|---|---|---|
| Connection | **Conjunction** | Mode of junction that is mutually reinforcing |
| Disconnection | **Disjunction** | Mode of junction that is mutually attenuating |
| Reconnection | **Re-conjunction** | Return to reinforcing mode |
| — | **Junction** | The fundamental relation (always present) |

**Junction** occurs when praxitive syntegration and syntegrative praxis are in mutually beneficial alterity. It is the baseline condition, not something achieved.

**Emergence** = adpression of anopression and anapression combined. It is the junction of upward and downward causal modes.

> **Implementation note**: This affects how we model agent relationships. Instead of a boolean "connected / not connected" adjacency matrix, use a continuous **junction mode** metric per pair: positive = conjunction, zero = neutral junction, negative = disjunction. No agent pair is ever truly "disconnected" — they are always juncted, just in different modes.

## §5.38 Conservation Law — Minimal Praxis for Maximal Syntegration

A variational principle governing the system:

**Praxis is always conserved** (via consummation in laminar flow).
**Syntegration is always consumed** (via consumption in turbulent flow).

**The law: the system tends toward minimal praxis for maximal syntegration.**

This is structurally analogous to the principle of least action: the system seeks the path that minimizes praxitive expenditure per unit of syntegrative yield. The cosmic wave function = cosmic praxis (praxis at the most universal scale is perfectly conserved).

For the simulation: this provides an **optimization target**. Agent behavior should tend toward efficiency — minimizing the praxitive cost of achieving syntegrative outcomes. Agents that violate this law (high praxis for low syntegration) should be less fit, drifting toward disintegration.

> **Implementation note**: Define a praxis-efficiency ratio `η_praxis = syntegrative_output / praxitive_cost` per agent. Track this ratio over time. Agents with declining η_praxis are approaching hexis (automatic but inefficient) or burnout (effortful but unproductive).

## §5.39 Transvolution Constraints — Timelike / Spacelike / Lightlike

The three fundamental directional pairs in transvolution have distinct causal characters:

| Direction pair | Causal character | Analogy |
|---|---|---|
| Evolution ↔ Involution | **Timelike** | Forward/backward arrows — sequential, causal, irreversible |
| Expansion ↔ Condension | **Spacelike** | Lateral spread/contraction — simultaneous, structural, reversible |
| Rarefaction ↔ Condensation | **Lightlike** | Arrow dynamics — propagation at the boundary of causal reach |

This constrains what kinds of transitions are possible:
- **Timelike** (evolution↔involution): Must follow causal ordering. You cannot involute before you have something to involute. Sequential.
- **Spacelike** (expansion↔condension): Can occur simultaneously across the system. Structural rearrangement without temporal ordering constraint.
- **Lightlike** (rarefaction↔condensation): Propagates at the system's "speed of information." The boundary between what can be influenced and what is out of causal reach.

> **Implementation note**: These constraints shape the allowed transition graph. Some metathetic transitions are timelike-only (must happen in order), some are spacelike (can happen in parallel), and some are lightlike (propagation-limited). The L-channel clock rates (§5.25) should respect these causal characters.

## §5.40 Praxis ↔ Syntegration Mutual Action

Praxis acts UPON syntegration and vice versa — they are not independent:

- The **consumptive aspect of praxis** literally *consumes* syntegrative structures. Praxis burns through syntegration as fuel. This is how praxis actualizes: by *using* structure.
- The **disintegrative aspect of syntegration** literally *consummates* praxitive structures. Syntegration completes praxis by absorbing it into structure. This is how syntegration actualizes: by *doing* what praxis has prepared.

| Direction | Mechanism | Mode | Actualizes via |
|---|---|---|---|
| Praxis → Syntegration | Consumptive | Consumption | Using (structure consumed by action) |
| Syntegration → Praxis | Disintegrative | Consummation | Doing (action absorbed into structure) |
| Both | Mutual | Flux/Change | Transformation |

This is not destruction — it is transformation. The consumed syntegration becomes actualized praxis; the consummated praxis becomes actualized syntegration. Conservation law (§5.38) holds: praxis is conserved through the cycle, syntegration is consumed and re-produced.

## §5.41 Law of Relativity (Proposed)

> "The goal direction of a system is always conserved as the minimal quantity of energy consumed to yield the maximal quality in production result per unit of energy expended."

This is the user's proposed fundamental law connecting conservation (§5.38) with directionality (§5.39) and the praxis↔syntegration cycle (§5.40). The system's telos is not a fixed endpoint but a **conserved optimization principle**: maximize quality per unit cost.

**"Goal direction"** is not teleological in the metaphysical sense — it is the system's tendency, analogous to how entropy increase is the "goal direction" of thermodynamic systems without implying conscious purpose.

> **Status**: Proposed. Needs formalization. The relationship between "quality of production result" and measurable simulation quantities (Youn ratio? TAPS diversity? Innovation rate?) must be specified before this can be tested.

## §5.42 Turbulence Reconceived — Affordance Overload

Turbulence is NOT random noise injection (as currently implemented in `turbulence.py`). Turbulence occurs when:

1. The possibility space becomes **overloaded with explicit affordances** — too many viable options.
2. The system is **overwhelmed with path-options** — decision paralysis.
3. The system **cannot decide within the available time span t**.
4. Time t **reconfigures the possibility space** without a decision having been made — the window closes, options shift, the landscape changes under the agent's feet.

This is the **abundance face of the scarcity-abundance janum** (§5.20): turbulence is what abundance looks like from the inside when the praxitive engine cannot process the fuel fast enough. The agent has too much gas and not enough engine capacity.

This connects to the conservation law (§5.38): turbulence is the failure mode of the minimal-praxis-for-maximal-syntegration principle. When syntegrative abundance exceeds praxitive processing capacity, the system enters turbulent flow — decisions become non-laminar, trajectories become unpredictable.

> **Implementation note**: Rethink `turbulence.py`. Instead of stochastic shock injection, model turbulence as a function of `affordance_count / decision_capacity`. When this ratio exceeds a threshold, the agent enters turbulent mode: its decisions become randomized not by external noise but by *internal overload*. The key difference: turbulence is endogenous (from the agent's relationship to its environment), not exogenous (from random perturbation).

## §5.43 Practico-Inert / Praxistatic Reversal — Gas / Engine

**Reversal of §5.24's motor/gas assignment**, after reflection on nature:

| Component | §5.24 (original) | §5.43 (revised) | Rationale |
|---|---|---|---|
| **Practico-inert** | Motor (drives via resistance) | **Gas / fuel** | Sedimented past praxis stores potential energy; gets consumed/burned by living praxis |
| **Praxistatic** | Gas (feeds the motor) | **Engine** | Active interface where combustion happens; transforms stored potential into directed work |

**Why the reversal is correct** (considering nature):

The practico-inert is sedimented praxis — dead labor, accumulated structure, worked matter that exists as a field. It has *stored energy* (the crystallized effort of past praxis). It IS the fuel. Like a cell's nutrient bath: raw materials that exist, have stored chemical energy, and get consumed by the cellular machinery.

The praxistatic is the living tension where praxis meets stasis — the active membrane, the enzymatic surface. It IS the engine. Like a cell's metabolic machinery: it transforms the nutrient substrate into directed work.

This coheres with §5.38 (conservation law): praxis is conserved (the engine persists), syntegration is consumed (the fuel gets burned through). The praxistatic-as-engine conserves the praxitive process. The practico-inert-as-gas gets consumed as the syntegrative substrate.

A river and its banks: the flow dynamics (praxistatic) = engine, converting gravitational potential into geological work. The sediment and rock (practico-inert) = fuel, the material the river works through.

## §5.44 Thetic Flux Feedback Loop

Synthetic time consummated into non-synthetic = **thetic flow**. This creates a feedback loop:

1. **Thetic** (present moment, L11) → creates the raw material for
2. **Synthesis** (combining thetic moments, L21) → which, when consummated, preserves time-boundedness →
3. **Metathesis-thesis** (meta-level becoming a new present, L22→L11) → producing a new present moment →
4. Back to (1)

This is Heraclitus's river, extended: **you cannot step into the same river even once**. The "same" present is never the same — it is always already the consummation of a prior synthetic cycle. The thetic moment is not a static point but a *flow* — the consummated output of the full cycle.

The deadzone at the center of the annular distribution (§5.35) is the formal expression of this: zero probability of generating an identical state because identical states do not exist. Every moment is a new thetic flow.

## §5.45 Alpha / Sigma Reinterpretation and Four Time-Modes

### α and σ as Phase Markers

The simulation parameters α and σ are not merely scaling constants — they mark **phases** of the metathetic process:

- **α (alpha)** = the **anapressive phase**: constant of increasing difficulty. Measures **desituation / dissociation** — how hard it is to leave the current state. As α increases, the agent faces greater resistance to departing its present configuration. α is the downward-pressing, inward-folding aspect.

- **σ (sigma)** = the **anopressive phase**: constant of decreasing difficulty. Measures **situatedness / association** — how readily the agent can settle into a new state. As σ increases, the agent finds it easier to arrive at and integrate with new configurations. σ is the upward-pressing, outward-unfolding aspect.

The existing sigma-TAP feedback `σ(Xi) = σ₀(1 + γ·Xi)` already encodes this: as affordance exposure (Xi) grows, situatedness increases — the agent becomes more able to associate with new states. The α parameter (innovation kernel difficulty) encodes the complementary resistance.

### Four Time-Modes Mapped to L-Channels

| Time-mode | Symbol | L-channel | Character | Description |
|---|---|---|---|---|
| Thetic time | **π** (pi) | L11 (self→self) | Fastest, most immediate | The self-referential present; the "clock tick" of self-encounter |
| Athetic time | **α** (alpha) | L12 (system→env) | Fast, dissociative | The time of departure/projection outward; desituation |
| Synthetic time | **σ** (sigma) | L21 (env→system) | Slow, associative | The time of arrival/integration inward; situatedness |
| Metathetic time | **Π** (Pi) | L22 (env→env) | Slowest, most encompassing | The environmental self-referential time; paradigm drift |

**Paradigm shifts** = collisions of all four time-modes. When π, α, σ, and Π briefly synchronize — the thetic present, athetic departure, synthetic arrival, and metathetic drift all align — the system undergoes a phase transition. This is why paradigm shifts feel like "everything at once": all four temporal registers are momentarily in phase.

The differential time dilation (§5.25) already proposed that L-channels run at different speeds. The four time-modes give those speeds semantic content: it is not just that L22 is slower than L11, but that metathetic time (paradigm drift) is intrinsically slower than thetic time (self-encounter) because it encompasses more of the system.

> **Implementation note**: Consider naming the existing α parameter `alpha_tap` and introducing `alpha_time` for the L12 clock rate, to avoid symbol collision. Similarly σ as `sigma_tap` (feedback) vs `sigma_time` (L21 clock rate). Or adopt the Greek letter convention consistently: π for L11 rate, α for L12 rate, σ for L21 rate, Π for L22 rate, and use separate Latin letters for the TAP kernel parameters.

---

### Deferred Items Summary (Updated through §5.45)

All forward notes §5.1–§5.45 are bookmarked for later stages. Key allocations:

**Stage 3B** (topology / trust / endogenous mu):
- §5.4, §5.5, §5.10 (topology, distance decay, alterity)
- §5.20, §5.32 (trust τ_pair + τ_self)
- §5.22 (endogenous mu)
- §5.29 (actor/artifact naming)
- §5.36 (seed entropy — low-cost, implement early)
- §5.37 (junction terminology — affects adjacency model)

**Stage 3C+** (deeper mechanisms):
- §5.31 (hexis autonomic/protonomic)
- §5.33 (counter-thesis participatory)
- §5.34, §5.35 (discontinuous leaps, annular distribution)
- §5.38 (conservation law)
- §5.42 (turbulence reconceived)
- §5.43 (practico-inert/praxistatic reversal)
- §5.44 (thetic flux)

**Stage 3X-RIP** (dedicated, per §5.30):
- §5.45 (four time-modes as L-channel semantics — inform RIP timing)

**Future architecture**:
- §5.39 (transvolution causal constraints)
- §5.40 (praxis↔syntegration mutual action)
- §5.41 (law of relativity — needs formalization)
