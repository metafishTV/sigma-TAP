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

### 5.8 Embedded Sub-Mode Combinatorics

The four-letter TAPS signature pulls from primary dimension labels. A
deeper combinatorial scheme using embedded sub-mode letters
(T→I/E/T, A→E/I/A, P→R/U/X, S→D/P/I/S) is noted for future
investigation. Could produce richer dispositional names but risks
confusing notation.

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
