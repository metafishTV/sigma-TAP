# Affordance-Gated Self-Metathesis + Annihilation Redistribution Design

**Claim policy label: exploratory**

> This design extends the metathetic multi-agent TAP framework (C8) with two
> mechanisms not present in the source TAP/biocosmology literature. Both are
> exploratory and require further validation.

## Motivation

Self-metathesis currently fires when an agent's innovation rate exceeds a
threshold. But for something to be built, what's needed to build it must be
available. The possibility must precede the actualization as an affordance.
A lone survivor from a collapsed interactive cluster can affect others
(cross-metathesize) but has near-zero success at reconstituting their own
identity alone. The metathetic aperture — the local interactive collective
that supports identity-level change — requires a minimum viable density.

Separately, annihilated agents (temporal state 0) currently sit inert. Their
types and knowledge contribute nothing. In reality, a defunct entity's
knowledge dissipates into interactively-proximate neighbors, weighted by
interaction history — not by raw activity level.

## Extension A: Affordance-Gated Self-Metathesis

### Mechanism

Per step, per active agent, compute a binary affordance tick:

1. Count `n_connected` = number of other active agents sharing >= 1 type
2. Check birth-term connectivity: is dM > 0 AND n_connected >= affordance_min_cluster?
3. `affordance_tick = 1` if both conditions met, else `0`

Maintain per-agent `_affordance_ticks: list[int]` (rolling window, last 10 steps).
`affordance_score = mean(_affordance_ticks)`.

Self-metathesis fires only when:
- dM > threshold * len(type_set) (existing, with temporal modulation)
- AND affordance_score > 0.0 (at least some recent environmental support)

Cross-metathesis is NOT gated by affordance. Agents can always affect and
be affected by others regardless of their local cluster density.

### Parameters

- `affordance_min_cluster: int = 2` (minimum connected agents for tick=1)
- `_AFFORDANCE_WINDOW: int = 10` (rolling window size, class constant)

### Agent fields

- `_affordance_ticks: list[int]` — rolling binary ticks, max length _AFFORDANCE_WINDOW

### New helper

- `_compute_affordance_tick(agent, active_agents) -> int`

### Effect on dynamics

- Isolated agents cannot self-metathesize (score stays at 0)
- Well-connected agents self-metathesize as before (score near 1.0)
- Agents losing connections see gradual affordance decay (rolling window)
- Spontaneous re-emergence possible if new connections form

## Extension B: Annihilation Redistribution

### Trigger

When `temporal_state_with_context()` returns 0 (annihilated) during the
ensemble run loop.

### Mechanism

1. Compute Jaccard similarity between annihilated agent and each active agent
2. If sum(weights) == 0: all types and k are LOST (no interactive neighbor)
3. Otherwise normalize: p_j = w_j / sum(w_j)
4. Redistribute types: each type assigned to agent with highest weight
   (ties broken by agent_id)
5. Redistribute knowledge: agent_j.k += annihilated.k * p_j
6. Mark agent as dissolved (remove from agents list or flag permanently)

### New diagnostics

- `n_annihilation_redistributions: int` — count of redistribution events
- `n_types_lost: int` — types that dissipated (no Jaccard neighbor)
- `k_lost: float` — knowledge that dissipated

### No new parameters

Uses existing Jaccard computation and existing annihilation detection.

### Effect on dynamics

- Knowledge flows to interactively-proximate neighbors
- Distant agents receive nothing (Jaccard = 0)
- Isolated knowledge is lost (realistic dissipation)
- Should affect Gini and Heaps diagnostics measurably

## Integration

Both mechanisms fire during the ensemble `run()` loop:
1. `_step_agents()` — existing TAP evolution
2. `_record_history()` — existing aggregate tracking
3. `_update_affordance_ticks()` — NEW: compute affordance per agent
4. `_check_self_metathesis()` — existing, now gated by affordance
5. `_check_cross_metathesis()` — existing, unchanged
6. `_check_annihilation_redistribution()` — NEW: detect state 0, redistribute
7. `_update_environment()` — existing environmental drift

## Diagnostics additions

Snapshot dict gains:
- `affordance_mean` — mean affordance_score across active agents
- `n_annihilation_redistributions` — cumulative count
- `n_types_lost` — cumulative types dissipated
- `k_lost` — cumulative knowledge dissipated

Summary JSON gains same fields.
