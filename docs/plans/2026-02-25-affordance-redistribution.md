# Affordance Gate + Disintegration Redistribution Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add affordance-gated self-metathesis and Jaccard-weighted disintegration redistribution to the metathetic ensemble.

**Architecture:** Extension A adds per-agent rolling affordance ticks computed from local interactive cluster density, gating self-metathesis. Extension B adds redistribution of disintegrated agents' types/knowledge to Jaccard-proximate neighbors. Both are wired into the ensemble run loop, with new diagnostics in snapshots and longrun summary.

**Tech Stack:** Python 3.12, unittest, existing simulator/metathetic.py infrastructure

---

### Task 1: Affordance tick computation + agent field

**Files:**
- Modify: `simulator/metathetic.py` (MetatheticAgent + new helper)
- Test: `tests/test_metathetic.py`

**Context:**
Add `_affordance_ticks` field to MetatheticAgent, `_AFFORDANCE_WINDOW` constant, and a module-level helper `_compute_affordance_tick()`. The helper checks two conditions: (1) does the agent have enough connected neighbors (sharing >=1 type), and (2) is the agent's recent dM positive? Returns 1 if both, else 0.

**Step 1: Write failing tests**

Add to `tests/test_metathetic.py`:

```python
class TestAffordanceTick(unittest.TestCase):
    """Tests for affordance tick computation."""

    def test_connected_agent_positive_dM_returns_1(self):
        """Agent with 2+ connected neighbors and positive dM gets tick=1."""
        from simulator.metathetic import _compute_affordance_tick, MetatheticAgent
        focal = MetatheticAgent(0, {0, 1}, 0.0, 10.0, dM_history=[1.0])
        others = [
            MetatheticAgent(1, {0, 2}, 0.0, 10.0, active=True),
            MetatheticAgent(2, {0, 3}, 0.0, 10.0, active=True),
        ]
        self.assertEqual(_compute_affordance_tick(focal, others, min_cluster=2), 1)

    def test_isolated_agent_returns_0(self):
        """Agent sharing no types with anyone gets tick=0."""
        from simulator.metathetic import _compute_affordance_tick, MetatheticAgent
        focal = MetatheticAgent(0, {99}, 0.0, 10.0, dM_history=[1.0])
        others = [
            MetatheticAgent(1, {0, 2}, 0.0, 10.0, active=True),
        ]
        self.assertEqual(_compute_affordance_tick(focal, others, min_cluster=2), 0)

    def test_negative_dM_returns_0(self):
        """Agent with negative dM gets tick=0 even if connected."""
        from simulator.metathetic import _compute_affordance_tick, MetatheticAgent
        focal = MetatheticAgent(0, {0, 1}, 0.0, 10.0, dM_history=[-1.0])
        others = [
            MetatheticAgent(1, {0, 2}, 0.0, 10.0, active=True),
            MetatheticAgent(2, {0, 3}, 0.0, 10.0, active=True),
        ]
        self.assertEqual(_compute_affordance_tick(focal, others, min_cluster=2), 0)

    def test_below_min_cluster_returns_0(self):
        """Agent with only 1 connected neighbor but min_cluster=2 gets tick=0."""
        from simulator.metathetic import _compute_affordance_tick, MetatheticAgent
        focal = MetatheticAgent(0, {0, 1}, 0.0, 10.0, dM_history=[1.0])
        others = [
            MetatheticAgent(1, {0, 2}, 0.0, 10.0, active=True),
        ]
        self.assertEqual(_compute_affordance_tick(focal, others, min_cluster=2), 0)

    def test_empty_dM_history_returns_0(self):
        """Agent with no dM_history gets tick=0."""
        from simulator.metathetic import _compute_affordance_tick, MetatheticAgent
        focal = MetatheticAgent(0, {0, 1}, 0.0, 10.0, dM_history=[])
        others = [
            MetatheticAgent(1, {0, 2}, 0.0, 10.0, active=True),
            MetatheticAgent(2, {0, 3}, 0.0, 10.0, active=True),
        ]
        self.assertEqual(_compute_affordance_tick(focal, others, min_cluster=2), 0)

    def test_affordance_ticks_field_exists(self):
        """Agent should have _affordance_ticks list."""
        from simulator.metathetic import MetatheticAgent
        a = MetatheticAgent(0, {0}, 0.0, 10.0)
        self.assertIsInstance(a._affordance_ticks, list)
        self.assertEqual(len(a._affordance_ticks), 0)

    def test_affordance_score_property(self):
        """affordance_score should be mean of _affordance_ticks."""
        from simulator.metathetic import MetatheticAgent
        a = MetatheticAgent(0, {0}, 0.0, 10.0)
        a._affordance_ticks = [1, 1, 0, 0, 1]
        self.assertAlmostEqual(a.affordance_score, 0.6)

    def test_affordance_score_empty_returns_zero(self):
        """Empty ticks list should return 0.0."""
        from simulator.metathetic import MetatheticAgent
        a = MetatheticAgent(0, {0}, 0.0, 10.0)
        self.assertEqual(a.affordance_score, 0.0)
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_metathetic.py::TestAffordanceTick -v`
Expected: FAIL (ImportError for _compute_affordance_tick, AttributeError for _affordance_ticks)

**Step 3: Implement**

In `simulator/metathetic.py`:

1. Add field to MetatheticAgent (after `_dormant_steps`):
```python
    _affordance_ticks: list[int] = field(default_factory=list)
```

2. Add constant (after existing temporal constants):
```python
    _AFFORDANCE_WINDOW: int = field(default=10, init=False, repr=False)
```

3. Add property (after temporal_state_with_context):
```python
    @property
    def affordance_score(self) -> float:
        """Rolling mean of affordance ticks. 0.0 if no ticks recorded."""
        if not self._affordance_ticks:
            return 0.0
        return sum(self._affordance_ticks) / len(self._affordance_ticks)
```

4. Add module-level helper (after _temporal_threshold_multiplier):
```python
def _compute_affordance_tick(
    agent: MetatheticAgent,
    other_active: list[MetatheticAgent],
    min_cluster: int = 2,
) -> int:
    """Binary affordance tick: 1 if agent has local interactive support, else 0.

    Conditions for tick=1:
      1. Agent has positive recent dM (growth happening)
      2. At least min_cluster other active agents share >= 1 type with agent

    The combination ensures that growth is occurring in a connected context,
    not in isolation. An agent whose types exist nowhere else cannot access
    the combinatorial building blocks for self-transformation.
    """
    if not agent.dM_history or agent.dM_history[-1] <= 0:
        return 0
    n_connected = sum(
        1 for other in other_active
        if other.agent_id != agent.agent_id and (agent.type_set & other.type_set)
    )
    return 1 if n_connected >= min_cluster else 0
```

**Step 4: Run tests**

Run: `python -m pytest tests/test_metathetic.py::TestAffordanceTick -v`
Expected: All 8 PASS.

**Step 5: Commit**

```bash
git add simulator/metathetic.py tests/test_metathetic.py
git commit -m "feat: add affordance tick computation + agent field"
```

---

### Task 2: Wire affordance gate into self-metathesis + ensemble loop

**Files:**
- Modify: `simulator/metathetic.py` (MetatheticEnsemble)
- Test: `tests/test_metathetic.py`

**Context:**
Add `_update_affordance_ticks()` method to MetatheticEnsemble, called every step in `run()`. Modify `_check_self_metathesis()` to add the affordance gate: self-metathesis only fires when `agent.affordance_score > 0.0`. Add `affordance_min_cluster` constructor param. Add `affordance_mean` to snapshot.

**Step 1: Write failing tests**

```python
class TestAffordanceGate(unittest.TestCase):
    """Tests for affordance-gated self-metathesis."""

    def test_isolated_agent_cannot_self_metathesize(self):
        """Agent with affordance_score=0 should not self-metathesize."""
        from simulator.metathetic import MetatheticEnsemble
        ens = MetatheticEnsemble(
            n_agents=3, initial_M=10.0,
            alpha=5e-3, a=3.0, mu=0.005,
            variant="logistic", carrying_capacity=2e5,
            seed=42, affordance_min_cluster=2,
        )
        # Force one agent to have unique types only (no overlap)
        ens.agents[0].type_set = {999}
        ens.agents[0].dM_history = [100.0]  # High dM
        ens.agents[0]._affordance_ticks = []  # No affordance history

        old_types = len(ens.agents[0].type_set)
        old_self = ens.n_self_metatheses
        ens._update_affordance_ticks()
        ens._check_self_metathesis()
        # Should not have self-metatheized despite high dM
        self.assertEqual(len(ens.agents[0].type_set), old_types)

    def test_connected_agent_can_self_metathesize(self):
        """Agent with affordance_score>0 and sufficient dM should fire."""
        from simulator.metathetic import MetatheticEnsemble
        ens = MetatheticEnsemble(
            n_agents=5, initial_M=10.0,
            alpha=5e-3, a=3.0, mu=0.005,
            variant="logistic", carrying_capacity=2e5,
            seed=42, affordance_min_cluster=2,
        )
        # All agents share type 0, so they're connected
        # Give one agent high dM and pre-fill affordance ticks
        ens.agents[0].dM_history = [100.0] * 5
        ens.agents[0]._affordance_ticks = [1, 1, 1]
        ens.agents[0].steps_since_metathesis = 10  # Past novelty window

        old_self = ens.n_self_metatheses
        ens._check_self_metathesis()
        self.assertGreater(ens.n_self_metatheses, old_self)

    def test_affordance_ticks_updated_in_run(self):
        """Running ensemble should populate affordance ticks."""
        from simulator.metathetic import MetatheticEnsemble
        ens = MetatheticEnsemble(
            n_agents=5, initial_M=10.0,
            alpha=5e-3, a=3.0, mu=0.005,
            variant="logistic", carrying_capacity=2e5,
            seed=42, affordance_min_cluster=2,
        )
        trajectory = ens.run(steps=20)
        for a in ens._active_agents():
            self.assertGreater(len(a._affordance_ticks), 0)
            self.assertLessEqual(len(a._affordance_ticks), a._AFFORDANCE_WINDOW)

    def test_affordance_mean_in_snapshot(self):
        """Snapshot should contain affordance_mean."""
        from simulator.metathetic import MetatheticEnsemble
        ens = MetatheticEnsemble(
            n_agents=5, initial_M=10.0,
            alpha=5e-3, a=3.0, mu=0.005,
            variant="logistic", carrying_capacity=2e5,
            seed=42,
        )
        trajectory = ens.run(steps=10)
        for snap in trajectory:
            self.assertIn("affordance_mean", snap)
            self.assertGreaterEqual(snap["affordance_mean"], 0.0)
            self.assertLessEqual(snap["affordance_mean"], 1.0)

    def test_affordance_ticks_capped_at_window(self):
        """Affordance ticks should not exceed _AFFORDANCE_WINDOW."""
        from simulator.metathetic import MetatheticEnsemble
        ens = MetatheticEnsemble(
            n_agents=3, initial_M=10.0,
            alpha=5e-3, a=3.0, mu=0.005,
            variant="logistic", carrying_capacity=2e5,
            seed=42,
        )
        ens.run(steps=30)
        for a in ens.agents:
            self.assertLessEqual(len(a._affordance_ticks), a._AFFORDANCE_WINDOW)
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_metathetic.py::TestAffordanceGate -v`
Expected: FAIL (missing affordance_min_cluster param, missing _update_affordance_ticks)

**Step 3: Implement**

1. Add `affordance_min_cluster` to `MetatheticEnsemble.__init__`:
```python
    def __init__(self, ..., affordance_min_cluster: int = 2):
        ...
        self.affordance_min_cluster = affordance_min_cluster
```

2. Add `_update_affordance_ticks()` method:
```python
    def _update_affordance_ticks(self) -> None:
        """Compute and record affordance tick for each active agent."""
        active = self._active_agents()
        for agent in active:
            tick = _compute_affordance_tick(
                agent, active, min_cluster=self.affordance_min_cluster
            )
            agent._affordance_ticks.append(tick)
            if len(agent._affordance_ticks) > agent._AFFORDANCE_WINDOW:
                agent._affordance_ticks = agent._affordance_ticks[-agent._AFFORDANCE_WINDOW:]
```

3. Modify `_check_self_metathesis()` — add affordance gate after temporal modulation:
```python
            # Affordance gate: agent must have recent environmental support
            if agent.affordance_score <= 0.0:
                continue
```
Insert this right after the temporal threshold line and before the firing check.

4. In `run()`, insert `_update_affordance_ticks()` after `_record_history()` and before `_check_self_metathesis()`.

5. Add `affordance_mean` to snapshot dict:
```python
            # Affordance diagnostics
            active_scores = [a.affordance_score for a in active]
            snapshot["affordance_mean"] = (
                sum(active_scores) / len(active_scores) if active_scores else 0.0
            )
```

**Step 4: Run tests**

Run: `python -m pytest tests/test_metathetic.py -v -k "Affordance"`
Expected: All 13 tests PASS (8 from Task 1 + 5 new).

**Step 5: Run full suite**

Run: `python -m pytest tests/ -q`
Expected: All pass (existing tests unaffected — affordance defaults to min_cluster=2, and since all agents share type 0, affordance_score should be >0 for connected agents).

**Step 6: Commit**

```bash
git add simulator/metathetic.py tests/test_metathetic.py
git commit -m "feat: wire affordance gate into self-metathesis + ensemble loop"
```

---

### Task 3: Disintegration redistribution

**Files:**
- Modify: `simulator/metathetic.py` (MetatheticEnsemble)
- Test: `tests/test_metathetic.py`

**Context:**
Add `_check_disintegration_redistribution()` method. When an agent's `temporal_state_with_context()` returns 0, redistribute its types and k to active agents weighted by Jaccard similarity. Types go to highest-weight agent (ties by agent_id). Knowledge split proportionally. Track new counters.

**Step 1: Write failing tests**

```python
class TestDisintegrationRedistribution(unittest.TestCase):
    """Tests for disintegration redistribution mechanism."""

    def test_redistribution_to_most_similar(self):
        """Types should go to the agent with highest Jaccard similarity."""
        from simulator.metathetic import MetatheticEnsemble, MetatheticAgent, _jaccard
        ens = MetatheticEnsemble(
            n_agents=3, initial_M=10.0,
            alpha=5e-3, a=3.0, mu=0.005,
            variant="logistic", carrying_capacity=2e5,
            seed=42,
        )
        # Make agent 0 disintegrated: inactive, dormant long enough, no shared types
        ens.agents[0].active = False
        ens.agents[0]._dormant_steps = 50
        ens.agents[0].type_set = {100, 101}  # unique types
        ens.agents[0].k = 50.0

        # Agent 1 shares type 100 (high Jaccard with disintegrated)
        ens.agents[1].type_set = {0, 1, 100}
        ens.agents[1].k = 10.0

        # Agent 2 shares nothing with disintegrated
        ens.agents[2].type_set = {0, 2}
        ens.agents[2].k = 10.0

        ens._check_disintegration_redistribution()

        # Agent 1 should have received the types
        self.assertIn(100, ens.agents[1].type_set)
        self.assertIn(101, ens.agents[1].type_set)
        # Agent 2 should NOT have received types (Jaccard=0)
        self.assertNotIn(101, ens.agents[2].type_set)

    def test_knowledge_split_proportionally(self):
        """Knowledge should be split proportional to Jaccard weights."""
        from simulator.metathetic import MetatheticEnsemble, MetatheticAgent
        ens = MetatheticEnsemble(
            n_agents=4, initial_M=10.0,
            alpha=5e-3, a=3.0, mu=0.005,
            variant="logistic", carrying_capacity=2e5,
            seed=42,
        )
        ens.agents[0].active = False
        ens.agents[0]._dormant_steps = 50
        ens.agents[0].type_set = {100, 101}
        ens.agents[0].k = 100.0

        # Two agents share type 100
        ens.agents[1].type_set = {0, 100}
        ens.agents[1].k = 0.0
        ens.agents[2].type_set = {0, 100}
        ens.agents[2].k = 0.0
        # One shares nothing
        ens.agents[3].type_set = {0, 3}
        ens.agents[3].k = 0.0

        ens._check_disintegration_redistribution()

        # Agents 1 and 2 have equal Jaccard, so should split k evenly
        self.assertAlmostEqual(ens.agents[1].k, 50.0, places=1)
        self.assertAlmostEqual(ens.agents[2].k, 50.0, places=1)
        # Agent 3 gets nothing
        self.assertAlmostEqual(ens.agents[3].k, 0.0)

    def test_no_neighbors_means_knowledge_lost(self):
        """If no active agent has Jaccard>0, types and k are lost."""
        from simulator.metathetic import MetatheticEnsemble
        ens = MetatheticEnsemble(
            n_agents=3, initial_M=10.0,
            alpha=5e-3, a=3.0, mu=0.005,
            variant="logistic", carrying_capacity=2e5,
            seed=42,
        )
        ens.agents[0].active = False
        ens.agents[0]._dormant_steps = 50
        ens.agents[0].type_set = {999}  # No overlap with anyone
        ens.agents[0].k = 100.0

        ens.agents[1].type_set = {0, 1}
        ens.agents[2].type_set = {0, 2}
        k1_before = ens.agents[1].k
        k2_before = ens.agents[2].k

        ens._check_disintegration_redistribution()

        # Knowledge should be lost (not redistributed)
        self.assertAlmostEqual(ens.agents[1].k, k1_before)
        self.assertAlmostEqual(ens.agents[2].k, k2_before)
        self.assertGreater(ens.n_types_lost, 0)
        self.assertGreater(ens.k_lost, 0)

    def test_counters_tracked(self):
        """Redistribution should increment event counters."""
        from simulator.metathetic import MetatheticEnsemble
        ens = MetatheticEnsemble(
            n_agents=3, initial_M=10.0,
            alpha=5e-3, a=3.0, mu=0.005,
            variant="logistic", carrying_capacity=2e5,
            seed=42,
        )
        ens.agents[0].active = False
        ens.agents[0]._dormant_steps = 50
        ens.agents[0].type_set = {100}
        ens.agents[0].k = 10.0
        ens.agents[1].type_set = {0, 1, 100}  # shares type 100

        ens._check_disintegration_redistribution()
        self.assertEqual(ens.n_disintegration_redistributions, 1)

    def test_dissolved_agent_removed(self):
        """Disintegrated agent should be marked dissolved after redistribution."""
        from simulator.metathetic import MetatheticEnsemble
        ens = MetatheticEnsemble(
            n_agents=3, initial_M=10.0,
            alpha=5e-3, a=3.0, mu=0.005,
            variant="logistic", carrying_capacity=2e5,
            seed=42,
        )
        ens.agents[0].active = False
        ens.agents[0]._dormant_steps = 50
        ens.agents[0].type_set = {100}
        ens.agents[0].k = 10.0
        ens.agents[1].type_set = {0, 1, 100}

        ens._check_disintegration_redistribution()
        # Agent should be flagged as dissolved
        self.assertTrue(hasattr(ens.agents[0], '_dissolved'))
        self.assertTrue(ens.agents[0]._dissolved)

    def test_diagnostics_in_snapshot(self):
        """Snapshot should include redistribution diagnostics."""
        from simulator.metathetic import MetatheticEnsemble
        ens = MetatheticEnsemble(
            n_agents=5, initial_M=10.0,
            alpha=5e-3, a=3.0, mu=0.005,
            variant="logistic", carrying_capacity=2e5,
            seed=42,
        )
        trajectory = ens.run(steps=10)
        for snap in trajectory:
            self.assertIn("n_disintegration_redistributions", snap)
            self.assertIn("n_types_lost", snap)
            self.assertIn("k_lost", snap)
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_metathetic.py::TestDisintegrationRedistribution -v`
Expected: FAIL (missing _check_disintegration_redistribution, missing counters)

**Step 3: Implement**

1. Add `_dissolved` field to MetatheticAgent:
```python
    _dissolved: bool = field(default=False, init=False, repr=False)
```

2. Add counters to MetatheticEnsemble.__init__ (after n_env_transitions):
```python
        self.n_disintegration_redistributions = 0
        self.n_types_lost = 0
        self.k_lost = 0.0
```

3. Add `_check_disintegration_redistribution()` method:
```python
    def _check_disintegration_redistribution(self) -> None:
        """Check for disintegrated agents and redistribute their types/knowledge.

        Redistribution is weighted by Jaccard similarity (interaction proximity).
        Types go to the most-similar active agent. Knowledge is split proportionally.
        If no active agent has Jaccard > 0, types and knowledge are lost.
        """
        type_counts = self._all_type_counts()
        active = self._active_agents()
        if not active:
            return

        for agent in list(self.agents):
            if agent._dissolved or agent.active:
                continue

            ts = agent.temporal_state_with_context(type_counts)
            if ts != 0:
                continue

            # Compute Jaccard weights to each active agent
            weights = {}
            for other in active:
                w = _jaccard(agent.type_set, other.type_set)
                if w > 0:
                    weights[other.agent_id] = w

            total_w = sum(weights.values())

            if total_w == 0:
                # No interactive neighbor — knowledge and types are lost
                self.n_types_lost += len(agent.type_set)
                self.k_lost += agent.k
            else:
                # Redistribute types to highest-weight agent per type
                agent_by_id = {a.agent_id: a for a in active}
                sorted_recipients = sorted(weights.keys(), key=lambda aid: (-weights[aid], aid))
                best_recipient = agent_by_id[sorted_recipients[0]]
                best_recipient.type_set = best_recipient.type_set | agent.type_set

                # Redistribute knowledge proportionally
                for aid, w in weights.items():
                    fraction = w / total_w
                    agent_by_id[aid].k += agent.k * fraction

            agent.type_set = set()
            agent.k = 0.0
            agent._dissolved = True
            self.n_disintegration_redistributions += 1
```

4. In `run()`, call `_check_disintegration_redistribution()` after `_check_cross_metathesis()` and before `_update_environment()`.

5. Add redistribution diagnostics to snapshot:
```python
            snapshot["n_disintegration_redistributions"] = self.n_disintegration_redistributions
            snapshot["n_types_lost"] = self.n_types_lost
            snapshot["k_lost"] = round(self.k_lost, 4)
```

**Step 4: Run tests**

Run: `python -m pytest tests/test_metathetic.py::TestDisintegrationRedistribution -v`
Expected: All 6 PASS.

**Step 5: Full suite**

Run: `python -m pytest tests/ -q`
Expected: All pass.

**Step 6: Commit**

```bash
git add simulator/metathetic.py tests/test_metathetic.py
git commit -m "feat: add disintegration redistribution with Jaccard-weighted knowledge flow"
```

---

### Task 4: Update longrun diagnostics + summary

**Files:**
- Modify: `scripts/longrun_diagnostics.py`
- Test: `tests/test_longrun.py`

**Context:**
Add the new snapshot fields to the longrun summary JSON and print output.

**Step 1: Write failing test**

```python
class TestAffordanceAndRedistributionDiagnostics(unittest.TestCase):
    def test_summary_contains_affordance_mean(self):
        """Summary should include final affordance_mean."""
        from scripts.longrun_diagnostics import run_and_diagnose
        _, summary = run_and_diagnose(n_agents=5, steps=20, seed=42)
        self.assertIn("affordance_mean_final", summary)

    def test_summary_contains_redistribution_counts(self):
        """Summary should include redistribution counters."""
        from scripts.longrun_diagnostics import run_and_diagnose
        _, summary = run_and_diagnose(n_agents=5, steps=20, seed=42)
        self.assertIn("n_disintegration_redistributions", summary)
        self.assertIn("n_types_lost", summary)
        self.assertIn("k_lost", summary)
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_longrun.py::TestAffordanceAndRedistributionDiagnostics -v`
Expected: FAIL (missing keys in summary)

**Step 3: Implement**

In `scripts/longrun_diagnostics.py`, add to the summary dict (after `n_env_transitions`):
```python
        "affordance_mean_final": round(last.get("affordance_mean", 0.0), 4),
        "n_disintegration_redistributions": last.get("n_disintegration_redistributions", 0),
        "n_types_lost": last.get("n_types_lost", 0),
        "k_lost": round(last.get("k_lost", 0.0), 4),
```

In the print output section (after temporal distribution), add:
```python
    aff = summary.get("affordance_mean_final", 0.0)
    print(f"  Affordance:       {aff:.4f}")
    print(f"  Disintegrations:    {summary.get('n_disintegration_redistributions', 0)}")
    tl = summary.get("n_types_lost", 0)
    kl = summary.get("k_lost", 0.0)
    if tl > 0 or kl > 0:
        print(f"  Types lost:       {tl}")
        print(f"  Knowledge lost:   {kl:.4f}")
```

**Step 4: Run tests**

Run: `python -m pytest tests/test_longrun.py -v`
Expected: All PASS.

**Step 5: Commit**

```bash
git add scripts/longrun_diagnostics.py tests/test_longrun.py
git commit -m "feat: add affordance + redistribution metrics to longrun diagnostics"
```

---

### Task 5: Final verification + push

**Step 1: Run full test suite**

Run: `python -m pytest tests/ -v`
Expected: All pass (~175+ tests).

**Step 2: Run longrun diagnostics to verify output**

Run: `python scripts/longrun_diagnostics.py --seed 42`
Expected: Completes with affordance and redistribution metrics in output.

**Step 3: Run audit**

Run: `python scripts/audit_claims.py`
Expected: PASS (no changes to claim annotations needed — C8 already exploratory).

**Step 4: Push**

```bash
git push origin unified-integration-20260225
```
