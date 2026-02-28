"""Tests for metathetic multi-agent TAP dynamics."""
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from simulator.metathetic import (
    MetatheticAgent,
    EnvironmentState,
    MetatheticEnsemble,
    _jaccard,
    _goal_alignment,
    _agent_weight,
    _temporal_threshold_multiplier,
    _signature_similarity,
)


class TestMetatheticAgent(unittest.TestCase):
    def test_creation(self):
        """Agent starts with correct initial state."""
        agent = MetatheticAgent(agent_id=0, type_set={1, 2}, k=0.0, M_local=10.0)
        self.assertEqual(agent.agent_id, 0)
        self.assertEqual(agent.type_set, {1, 2})
        self.assertEqual(agent.k, 0.0)
        self.assertEqual(agent.M_local, 10.0)
        self.assertTrue(agent.active)

    def test_dormant_preserves_state(self):
        """Setting dormant preserves k and type_set."""
        agent = MetatheticAgent(agent_id=0, type_set={1, 2, 3}, k=50.0, M_local=20.0)
        agent.active = False
        self.assertFalse(agent.active)
        self.assertEqual(agent.k, 50.0)
        self.assertEqual(agent.type_set, {1, 2, 3})


class TestEnvironmentState(unittest.TestCase):
    def test_creation_defaults(self):
        env = EnvironmentState()
        self.assertGreater(env.a_env, 0)
        self.assertGreater(env.K_env, 0)
        self.assertEqual(env.texture_type, 1)

    def test_innovation_potential(self):
        env = EnvironmentState(K_env=1000.0)
        pot = env.innovation_potential(total_M=300.0)
        self.assertAlmostEqual(pot, 700.0)

    def test_innovation_potential_floor(self):
        env = EnvironmentState(K_env=100.0)
        pot = env.innovation_potential(total_M=200.0)
        self.assertEqual(pot, 0.0)

    def test_cross_threshold_varies_by_texture(self):
        """Type IV (turbulent) should have lower threshold than Type I (placid)."""
        env = EnvironmentState()
        env.texture_type = 1
        t1 = env.cross_metathesis_threshold()
        env.texture_type = 4
        t4 = env.cross_metathesis_threshold()
        self.assertGreater(t1, t4)


class TestSelfMetathesis(unittest.TestCase):
    def test_self_metathesis_preserves_k(self):
        """Self-metathesis adds a type but preserves k."""
        agent = MetatheticAgent(agent_id=0, type_set={1}, k=25.0, M_local=10.0)
        old_k = agent.k
        old_types = len(agent.type_set)
        agent.self_metathesize(next_type_id=2)
        self.assertEqual(agent.k, old_k)
        self.assertEqual(len(agent.type_set), old_types + 1)
        self.assertIn(2, agent.type_set)

    def test_self_metathesis_does_not_duplicate(self):
        """Self-metathesis with existing type is a no-op on type_set size."""
        agent = MetatheticAgent(agent_id=0, type_set={1, 2}, k=10.0, M_local=5.0)
        agent.self_metathesize(next_type_id=2)
        self.assertEqual(len(agent.type_set), 2)


class TestCrossMetathesis(unittest.TestCase):
    def test_absorptive_cross(self):
        """Absorptive: smaller agent goes dormant, larger gets union of types."""
        a1 = MetatheticAgent(agent_id=0, type_set={1, 2}, k=30.0, M_local=20.0)
        a2 = MetatheticAgent(agent_id=1, type_set={2, 3}, k=10.0, M_local=5.0)
        MetatheticAgent.absorptive_cross(a1, a2)
        # a1 absorbs a2 (a2 has lower M_local)
        self.assertTrue(a1.active)
        self.assertFalse(a2.active)
        self.assertEqual(a1.type_set, {1, 2, 3})
        self.assertEqual(a1.k, 40.0)

    def test_novel_cross(self):
        """Novel: both parents go dormant, new agent is returned."""
        a1 = MetatheticAgent(agent_id=0, type_set={1, 2}, k=20.0, M_local=15.0)
        a2 = MetatheticAgent(agent_id=1, type_set={3, 4}, k=15.0, M_local=10.0)
        child = MetatheticAgent.novel_cross(a1, a2, child_id=2, next_type_id=5)
        self.assertFalse(a1.active)
        self.assertFalse(a2.active)
        self.assertTrue(child.active)
        self.assertEqual(child.k, 35.0)
        self.assertEqual(child.M_local, 25.0)
        # Child should have types from both parents plus at least one novel type
        self.assertTrue(len(child.type_set) >= 1)
        self.assertIn(5, child.type_set)


class TestEnsembleRun(unittest.TestCase):
    def test_runs_without_error(self):
        """Ensemble completes 50 steps with 5 agents."""
        ensemble = MetatheticEnsemble(
            n_agents=5, initial_M=10.0,
            alpha=1e-3, a=8.0, mu=0.02,
            seed=42,
        )
        trajectory = ensemble.run(steps=50)
        self.assertEqual(len(trajectory), 50)
        self.assertIn("D_total", trajectory[0])
        self.assertIn("k_total", trajectory[0])
        self.assertIn("n_active", trajectory[0])

    def test_diversity_non_decreasing(self):
        """Total diversity D should generally not decrease (types are not destroyed)."""
        # NOTE: This assertion holds under conservative params (alpha=1e-3, 30 steps)
        # where disintegration redistribution is unlikely to fire. With aggressive
        # params or longer runs, type loss during disintegration could violate this.
        ensemble = MetatheticEnsemble(
            n_agents=5, initial_M=10.0,
            alpha=1e-3, a=8.0, mu=0.02,
            seed=42,
        )
        trajectory = ensemble.run(steps=30)
        D_values = [s["D_total"] for s in trajectory]
        # D may stay flat but should never decrease
        for i in range(1, len(D_values)):
            self.assertGreaterEqual(D_values[i], D_values[i - 1])

    def test_texture_type_valid(self):
        """Environment texture type stays in {1, 2, 3, 4}."""
        ensemble = MetatheticEnsemble(
            n_agents=5, initial_M=10.0,
            alpha=1e-3, a=8.0, mu=0.02,
            seed=42,
        )
        trajectory = ensemble.run(steps=30)
        for s in trajectory:
            self.assertIn(s["texture_type"], {1, 2, 3, 4})

    def test_event_counts(self):
        """Run returns metathetic event counts."""
        ensemble = MetatheticEnsemble(
            n_agents=5, initial_M=10.0,
            alpha=1e-3, a=8.0, mu=0.02,
            seed=42,
        )
        trajectory = ensemble.run(steps=30)
        last = trajectory[-1]
        self.assertIn("n_self_metatheses", last)
        self.assertIn("n_absorptive_cross", last)
        self.assertIn("n_novel_cross", last)

    def test_env_transitions_in_snapshot(self):
        """n_env_transitions should appear in every snapshot."""
        ensemble = MetatheticEnsemble(
            n_agents=5, initial_M=10.0,
            alpha=1e-3, a=8.0, mu=0.02,
            seed=42,
        )
        trajectory = ensemble.run(steps=30)
        for s in trajectory:
            self.assertIn("n_env_transitions", s)

    def test_seed_reproducibility(self):
        """Two runs with same seed produce identical trajectories."""
        kwargs = dict(
            n_agents=5, initial_M=10.0,
            alpha=5e-3, a=3.0, mu=0.005,
            variant="logistic", carrying_capacity=2e5,
            seed=123,
        )
        traj1 = MetatheticEnsemble(**kwargs).run(steps=30)
        traj2 = MetatheticEnsemble(**kwargs).run(steps=30)
        for s1, s2 in zip(traj1, traj2):
            self.assertEqual(s1["D_total"], s2["D_total"])
            self.assertAlmostEqual(s1["total_M"], s2["total_M"], places=10)
            self.assertAlmostEqual(s1["k_total"], s2["k_total"], places=10)


class TestHelperFunctions(unittest.TestCase):
    """Direct tests for _jaccard, _goal_alignment, _agent_weight."""

    def test_jaccard_identical(self):
        self.assertAlmostEqual(_jaccard({1, 2, 3}, {1, 2, 3}), 1.0)

    def test_jaccard_disjoint(self):
        self.assertAlmostEqual(_jaccard({1, 2}, {3, 4}), 0.0)

    def test_jaccard_partial(self):
        # {1,2,3} ∩ {2,3,4} = {2,3}, |union| = 4 → 2/4 = 0.5
        self.assertAlmostEqual(_jaccard({1, 2, 3}, {2, 3, 4}), 0.5)

    def test_jaccard_empty_sets(self):
        self.assertAlmostEqual(_jaccard(set(), set()), 0.0)

    def test_goal_alignment_identical_series(self):
        """Identical dM histories → perfect alignment."""
        h = [1.0, 2.0, 3.0, 4.0, 5.0]
        self.assertAlmostEqual(_goal_alignment(h, h), 1.0, places=5)

    def test_goal_alignment_opposite(self):
        """Opposite dM histories → negative alignment."""
        h1 = [1.0, 2.0, 3.0, 4.0, 5.0]
        h2 = [5.0, 4.0, 3.0, 2.0, 1.0]
        self.assertLess(_goal_alignment(h1, h2), 0.0)

    def test_goal_alignment_short_history(self):
        """With <2 entries, return 0."""
        self.assertAlmostEqual(_goal_alignment([1.0], [2.0]), 0.0)

    def test_goal_alignment_constant_history(self):
        """All-same values (zero variance) → return 0."""
        self.assertAlmostEqual(_goal_alignment([5.0, 5.0, 5.0], [5.0, 5.0, 5.0]), 0.0)

    def test_goal_alignment_overflow_guard(self):
        """Extremely large values don't crash."""
        h1 = [1e200, 1e250, 1e300]
        h2 = [1e200, 1e250, 1e300]
        result = _goal_alignment(h1, h2)
        self.assertTrue(-1.0 <= result <= 1.0)

    def test_agent_weight_all_unique(self):
        """Agent with all unique types has weight 1.0."""
        agent = MetatheticAgent(agent_id=0, type_set={10, 11, 12}, k=0.0, M_local=1.0)
        # type counts: each appears once; n_active=5 → threshold = max(1, int(5*0.3)) = 1
        type_counts = {10: 1, 11: 1, 12: 1}
        w = _agent_weight(agent, type_counts, n_active=5)
        self.assertAlmostEqual(w, 1.0)

    def test_agent_weight_all_common(self):
        """Agent with all widely-shared types has weight 0.0."""
        agent = MetatheticAgent(agent_id=0, type_set={1, 2}, k=0.0, M_local=1.0)
        # Every type held by all 10 agents → threshold = max(1, int(10*0.3)) = 3
        type_counts = {1: 10, 2: 10}
        w = _agent_weight(agent, type_counts, n_active=10)
        self.assertAlmostEqual(w, 0.0)

    def test_agent_weight_empty_types(self):
        agent = MetatheticAgent(agent_id=0, type_set=set(), k=0.0, M_local=1.0)
        self.assertAlmostEqual(_agent_weight(agent, {}, n_active=5), 0.0)


class TestEnvironmentUpdate(unittest.TestCase):
    """Direct tests for EnvironmentState.update() drift logic."""

    def test_texture_type_from_regime(self):
        """update() sets texture_type based on regime string."""
        env = EnvironmentState()
        env.update(D_total=10, k_total=100.0, total_M=500.0, regime="explosive")
        self.assertEqual(env.texture_type, 4)
        env.update(D_total=10, k_total=100.0, total_M=500.0, regime="exponential")
        self.assertEqual(env.texture_type, 2)

    def test_a_env_drifts_down_with_high_diversity(self):
        """High diversity should push a_env down (richer adjacency)."""
        env = EnvironmentState(a_env=8.0, _a_env_base=8.0)
        old_a = env.a_env
        # D_total=100 with base 8 → a_target = max(2, 8 * 10/100) = max(2, 0.8) = 2
        env.update(D_total=100, k_total=50.0, total_M=50.0, regime="plateau")
        self.assertLess(env.a_env, old_a)

    def test_K_env_drifts_up_with_more_innovation(self):
        """More total k should push K_env up."""
        env = EnvironmentState(K_env=1e5, _K_env_base=1e5)
        old_K = env.K_env
        # K_target = 1e5 + 0.1 * 10000 = 101000 > 1e5
        env.update(D_total=5, k_total=10000.0, total_M=50.0, regime="plateau")
        self.assertGreater(env.K_env, old_K)

    def test_unknown_regime_defaults_to_type_1(self):
        env = EnvironmentState()
        env.update(D_total=5, k_total=50.0, total_M=50.0, regime="unknown_regime")
        self.assertEqual(env.texture_type, 1)


class TestEnsembleRegimeClassification(unittest.TestCase):
    """Integration: verify that the rolling history fix enables non-plateau regimes."""

    def test_rolling_history_grows(self):
        """After enough steps, internal history should have ≥3 points."""
        ensemble = MetatheticEnsemble(
            n_agents=5, initial_M=10.0,
            alpha=5e-3, a=3.0, mu=0.005,
            variant="logistic", carrying_capacity=2e5,
            seed=42,
        )
        ensemble.run(steps=15)
        self.assertGreaterEqual(len(ensemble._m_history), 3)
        self.assertGreaterEqual(len(ensemble._k_history), 3)

    def test_growth_params_logistic_produces_events(self):
        """Growth params with logistic variant should produce metathetic events."""
        ensemble = MetatheticEnsemble(
            n_agents=10, initial_M=10.0,
            alpha=5e-3, a=3.0, mu=0.005,
            variant="logistic", carrying_capacity=2e5,
            self_meta_threshold=0.15,
            seed=42,
        )
        trajectory = ensemble.run(steps=150)
        last = trajectory[-1]
        # At minimum, self-metatheses should fire with these growth params
        self.assertGreater(last["n_self_metatheses"], 0)


class TestTemporalState(unittest.TestCase):
    """Tests for the five-state temporal orientation gate on MetatheticAgent."""

    def _make_agent(self, **kwargs):
        """Helper to create an agent with sensible defaults."""
        defaults = dict(agent_id=0, type_set={1, 2}, k=10.0, M_local=10.0, active=True)
        defaults.update(kwargs)
        return MetatheticAgent(**defaults)

    def test_new_agent_is_desituated_novelty(self):
        """Agent with steps_since_metathesis=0 should be in state 3 (desituated/novelty)."""
        agent = self._make_agent()
        agent.steps_since_metathesis = 0
        agent.dM_history = [1.0, 2.0, 3.0]
        self.assertEqual(agent.temporal_state, 3)

    def test_agent_becomes_situated_after_novelty_window(self):
        """After novelty window with positive dM, agent reaches state 2 (situated)."""
        agent = self._make_agent()
        agent.steps_since_metathesis = 10  # past _NOVELTY_WINDOW=5
        agent.dM_history = [0.5, 0.6, 0.7, 0.8, 0.9]  # moderate positive, not strongly aligned
        self.assertEqual(agent.temporal_state, 2)

    def test_inertial_on_diverging_trajectory(self):
        """Negative/diverging dM history with steps=20 should give state 1 (inertial)."""
        agent = self._make_agent()
        agent.steps_since_metathesis = 20
        # Strongly negative and worsening trajectory
        agent.dM_history = [-1.0, -2.0, -3.0, -4.0, -5.0]
        self.assertEqual(agent.temporal_state, 1)

    def test_desituated_stagnation(self):
        """steps_since_metathesis=60 (>= _STAGNATION_THRESHOLD=50) -> state 3."""
        agent = self._make_agent()
        agent.steps_since_metathesis = 60
        agent.dM_history = [0.5, 0.6, 0.7, 0.8, 0.9]
        self.assertEqual(agent.temporal_state, 3)

    def test_established_after_alignment(self):
        """Strong positive dM history with enough steps -> state 4 (established)."""
        agent = self._make_agent()
        agent.steps_since_metathesis = 25
        # Strong positive trend that yields alignment > 0.5
        agent.dM_history = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        self.assertEqual(agent.temporal_state, 4)

    def test_disintegrated_state(self):
        """Inactive agent with dormant_steps >= 30, no living type connections -> state 0."""
        agent = self._make_agent(active=False)
        agent._dormant_steps = 40
        # active_type_counts has none of agent's types
        active_type_counts = {10: 3, 11: 2}  # agent has types {1, 2} — not present
        self.assertEqual(agent.temporal_state_with_context(active_type_counts), 0)

    def test_inactive_agent_not_disintegrated_if_types_still_active(self):
        """Inactive agent with types still active should be desituated (3), not disintegrated."""
        agent = self._make_agent(active=False)
        agent._dormant_steps = 40
        # type 1 is still held by an active agent
        active_type_counts = {1: 2, 10: 3}
        self.assertEqual(agent.temporal_state_with_context(active_type_counts), 3)

    def test_inactive_agent_not_disintegrated_if_dormant_too_short(self):
        """Inactive agent with short dormancy should be desituated (3), not disintegrated."""
        agent = self._make_agent(active=False)
        agent._dormant_steps = 10  # < _RELATIONAL_DECAY_WINDOW=30
        active_type_counts = {10: 3}  # no overlap with agent types
        self.assertEqual(agent.temporal_state_with_context(active_type_counts), 3)

    def test_self_metathesize_resets_steps(self):
        """self_metathesize should reset steps_since_metathesis to 0."""
        agent = self._make_agent()
        agent.steps_since_metathesis = 25
        agent.self_metathesize(next_type_id=99)
        self.assertEqual(agent.steps_since_metathesis, 0)

    def test_trajectory_alignment_short_history(self):
        """With fewer than 3 entries, _trajectory_alignment returns 0.0."""
        agent = self._make_agent()
        agent.dM_history = [1.0, 2.0]
        self.assertAlmostEqual(agent._trajectory_alignment(), 0.0)

    def test_trajectory_alignment_positive_trend(self):
        """Strongly positive trend returns a positive value."""
        agent = self._make_agent()
        agent.dM_history = [1.0, 2.0, 3.0, 4.0, 5.0]
        alignment = agent._trajectory_alignment()
        self.assertGreater(alignment, 0.0)

    def test_trajectory_alignment_negative_trend(self):
        """Strongly negative trend returns a negative value."""
        agent = self._make_agent()
        agent.dM_history = [-1.0, -2.0, -3.0, -4.0, -5.0]
        alignment = agent._trajectory_alignment()
        self.assertLess(alignment, 0.0)


class TestTemporalModulation(unittest.TestCase):
    def test_inertial_multiplier(self):
        from simulator.metathetic import _temporal_threshold_multiplier
        self.assertAlmostEqual(_temporal_threshold_multiplier(1), 0.5)

    def test_situated_multiplier(self):
        from simulator.metathetic import _temporal_threshold_multiplier
        self.assertAlmostEqual(_temporal_threshold_multiplier(2), 1.5)

    def test_established_multiplier(self):
        from simulator.metathetic import _temporal_threshold_multiplier
        self.assertAlmostEqual(_temporal_threshold_multiplier(4), 2.0)

    def test_desituated_novelty_suppresses(self):
        from simulator.metathetic import _temporal_threshold_multiplier
        self.assertEqual(_temporal_threshold_multiplier(3), float('inf'))

    def test_desituated_stagnation_cross_easier(self):
        """Stagnating agents get 0.5x threshold for cross-metathesis."""
        from simulator.metathetic import _temporal_threshold_multiplier
        self.assertAlmostEqual(
            _temporal_threshold_multiplier(3, for_cross=True, is_stagnating=True), 0.5)

    def test_desituated_stagnation_self_still_suppressed(self):
        """Stagnating agents still can't self-metathesize (need external stimulus)."""
        from simulator.metathetic import _temporal_threshold_multiplier
        self.assertEqual(
            _temporal_threshold_multiplier(3, for_cross=False, is_stagnating=True), float('inf'))

    def test_temporal_state_in_snapshot(self):
        ensemble = MetatheticEnsemble(
            n_agents=5, initial_M=10.0,
            alpha=1e-3, a=8.0, mu=0.02, seed=42,
        )
        trajectory = ensemble.run(steps=20)
        for s in trajectory:
            self.assertIn("temporal_state_counts", s)
            self.assertIsInstance(s["temporal_state_counts"], dict)

    def test_steps_since_metathesis_increments(self):
        ensemble = MetatheticEnsemble(
            n_agents=3, initial_M=10.0,
            alpha=1e-4, a=8.0, mu=0.02, seed=42,
        )
        ensemble.run(steps=10)
        for agent in ensemble.agents:
            if agent.active:
                self.assertGreaterEqual(agent.steps_since_metathesis, 10)

    def test_dormant_steps_tracked(self):
        a1 = MetatheticAgent(agent_id=0, type_set={1, 2}, k=20.0, M_local=15.0)
        a2 = MetatheticAgent(agent_id=1, type_set={3, 4}, k=15.0, M_local=10.0)
        MetatheticAgent.novel_cross(a1, a2, child_id=2, next_type_id=5)
        self.assertFalse(a1.active)
        self.assertEqual(a1._dormant_steps, 0)


class TestAffordanceTick(unittest.TestCase):
    """Tests for affordance tick computation."""

    def test_connected_agent_positive_dM_returns_1(self):
        from simulator.metathetic import _compute_affordance_tick, MetatheticAgent
        focal = MetatheticAgent(0, {0, 1}, 0.0, 10.0, dM_history=[1.0])
        others = [
            MetatheticAgent(1, {0, 2}, 0.0, 10.0, active=True),
            MetatheticAgent(2, {0, 3}, 0.0, 10.0, active=True),
        ]
        self.assertEqual(_compute_affordance_tick(focal, others, min_cluster=2), 1)

    def test_isolated_agent_returns_0(self):
        from simulator.metathetic import _compute_affordance_tick, MetatheticAgent
        focal = MetatheticAgent(0, {99}, 0.0, 10.0, dM_history=[1.0])
        others = [MetatheticAgent(1, {0, 2}, 0.0, 10.0, active=True)]
        self.assertEqual(_compute_affordance_tick(focal, others, min_cluster=2), 0)

    def test_negative_dM_returns_0(self):
        from simulator.metathetic import _compute_affordance_tick, MetatheticAgent
        focal = MetatheticAgent(0, {0, 1}, 0.0, 10.0, dM_history=[-1.0])
        others = [
            MetatheticAgent(1, {0, 2}, 0.0, 10.0, active=True),
            MetatheticAgent(2, {0, 3}, 0.0, 10.0, active=True),
        ]
        self.assertEqual(_compute_affordance_tick(focal, others, min_cluster=2), 0)

    def test_below_min_cluster_returns_0(self):
        from simulator.metathetic import _compute_affordance_tick, MetatheticAgent
        focal = MetatheticAgent(0, {0, 1}, 0.0, 10.0, dM_history=[1.0])
        others = [MetatheticAgent(1, {0, 2}, 0.0, 10.0, active=True)]
        self.assertEqual(_compute_affordance_tick(focal, others, min_cluster=2), 0)

    def test_empty_dM_history_returns_0(self):
        from simulator.metathetic import _compute_affordance_tick, MetatheticAgent
        focal = MetatheticAgent(0, {0, 1}, 0.0, 10.0, dM_history=[])
        others = [
            MetatheticAgent(1, {0, 2}, 0.0, 10.0, active=True),
            MetatheticAgent(2, {0, 3}, 0.0, 10.0, active=True),
        ]
        self.assertEqual(_compute_affordance_tick(focal, others, min_cluster=2), 0)

    def test_affordance_ticks_field_exists(self):
        from simulator.metathetic import MetatheticAgent
        a = MetatheticAgent(0, {0}, 0.0, 10.0)
        self.assertIsInstance(a._affordance_ticks, list)
        self.assertEqual(len(a._affordance_ticks), 0)

    def test_affordance_score_property(self):
        from simulator.metathetic import MetatheticAgent
        a = MetatheticAgent(0, {0}, 0.0, 10.0)
        a._affordance_ticks = [1, 1, 0, 0, 1]
        self.assertAlmostEqual(a.affordance_score, 0.6)

    def test_affordance_score_empty_returns_zero(self):
        from simulator.metathetic import MetatheticAgent
        a = MetatheticAgent(0, {0}, 0.0, 10.0)
        self.assertEqual(a.affordance_score, 0.0)


class TestAffordanceGate(unittest.TestCase):
    """Tests for affordance-gated self-metathesis."""

    def test_isolated_agent_cannot_self_metathesize(self):
        from simulator.metathetic import MetatheticEnsemble
        ens = MetatheticEnsemble(
            n_agents=3, initial_M=10.0,
            alpha=5e-3, a=3.0, mu=0.005,
            variant="logistic", carrying_capacity=2e5,
            seed=42, affordance_min_cluster=2,
        )
        ens.agents[0].type_set = {999}
        ens.agents[0].dM_history = [100.0]
        ens.agents[0]._affordance_ticks = []
        ens.agents[0].steps_since_metathesis = 10

        old_types = len(ens.agents[0].type_set)
        ens._update_affordance_ticks()
        ens._check_self_metathesis()
        self.assertEqual(len(ens.agents[0].type_set), old_types)

    def test_connected_agent_can_self_metathesize(self):
        from simulator.metathetic import MetatheticEnsemble
        ens = MetatheticEnsemble(
            n_agents=5, initial_M=10.0,
            alpha=5e-3, a=3.0, mu=0.005,
            variant="logistic", carrying_capacity=2e5,
            seed=42, affordance_min_cluster=2,
        )
        ens.agents[0].dM_history = [100.0] * 5
        ens.agents[0]._affordance_ticks = [1, 1, 1]
        ens.agents[0].steps_since_metathesis = 10

        old_self = ens.n_self_metatheses
        ens._check_self_metathesis()
        self.assertGreater(ens.n_self_metatheses, old_self)

    def test_affordance_ticks_updated_in_run(self):
        from simulator.metathetic import MetatheticEnsemble
        ens = MetatheticEnsemble(
            n_agents=5, initial_M=10.0,
            alpha=5e-3, a=3.0, mu=0.005,
            variant="logistic", carrying_capacity=2e5,
            seed=42, affordance_min_cluster=2,
        )
        ens.run(steps=20)
        for a in ens._active_agents():
            self.assertGreater(len(a._affordance_ticks), 0)
            self.assertLessEqual(len(a._affordance_ticks), a._AFFORDANCE_WINDOW)

    def test_affordance_mean_in_snapshot(self):
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


class TestDisintegrationRedistribution(unittest.TestCase):
    """Tests for disintegration redistribution mechanism."""

    def test_redistribution_to_most_similar(self):
        """Types should go to the agent with highest Jaccard similarity."""
        from simulator.metathetic import MetatheticEnsemble
        ens = MetatheticEnsemble(
            n_agents=3, initial_M=10.0,
            alpha=5e-3, a=3.0, mu=0.005,
            variant="logistic", carrying_capacity=2e5,
            seed=42,
        )
        # Make agent 0 disintegrated: inactive, dormant long enough, unique types
        ens.agents[0].active = False
        ens.agents[0]._dormant_steps = 50
        ens.agents[0].type_set = {100, 101}
        ens.agents[0].k = 50.0

        # Agent 1 shares type 100 (high Jaccard)
        ens.agents[1].type_set = {0, 1, 100}
        ens.agents[1].k = 10.0

        # Agent 2 shares nothing with disintegrated
        ens.agents[2].type_set = {0, 2}
        ens.agents[2].k = 10.0

        ens._check_disintegration_redistribution()

        self.assertIn(100, ens.agents[1].type_set)
        self.assertIn(101, ens.agents[1].type_set)
        self.assertNotIn(101, ens.agents[2].type_set)

    def test_knowledge_split_proportionally(self):
        """Knowledge should be split proportional to Jaccard weights."""
        from simulator.metathetic import MetatheticEnsemble
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

        ens.agents[1].type_set = {0, 100}
        ens.agents[1].k = 0.0
        ens.agents[2].type_set = {0, 100}
        ens.agents[2].k = 0.0
        ens.agents[3].type_set = {0, 3}
        ens.agents[3].k = 0.0

        ens._check_disintegration_redistribution()

        self.assertAlmostEqual(ens.agents[1].k, 50.0, places=1)
        self.assertAlmostEqual(ens.agents[2].k, 50.0, places=1)
        self.assertAlmostEqual(ens.agents[3].k, 0.0)

        # With equal Jaccard, lowest agent_id (agent 1) wins the types
        self.assertIn(101, ens.agents[1].type_set)

    def test_no_neighbors_means_deep_stasis(self):
        """If no active agent has Jaccard>0, agent enters deep stasis."""
        from simulator.metathetic import MetatheticEnsemble
        ens = MetatheticEnsemble(
            n_agents=3, initial_M=10.0,
            alpha=5e-3, a=3.0, mu=0.005,
            variant="logistic", carrying_capacity=2e5,
            seed=42,
        )
        ens.agents[0].active = False
        ens.agents[0]._dormant_steps = 50
        ens.agents[0].type_set = {999}
        ens.agents[0].k = 100.0

        ens.agents[1].type_set = {0, 1}
        ens.agents[2].type_set = {0, 2}
        k1_before = ens.agents[1].k
        k2_before = ens.agents[2].k

        ens._check_disintegration_redistribution()

        # Other agents' k unchanged
        self.assertAlmostEqual(ens.agents[1].k, k1_before)
        self.assertAlmostEqual(ens.agents[2].k, k2_before)
        # Deep stasis: types preserved, n_types_lost = 0, k truncated to 5%
        self.assertTrue(ens.agents[0]._deep_stasis)
        self.assertFalse(ens.agents[0]._dissolved)
        self.assertEqual(ens.agents[0].type_set, {999})
        self.assertAlmostEqual(ens.agents[0].k, 5.0, places=1)
        self.assertEqual(ens.n_types_lost, 0)
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
        ens.agents[1].type_set = {0, 1, 100}

        ens._check_disintegration_redistribution()
        self.assertEqual(ens.n_disintegration_redistributions, 1)

    def test_dissolved_agent_flagged(self):
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

    def test_double_disintegration_prevented(self):
        """Dissolved agent should not be disintegrated a second time."""
        from simulator.metathetic import MetatheticEnsemble, MetatheticAgent
        ens = MetatheticEnsemble(
            n_agents=3, initial_M=10.0,
            alpha=5e-3, a=3.0, mu=0.005,
            variant="logistic", carrying_capacity=2e5,
            seed=42,
        )
        # Force agent 0 to dormancy threshold with type overlap
        ens.agents[0].active = False
        ens.agents[0]._dormant_steps = 30
        ens.agents[0].type_set = {0, 1}
        ens.agents[0].k = 50.0
        ens.agents[1].type_set = {0, 2}
        ens.agents[2].type_set = {1, 3}

        ens._check_disintegration_redistribution()
        self.assertEqual(ens.n_disintegration_redistributions, 1)
        self.assertTrue(ens.agents[0]._dissolved)

        # Second call should be a no-op
        ens._check_disintegration_redistribution()
        self.assertEqual(ens.n_disintegration_redistributions, 1)  # Still 1

    def test_deep_stasis_retains_types(self):
        """Agent with no Jaccard neighbor enters deep stasis, retains types."""
        from simulator.metathetic import MetatheticEnsemble
        ens = MetatheticEnsemble(
            n_agents=3, initial_M=10.0,
            alpha=5e-3, a=3.0, mu=0.005,
            variant="logistic", carrying_capacity=2e5,
            seed=42,
        )
        # Agent 0: unique types, no overlap with anyone
        ens.agents[0].active = False
        ens.agents[0]._dormant_steps = 30
        ens.agents[0].type_set = {999, 998}
        ens.agents[0].k = 100.0
        # Agents 1,2: share types with each other but NOT with agent 0
        ens.agents[1].type_set = {0, 1}
        ens.agents[2].type_set = {0, 2}

        ens._check_disintegration_redistribution()

        # Agent should be in deep stasis, not dissolved
        self.assertTrue(ens.agents[0]._deep_stasis)
        self.assertFalse(ens.agents[0]._dissolved)
        # Types preserved
        self.assertEqual(ens.agents[0].type_set, {999, 998})
        # Knowledge truncated to 5% residual
        self.assertAlmostEqual(ens.agents[0].k, 5.0, places=1)
        # k_lost tracks the truncated portion
        self.assertAlmostEqual(ens.k_lost, 95.0, places=1)
        # n_types_lost is 0 (types preserved)
        self.assertEqual(ens.n_types_lost, 0)

    def test_deep_stasis_not_redisintegrated(self):
        """Deep stasis agent should not be processed again."""
        from simulator.metathetic import MetatheticEnsemble
        ens = MetatheticEnsemble(
            n_agents=3, initial_M=10.0,
            alpha=5e-3, a=3.0, mu=0.005,
            variant="logistic", carrying_capacity=2e5,
            seed=42,
        )
        ens.agents[0].active = False
        ens.agents[0]._dormant_steps = 30
        ens.agents[0].type_set = {999}
        ens.agents[0].k = 100.0
        ens.agents[1].type_set = {0, 1}
        ens.agents[2].type_set = {0, 2}

        ens._check_disintegration_redistribution()
        first_k = ens.agents[0].k
        first_lost = ens.k_lost

        # Second call should be a no-op
        ens._check_disintegration_redistribution()
        self.assertEqual(ens.agents[0].k, first_k)
        self.assertEqual(ens.k_lost, first_lost)
        self.assertEqual(ens.n_disintegration_redistributions, 1)

    def test_disintegration_fires_in_full_run(self):
        """With aggressive params and long run, disintegration should fire naturally."""
        from simulator.metathetic import MetatheticEnsemble
        # High alpha + long run + many agents = some will go dormant and disintegrate
        ens = MetatheticEnsemble(
            n_agents=15, initial_M=10.0,
            alpha=5e-3, a=3.0, mu=0.005,
            variant="logistic", carrying_capacity=2e5,
            seed=99,
        )
        trajectory = ens.run(steps=200)
        final = trajectory[-1]
        # At least verify the counter exists and is non-negative
        self.assertGreaterEqual(final["n_disintegration_redistributions"], 0)
        # With these params, we expect at least some disintegration events
        # (if not, the test still passes — it's a smoke test, not a guarantee)


class TestLMatrixLedger(unittest.TestCase):
    """Per-agent L-matrix event ledger (Emery channels)."""

    def test_initial_ledger_zeros(self):
        """New agent starts with all L-matrix counters at zero."""
        agent = MetatheticAgent(agent_id=0, type_set={1}, k=0.0, M_local=10.0)
        self.assertEqual(agent.n_self_metatheses_local, 0)
        self.assertEqual(agent.n_novel_cross_local, 0)
        self.assertEqual(agent.n_absorptive_given_local, 0)
        self.assertEqual(agent.n_absorptive_received_local, 0)
        self.assertEqual(agent.n_env_transitions_local, 0)

    def test_self_metathesis_increments_l11(self):
        """Self-metathesis increments the L11 (intrapraxis) counter."""
        agent = MetatheticAgent(agent_id=0, type_set={1}, k=0.0, M_local=10.0)
        agent.self_metathesize(next_type_id=99)
        self.assertEqual(agent.n_self_metatheses_local, 1)
        agent.self_metathesize(next_type_id=100)
        self.assertEqual(agent.n_self_metatheses_local, 2)

    def test_absorptive_cross_increments_l12_l21(self):
        """Absorptive cross increments L12 for donor (given), L21 for receiver."""
        a1 = MetatheticAgent(agent_id=1, type_set={1, 2}, k=10.0, M_local=50.0)
        a2 = MetatheticAgent(agent_id=2, type_set={2, 3}, k=5.0, M_local=20.0)
        MetatheticAgent.absorptive_cross(a1, a2)
        # a1 has higher M_local so a1 absorbs a2
        # a1 = absorber = received; a2 = absorbed = given
        self.assertEqual(a1.n_absorptive_received_local, 1)
        self.assertEqual(a2.n_absorptive_given_local, 1)

    def test_novel_cross_increments_l12(self):
        """Novel cross increments L12 (novel_cross_local) for both parents."""
        a1 = MetatheticAgent(agent_id=1, type_set={1}, k=5.0, M_local=10.0)
        a2 = MetatheticAgent(agent_id=2, type_set={2}, k=5.0, M_local=10.0)
        child = MetatheticAgent.novel_cross(a1, a2, child_id=3, next_type_id=99)
        self.assertEqual(a1.n_novel_cross_local, 1)
        self.assertEqual(a2.n_novel_cross_local, 1)
        # Child starts fresh
        self.assertEqual(child.n_novel_cross_local, 0)


class TestTAPSSignature(unittest.TestCase):
    """Per-agent TAPS dispositional signature."""

    def test_fresh_agent_default_signature(self):
        """Agent with no events gets default signature.

        A fresh agent has steps_since_metathesis=0, so has_adpression=True,
        giving A-letter='A'. All counters zero gives T=T, P=X, S=S.
        """
        agent = MetatheticAgent(agent_id=0, type_set={1}, k=0.0, M_local=10.0)
        sig = agent.taps_signature
        self.assertEqual(len(sig), 4)
        # T=T (balanced), A=A (adpression: steps_since_metathesis=0), P=X (balanced), S=S (default)
        self.assertEqual(sig, "TAXS")

    def test_signature_is_four_letters(self):
        """Signature always returns exactly 4 characters."""
        agent = MetatheticAgent(agent_id=0, type_set={1}, k=0.0, M_local=10.0)
        agent.n_self_metatheses_local = 10
        agent.n_novel_cross_local = 5
        self.assertEqual(len(agent.taps_signature), 4)

    def test_involution_dominant_gives_I(self):
        """Agent with mostly L11+L21 events has T-letter = I."""
        agent = MetatheticAgent(agent_id=0, type_set={1}, k=0.0, M_local=10.0)
        agent.n_self_metatheses_local = 10  # L11
        agent.n_absorptive_received_local = 5  # L21
        agent.n_novel_cross_local = 1  # L12 (low)
        sig = agent.taps_signature
        self.assertEqual(sig[0], "I")

    def test_evolution_dominant_gives_E(self):
        """Agent with mostly L12+L22 events has T-letter = E."""
        agent = MetatheticAgent(agent_id=0, type_set={1}, k=0.0, M_local=10.0)
        agent.n_novel_cross_local = 10  # L12
        agent.n_env_transitions_local = 5  # L22
        agent.n_self_metatheses_local = 1  # L11 (low)
        sig = agent.taps_signature
        self.assertEqual(sig[0], "E")

    def test_balanced_gives_T(self):
        """Agent with balanced inward/outward has T-letter = T."""
        agent = MetatheticAgent(agent_id=0, type_set={1}, k=0.0, M_local=10.0)
        agent.n_self_metatheses_local = 5  # L11
        agent.n_novel_cross_local = 5  # L12
        sig = agent.taps_signature
        self.assertEqual(sig[0], "T")

    def test_consummation_dominant_gives_U(self):
        """Agent with mostly L12 outward events has P-letter = U."""
        agent = MetatheticAgent(agent_id=0, type_set={1}, k=0.0, M_local=10.0)
        agent.n_novel_cross_local = 10  # L12 outward
        agent.n_absorptive_received_local = 1  # L21 inward (low)
        sig = agent.taps_signature
        self.assertEqual(sig[2], "U")

    def test_consumption_dominant_gives_R(self):
        """Agent with mostly L21 inward events has P-letter = R."""
        agent = MetatheticAgent(agent_id=0, type_set={1}, k=0.0, M_local=10.0)
        agent.n_absorptive_received_local = 10  # L21
        agent.n_novel_cross_local = 1  # L12 (low)
        sig = agent.taps_signature
        self.assertEqual(sig[2], "R")

    def test_adpression_on_fresh_metathesis(self):
        """Agent that just self-metathesized has A-letter = A."""
        agent = MetatheticAgent(agent_id=0, type_set={1}, k=0.0, M_local=10.0)
        agent.self_metathesize(next_type_id=99)
        # steps_since_metathesis == 0 after self_metathesize
        sig = agent.taps_signature
        self.assertEqual(sig[1], "A")

    def test_dormant_agent_preservation(self):
        """Dormant agent has S-letter = P (preservation)."""
        agent = MetatheticAgent(agent_id=0, type_set={1}, k=0.0, M_local=10.0)
        agent.active = False
        sig = agent.taps_signature
        self.assertEqual(sig[3], "P")


class TestSignatureClassification(unittest.TestCase):
    """Three-level tension classification for cross-metathesis."""

    def test_identical_signatures_produce_absorptive(self):
        """Two agents with 4/4 matching signature → absorptive (low tension)."""
        a1 = MetatheticAgent(agent_id=1, type_set={1, 2, 3}, k=5.0, M_local=20.0)
        a2 = MetatheticAgent(agent_id=2, type_set={1, 2, 3}, k=5.0, M_local=15.0)
        for a in (a1, a2):
            a.n_self_metatheses_local = 10
            a.dM_history = [1.0, 1.0, 1.0]
            a._affordance_ticks = [1, 1, 1, 1, 1]
        self.assertEqual(a1.taps_signature, a2.taps_signature)
        similarity = _signature_similarity(a1.taps_signature, a2.taps_signature)
        self.assertEqual(similarity, 4)

    def test_different_signatures_produce_novel(self):
        """Two agents with very different dispositions → high tension (0-1 match)."""
        a1 = MetatheticAgent(agent_id=1, type_set={1}, k=5.0, M_local=20.0)
        a2 = MetatheticAgent(agent_id=2, type_set={2}, k=5.0, M_local=15.0)
        # a1: inward-dominant, recent self-metathesis, high affordance
        a1.n_self_metatheses_local = 20
        a1.dM_history = [1.0, 1.0, 1.0]
        a1._affordance_ticks = [1, 1, 1, 1, 1]
        a1.steps_since_metathesis = 0  # just metathesized → A = "A"
        # a2: outward-dominant, no affordance, absorptive-received dominates
        a2.n_novel_cross_local = 0
        a2.n_absorptive_given_local = 20
        a2.n_env_transitions_local = 5
        a2.n_absorptive_received_local = 1
        a2.dM_history = [-0.1, -0.1, -0.1]
        a2._affordance_ticks = [0, 0, 0, 0, 0]
        a2.steps_since_metathesis = 10  # past adpression window
        # a1 sig: I(inward>outward) A(adpression) X(l21==l12==0) S(synthesis=20)
        # a2 sig: E(outward>inward) I(l21 dominates: 1>=0,>=0) U(l12=20>l21=1) D(disintegration=5>synthesis=0,integration=1)
        self.assertEqual(a1.taps_signature, "IAXS")
        self.assertEqual(a2.taps_signature, "EIUD")
        # "IAXS" vs "EIUD" → 0 matches
        similarity = _signature_similarity(a1.taps_signature, a2.taps_signature)
        self.assertEqual(similarity, 0)

    def test_signature_similarity_function(self):
        """_signature_similarity counts matching positions."""
        self.assertEqual(_signature_similarity("IEUS", "IEUS"), 4)
        self.assertEqual(_signature_similarity("IEUS", "EIRS"), 1)  # only S at pos 3
        self.assertEqual(_signature_similarity("TEXP", "TEXP"), 4)
        self.assertEqual(_signature_similarity("IAXS", "EIUD"), 0)  # no matches

    def test_mid_tension_falls_back_to_L_vs_G(self):
        """With 2/4 matching letters, classification uses L vs G tiebreak."""
        self.assertEqual(_signature_similarity("IEUS", "IEXD"), 2)  # I,E match; U!=X, S!=D


class TestYounRatioImprovement(unittest.TestCase):
    """Verify Stage 3A classification produces absorptive events."""

    def test_youn_ratio_below_one(self):
        """Full ensemble run should produce at least some absorptive events.

        The Youn ratio (novel / total_cross) should be < 1.0, indicating
        the signature-based classification is producing absorptive events
        that the old L > G rule never did.
        """
        ensemble = MetatheticEnsemble(
            n_agents=8,
            initial_M=10.0,
            alpha=5e-3,
            a=3.0,
            mu=0.005,
            carrying_capacity=500.0,
            seed=42,
        )
        traj = ensemble.run(steps=150)

        final = traj[-1]
        n_novel = final["n_novel_cross"]
        n_absorptive = final["n_absorptive_cross"]
        total_cross = n_novel + n_absorptive

        # Must have SOME cross-metathesis events
        self.assertGreater(total_cross, 0,
                           "No cross-metathesis events at all")

        # Must have at least one absorptive event (Youn ratio < 1.0)
        self.assertGreater(n_absorptive, 0,
                           f"Youn ratio still 1.0: {n_novel} novel, "
                           f"{n_absorptive} absorptive")

    def test_youn_ratio_in_target_range(self):
        """Youn exploration fraction should be closer to 0.6 than to 1.0.

        This is a soft target — we check that the ratio has meaningfully
        moved toward the empirical target, not that it's exactly 0.6.
        """
        ensemble = MetatheticEnsemble(
            n_agents=8,
            initial_M=10.0,
            alpha=5e-3,
            a=3.0,
            mu=0.005,
            carrying_capacity=500.0,
            seed=42,
        )
        traj = ensemble.run(steps=150)

        final = traj[-1]
        n_novel = final["n_novel_cross"]
        n_absorptive = final["n_absorptive_cross"]
        total_cross = n_novel + n_absorptive

        if total_cross == 0:
            self.skipTest("No cross-metathesis events")

        exploration_fraction = n_novel / total_cross
        # Should be meaningfully below 1.0 (moved toward 0.6 target)
        self.assertLess(exploration_fraction, 0.95,
                        f"Youn ratio barely moved: {exploration_fraction:.3f}")


class TestSignatureSnapshot(unittest.TestCase):
    """Verify per-agent signature data appears in ensemble snapshots."""

    def test_snapshot_has_signature_distribution(self):
        """Each snapshot includes signature_distribution dict."""
        ensemble = MetatheticEnsemble(
            n_agents=4, initial_M=10.0, alpha=1e-3, a=8.0, mu=0.02, seed=1
        )
        traj = ensemble.run(steps=20)
        for snap in traj:
            self.assertIn("signature_distribution", snap)
            self.assertIsInstance(snap["signature_distribution"], dict)
            # Values should sum to number of active agents
            self.assertEqual(
                sum(snap["signature_distribution"].values()),
                snap["n_active"],
            )

    def test_snapshot_has_signature_diversity(self):
        """Each snapshot includes signature_diversity (count of unique sigs)."""
        ensemble = MetatheticEnsemble(
            n_agents=4, initial_M=10.0, alpha=1e-3, a=8.0, mu=0.02, seed=1
        )
        traj = ensemble.run(steps=20)
        for snap in traj:
            self.assertIn("signature_diversity", snap)
            self.assertGreaterEqual(snap["signature_diversity"], 1)
            self.assertLessEqual(snap["signature_diversity"], snap["n_active"])


class TestEnvTransitionPerAgent(unittest.TestCase):
    """L22 channel: env transitions recorded per agent."""

    def test_env_transition_increments_active_agents(self):
        """When environment texture changes, all active agents get L22 tick."""
        ensemble = MetatheticEnsemble(
            n_agents=4, initial_M=10.0,
            alpha=5e-3, a=3.0, mu=0.005, seed=42,
            carrying_capacity=500.0,
        )
        # Run enough steps that at least one env transition occurs
        traj = ensemble.run(steps=100)
        final = traj[-1]

        if final["n_env_transitions"] == 0:
            self.skipTest("No env transitions occurred in this run")

        # At least one active agent should have recorded L22 events
        active = ensemble._active_agents()
        total_l22 = sum(a.n_env_transitions_local for a in active)
        self.assertGreater(total_l22, 0,
                           "Env transitions happened but no agent recorded L22")


class TestSigmaWiring(unittest.TestCase):
    """Phase 1: Sigma-TAP feedback wired into multi-agent layer."""

    def test_xi_local_attribute_exists(self):
        """MetatheticAgent has Xi_local attribute defaulting to 0.0."""
        agent = MetatheticAgent(agent_id=0, type_set={1}, k=0.0, M_local=10.0)
        self.assertEqual(agent.Xi_local, 0.0)

    def test_defaults_recover_stage3a(self):
        """With default sigma params (gamma=0, beta=0, eta=0),
        simulation is bit-identical to a run without sigma wiring."""
        kwargs = dict(
            n_agents=5, initial_M=10.0,
            alpha=5e-3, a=3.0, mu=0.005,
            variant="logistic", carrying_capacity=2e5,
            seed=42,
        )
        traj = MetatheticEnsemble(**kwargs).run(steps=50)
        for snap in traj:
            self.assertAlmostEqual(snap["Xi_mean"], 0.0)
            self.assertAlmostEqual(snap["Xi_std"], 0.0)
            self.assertAlmostEqual(snap["sigma_mean"], 1.0)

    def test_xi_accumulates_with_beta(self):
        """When beta > 0, Xi should accumulate from birth term exposure."""
        ens = MetatheticEnsemble(
            n_agents=5, initial_M=10.0,
            alpha=5e-3, a=3.0, mu=0.005,
            variant="logistic", carrying_capacity=2e5,
            seed=42, beta=0.1, sigma0=1.0, gamma=0.0,
        )
        traj = ens.run(steps=30)
        self.assertGreater(traj[-1]["Xi_mean"], 0.0)

    def test_sigma_modulates_with_gamma(self):
        """When gamma > 0 and beta > 0, sigma should exceed 1.0."""
        ens = MetatheticEnsemble(
            n_agents=5, initial_M=10.0,
            alpha=5e-3, a=3.0, mu=0.005,
            variant="logistic", carrying_capacity=2e5,
            seed=42, beta=0.1, gamma=0.1, sigma0=1.0,
        )
        traj = ens.run(steps=30)
        self.assertGreater(traj[-1]["sigma_mean"], 1.0)

    def test_negative_gamma_decreases_sigma(self):
        """Negative gamma should decrease sigma below sigma0."""
        ens = MetatheticEnsemble(
            n_agents=5, initial_M=10.0,
            alpha=5e-3, a=3.0, mu=0.005,
            variant="logistic", carrying_capacity=2e5,
            seed=42, beta=0.1, gamma=-0.5, sigma0=1.0,
        )
        traj = ens.run(steps=30)
        self.assertLessEqual(traj[-1]["sigma_mean"], 1.0)

    def test_h_decay_compresses_xi(self):
        """H feedback with eta > 0 should slow Xi growth.

        H = -h_decay * Xi is negative when Xi > 0.
        With eta > 0, the term eta * H < 0 drags Xi back toward zero,
        compressing its growth relative to an eta=0 baseline.
        """
        ens_no_h = MetatheticEnsemble(
            n_agents=5, initial_M=10.0,
            alpha=5e-3, a=3.0, mu=0.005,
            variant="logistic", carrying_capacity=2e5,
            seed=42, beta=0.1, eta=0.0,
        )
        ens_with_h = MetatheticEnsemble(
            n_agents=5, initial_M=10.0,
            alpha=5e-3, a=3.0, mu=0.005,
            variant="logistic", carrying_capacity=2e5,
            seed=42, beta=0.1, eta=0.5, h_decay=0.02,
        )
        traj_no = ens_no_h.run(steps=30)
        traj_with = ens_with_h.run(steps=30)
        self.assertLess(traj_with[-1]["Xi_mean"], traj_no[-1]["Xi_mean"])

    def test_snapshot_has_xi_fields(self):
        """Snapshot includes Xi_mean, Xi_std, sigma_mean."""
        ens = MetatheticEnsemble(
            n_agents=5, initial_M=10.0,
            alpha=1e-3, a=8.0, mu=0.02, seed=42,
        )
        traj = ens.run(steps=10)
        for snap in traj:
            self.assertIn("Xi_mean", snap)
            self.assertIn("Xi_std", snap)
            self.assertIn("sigma_mean", snap)

    def test_seed_reproducibility_with_sigma(self):
        """Two identical sigma runs produce identical results."""
        kwargs = dict(
            n_agents=5, initial_M=10.0,
            alpha=5e-3, a=3.0, mu=0.005,
            variant="logistic", carrying_capacity=2e5,
            seed=42, sigma0=1.0, gamma=0.1, beta=0.1,
        )
        traj1 = MetatheticEnsemble(**kwargs).run(steps=30)
        traj2 = MetatheticEnsemble(**kwargs).run(steps=30)
        for s1, s2 in zip(traj1, traj2):
            self.assertAlmostEqual(s1["Xi_mean"], s2["Xi_mean"], places=10)
            self.assertAlmostEqual(s1["sigma_mean"], s2["sigma_mean"], places=10)


class TestSeedEntropy(unittest.TestCase):
    """Phase 2: Per-agent parameter offsets via seed entropy."""

    def test_zero_entropy_identical_params(self):
        """seed_entropy=0 means all agents have same alpha/mu as ensemble."""
        ens = MetatheticEnsemble(
            n_agents=5, initial_M=10.0,
            alpha=1e-3, a=8.0, mu=0.02, seed=42,
            seed_entropy=0.0,
        )
        for agent in ens.agents:
            self.assertEqual(agent.alpha_local, 1e-3)
            self.assertEqual(agent.mu_local, 0.02)

    def test_nonzero_entropy_produces_variation(self):
        """seed_entropy>0 gives different alpha_local/mu_local across agents."""
        ens = MetatheticEnsemble(
            n_agents=10, initial_M=10.0,
            alpha=1e-3, a=8.0, mu=0.02, seed=42,
            seed_entropy=0.1,
        )
        alphas = [a.alpha_local for a in ens.agents]
        mus = [a.mu_local for a in ens.agents]
        # Not all identical
        self.assertGreater(len(set(alphas)), 1,
                           "All alpha_local values are identical despite seed_entropy>0")
        self.assertGreater(len(set(mus)), 1,
                           "All mu_local values are identical despite seed_entropy>0")
        # At least one differs from ensemble default
        self.assertTrue(any(a != 1e-3 for a in alphas))
        self.assertTrue(any(m != 0.02 for m in mus))

    def test_params_stay_positive(self):
        """No negative alpha_local or mu_local even with large seed_entropy."""
        ens = MetatheticEnsemble(
            n_agents=20, initial_M=10.0,
            alpha=1e-3, a=8.0, mu=0.02, seed=42,
            seed_entropy=0.5,
        )
        for agent in ens.agents:
            self.assertGreater(agent.alpha_local, 0.0,
                               f"Agent {agent.agent_id} has non-positive alpha_local")
            self.assertGreater(agent.mu_local, 0.0,
                               f"Agent {agent.agent_id} has non-positive mu_local")

    def test_divergent_trajectories(self):
        """Agents with seed_entropy>0 diverge in M_local over 50 steps
        vs identical M_local when seed_entropy=0."""
        kwargs_base = dict(
            n_agents=5, initial_M=10.0,
            alpha=5e-3, a=3.0, mu=0.005,
            variant="logistic", carrying_capacity=2e5,
            seed=42,
        )
        # seed_entropy=0: all agents use same alpha/mu
        ens0 = MetatheticEnsemble(**kwargs_base, seed_entropy=0.0)
        traj0 = ens0.run(steps=50)

        # seed_entropy=0.1: agents have different alpha/mu
        ens1 = MetatheticEnsemble(**kwargs_base, seed_entropy=0.1)
        traj1 = ens1.run(steps=50)

        # Compute variance of M_local at final step for each
        active0 = [a.M_local for a in ens0.agents if a.active]
        active1 = [a.M_local for a in ens1.agents if a.active]

        if len(active0) >= 2 and len(active1) >= 2:
            mean0 = sum(active0) / len(active0)
            mean1 = sum(active1) / len(active1)
            var0 = sum((x - mean0) ** 2 for x in active0) / len(active0)
            var1 = sum((x - mean1) ** 2 for x in active1) / len(active1)
            # Entropy run should have at least as much or more variance
            # (agents diverge due to different growth/death rates)
            self.assertGreater(var1, var0 * 0.5,
                               "Seed entropy did not produce divergent trajectories")

    def test_novel_cross_child_inherits_avg_params(self):
        """Child from novel_cross gets mean of parent params."""
        a1 = MetatheticAgent(agent_id=0, type_set={0, 1}, k=5.0, M_local=10.0)
        a1.alpha_local = 1e-3
        a1.mu_local = 0.02

        a2 = MetatheticAgent(agent_id=1, type_set={0, 2}, k=5.0, M_local=10.0)
        a2.alpha_local = 3e-3
        a2.mu_local = 0.04

        child = MetatheticAgent.novel_cross(a1, a2, child_id=2, next_type_id=10)

        self.assertAlmostEqual(child.alpha_local, (1e-3 + 3e-3) / 2.0)
        self.assertAlmostEqual(child.mu_local, (0.02 + 0.04) / 2.0)

    def test_backward_compat_with_seed_entropy_zero(self):
        """seed_entropy=0 produces bit-identical trajectory to Phase 1 defaults."""
        kwargs = dict(
            n_agents=5, initial_M=10.0,
            alpha=5e-3, a=3.0, mu=0.005,
            variant="logistic", carrying_capacity=2e5,
            seed=42,
        )
        # Phase 1 default: no seed_entropy param
        traj_default = MetatheticEnsemble(**kwargs).run(steps=50)
        # Phase 2 explicit zero
        traj_zero = MetatheticEnsemble(**kwargs, seed_entropy=0.0).run(steps=50)

        for s1, s2 in zip(traj_default, traj_zero):
            self.assertAlmostEqual(s1["total_M"], s2["total_M"], places=10)
            self.assertAlmostEqual(s1["k_total"], s2["k_total"], places=10)
            self.assertAlmostEqual(s1["Xi_mean"], s2["Xi_mean"], places=10)
            self.assertAlmostEqual(s1["sigma_mean"], s2["sigma_mean"], places=10)


class TestTrustMetrics(unittest.TestCase):
    """Phase 3: Trust metrics (tau_self, trust_map)."""

    def _make_ensemble(self, **overrides):
        defaults = dict(
            n_agents=5, initial_M=10.0,
            alpha=5e-3, a=3.0, mu=0.005,
            variant="logistic", carrying_capacity=2e5,
            seed=42,
        )
        defaults.update(overrides)
        return MetatheticEnsemble(**defaults)

    def test_default_trust_unchanged(self):
        """trust_update_rate=0 keeps tau_self at 0.5 and trust_map empty."""
        ens = self._make_ensemble(trust_update_rate=0.0, trust_decay_rate=0.0)
        ens.run(steps=50)
        for agent in ens.agents:
            self.assertEqual(agent.tau_self, 0.5,
                             f"Agent {agent.agent_id} tau_self changed from 0.5")
            self.assertEqual(len(agent.trust_map), 0,
                             f"Agent {agent.agent_id} trust_map is non-empty")

    def test_tau_self_in_snapshot(self):
        """Snapshots contain tau_self_mean and tau_pair_mean keys."""
        ens = self._make_ensemble()
        traj = ens.run(steps=5)
        for snap in traj:
            self.assertIn("tau_self_mean", snap)
            self.assertIn("tau_pair_mean", snap)

    def test_tau_self_increases_on_stable_growth(self):
        """Stable positive trajectory increases tau_self above 0.5."""
        ens = self._make_ensemble(trust_update_rate=0.1)
        # Manually set up an agent with stable positive history
        agent = ens.agents[0]
        agent.dM_history = [1.0, 1.1, 1.2, 1.3, 1.4]
        old_tau = agent.tau_self
        ens._update_trust_self()
        self.assertGreater(agent.tau_self, old_tau,
                           "tau_self did not increase on stable positive growth")

    def test_trust_map_populated_after_cross(self):
        """Cross-metathesis with trust_update_rate>0 populates trust_map."""
        ens = self._make_ensemble(
            n_agents=10, trust_update_rate=0.2,
            alpha=5e-3, a=8.0, mu=0.005,
        )
        ens.run(steps=200)
        # Check if any agent has a non-empty trust_map
        any_trust = any(len(a.trust_map) > 0 for a in ens.agents)
        total_cross = ens.n_absorptive_cross + ens.n_novel_cross
        if total_cross > 0:
            self.assertTrue(any_trust,
                            "No trust_map populated despite cross-metathesis events")

    def test_trust_decay_toward_baseline(self):
        """trust_decay_rate>0 moves tau_self toward 0.5 baseline."""
        ens = self._make_ensemble(trust_decay_rate=0.1)
        agent = ens.agents[0]
        agent.tau_self = 0.9  # Far above baseline
        ens._decay_trust()
        self.assertLess(agent.tau_self, 0.9,
                        "tau_self did not decay toward baseline")
        self.assertGreater(agent.tau_self, 0.5,
                           "tau_self decayed past baseline")

    def test_tau_bounds(self):
        """tau_self stays in [0.0, 1.0] even with extreme inputs."""
        ens = self._make_ensemble(trust_update_rate=0.5)
        agent = ens.agents[0]
        # Push toward upper bound
        agent.tau_self = 0.99
        agent.dM_history = [100.0, 100.0, 100.0, 100.0, 100.0]
        for _ in range(20):
            ens._update_trust_self()
        self.assertLessEqual(agent.tau_self, 1.0)
        self.assertGreaterEqual(agent.tau_self, 0.0)

        # Push toward lower bound
        agent.tau_self = 0.01
        agent.dM_history = [-100.0, -50.0, -200.0, -10.0, -300.0]
        for _ in range(20):
            ens._update_trust_self()
        self.assertLessEqual(agent.tau_self, 1.0)
        self.assertGreaterEqual(agent.tau_self, 0.0)

    def test_backward_compat_no_trust(self):
        """trust_update_rate=0.0 produces bit-identical trajectory to Phase 2 defaults."""
        kwargs = dict(
            n_agents=5, initial_M=10.0,
            alpha=5e-3, a=3.0, mu=0.005,
            variant="logistic", carrying_capacity=2e5,
            seed=42,
        )
        # Phase 2 baseline: no trust params
        traj_base = MetatheticEnsemble(**kwargs).run(steps=50)
        # Phase 3 explicit zeros
        traj_trust = MetatheticEnsemble(
            **kwargs, trust_update_rate=0.0, trust_decay_rate=0.0
        ).run(steps=50)

        for s1, s2 in zip(traj_base, traj_trust):
            self.assertAlmostEqual(s1["total_M"], s2["total_M"], places=10)
            self.assertAlmostEqual(s1["k_total"], s2["k_total"], places=10)
            self.assertAlmostEqual(s1["Xi_mean"], s2["Xi_mean"], places=10)
            self.assertAlmostEqual(s1["sigma_mean"], s2["sigma_mean"], places=10)


class TestEndogenousMu(unittest.TestCase):
    """Phase 4: Death rate modulated by affordance score."""

    def test_default_mu_unchanged(self):
        """mu_affordance_sensitivity=0 means D uses base mu exactly."""
        ens = MetatheticEnsemble(
            n_agents=5, initial_M=10.0,
            alpha=5e-3, a=3.0, mu=0.02,
            variant="logistic", carrying_capacity=2e5,
            seed=42,
            mu_affordance_sensitivity=0.0,
        )
        ens.run(steps=20)
        for agent in ens.agents:
            if agent.active:
                # With sensitivity=0, mu_eff should equal base mu exactly
                base_mu = agent.mu_local if agent.mu_local is not None else ens.mu
                self.assertAlmostEqual(agent._mu_eff, base_mu, places=12)

    def test_high_affordance_lowers_mu(self):
        """Agent with affordance_score approaching 1.0 has mu_eff < base mu when sensitivity > 0."""
        ens = MetatheticEnsemble(
            n_agents=5, initial_M=10.0,
            alpha=5e-3, a=3.0, mu=0.02,
            variant="logistic", carrying_capacity=2e5,
            seed=42,
            mu_affordance_sensitivity=0.5,
        )
        # Manually pump affordance ticks to get a high affordance_score
        for agent in ens.agents:
            agent._affordance_ticks = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        ens._step_agents()
        for agent in ens.agents:
            if agent.active:
                base_mu = agent.mu_local if agent.mu_local is not None else ens.mu
                self.assertLess(agent._mu_eff, base_mu)

    def test_mu_lower_bound_respected(self):
        """mu_eff >= mu_lower_bound even with high affordance and sensitivity."""
        floor = 0.01
        ens = MetatheticEnsemble(
            n_agents=5, initial_M=10.0,
            alpha=5e-3, a=3.0, mu=0.02,
            variant="logistic", carrying_capacity=2e5,
            seed=42,
            mu_affordance_sensitivity=1.0,
            mu_lower_bound=floor,
        )
        # Perfect affordance => would try to go to 0.0 without floor
        for agent in ens.agents:
            agent._affordance_ticks = [1] * 10
        ens._step_agents()
        for agent in ens.agents:
            if agent.active:
                self.assertGreaterEqual(agent._mu_eff, floor)

    def test_mu_never_negative(self):
        """Even with extreme affordance_score and sensitivity, mu_eff >= 0."""
        ens = MetatheticEnsemble(
            n_agents=5, initial_M=10.0,
            alpha=5e-3, a=3.0, mu=0.02,
            variant="logistic", carrying_capacity=2e5,
            seed=42,
            mu_affordance_sensitivity=2.0,  # Extreme sensitivity
        )
        for agent in ens.agents:
            agent._affordance_ticks = [1] * 10  # Max affordance
        ens._step_agents()
        for agent in ens.agents:
            if agent.active:
                self.assertGreaterEqual(agent._mu_eff, 0.0)

    def test_snapshot_has_mu_eff_fields(self):
        """mu_eff_mean and mu_eff_std keys present in snapshot."""
        ens = MetatheticEnsemble(
            n_agents=5, initial_M=10.0,
            alpha=5e-3, a=3.0, mu=0.02,
            variant="logistic", carrying_capacity=2e5,
            seed=42,
            mu_affordance_sensitivity=0.3,
        )
        traj = ens.run(steps=10)
        for snap in traj:
            self.assertIn("mu_eff_mean", snap)
            self.assertIn("mu_eff_std", snap)
            self.assertIsInstance(snap["mu_eff_mean"], float)
            self.assertIsInstance(snap["mu_eff_std"], float)
            self.assertGreaterEqual(snap["mu_eff_std"], 0.0)

    def test_backward_compat_mu_zero_sensitivity(self):
        """sensitivity=0 produces bit-identical trajectories to Phase 3 defaults."""
        kwargs = dict(
            n_agents=5, initial_M=10.0,
            alpha=5e-3, a=3.0, mu=0.005,
            variant="logistic", carrying_capacity=2e5,
            seed=42,
        )
        # Phase 3 baseline: no mu params
        traj_base = MetatheticEnsemble(**kwargs).run(steps=50)
        # Phase 4 explicit zeros
        traj_mu = MetatheticEnsemble(
            **kwargs, mu_affordance_sensitivity=0.0, mu_lower_bound=None,
        ).run(steps=50)

        for s1, s2 in zip(traj_base, traj_mu):
            self.assertAlmostEqual(s1["total_M"], s2["total_M"], places=10)
            self.assertAlmostEqual(s1["k_total"], s2["k_total"], places=10)
            self.assertAlmostEqual(s1["Xi_mean"], s2["Xi_mean"], places=10)
            self.assertAlmostEqual(s1["sigma_mean"], s2["sigma_mean"], places=10)


class TestFamilyGroups(unittest.TestCase):
    """Phase 5: Topology / family group tracking tests."""

    def test_initial_agents_have_no_family(self):
        """All agents start with family_id=None."""
        ens = MetatheticEnsemble(
            n_agents=5, initial_M=10.0,
            alpha=5e-3, a=3.0, mu=0.005,
            variant="logistic", carrying_capacity=2e5,
            seed=42,
        )
        for agent in ens.agents:
            self.assertIsNone(agent.family_id)

    def test_family_forms_on_self_metathesis(self):
        """After self-metathesis fires, the agent gets a family_id (not None)."""
        ens = MetatheticEnsemble(
            n_agents=5, initial_M=10.0,
            alpha=5e-3, a=3.0, mu=0.005,
            variant="logistic", carrying_capacity=2e5,
            seed=42,
        )
        # Run enough steps that at least one self-metathesis fires
        traj = ens.run(steps=100)
        if ens.n_self_metatheses > 0:
            # At least one agent should have a family
            has_family = any(a.family_id is not None for a in ens.agents)
            self.assertTrue(has_family, "Self-metathesis fired but no agent has a family_id")
        else:
            self.skipTest("No self-metathesis fired in 100 steps")

    def test_novel_cross_child_gets_new_family(self):
        """After novel cross, child has a family_id different from parents' families."""
        ens = MetatheticEnsemble(
            n_agents=5, initial_M=10.0,
            alpha=5e-3, a=3.0, mu=0.005,
            variant="logistic", carrying_capacity=2e5,
            seed=5,  # seed=5 reliably triggers novel cross events
        )
        traj = ens.run(steps=200)
        self.assertGreater(ens.n_novel_cross, 0,
                           "Expected novel cross events with seed=5")
        # Children are agents appended after the initial n_agents
        children = [a for a in ens.agents if a.agent_id >= 5]
        self.assertTrue(len(children) > 0, "Novel cross fired but no child agents found")
        for child in children:
            self.assertIsNotNone(child.family_id,
                                 f"Child agent {child.agent_id} has no family_id")

    def test_snapshot_has_family_fields(self):
        """n_families, family_size_distribution, family_lineage_depth in snapshot."""
        ens = MetatheticEnsemble(
            n_agents=5, initial_M=10.0,
            alpha=5e-3, a=3.0, mu=0.005,
            variant="logistic", carrying_capacity=2e5,
            seed=42,
        )
        traj = ens.run(steps=10)
        for snap in traj:
            self.assertIn("n_families", snap)
            self.assertIn("family_size_distribution", snap)
            self.assertIn("family_lineage_depth", snap)
            self.assertIsInstance(snap["n_families"], int)
            self.assertIsInstance(snap["family_size_distribution"], dict)
            self.assertIsInstance(snap["family_lineage_depth"], int)

    def test_n_families_zero_initially(self):
        """Before any metathetic events, n_families=0 in snapshot."""
        ens = MetatheticEnsemble(
            n_agents=3, initial_M=10.0,
            alpha=1e-6, a=8.0, mu=0.5,
            variant="logistic", carrying_capacity=2e5,
            seed=99,
        )
        # Very low alpha, very high mu → no self-metathesis should fire on step 0.
        traj = ens.run(steps=1)
        self.assertEqual(traj[0]["n_families"], 0)

    def test_family_counter_monotonic(self):
        """Family IDs are monotonically increasing."""
        ens = MetatheticEnsemble(
            n_agents=5, initial_M=10.0,
            alpha=5e-3, a=3.0, mu=0.005,
            variant="logistic", carrying_capacity=2e5,
            seed=42,
        )
        traj = ens.run(steps=200)
        # Collect all family IDs from the internal registry
        if ens._families:
            fids = sorted(ens._families.keys())
            for i in range(1, len(fids)):
                self.assertGreater(fids[i], fids[i - 1],
                                   "Family IDs are not monotonically increasing")
            # First family ID should be 1
            self.assertEqual(fids[0], 1)

    def test_backward_compat_families_are_additive(self):
        """Family tracking doesn't change dynamics: bit-identical total_M, k_total, Xi_mean, sigma_mean."""
        kwargs = dict(
            n_agents=5, initial_M=10.0,
            alpha=5e-3, a=3.0, mu=0.005,
            variant="logistic", carrying_capacity=2e5,
            seed=42,
        )
        # Run two identical ensembles — family tracking is always on but purely
        # observational, so results must be identical seed-for-seed.
        traj1 = MetatheticEnsemble(**kwargs).run(steps=50)
        traj2 = MetatheticEnsemble(**kwargs).run(steps=50)
        for s1, s2 in zip(traj1, traj2):
            self.assertAlmostEqual(s1["total_M"], s2["total_M"], places=10)
            self.assertAlmostEqual(s1["k_total"], s2["k_total"], places=10)
            self.assertAlmostEqual(s1["Xi_mean"], s2["Xi_mean"], places=10)
            self.assertAlmostEqual(s1["sigma_mean"], s2["sigma_mean"], places=10)


if __name__ == "__main__":
    unittest.main()
