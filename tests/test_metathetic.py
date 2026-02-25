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

    def test_annihilated_state(self):
        """Inactive agent with dormant_steps >= 30, no living type connections -> state 0."""
        agent = self._make_agent(active=False)
        agent._dormant_steps = 40
        # active_type_counts has none of agent's types
        active_type_counts = {10: 3, 11: 2}  # agent has types {1, 2} — not present
        self.assertEqual(agent.temporal_state_with_context(active_type_counts), 0)

    def test_inactive_agent_not_annihilated_if_types_still_active(self):
        """Inactive agent with types still active should be desituated (3), not annihilated."""
        agent = self._make_agent(active=False)
        agent._dormant_steps = 40
        # type 1 is still held by an active agent
        active_type_counts = {1: 2, 10: 3}
        self.assertEqual(agent.temporal_state_with_context(active_type_counts), 3)

    def test_inactive_agent_not_annihilated_if_dormant_too_short(self):
        """Inactive agent with short dormancy should be desituated (3), not annihilated."""
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

    def test_desituated_suppresses(self):
        from simulator.metathetic import _temporal_threshold_multiplier
        self.assertEqual(_temporal_threshold_multiplier(3), float('inf'))

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


if __name__ == "__main__":
    unittest.main()
