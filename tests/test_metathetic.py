"""Tests for metathetic multi-agent TAP dynamics."""
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from simulator.metathetic import (
    MetatheticAgent,
    EnvironmentState,
    MetatheticEnsemble,
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


if __name__ == "__main__":
    unittest.main()
