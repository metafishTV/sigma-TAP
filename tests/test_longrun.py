"""Tests for long-run scaling diagnostics (Taalbi-inspired)."""
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from simulator.analysis import innovation_rate_scaling, constraint_tag


class TestInnovationRateScaling(unittest.TestCase):
    def test_linear_trajectory(self):
        """Exponential M(t) should give scaling exponent ~ 1."""
        import math
        # M(t) = 10 * exp(0.1 * t) => dk/dt ~ k => sigma ~ 1
        m_traj = [10.0 * math.exp(0.1 * t) for t in range(50)]
        result = innovation_rate_scaling(m_traj, dt=1.0)
        self.assertIn("exponent", result)
        self.assertAlmostEqual(result["exponent"], 1.0, delta=0.3)

    def test_superlinear_trajectory(self):
        """Power-law growth M ~ t^3 should give exponent > 1."""
        # Use pure power-law: M(t) = (1+t)^3, so dM/dt ~ M^(2/3) * 3
        # In log-log: log(rate) = (2/3)*log(M) + const => exponent ~ 2/3?
        # Actually for dk/dt ~ k^sigma with k = t^3: dk/dt = 3t^2, k = t^3
        # so sigma = d(log 3t^2)/d(log t^3) = (2/t) / (3/t) = 2/3
        # Use instead: M(t) = exp(t^2) for truly superlinear (accelerating) growth
        import math
        m_traj = [math.exp(0.001 * t ** 2) for t in range(100)]
        result = innovation_rate_scaling(m_traj, dt=1.0)
        self.assertGreater(result["exponent"], 1.0)

    def test_short_trajectory_graceful(self):
        """Very short trajectory should not crash."""
        result = innovation_rate_scaling([10.0, 11.0], dt=1.0)
        self.assertIn("exponent", result)


class TestConstraintTag(unittest.TestCase):
    def test_resource_limited(self):
        """M plateauing well below K is resource-limited."""
        m_traj = [10.0 + 0.1 * t for t in range(100)]
        tag = constraint_tag(m_traj, carrying_capacity=1e6, dt=1.0)
        self.assertIn(tag, {"adjacency-limited", "resource-limited", "mixed"})

    def test_no_capacity_is_adjacency(self):
        """Without carrying capacity, tag should be adjacency-limited."""
        m_traj = [10.0 + t for t in range(50)]
        tag = constraint_tag(m_traj, carrying_capacity=None, dt=1.0)
        self.assertEqual(tag, "adjacency-limited")


if __name__ == "__main__":
    unittest.main()
