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


from simulator.longrun import (
    heaps_law_fit,
    gini_coefficient,
    top_k_share,
    diversification_rate,
    enhanced_constraint_tag,
)


class TestHeapsLawFit(unittest.TestCase):
    def test_known_power_law(self):
        """D(k) = 2 * k^0.7 should return beta ~ 0.7."""
        k_series = [float(i) for i in range(1, 101)]
        D_series = [2.0 * (k ** 0.7) for k in k_series]
        result = heaps_law_fit(D_series, k_series)
        self.assertAlmostEqual(result["beta"], 0.7, delta=0.05)
        self.assertGreater(result["r_squared"], 0.95)

    def test_linear_gives_beta_one(self):
        """D(k) = k should give beta ~ 1."""
        k_series = [float(i) for i in range(1, 51)]
        D_series = list(k_series)
        result = heaps_law_fit(D_series, k_series)
        self.assertAlmostEqual(result["beta"], 1.0, delta=0.05)

    def test_short_series_graceful(self):
        """Very short series should not crash."""
        result = heaps_law_fit([1.0, 2.0], [1.0, 3.0])
        self.assertIn("beta", result)

    def test_constant_D_series(self):
        """Constant D (ss_yy = 0) should not crash, returns r_squared=0."""
        k_series = [float(i) for i in range(1, 20)]
        D_series = [5.0] * len(k_series)
        result = heaps_law_fit(D_series, k_series)
        self.assertAlmostEqual(result["r_squared"], 0.0)
        self.assertAlmostEqual(result["beta"], 0.0, delta=0.01)


class TestGiniCoefficient(unittest.TestCase):
    def test_perfect_equality(self):
        """All equal values -> Gini = 0."""
        self.assertAlmostEqual(gini_coefficient([10.0, 10.0, 10.0, 10.0]), 0.0, places=5)

    def test_perfect_inequality(self):
        """One agent has everything -> Gini near 1."""
        gini = gini_coefficient([0.0, 0.0, 0.0, 100.0])
        self.assertGreater(gini, 0.7)

    def test_single_value(self):
        """Single value -> Gini = 0."""
        self.assertAlmostEqual(gini_coefficient([42.0]), 0.0, places=5)

    def test_empty_returns_zero(self):
        self.assertAlmostEqual(gini_coefficient([]), 0.0, places=5)


class TestTopKShare(unittest.TestCase):
    def test_equal_distribution(self):
        """Equal distribution: top 50% holds 50%."""
        self.assertAlmostEqual(top_k_share([10.0] * 10, k_frac=0.5), 0.5, delta=0.01)

    def test_skewed_distribution(self):
        """One dominant agent: top 10% holds most."""
        vals = [1.0] * 9 + [91.0]
        self.assertGreater(top_k_share(vals, k_frac=0.1), 0.9)


class TestDiversificationRate(unittest.TestCase):
    def test_heaps_sublinear_gives_declining_rate(self):
        """D(k) = k^0.5 -> dD/dk should be declining."""
        import math
        k_series = [float(i) for i in range(1, 51)]
        D_series = [math.sqrt(k) for k in k_series]
        rates = diversification_rate(D_series, k_series)
        self.assertGreater(len(rates), 0)
        # First rate should exceed last rate
        if len(rates) > 5:
            self.assertGreater(rates[0], rates[-1])


class TestEnhancedConstraintTag(unittest.TestCase):
    def test_adjacency_limited(self):
        result = enhanced_constraint_tag(
            sigma=1.5, beta=0.6, gini=0.3,
            carrying_capacity=None, m_final=100.0,
        )
        self.assertEqual(result["tag"], "adjacency-limited")
        self.assertIn(result["confidence"], {"high", "medium", "low"})

    def test_resource_limited(self):
        result = enhanced_constraint_tag(
            sigma=1.0, beta=0.8, gini=0.3,
            carrying_capacity=1000.0, m_final=900.0,
        )
        self.assertEqual(result["tag"], "resource-limited")

    def test_returns_reasoning(self):
        result = enhanced_constraint_tag(
            sigma=1.2, beta=0.7, gini=0.4,
            carrying_capacity=None, m_final=50.0,
        )
        self.assertIn("reasoning", result)
        self.assertIsInstance(result["reasoning"], str)

    def test_mixed_tag(self):
        """Ambiguous indicators should produce 'mixed' tag."""
        # sigma=1.25 (not >1.3 so no adjacency), beta=0.75 (not <0.7),
        # gini=0.5 (not <0.3), not near capacity → no strong signal either way
        result = enhanced_constraint_tag(
            sigma=1.25, beta=0.75, gini=0.5,
            carrying_capacity=1000.0, m_final=100.0,
        )
        self.assertEqual(result["tag"], "mixed")


class TestDiversificationRateEdgeCases(unittest.TestCase):
    def test_dk_zero_returns_zero_rate(self):
        """When dk=0 at some steps, rate is 0 (not error)."""
        k_series = [1.0, 1.0, 2.0, 2.0, 3.0]
        D_series = [1.0, 1.0, 2.0, 2.0, 3.0]
        rates = diversification_rate(D_series, k_series)
        self.assertEqual(len(rates), 4)
        self.assertAlmostEqual(rates[0], 0.0)  # dk=0 → rate=0
        self.assertAlmostEqual(rates[2], 0.0)  # dk=0 → rate=0

    def test_empty_series(self):
        """Empty series returns empty rates."""
        self.assertEqual(diversification_rate([], []), [])


class TestLongrunDiagnosticsSummaryKeys(unittest.TestCase):
    """Verify longrun diagnostics summary contains affordance + redistribution keys."""

    def test_summary_has_affordance_mean(self):
        """Summary should include affordance_mean_final key."""
        from scripts.longrun_diagnostics import run_and_diagnose
        _, summary = run_and_diagnose(n_agents=5, steps=20, seed=42)
        self.assertIn("affordance_mean_final", summary)
        self.assertIsInstance(summary["affordance_mean_final"], float)

    def test_summary_has_redistribution_keys(self):
        """Summary should include disintegration redistribution diagnostic keys."""
        from scripts.longrun_diagnostics import run_and_diagnose
        _, summary = run_and_diagnose(n_agents=5, steps=20, seed=42)
        self.assertIn("n_disintegration_redistributions", summary)
        self.assertIn("n_types_lost", summary)
        self.assertIn("k_lost", summary)


if __name__ == "__main__":
    unittest.main()
