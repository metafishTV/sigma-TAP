"""Tests for TAPS/RIP diagnostic overlay."""
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def _make_trajectory(steps=10, with_events=True):
    """Build a synthetic trajectory for testing.

    Creates a trajectory where events happen at specific steps so test
    assertions are deterministic.
    """
    traj = []
    cumul = {"self": 0, "absorptive": 0, "novel": 0, "env": 0, "disint": 0,
             "types_lost": 0, "k_lost": 0.0, "deep_stasis": 0}
    for t in range(steps):
        if with_events and t == 3:
            cumul["self"] += 1
            cumul["absorptive"] += 1
        if with_events and t == 6:
            cumul["novel"] += 1
            cumul["disint"] += 1
        traj.append({
            "step": t,
            "D_total": 10 + t,
            "k_total": 50.0 + t * 5,
            "total_M": 100.0 + t * 10,
            "n_active": 8,
            "n_dormant": 2,
            "agent_k_list": [10.0] * 8,
            "convergence": 0.5,
            "texture_type": 1,
            "a_env": 3.0,
            "K_env": 2e5,
            "innovation_potential": 0.8 - t * 0.01,
            "n_self_metatheses": cumul["self"],
            "n_absorptive_cross": cumul["absorptive"],
            "n_novel_cross": cumul["novel"],
            "n_env_transitions": cumul["env"],
            "n_disintegration_redistributions": cumul["disint"],
            "n_types_lost": cumul["types_lost"],
            "k_lost": cumul["k_lost"],
            "n_deep_stasis": cumul["deep_stasis"],
            "affordance_mean": 0.6,
            "temporal_state_counts": {0: 0, 1: 2, 2: 3, 3: 3, 4: 2},
        })
    return traj


class TestTransvolution(unittest.TestCase):
    """Tests for T mode (involution/evolution/condensation)."""

    def test_involution_includes_self_and_absorptive(self):
        """Involution score should count self + absorptive events."""
        from simulator.taps import compute_transvolution
        traj = _make_trajectory(steps=10)
        result = compute_transvolution(traj)
        # At step 3: delta_self=1, delta_absorptive=1, total_events=2
        # involution = (1+1)/2 = 1.0
        self.assertAlmostEqual(result["involution"][3], 1.0)

    def test_evolution_includes_novel_and_disintegration(self):
        """Evolution score should count novel + disintegration events."""
        from simulator.taps import compute_transvolution
        traj = _make_trajectory(steps=10)
        result = compute_transvolution(traj)
        # At step 6: delta_novel=1, delta_disint=1, total_events=2
        # evolution = (1+1)/2 = 1.0
        self.assertAlmostEqual(result["evolution"][6], 1.0)

    def test_condensation_is_product(self):
        """Condensation = involution * evolution."""
        from simulator.taps import compute_transvolution
        traj = _make_trajectory(steps=10)
        result = compute_transvolution(traj)
        for t in range(len(result["condensation"])):
            self.assertAlmostEqual(
                result["condensation"][t],
                result["involution"][t] * result["evolution"][t],
            )


class TestAnopression(unittest.TestCase):
    """Tests for A mode (pressure decomposition)."""

    def test_anopressive_sums_to_one(self):
        """Anopressive scores (expression+impression+adpression) should sum to 1."""
        from simulator.taps import compute_anopression
        traj = _make_trajectory(steps=10)
        result = compute_anopression(traj, mu=0.005)
        for t in range(len(result["expression"])):
            ano_total = (result["expression"][t]
                         + result["impression"][t]
                         + result["adpression"][t])
            if ano_total > 0:  # only when events happen
                self.assertAlmostEqual(ano_total, 1.0, places=5)

    def test_anapressive_can_exceed_one(self):
        """Anapressive total can exceed 1.0 (net entropy)."""
        from simulator.taps import compute_anopression
        # Create trajectory with high decay, low growth -> anapressive > 1
        traj = _make_trajectory(steps=10, with_events=False)
        for s in traj:
            s["total_M"] = 1000.0  # high M
            s["n_active"] = 8
            s["K_env"] = 1100.0  # close to capacity
            s["affordance_mean"] = 0.1  # low affordance = high suppression
        result = compute_anopression(traj, mu=0.05)  # high mu
        # At least some steps should have anapressive > 1
        anapressive_totals = [
            result["oppression"][t] + result["suppression"][t]
            + result["depression"][t] + result["compression"][t]
            for t in range(len(result["oppression"]))
        ]
        self.assertTrue(any(a > 1.0 for a in anapressive_totals),
                        f"Expected some anapressive > 1.0, got max={max(anapressive_totals):.4f}")


class TestPressureRatio(unittest.TestCase):
    """Tests for pressure ratio (net entropy/extropy indicator)."""

    def test_pressure_ratio_computed(self):
        """Pressure ratio should be sum of anapressive scores."""
        from simulator.taps import compute_anopression, pressure_ratio
        traj = _make_trajectory(steps=10)
        ano = compute_anopression(traj, mu=0.005)
        ratios = pressure_ratio(ano)
        self.assertEqual(len(ratios), len(traj))

    def test_high_pressure_means_entropy(self):
        """Pressure ratio > 1 indicates net entropy."""
        from simulator.taps import compute_anopression, pressure_ratio
        traj = _make_trajectory(steps=10, with_events=False)
        for s in traj:
            s["total_M"] = 1000.0
            s["K_env"] = 1100.0
            s["affordance_mean"] = 0.1
        ano = compute_anopression(traj, mu=0.05)
        ratios = pressure_ratio(ano)
        self.assertTrue(any(r > 1.0 for r in ratios))


class TestRIPDominance(unittest.TestCase):
    """Tests for RIP mode classification."""

    def test_recursion_dominant_no_events(self):
        """Steps with no metathesis events should be recursion-dominant."""
        from simulator.taps import compute_rip
        traj = _make_trajectory(steps=10, with_events=False)
        # Make total_M change so recursion_score > 0
        for i, s in enumerate(traj):
            s["total_M"] = 100.0 + i * 10
        result = compute_rip(traj)
        # All steps should be recursion-dominant (no events)
        for t in range(1, len(result["dominance"])):
            self.assertEqual(result["dominance"][t], "recursion")

    def test_praxis_dominant_with_events(self):
        """Steps with metathesis events should be praxis-dominant."""
        from simulator.taps import compute_rip
        traj = _make_trajectory(steps=10, with_events=True)
        result = compute_rip(traj)
        # Step 3 has events -> should be praxis-dominant
        self.assertEqual(result["dominance"][3], "praxis")


class TestCorrelationMatrix(unittest.TestCase):
    """Tests for correlation analysis."""

    def test_output_shape(self):
        """Correlation matrix should be square with correct dimensions."""
        from simulator.taps import compute_all_scores, correlation_matrix
        traj = _make_trajectory(steps=50)
        scores = compute_all_scores(traj, mu=0.005)
        result = correlation_matrix(scores)
        n = len(result["labels"])
        self.assertEqual(len(result["matrix"]), n)
        for row in result["matrix"]:
            self.assertEqual(len(row), n)

    def test_highly_correlated_detection(self):
        """Should detect pairs with |r| > 0.85."""
        from simulator.taps import correlation_matrix
        # Construct scores with two perfectly correlated series
        import numpy as np
        x = list(np.linspace(0, 1, 50))
        scores = {
            "mode_a": x,
            "mode_b": x,  # identical = r=1.0
            "mode_c": list(np.random.RandomState(42).rand(50)),
        }
        result = correlation_matrix(scores)
        correlated_pairs = [(a, b) for a, b, r in result["highly_correlated"]]
        self.assertIn(("mode_a", "mode_b"), correlated_pairs)


if __name__ == "__main__":
    unittest.main()
