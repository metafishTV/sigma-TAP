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


class TestActionModalities(unittest.TestCase):
    """Tests for consumption/consummation decomposition of action."""

    def test_praxis_returns_all_keys(self):
        """compute_praxis should return all 7 keys including modalities."""
        from simulator.taps import compute_praxis
        traj = _make_trajectory(steps=10)
        result = compute_praxis(traj)
        expected = {"projection", "reflection", "action",
                    "consumption", "consummation", "pure_action",
                    "action_balance"}
        self.assertEqual(set(result.keys()), expected)

    def test_all_scores_includes_modalities(self):
        """compute_all_scores should propagate modality keys."""
        from simulator.taps import compute_all_scores
        traj = _make_trajectory(steps=10)
        scores = compute_all_scores(traj, mu=0.005)
        for key in ("consumption", "consummation", "pure_action", "action_balance"):
            self.assertIn(key, scores, f"Missing key: {key}")
            self.assertEqual(len(scores[key]), 10)

    def test_no_events_yields_zero_modalities(self):
        """Steps with no events should have zero consumption/consummation."""
        from simulator.taps import compute_praxis
        traj = _make_trajectory(steps=10, with_events=False)
        result = compute_praxis(traj)
        for t in range(10):
            self.assertAlmostEqual(result["consumption"][t], 0.0)
            self.assertAlmostEqual(result["consummation"][t], 0.0)
            self.assertAlmostEqual(result["pure_action"][t], 0.0)

    def test_no_events_balance_is_neutral(self):
        """action_balance should be 0.5 (neutral) when no events occur."""
        from simulator.taps import compute_praxis
        traj = _make_trajectory(steps=10, with_events=False)
        result = compute_praxis(traj)
        for t in range(10):
            self.assertAlmostEqual(result["action_balance"][t], 0.5)

    def test_pure_action_is_min(self):
        """pure_action = min(consumption, consummation) at every step."""
        from simulator.taps import compute_praxis
        traj = _make_trajectory(steps=10)
        result = compute_praxis(traj)
        for t in range(10):
            self.assertAlmostEqual(
                result["pure_action"][t],
                min(result["consumption"][t], result["consummation"][t]),
            )

    def test_absorptive_step_is_consumptive(self):
        """A step with only absorptive events should be consumption-dominant.

        At step 3: delta_self=1, delta_absorptive=1.
        self weights:       (0.45, 0.55)
        absorptive weights: (0.60, 0.40)
        consumption  = (1*0.45 + 1*0.60) / 2 = 0.525
        consummation = (1*0.55 + 1*0.40) / 2 = 0.475
        action_balance = 0.475 / (0.525 + 0.475) = 0.475 < 0.5
        """
        from simulator.taps import compute_praxis
        traj = _make_trajectory(steps=10)
        result = compute_praxis(traj)
        # Step 3 has self + absorptive events -> consumption > consummation
        self.assertGreater(result["consumption"][3], result["consummation"][3])
        self.assertLess(result["action_balance"][3], 0.5)

    def test_novel_step_is_consummative(self):
        """A step with only novel+disintegration events should be
        consummation-dominant.

        At step 6: delta_novel=1, delta_disint=1.
        novel weights:  (0.35, 0.65)
        disint weights: (0.40, 0.60)
        consumption  = (1*0.35 + 1*0.40) / 2 = 0.375
        consummation = (1*0.65 + 1*0.60) / 2 = 0.625
        action_balance = 0.625 / (0.375 + 0.625) = 0.625 > 0.5
        """
        from simulator.taps import compute_praxis
        traj = _make_trajectory(steps=10)
        result = compute_praxis(traj)
        # Step 6 has novel + disintegration events -> consummation > consumption
        self.assertGreater(result["consummation"][6], result["consumption"][6])
        self.assertGreater(result["action_balance"][6], 0.5)

    def test_action_balance_range(self):
        """action_balance should always be in [0, 1]."""
        from simulator.taps import compute_praxis
        traj = _make_trajectory(steps=10)
        result = compute_praxis(traj)
        for t in range(10):
            self.assertGreaterEqual(result["action_balance"][t], 0.0)
            self.assertLessEqual(result["action_balance"][t], 1.0)

    def test_consumption_plus_consummation_equals_one(self):
        """Normalized consumption + consummation should sum to ~1.0 when
        events occur (since weights are (w, 1-w) pairs that average to 1)."""
        from simulator.taps import compute_praxis
        traj = _make_trajectory(steps=10)
        result = compute_praxis(traj)
        for t in range(10):
            if result["action"][t] > 0:
                total = result["consumption"][t] + result["consummation"][t]
                self.assertAlmostEqual(total, 1.0, places=10,
                                       msg=f"Step {t}: cons+consu={total}")

    def test_weights_constant_reference(self):
        """ACTION_MODALITY_WEIGHTS should contain all 5 event types."""
        from simulator.taps import ACTION_MODALITY_WEIGHTS
        expected_keys = {"self", "absorptive", "novel", "disintegration", "env"}
        self.assertEqual(set(ACTION_MODALITY_WEIGHTS.keys()), expected_keys)
        # Each weight pair should sum to 1.0 (obverse + reverse = unity)
        for key, (w_cons, w_consu) in ACTION_MODALITY_WEIGHTS.items():
            self.assertAlmostEqual(w_cons + w_consu, 1.0,
                                   msg=f"Weights for {key} don't sum to 1.0")


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
