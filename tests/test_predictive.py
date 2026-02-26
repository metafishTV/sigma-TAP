"""Tests for predictive orientation diagnostic."""
import os
import sys
import unittest
import math

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def _make_trajectory(steps=10, with_events=True):
    """Build a synthetic trajectory for testing."""
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


class TestPredictStep(unittest.TestCase):
    """Tests for predict_step()."""

    def test_known_2x2_matrix(self):
        """Given state 'a' in a known 2x2 matrix, predict correct distribution."""
        from simulator.predictive import predict_step

        P = np.array([[0.7, 0.3], [0.4, 0.6]])
        states = ["a", "b"]
        result = predict_step("a", P, states, horizon=1)
        self.assertAlmostEqual(result["a"], 0.7, places=5)
        self.assertAlmostEqual(result["b"], 0.3, places=5)

    def test_horizon_2_uses_matrix_power(self):
        """horizon=2 should use P^2 for two-step prediction."""
        from simulator.predictive import predict_step

        P = np.array([[0.7, 0.3], [0.4, 0.6]])
        states = ["a", "b"]
        result = predict_step("a", P, states, horizon=2)
        self.assertAlmostEqual(result["a"], 0.61, places=5)
        self.assertAlmostEqual(result["b"], 0.39, places=5)

    def test_unknown_state_returns_uniform(self):
        """If current_state is not in states, return uniform distribution."""
        from simulator.predictive import predict_step

        P = np.array([[0.7, 0.3], [0.4, 0.6]])
        states = ["a", "b"]
        result = predict_step("unknown", P, states, horizon=1)
        self.assertAlmostEqual(result["a"], 0.5, places=5)
        self.assertAlmostEqual(result["b"], 0.5, places=5)

    def test_distribution_sums_to_one(self):
        """Predicted distribution must sum to 1.0."""
        from simulator.predictive import predict_step

        P = np.array([[0.5, 0.3, 0.2], [0.1, 0.6, 0.3], [0.2, 0.2, 0.6]])
        states = ["a", "b", "c"]
        result = predict_step("b", P, states, horizon=1)
        self.assertAlmostEqual(sum(result.values()), 1.0, places=10)


class TestComputeSurprisal(unittest.TestCase):
    """Tests for compute_surprisal()."""

    def test_known_probability(self):
        """P=0.5 should give exactly 1.0 bit of surprisal."""
        from simulator.predictive import compute_surprisal

        result = compute_surprisal({"a": 0.5, "b": 0.5}, "a")
        self.assertAlmostEqual(result, 1.0, places=10)

    def test_certain_event_zero_surprisal(self):
        """P=1.0 should give 0.0 bits of surprisal."""
        from simulator.predictive import compute_surprisal

        result = compute_surprisal({"a": 1.0}, "a")
        self.assertAlmostEqual(result, 0.0, places=10)

    def test_zero_probability_caps_at_max(self):
        """P=0.0 should cap at max_surprisal (default 10.0 bits)."""
        from simulator.predictive import compute_surprisal

        result = compute_surprisal({"a": 0.0, "b": 1.0}, "a")
        self.assertAlmostEqual(result, 10.0, places=10)

    def test_unknown_state_caps_at_max(self):
        """State not in distribution should cap at max_surprisal."""
        from simulator.predictive import compute_surprisal

        result = compute_surprisal({"a": 0.5, "b": 0.5}, "c")
        self.assertAlmostEqual(result, 10.0, places=10)

    def test_custom_max_surprisal(self):
        """Custom max_surprisal should be respected."""
        from simulator.predictive import compute_surprisal

        result = compute_surprisal({"a": 0.0}, "a", max_surprisal=5.0)
        self.assertAlmostEqual(result, 5.0, places=10)


class TestDetectAdpression(unittest.TestCase):
    """Tests for detect_adpression()."""

    def test_spike_detected(self):
        """A single spike in otherwise uniform surprisal should be flagged."""
        from simulator.predictive import detect_adpression

        # 20 uniform values at 1.0, then one spike at 8.0
        surprisals = [1.0] * 20 + [8.0] + [1.0] * 5
        events = detect_adpression(surprisals, threshold_sd=2.0, burn_in=5)
        # The spike at index 20 should be detected
        flagged_steps = [e[0] for e in events]
        self.assertIn(20, flagged_steps)

    def test_burn_in_excludes_early_steps(self):
        """No events should be flagged within the burn-in period."""
        from simulator.predictive import detect_adpression

        # Put a spike in burn-in period
        surprisals = [8.0] + [1.0] * 20
        events = detect_adpression(surprisals, threshold_sd=2.0, burn_in=5)
        flagged_steps = [e[0] for e in events]
        self.assertNotIn(0, flagged_steps)

    def test_uniform_surprisal_no_events(self):
        """Perfectly uniform surprisal should produce no adpression events."""
        from simulator.predictive import detect_adpression

        surprisals = [1.0] * 30
        events = detect_adpression(surprisals, threshold_sd=2.0, burn_in=5)
        self.assertEqual(len(events), 0)

    def test_empty_input(self):
        """Empty surprisal list should return empty events."""
        from simulator.predictive import detect_adpression

        events = detect_adpression([], threshold_sd=2.0, burn_in=5)
        self.assertEqual(len(events), 0)


class TestDetectStateCollapse(unittest.TestCase):
    """Tests for detect_state_collapse()."""

    def test_collapse_detected(self):
        """Axis transitioning from multi-state to single-state should be flagged."""
        from simulator.predictive import detect_state_collapse

        # 15 steps with variety, then 15 steps stuck in one state
        states = (["a", "b", "c"] * 5) + (["a"] * 15)
        events = detect_state_collapse(states, "test_axis", window=10)
        self.assertGreater(len(events), 0)
        self.assertEqual(events[0].event_type, "state_collapse")

    def test_always_single_state_no_event(self):
        """An axis that was always single-state should NOT be flagged."""
        from simulator.predictive import detect_state_collapse

        states = ["a"] * 30
        events = detect_state_collapse(states, "test_axis", window=10)
        self.assertEqual(len(events), 0)

    def test_diverse_axis_no_event(self):
        """An axis maintaining diversity should produce no collapse events."""
        from simulator.predictive import detect_state_collapse

        states = (["a", "b", "c"] * 10)
        events = detect_state_collapse(states, "test_axis", window=10)
        self.assertEqual(len(events), 0)


class TestRunPredictiveDiagnostic(unittest.TestCase):
    """Tests for run_predictive_diagnostic()."""

    def _compute_inputs(self, steps=50):
        """Helper to compute TAPS inputs from synthetic trajectory."""
        from simulator.taps import (
            compute_all_scores, compute_anopression, compute_rip,
            pressure_ratio,
        )
        traj = _make_trajectory(steps=steps, with_events=True)
        scores = compute_all_scores(traj, mu=0.005)
        ano_scores = compute_anopression(traj, mu=0.005)
        rip = compute_rip(traj)
        ratios = pressure_ratio(ano_scores)
        return traj, scores, ano_scores, rip, ratios

    def test_coarse_grain_selects_3_axes(self):
        """Coarse grain should analyze syntegration_phase, rip_dominance, ano_dominant."""
        from simulator.predictive import run_predictive_diagnostic

        traj, scores, ano_scores, rip, ratios = self._compute_inputs()
        result = run_predictive_diagnostic(
            traj, scores, ano_scores, rip, ratios, grain="coarse")
        self.assertEqual(len(result.axes_analyzed), 3)
        expected = {"syntegration_phase", "rip_dominance", "ano_dominant"}
        self.assertEqual(set(result.axes_analyzed), expected)

    def test_fine_grain_selects_5_axes(self):
        """Fine grain should analyze all 5 transition axes."""
        from simulator.predictive import run_predictive_diagnostic

        traj, scores, ano_scores, rip, ratios = self._compute_inputs()
        result = run_predictive_diagnostic(
            traj, scores, ano_scores, rip, ratios, grain="fine")
        self.assertEqual(len(result.axes_analyzed), 5)

    def test_result_has_correct_structure(self):
        """Result should have all expected fields with correct types."""
        from simulator.predictive import (
            run_predictive_diagnostic, PredictiveDiagnosticResult,
        )

        traj, scores, ano_scores, rip, ratios = self._compute_inputs()
        result = run_predictive_diagnostic(
            traj, scores, ano_scores, rip, ratios)
        self.assertIsInstance(result, PredictiveDiagnosticResult)
        self.assertIsInstance(result.predictions, list)
        self.assertIsInstance(result.parallel_matching_rate, dict)
        self.assertIsInstance(result.mean_surprisal, dict)
        self.assertIsInstance(result.adpression_events, list)
        self.assertIsInstance(result.state_collapse_events, list)

    def test_parallel_matching_rate_in_range(self):
        """Parallel matching rate should be between 0.0 and 1.0 for each axis."""
        from simulator.predictive import run_predictive_diagnostic

        traj, scores, ano_scores, rip, ratios = self._compute_inputs()
        result = run_predictive_diagnostic(
            traj, scores, ano_scores, rip, ratios)
        for axis, rate in result.parallel_matching_rate.items():
            self.assertGreaterEqual(rate, 0.0,
                f"Axis '{axis}' match rate should be >= 0")
            self.assertLessEqual(rate, 1.0,
                f"Axis '{axis}' match rate should be <= 1")

    def test_mean_surprisal_non_negative(self):
        """Mean surprisal should be >= 0 for each axis."""
        from simulator.predictive import run_predictive_diagnostic

        traj, scores, ano_scores, rip, ratios = self._compute_inputs()
        result = run_predictive_diagnostic(
            traj, scores, ano_scores, rip, ratios)
        for axis, surp in result.mean_surprisal.items():
            self.assertGreaterEqual(surp, 0.0,
                f"Axis '{axis}' mean surprisal should be >= 0")

    def test_top_k_predictions_ordered(self):
        """Top predictions should be in descending probability order."""
        from simulator.predictive import run_predictive_diagnostic

        traj, scores, ano_scores, rip, ratios = self._compute_inputs()
        result = run_predictive_diagnostic(
            traj, scores, ano_scores, rip, ratios, top_k=3)
        for pred in result.predictions:
            probs = [p for _, p in pred.top_predictions]
            for i in range(len(probs) - 1):
                self.assertGreaterEqual(probs[i], probs[i + 1],
                    f"Step {pred.step}, axis {pred.axis}: "
                    f"top predictions not in descending order")

    def test_horizon_parameter_stored(self):
        """Different horizons should be stored correctly in result."""
        from simulator.predictive import run_predictive_diagnostic

        traj, scores, ano_scores, rip, ratios = self._compute_inputs(steps=50)
        r1 = run_predictive_diagnostic(
            traj, scores, ano_scores, rip, ratios, horizon=1)
        r3 = run_predictive_diagnostic(
            traj, scores, ano_scores, rip, ratios, horizon=3)
        self.assertEqual(r1.horizon, 1)
        self.assertEqual(r3.horizon, 3)

    def test_step_count_matches_trajectory(self):
        """step_count should equal len(trajectory) - 1."""
        from simulator.predictive import run_predictive_diagnostic

        steps = 30
        traj, scores, ano_scores, rip, ratios = self._compute_inputs(steps=steps)
        result = run_predictive_diagnostic(
            traj, scores, ano_scores, rip, ratios)
        self.assertEqual(result.step_count, steps - 1)


if __name__ == "__main__":
    unittest.main()
