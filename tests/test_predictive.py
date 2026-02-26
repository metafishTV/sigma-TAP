"""Tests for predictive orientation diagnostic."""
import os
import sys
import unittest
import math

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


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


if __name__ == "__main__":
    unittest.main()
