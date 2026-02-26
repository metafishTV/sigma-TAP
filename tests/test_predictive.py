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


if __name__ == "__main__":
    unittest.main()
