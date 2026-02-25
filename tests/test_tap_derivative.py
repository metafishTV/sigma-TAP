"""Tests for innovation_kernel_derivative."""
import math
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from simulator.tap import innovation_kernel_closed, innovation_kernel_derivative


class TestKernelDerivative(unittest.TestCase):
    def test_derivative_matches_numerical(self):
        """f'(M) via our function should match (f(M+h)-f(M-h))/(2h)."""
        alpha, a = 1e-3, 8.0
        for M in [5.0, 10.0, 20.0, 50.0]:
            h = 1e-6
            numerical = (
                innovation_kernel_closed(M + h, alpha, a)
                - innovation_kernel_closed(M - h, alpha, a)
            ) / (2 * h)
            analytic = innovation_kernel_derivative(M, alpha, a)
            self.assertAlmostEqual(analytic, numerical, places=4,
                msg=f"Derivative mismatch at M={M}")

    def test_derivative_positive_for_positive_M(self):
        """f'(M) > 0 for M > 1 (innovation kernel is monotonically increasing)."""
        alpha, a = 1e-4, 8.0
        for M in [2.0, 10.0, 100.0]:
            self.assertGreater(innovation_kernel_derivative(M, alpha, a), 0.0)

    def test_derivative_zero_for_small_M(self):
        """f'(M) = 0 for M <= 1."""
        self.assertEqual(innovation_kernel_derivative(0.5, 1e-3, 8.0), 0.0)
        self.assertEqual(innovation_kernel_derivative(1.0, 1e-3, 8.0), 0.0)

    def test_derivative_overflow_returns_inf(self):
        """Very large M should return inf, not crash."""
        result = innovation_kernel_derivative(10000.0, 1e-3, 2.0)
        self.assertEqual(result, float("inf"))


if __name__ == "__main__":
    unittest.main()
