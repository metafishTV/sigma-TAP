"""Tests for continuous-time ODE solver."""
import math
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from simulator.continuous import ContinuousResult, run_continuous
from simulator.simulate import run_sigma_tap
from simulator.state import ModelParams


class TestContinuousSolver(unittest.TestCase):
    def _make_params(self, variant="baseline"):
        return ModelParams(
            alpha=1e-3, a=8.0, mu=0.02,
            beta=0.0, eta=0.0,
            tap_variant=variant,
            alpha1=1e-2 if variant == "two_scale" else 0.0,
            carrying_capacity=1e5 if variant == "logistic" else None,
        )

    def test_returns_continuous_result(self):
        params = self._make_params()
        result = run_continuous(initial_M=10.0, t_span=(0, 5), params=params)
        self.assertIsInstance(result, ContinuousResult)
        self.assertGreater(len(result.t), 0)
        self.assertEqual(len(result.t), len(result.M))
        self.assertEqual(len(result.t), len(result.Xi))

    def test_baseline_matches_discrete_qualitatively(self):
        """Continuous and discrete should agree in trend for baseline variant."""
        params = self._make_params("baseline")
        # Discrete: 10 steps
        discrete = run_sigma_tap(
            initial_M=10.0, steps=10, params=params,
            sigma0=1.0, gamma=0.0, append_terminal_state=False,
        )
        m_discrete_final = discrete[-1]["M_t1"]
        # Continuous: same time span
        result = run_continuous(
            initial_M=10.0, t_span=(0, 10), params=params,
            sigma0=1.0, gamma=0.0,
            t_eval=np.array([10.0]),
        )
        m_continuous_final = result.M[-1]
        # Should be in the same order of magnitude
        if m_discrete_final > 0 and m_continuous_final > 0:
            ratio = m_continuous_final / m_discrete_final
            self.assertGreater(ratio, 0.1)
            self.assertLess(ratio, 10.0)

    def test_logistic_variant_bounded(self):
        """Logistic variant should not exceed carrying capacity."""
        params = self._make_params("logistic")
        result = run_continuous(
            initial_M=10.0, t_span=(0, 200), params=params,
            sigma0=1.0, gamma=0.0,
        )
        self.assertTrue(all(m <= params.carrying_capacity * 1.01 for m in result.M))

    def test_two_scale_variant_runs(self):
        """Two-scale variant should run without error."""
        params = self._make_params("two_scale")
        result = run_continuous(
            initial_M=10.0, t_span=(0, 20), params=params,
            sigma0=1.0, gamma=0.0,
        )
        self.assertGreater(len(result.t), 0)

    def test_overflow_terminates(self):
        """Moderate growth should trigger overflow termination at low cap."""
        params = ModelParams(alpha=1e-2, a=8.0, mu=0.0, tap_variant="baseline")
        result = run_continuous(
            initial_M=10.0, t_span=(0, 200), params=params,
            sigma0=1.0, gamma=0.0, m_cap=50.0,
        )
        self.assertTrue(result.terminated_by_overflow)
        self.assertIsNotNone(result.blowup_time)

    def test_sigma_feedback_accelerates(self):
        """Positive gamma should produce faster growth than gamma=0."""
        params = self._make_params("baseline")
        params.beta = 0.05
        r0 = run_continuous(
            initial_M=10.0, t_span=(0, 20), params=params,
            sigma0=1.0, gamma=0.0,
        )
        r1 = run_continuous(
            initial_M=10.0, t_span=(0, 20), params=params,
            sigma0=1.0, gamma=0.5,
        )
        # With sigma feedback, M should be >= M without it
        m0_final = r0.M[-1]
        m1_final = r1.M[-1]
        self.assertGreaterEqual(m1_final, m0_final * 0.99)


if __name__ == "__main__":
    unittest.main()
