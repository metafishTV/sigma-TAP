"""Tests for turbulence diagnostics (Interpretive layer)."""
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from simulator.continuous import run_continuous
from simulator.state import ModelParams
from simulator.turbulence import TurbulenceDiagnostics, compute_turbulence_diagnostics


class TestTurbulenceDiagnostics(unittest.TestCase):
    def _run_trajectory(self, alpha=1e-3, a=8.0, mu=0.02, t_end=30):
        params = ModelParams(alpha=alpha, a=a, mu=mu, tap_variant="baseline")
        return run_continuous(
            initial_M=10.0, t_span=(0, t_end), params=params,
            sigma0=1.0, gamma=0.0,
        ), params

    def test_returns_diagnostics(self):
        result, params = self._run_trajectory()
        diag = compute_turbulence_diagnostics(result, params, sigma0=1.0, gamma=0.0)
        self.assertIsInstance(diag, TurbulenceDiagnostics)
        self.assertEqual(len(diag.B_decision), len(result.t))
        self.assertEqual(len(diag.Re_prax), len(result.t))

    def test_B_decreases_as_M_grows(self):
        """Decision bandwidth should decrease as innovation rate overwhelms decision capacity."""
        result, params = self._run_trajectory(alpha=1e-2, a=4.0, mu=0.001, t_end=50)
        diag = compute_turbulence_diagnostics(result, params, sigma0=1.0, gamma=0.0)
        # Filter to points where M is meaningfully growing
        valid = result.M > 2.0
        if sum(valid) > 5:
            B_valid = diag.B_decision[valid]
            # B should trend downward (allow noise from numerics)
            self.assertLess(B_valid[-1], B_valid[0] * 1.1)

    def test_laminar_fraction_one_when_M_small(self):
        """When M stays small, everything is laminar (B > 1)."""
        # High mu keeps M small
        result, params = self._run_trajectory(alpha=1e-5, mu=0.1, t_end=10)
        diag = compute_turbulence_diagnostics(result, params, sigma0=1.0, gamma=0.0, tau_decision=10.0)
        self.assertEqual(diag.laminar_fraction, 1.0)

    def test_Re_prax_positive(self):
        """Reynolds number should be non-negative."""
        result, params = self._run_trajectory()
        diag = compute_turbulence_diagnostics(result, params, sigma0=1.0, gamma=0.0)
        self.assertTrue(all(r >= 0 for r in diag.Re_prax))


if __name__ == "__main__":
    unittest.main()
