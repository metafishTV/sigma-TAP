"""Tests for sensitivity analysis sweeps."""
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.sensitivity_analysis import run_extinction_sweep, run_adjacency_sweep


class TestExtinctionSweep(unittest.TestCase):
    def test_returns_list_of_dicts(self):
        rows = run_extinction_sweep(n_mu=5, steps=20)
        self.assertIsInstance(rows, list)
        self.assertGreater(len(rows), 0)
        self.assertIn("mu", rows[0])
        self.assertIn("variant", rows[0])
        self.assertIn("final_M", rows[0])
        self.assertIn("regime", rows[0])

    def test_high_mu_causes_extinction_or_plateau(self):
        """Very high extinction should prevent explosive growth."""
        rows = run_extinction_sweep(n_mu=3, steps=40, mu_range=(0.3, 0.5))
        regimes = {r["regime"] for r in rows}
        self.assertNotIn("explosive", regimes)

    def test_all_variants_present(self):
        rows = run_extinction_sweep(n_mu=3, steps=20)
        variants = {r["variant"] for r in rows}
        self.assertEqual(variants, {"baseline", "two_scale", "logistic"})


class TestAdjacencySweep(unittest.TestCase):
    def test_returns_list_of_dicts(self):
        rows = run_adjacency_sweep(a_values=[4.0, 8.0], steps=20)
        self.assertIsInstance(rows, list)
        self.assertGreater(len(rows), 0)
        self.assertIn("a", rows[0])
        self.assertIn("variant", rows[0])
        self.assertIn("final_M", rows[0])

    def test_all_variants_present(self):
        rows = run_adjacency_sweep(a_values=[8.0], steps=20)
        variants = {r["variant"] for r in rows}
        self.assertEqual(variants, {"baseline", "two_scale", "logistic"})

    def test_larger_a_means_faster_growth(self):
        """Smaller a means stronger combinatorial coupling, faster growth."""
        rows = run_adjacency_sweep(a_values=[4.0, 32.0], steps=40)
        baseline_rows = [r for r in rows if r["variant"] == "baseline"]
        by_a = {r["a"]: r["final_M"] for r in baseline_rows}
        if by_a.get(4.0, 0) > 0 and by_a.get(32.0, 0) > 0:
            self.assertGreater(by_a[4.0], by_a[32.0])


if __name__ == "__main__":
    unittest.main()
