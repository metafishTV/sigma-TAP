"""Smoke tests for publication figure generation."""
import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.generate_figures import (
    fig_trajectory_variants,
    fig_extinction_sensitivity,
    fig_adjacency_sensitivity,
    fig_turbulence_bandwidth,
    COLORS,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "figures"


class TestFigureHelpers(unittest.TestCase):
    def test_colors_defined(self):
        self.assertIn("baseline", COLORS)
        self.assertIn("two_scale", COLORS)
        self.assertIn("logistic", COLORS)

    def test_trajectory_variants_creates_file(self):
        path = OUT / "_test_trajectory.png"
        fig_trajectory_variants(str(path))
        self.assertTrue(path.exists())
        self.assertGreater(path.stat().st_size, 0)
        path.unlink()

    def test_extinction_sensitivity_creates_file(self):
        path = OUT / "_test_extinction.png"
        # Generate minimal data inline
        from scripts.sensitivity_analysis import run_extinction_sweep
        rows = run_extinction_sweep(n_mu=5, steps=20)
        fig_extinction_sensitivity(rows, str(path))
        self.assertTrue(path.exists())
        self.assertGreater(path.stat().st_size, 0)
        path.unlink()

    def test_adjacency_sensitivity_creates_file(self):
        path = OUT / "_test_adjacency.png"
        from scripts.sensitivity_analysis import run_adjacency_sweep
        rows = run_adjacency_sweep(a_values=[4.0, 8.0], steps=20)
        fig_adjacency_sensitivity(rows, str(path))
        self.assertTrue(path.exists())
        self.assertGreater(path.stat().st_size, 0)
        path.unlink()

    def test_turbulence_bandwidth_creates_file(self):
        path = OUT / "_test_turbulence.png"
        fig_turbulence_bandwidth(str(path))
        self.assertTrue(path.exists())
        self.assertGreater(path.stat().st_size, 0)
        path.unlink()


if __name__ == "__main__":
    unittest.main()
