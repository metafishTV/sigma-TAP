"""Smoke tests for real-world fitting pipeline."""
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.fit_realworld import fit_single_variant, load_datasets, fit_null_models


class TestFitRealworld(unittest.TestCase):
    def test_load_datasets(self):
        datasets = load_datasets()
        self.assertIn("wikipedia_articles", datasets)
        self.assertIn("npm_packages", datasets)
        self.assertIn("described_species", datasets)
        for name, ds in datasets.items():
            self.assertEqual(len(ds["years"]), len(ds["counts"]))

    def test_fit_single_variant_runs(self):
        """Fit baseline variant to npm with minimal grid — should not crash."""
        datasets = load_datasets()
        ds = datasets["npm_packages"]
        result = fit_single_variant(
            years=ds["years"], counts=ds["counts"],
            variant="baseline",
            grid_size=3,  # tiny grid for speed
        )
        self.assertIn("rmse", result)
        self.assertIn("params", result)
        self.assertGreater(result["rmse"], 0)

    def test_fit_null_models_runs(self):
        """Fit null models to species data."""
        datasets = load_datasets()
        ds = datasets["described_species"]
        results = fit_null_models(ds["years"], ds["counts"])
        self.assertIn("exponential", results)
        self.assertIn("logistic_growth", results)
        self.assertIn("power_law", results)


if __name__ == "__main__":
    unittest.main()
