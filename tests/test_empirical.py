"""Tests for empirical validation metrics.

CLAIM POLICY LABEL: exploratory
"""
from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
import numpy as np

from simulator.empirical import (
    YounRatioResult,
    TaalbiLinearityResult,
    HeapsLawResult,
    PowerLawResult,
    EmpiricalValidationResult,
    youn_ratio,
    taalbi_linearity,
    heaps_exponent,
    power_law_fit,
)


# ===================================================================
# youn_ratio tests
# ===================================================================

class TestYounRatio:
    def test_known_ratio(self):
        """60 novel + 40 absorptive -> fraction ~0.6, deviation ~0.0."""
        result = youn_ratio([0, 30, 60], [0, 20, 40])
        assert result.exploration_count == 60
        assert result.exploitation_count == 40
        assert result.exploration_fraction == pytest.approx(0.6, abs=1e-9)
        assert result.deviation == pytest.approx(0.0, abs=1e-9)
        assert result.target == 0.6

    def test_fifty_fifty(self):
        """Equal counts -> fraction ~0.5, deviation ~0.1."""
        result = youn_ratio([0, 25, 50], [0, 25, 50])
        assert result.exploration_fraction == pytest.approx(0.5, abs=1e-9)
        assert result.deviation == pytest.approx(0.1, abs=1e-9)

    def test_all_novel(self):
        """All novel -> fraction ~1.0."""
        result = youn_ratio([0, 50, 100], [0, 0, 0])
        assert result.exploration_fraction == pytest.approx(1.0, abs=1e-9)

    def test_no_events(self):
        """All zeros -> NaN fraction."""
        result = youn_ratio([0, 0, 0], [0, 0, 0])
        assert math.isnan(result.exploration_fraction)
        assert math.isnan(result.deviation)

    def test_trajectory_ratios_length(self):
        """Trajectory ratios list has correct length."""
        novel = [0, 10, 20, 30]
        absorptive = [0, 5, 10, 15]
        result = youn_ratio(novel, absorptive)
        assert len(result.trajectory_ratios) == len(novel)


# ===================================================================
# taalbi_linearity tests
# ===================================================================

class TestTaalbiLinearity:
    def test_linear_growth(self):
        """Exponential k (k = 10 * 1.1^t) -> slope ~1.0, R^2 > 0.8."""
        k_total = [int(10 * 1.1 ** t) for t in range(50)]
        result = taalbi_linearity(k_total)
        assert result.slope == pytest.approx(1.0, abs=0.3)
        assert result.r_squared > 0.8
        assert result.target_slope == 1.0

    def test_quadratic_growth(self):
        """k = t^2 -> slope ~0.5."""
        k_total = [t * t for t in range(2, 52)]
        result = taalbi_linearity(k_total)
        # For k = t^2: dk/dt = 2t, k = t^2, so log(dk) ~ log(2) + 0.5*log(k)
        assert result.slope == pytest.approx(0.5, abs=0.15)

    def test_constant_k(self):
        """k never changes -> NaN slope, n_points == 0."""
        k_total = [100] * 20
        result = taalbi_linearity(k_total)
        assert math.isnan(result.slope)
        assert result.n_points == 0


# ===================================================================
# heaps_exponent tests
# ===================================================================

class TestHeapsExponent:
    def test_sqrt_relationship(self):
        """k = i^2, D = i -> exponent ~0.5, is_sublinear=True, R^2 > 0.99."""
        i_vals = list(range(1, 101))
        k_total = [i * i for i in i_vals]
        D_total = i_vals
        result = heaps_exponent(k_total, D_total)
        assert result.exponent == pytest.approx(0.5, abs=0.01)
        assert result.is_sublinear is True
        assert result.r_squared > 0.99

    def test_linear_relationship(self):
        """D = k -> exponent ~1.0, is_sublinear=False."""
        k_total = list(range(1, 101))
        D_total = list(range(1, 101))
        result = heaps_exponent(k_total, D_total)
        assert result.exponent == pytest.approx(1.0, abs=0.01)
        assert result.is_sublinear is False

    def test_single_point(self):
        """Single data point -> NaN exponent."""
        result = heaps_exponent([5], [3])
        assert math.isnan(result.exponent)
        assert result.n_points == 1


# ===================================================================
# power_law_fit tests
# ===================================================================

class TestPowerLawFit:
    def test_zipf_distribution(self):
        """Zipf with alpha=2.0 via inverse CDF -> exponent within 0.3 of 2.0."""
        rng = np.random.default_rng(42)
        alpha_true = 2.0
        k_min = 1
        u = rng.uniform(0, 1, size=500)
        k_samples = (k_min * u ** (-1.0 / (alpha_true - 1.0))).astype(int)
        k_samples = np.maximum(k_samples, k_min)  # ensure >= k_min

        result = power_law_fit(k_samples.tolist(), k_min=k_min)
        assert result.exponent == pytest.approx(2.0, abs=0.3)
        assert result.n_agents == 500
        assert result.target_exponent == 2.0

    def test_uniform_distribution(self):
        """Random integers 1-100 -> just check n_agents correct."""
        rng = np.random.default_rng(123)
        data = rng.integers(1, 101, size=200).tolist()
        result = power_law_fit(data, k_min=1)
        assert result.n_agents == 200

    def test_all_same_at_kmin(self):
        """All values = k_min -> log_sum = 0 -> NaN exponent."""
        result = power_law_fit([5, 5, 5, 5, 5], k_min=5)
        assert math.isnan(result.exponent)
        assert result.n_agents == 5

    def test_single_agent(self):
        """Single value -> NaN (n < 2)."""
        result = power_law_fit([10], k_min=1)
        assert math.isnan(result.exponent)
        assert result.n_agents == 1


# ===================================================================
# Integration tests
# ===================================================================

class TestIntegration:
    """Integration tests with actual simulation data."""

    def test_full_validation_from_simulation(self):
        """Run a short simulation and validate all 4 metrics produce results."""
        from simulator.metathetic import MetatheticEnsemble
        ens = MetatheticEnsemble(
            n_agents=10, initial_M=10.0,
            alpha=5e-3, a=3.0, mu=0.005,
            variant="logistic", carrying_capacity=2e5, seed=42,
        )
        trajectory = ens.run(steps=100)

        n_novel = [s["n_novel_cross"] for s in trajectory]
        n_absorptive = [s["n_absorptive_cross"] for s in trajectory]
        k_total_list = [s["k_total"] for s in trajectory]
        D_total_list = [s["D_total"] for s in trajectory]
        agent_k_list = trajectory[-1]["agent_k_list"]

        yr = youn_ratio(n_novel, n_absorptive)
        assert 0.0 <= yr.exploration_fraction <= 1.0 or math.isnan(yr.exploration_fraction)

        tl = taalbi_linearity(k_total_list)
        assert isinstance(tl.slope, float)

        he = heaps_exponent(k_total_list, D_total_list)
        assert isinstance(he.exponent, float)

        pl = power_law_fit(agent_k_list)
        assert isinstance(pl.exponent, float)

    def test_status_classification(self):
        """MATCH/CLOSE/DIVERGENT thresholds."""
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts")))
        from empirical_validation import classify_status
        assert classify_status(0.05, 0.10) == "MATCH"
        assert classify_status(0.15, 0.10) == "CLOSE"
        assert classify_status(0.30, 0.10) == "DIVERGENT"
