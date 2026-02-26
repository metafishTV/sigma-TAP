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


# ===================================================================
# Sweep scoring and ranking tests
# ===================================================================

class TestSweep:
    """Tests for parameter sweep scoring and ranking."""

    def test_composite_score_known(self):
        """Known deviations produce correct composite score."""
        from scripts.empirical_sweep import compute_sweep_point, NAN_PENALTY
        # Create mock results with known values
        yr = YounRatioResult(30, 20, 0.6, 0.6, 0.0, [0.6])
        tl = TaalbiLinearityResult(1.0, 0.5, 0.95, 1.0, 0.001, 50)
        he = HeapsLawResult(0.5, 1.0, 0.99, "< 1.0", True, 50)
        pl = PowerLawResult(2.0, 2.0, 0.05, 0.8, 20, 1)
        result = EmpiricalValidationResult(yr, tl, he, pl, {}, 100, 10)
        point = compute_sweep_point({"alpha": 0.005}, [result])
        # All deviations are 0: youn=0.0, linearity=0.0, heaps=0.0 (match), powerlaw=0.0
        assert point.composite_score == pytest.approx(0.0, abs=0.01)

    def test_composite_score_nan_penalty(self):
        """NaN metrics get NAN_PENALTY in deviation terms."""
        from scripts.empirical_sweep import compute_sweep_point, NAN_PENALTY
        yr = YounRatioResult(0, 0, float("nan"), 0.6, float("nan"), [])
        tl = TaalbiLinearityResult(float("nan"), 0, 0, 1.0, 0, 0)
        he = HeapsLawResult(float("nan"), 0, 0, "< 1.0", False, 0)
        pl = PowerLawResult(float("nan"), 2.0, 0, 0, 0, 1)
        result = EmpiricalValidationResult(yr, tl, he, pl, {}, 100, 10)
        point = compute_sweep_point({"alpha": 0.005}, [result])
        # NaN heaps -> exponent=NAN_PENALTY=1.0, which is NOT < 1.0 so
        # heaps_match=False, heaps_dev = 1.0 - 1.0 = 0.0.
        # Composite = youn(1.0) + linearity(1.0) + heaps(0.0) + powerlaw(1.0) = 3.0
        assert point.mean_youn_deviation == pytest.approx(NAN_PENALTY, abs=0.01)
        assert point.mean_linearity_deviation == pytest.approx(NAN_PENALTY, abs=0.01)
        assert point.mean_powerlaw_deviation == pytest.approx(NAN_PENALTY, abs=0.01)
        assert point.heaps_match is False
        assert point.composite_score == pytest.approx(3 * NAN_PENALTY, abs=0.01)

    def test_rank_results(self):
        """Points ranked by composite_score ascending."""
        from scripts.empirical_sweep import SweepPoint, rank_sweep_results
        p1 = SweepPoint({}, [], 0.5, 0.5, 0.5, True, 0.5, 2.0)
        p2 = SweepPoint({}, [], 0.1, 0.1, 0.3, True, 0.1, 0.6)
        p3 = SweepPoint({}, [], 0.3, 0.3, 0.4, True, 0.3, 1.3)
        ranked = rank_sweep_results([p1, p2, p3], top_n=2)
        assert len(ranked) == 2
        assert ranked[0].composite_score < ranked[1].composite_score
        assert ranked[0] is p2

    def test_sensitivity_identifies_dominant(self):
        """Sensitivity correctly identifies most-sensitive parameter."""
        from scripts.empirical_sweep import SweepPoint, compute_sensitivity
        # alpha=low has youn_dev=0.1, alpha=high has youn_dev=0.9
        # a is constant -> sensitivity=0
        points = [
            SweepPoint({"alpha": 0.001, "a": 3.0}, [], 0.1, 0.1, 0.5, True, 0.1, 0.3),
            SweepPoint({"alpha": 0.01, "a": 3.0}, [], 0.9, 0.1, 0.5, True, 0.1, 1.1),
        ]
        grid = {"alpha": [0.001, 0.01], "a": [3.0]}
        sens = compute_sensitivity(points, grid)
        assert sens["alpha"]["youn"] > sens["a"]["youn"]

    def test_sweep_small_grid(self):
        """Quick sweep with tiny grid completes without error."""
        from scripts.empirical_sweep import run_sweep, SweepResult
        result = run_sweep(
            grid={"alpha": [5e-3], "a": [3.0], "mu": [0.005], "n_agents": [10]},
            seeds=[42],
            steps=50,
            top_n=1,
        )
        assert isinstance(result, SweepResult)
        assert len(result.points) == 1
        assert len(result.best) == 1
        assert result.total_sims == 1
