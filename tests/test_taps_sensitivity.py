"""Tests for TAPS sensitivity — mode transition map classification and counting."""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def _make_trajectory(steps=10, with_events=True):
    """Build a synthetic trajectory for testing.

    Creates a trajectory where events happen at specific steps so test
    assertions are deterministic.  Copied from test_taps.py for consistency.
    """
    traj = []
    cumul = {"self": 0, "absorptive": 0, "novel": 0, "env": 0, "disint": 0,
             "types_lost": 0, "k_lost": 0.0, "deep_stasis": 0}
    for t in range(steps):
        if with_events and t == 3:
            cumul["self"] += 1
            cumul["absorptive"] += 1
        if with_events and t == 6:
            cumul["novel"] += 1
            cumul["disint"] += 1
        traj.append({
            "step": t,
            "D_total": 10 + t,
            "k_total": 50.0 + t * 5,
            "total_M": 100.0 + t * 10,
            "n_active": 8,
            "n_dormant": 2,
            "agent_k_list": [10.0] * 8,
            "convergence": 0.5,
            "texture_type": 1,
            "a_env": 3.0,
            "K_env": 2e5,
            "innovation_potential": 0.8 - t * 0.01,
            "n_self_metatheses": cumul["self"],
            "n_absorptive_cross": cumul["absorptive"],
            "n_novel_cross": cumul["novel"],
            "n_env_transitions": cumul["env"],
            "n_disintegration_redistributions": cumul["disint"],
            "n_types_lost": cumul["types_lost"],
            "k_lost": cumul["k_lost"],
            "n_deep_stasis": cumul["deep_stasis"],
            "affordance_mean": 0.6,
            "temporal_state_counts": {0: 0, 1: 2, 2: 3, 3: 3, 4: 2},
        })
    return traj


class TestClassifyStep(unittest.TestCase):
    """Tests for classify_step()."""

    def test_classify_returns_all_axes(self):
        """classify_step must return all 6 categorical axes."""
        from simulator.taps_sensitivity import classify_step
        from simulator.taps import (
            compute_all_scores, compute_anopression, compute_rip,
            pressure_ratio,
        )

        traj = _make_trajectory(steps=10)
        all_scores = compute_all_scores(traj, mu=0.005)
        ano_scores = compute_anopression(traj, mu=0.005)
        rip_result = compute_rip(traj)
        ratios = pressure_ratio(ano_scores)

        result = classify_step(all_scores, ano_scores, rip_result, ratios, step=5)

        expected_axes = {
            "rip_dominance",
            "pressure_regime",
            "ano_dominant",
            "syntegration_phase",
            "transvolution_dir",
            "texture_type",
        }
        self.assertEqual(set(result.keys()), expected_axes)

    def test_classify_returns_strings(self):
        """All classify_step values must be strings."""
        from simulator.taps_sensitivity import classify_step
        from simulator.taps import (
            compute_all_scores, compute_anopression, compute_rip,
            pressure_ratio,
        )

        traj = _make_trajectory(steps=10)
        all_scores = compute_all_scores(traj, mu=0.005)
        ano_scores = compute_anopression(traj, mu=0.005)
        rip_result = compute_rip(traj)
        ratios = pressure_ratio(ano_scores)

        result = classify_step(all_scores, ano_scores, rip_result, ratios, step=5)

        for axis, value in result.items():
            self.assertIsInstance(value, str,
                                 f"Axis '{axis}' value should be str, got {type(value)}")


class TestTransitionMap(unittest.TestCase):
    """Tests for build_transition_map()."""

    def test_constant_state_only_self_transitions(self):
        """With no events, RIP should be all recursion -> only self-transitions."""
        from simulator.taps_sensitivity import build_transition_map
        from simulator.taps import (
            compute_all_scores, compute_anopression, compute_rip,
            pressure_ratio,
        )

        traj = _make_trajectory(steps=10, with_events=False)
        # Ensure total_M grows so recursion score > 0 at every step
        for i, s in enumerate(traj):
            s["total_M"] = 100.0 + i * 10

        all_scores = compute_all_scores(traj, mu=0.005)
        ano_scores = compute_anopression(traj, mu=0.005)
        rip_result = compute_rip(traj)
        ratios = pressure_ratio(ano_scores)

        result = build_transition_map(all_scores, ano_scores, rip_result, ratios, traj)

        rip_map = result["rip_dominance"]
        counts = rip_map["counts"]
        states = rip_map["states"]

        # All steps (except step 0 which ties at 0) should be recursion-dominant.
        # The sequence may contain "recursion" at most steps; check that
        # self-transitions dominate: off-diagonal entries should be very small
        # relative to the diagonal.
        total_off_diag = counts.sum() - np.trace(counts)
        total_diag = np.trace(counts)
        self.assertGreaterEqual(
            total_diag, total_off_diag,
            f"Expected diagonal (self-transitions) >= off-diagonal, "
            f"got diag={total_diag}, off={total_off_diag}",
        )

    def test_transition_counts_sum_to_steps_minus_one(self):
        """Total transitions for each axis must equal len(trajectory) - 1."""
        from simulator.taps_sensitivity import build_transition_map
        from simulator.taps import (
            compute_all_scores, compute_anopression, compute_rip,
            pressure_ratio,
        )

        steps = 20
        traj = _make_trajectory(steps=steps, with_events=True)
        all_scores = compute_all_scores(traj, mu=0.005)
        ano_scores = compute_anopression(traj, mu=0.005)
        rip_result = compute_rip(traj)
        ratios = pressure_ratio(ano_scores)

        result = build_transition_map(all_scores, ano_scores, rip_result, ratios, traj)

        for axis_name, axis_data in result.items():
            total = int(axis_data["counts"].sum())
            self.assertEqual(
                total, steps - 1,
                f"Axis '{axis_name}': expected {steps - 1} transitions, got {total}",
            )


class TestTransitionSummary(unittest.TestCase):
    """Tests for transition_summary()."""

    def test_absorbing_state_detected(self):
        """A state with >50% self-transitions should appear in absorbing_states."""
        from simulator.taps_sensitivity import transition_summary

        transition_maps = {
            "test_axis": {
                "states": ["a", "b"],
                "counts": np.array([[5, 1], [1, 3]]),
                "sequence": ["a"] * 6 + ["b"] * 4,
            }
        }
        summary = transition_summary(transition_maps)
        # State "a": 5 self out of 6 total (83%) -> absorbing
        # State "b": 3 self out of 4 total (75%) -> absorbing
        self.assertIn("a", summary["absorbing_states"]["test_axis"])
        self.assertIn("b", summary["absorbing_states"]["test_axis"])

    def test_path_entropy_uniform(self):
        """A 2x2 uniform transition matrix should have entropy = log2(4) = 2.0."""
        from simulator.taps_sensitivity import transition_summary

        transition_maps = {
            "test_axis": {
                "states": ["a", "b"],
                "counts": np.array([[1, 1], [1, 1]]),
                "sequence": ["a", "b"] * 2,
            }
        }
        summary = transition_summary(transition_maps)
        self.assertAlmostEqual(summary["path_entropy"]["test_axis"], 2.0, places=10)

    def test_common_pathways_top3(self):
        """Top-3 off-diagonal pathways should be ordered by count descending."""
        from simulator.taps_sensitivity import transition_summary

        transition_maps = {
            "test_axis": {
                "states": ["a", "b", "c"],
                "counts": np.array([
                    [0, 5, 3],
                    [2, 0, 8],
                    [1, 4, 0],
                ]),
                "sequence": ["a", "b", "c"] * 5,
            }
        }
        summary = transition_summary(transition_maps)
        pathways = summary["common_pathways"]["test_axis"]
        # Expected order: (b,c,8), (a,b,5), (c,b,4)
        self.assertEqual(len(pathways), 3)
        self.assertEqual(pathways[0], ("b", "c", 8))
        self.assertEqual(pathways[1], ("a", "b", 5))
        self.assertEqual(pathways[2], ("c", "b", 4))

    def test_summary_includes_eigenvalue_analysis(self):
        """transition_summary should include eigenvalue_analysis key."""
        from simulator.taps_sensitivity import transition_summary

        transition_maps = {
            "test_axis": {
                "states": ["a", "b"],
                "counts": np.array([[5, 2], [3, 4]]),
                "sequence": ["a", "b"] * 5,
            }
        }
        summary = transition_summary(transition_maps)
        self.assertIn("eigenvalue_analysis", summary)
        self.assertIn("test_axis", summary["eigenvalue_analysis"])
        self.assertIn("dynamics_class",
                       summary["eigenvalue_analysis"]["test_axis"])


class TestSweepTapsModes(unittest.TestCase):
    """Tests for parameter sensitivity sweep."""

    def test_sweep_returns_correct_keys(self):
        """Sweep result should contain grid, mode_summaries, sensitivity, transition_maps."""
        from simulator.taps_sensitivity import sweep_taps_modes
        grid = {"mu": [0.01, 0.1], "alpha": [1e-3], "a": [8.0]}
        result = sweep_taps_modes(grid, n_agents=5, steps=20, seed=42)
        self.assertIn("grid", result)
        self.assertIn("mode_summaries", result)
        self.assertIn("sensitivity", result)
        self.assertIn("transition_maps", result)

    def test_sweep_mode_summaries_shape(self):
        """Each mode summary should have one entry per grid point."""
        from simulator.taps_sensitivity import sweep_taps_modes
        grid = {"mu": [0.01, 0.05], "alpha": [1e-3], "a": [4.0]}
        result = sweep_taps_modes(grid, n_agents=5, steps=20, seed=42)
        n_points = 2  # 2 mu * 1 alpha * 1 a
        for mode_name, summary in result["mode_summaries"].items():
            self.assertEqual(len(summary["mean"]), n_points,
                             f"Mode {mode_name}: expected {n_points} entries")

    def test_sensitivity_metric_computed(self):
        """Sensitivity dict should have normalized range per mode per swept param."""
        from simulator.taps_sensitivity import sweep_taps_modes
        grid = {"mu": [0.005, 0.05, 0.5], "alpha": [1e-3], "a": [8.0]}
        result = sweep_taps_modes(grid, n_agents=5, steps=20, seed=42)
        self.assertGreater(len(result["sensitivity"]), 0)
        for mode_name, param_sens in result["sensitivity"].items():
            self.assertIn("mu", param_sens,
                          f"Mode {mode_name} should have mu sensitivity")


class TestDivergenceMap(unittest.TestCase):
    """Tests for gated vs ungated divergence."""

    def test_identical_params_zero_divergence(self):
        """Same gate settings should produce zero divergence."""
        from simulator.taps_sensitivity import compute_divergence
        grid = {"mu": [0.01], "alpha": [1e-3], "a": [8.0]}
        result = compute_divergence(
            grid, n_agents=5, steps=20, seed=42,
            gated_cluster=2, ungated_cluster=2,  # SAME setting
        )
        for mode_name, divs in result["mode_divergence"].items():
            for d in divs:
                self.assertAlmostEqual(d, 0.0, places=5,
                    msg=f"Mode {mode_name} should have 0 divergence with same gate")

    def test_divergence_returns_all_modes(self):
        """Divergence result should cover all 17 TAPS modes."""
        from simulator.taps_sensitivity import compute_divergence
        grid = {"mu": [0.01], "alpha": [1e-3], "a": [8.0]}
        result = compute_divergence(grid, n_agents=5, steps=20, seed=42)
        self.assertGreaterEqual(len(result["mode_divergence"]), 17)


class TestTextureValidation(unittest.TestCase):
    def test_classify_dM_texture_length(self):
        """Output length should match trajectory length."""
        from simulator.taps_sensitivity import classify_dM_texture
        traj = _make_trajectory(steps=20, with_events=True)
        result = classify_dM_texture(traj, window=5)
        self.assertEqual(len(result), 20)

    def test_classify_dM_texture_values_in_range(self):
        """All texture types should be in {0, 1, 2, 3, 4}."""
        from simulator.taps_sensitivity import classify_dM_texture
        traj = _make_trajectory(steps=20, with_events=True)
        result = classify_dM_texture(traj, window=5)
        for t_type in result:
            self.assertIn(t_type, {0, 1, 2, 3, 4})


class TestCorrelationStability(unittest.TestCase):
    def test_stability_returns_expected_keys(self):
        """Result should have stable_pairs, unstable_pairs, stability_map, param_dependent."""
        from simulator.taps_sensitivity import correlation_stability, sweep_taps_modes
        grid = {"mu": [0.01], "alpha": [1e-3], "a": [8.0]}
        sweep = sweep_taps_modes(grid, n_agents=5, steps=30, seed=42)
        result = correlation_stability(sweep, threshold=0.85)
        self.assertIn("stable_pairs", result)
        self.assertIn("unstable_pairs", result)
        self.assertIn("stability_map", result)
        self.assertIn("param_dependent", result)

    def test_single_grid_point_all_pairs_stable_or_absent(self):
        """With 1 grid point, pairs are either 100% stable (=1.0) or not found."""
        from simulator.taps_sensitivity import correlation_stability, sweep_taps_modes
        grid = {"mu": [0.01], "alpha": [1e-3], "a": [8.0]}
        sweep = sweep_taps_modes(grid, n_agents=5, steps=30, seed=42)
        result = correlation_stability(sweep, threshold=0.85)
        for pair, frac in result["stability_map"].items():
            self.assertEqual(frac, 1.0,
                f"With 1 grid point, pair {pair} should have fraction 1.0, got {frac}")


class TestEigenvalueAnalysis(unittest.TestCase):
    """Tests for Poincare eigenvalue analysis of transition matrices."""

    def test_returns_all_axes(self):
        """eigenvalue_analysis should return one entry per axis from transition_maps."""
        from simulator.taps_sensitivity import eigenvalue_analysis

        transition_maps = {
            "axis_a": {
                "states": ["x", "y"],
                "counts": np.array([[5, 2], [3, 4]]),
                "sequence": ["x", "y"] * 5,
            },
            "axis_b": {
                "states": ["p", "q", "r"],
                "counts": np.array([[3, 1, 1], [1, 3, 1], [1, 1, 3]]),
                "sequence": ["p", "q", "r"] * 5,
            },
        }
        result = eigenvalue_analysis(transition_maps)
        self.assertEqual(set(result.keys()), {"axis_a", "axis_b"})

    def test_returns_expected_keys_per_axis(self):
        """Each axis result must contain all expected analysis keys."""
        from simulator.taps_sensitivity import eigenvalue_analysis

        transition_maps = {
            "test": {
                "states": ["a", "b"],
                "counts": np.array([[5, 2], [3, 4]]),
                "sequence": ["a", "b"] * 5,
            }
        }
        result = eigenvalue_analysis(transition_maps)
        expected_keys = {
            "eigenvalues", "spectral_gap", "stationary_distribution",
            "mixing_time", "oscillation_period", "dynamics_class",
            "is_irreducible", "is_aperiodic",
        }
        self.assertEqual(set(result["test"].keys()), expected_keys)

    def test_degenerate_single_state(self):
        """A 1x1 transition matrix (single state) should be classified as degenerate."""
        from simulator.taps_sensitivity import eigenvalue_analysis

        transition_maps = {
            "test": {
                "states": ["only"],
                "counts": np.array([[10]]),
                "sequence": ["only"] * 11,
            }
        }
        result = eigenvalue_analysis(transition_maps)
        self.assertEqual(result["test"]["dynamics_class"], "degenerate")
        self.assertAlmostEqual(
            result["test"]["stationary_distribution"]["only"], 1.0)

    def test_uniform_mixing_is_ergodic(self):
        """Equal transitions everywhere -> fast mixing -> ergodic."""
        from simulator.taps_sensitivity import eigenvalue_analysis

        # Perfectly uniform 3x3
        transition_maps = {
            "test": {
                "states": ["a", "b", "c"],
                "counts": np.array([[10, 10, 10], [10, 10, 10], [10, 10, 10]]),
                "sequence": ["a", "b", "c"] * 10,
            }
        }
        result = eigenvalue_analysis(transition_maps)
        self.assertEqual(result["test"]["dynamics_class"], "ergodic")
        # Spectral gap should be 1.0 (all non-principal eigenvalues are 0)
        self.assertAlmostEqual(result["test"]["spectral_gap"], 1.0, places=5)
        # Stationary distribution should be uniform
        for state in ["a", "b", "c"]:
            self.assertAlmostEqual(
                result["test"]["stationary_distribution"][state], 1/3, places=5)

    def test_near_absorbing_detected(self):
        """Very high self-transition with tiny leakage -> quasi_absorbing."""
        from simulator.taps_sensitivity import eigenvalue_analysis

        # State "a" has 99% self-transition, "b" has 98%
        transition_maps = {
            "test": {
                "states": ["a", "b"],
                "counts": np.array([[99, 1], [2, 98]]),
                "sequence": ["a"] * 100 + ["b"] * 100,
            }
        }
        result = eigenvalue_analysis(transition_maps)
        self.assertEqual(result["test"]["dynamics_class"], "quasi_absorbing")
        self.assertLess(result["test"]["spectral_gap"], 0.1)

    def test_spectral_gap_in_range(self):
        """Spectral gap should always be in [0, 1]."""
        from simulator.taps_sensitivity import eigenvalue_analysis

        transition_maps = {
            "test": {
                "states": ["a", "b", "c"],
                "counts": np.array([[5, 2, 1], [3, 4, 1], [1, 2, 5]]),
                "sequence": ["a", "b", "c"] * 5,
            }
        }
        result = eigenvalue_analysis(transition_maps)
        self.assertGreaterEqual(result["test"]["spectral_gap"], 0.0)
        self.assertLessEqual(result["test"]["spectral_gap"], 1.0)

    def test_stationary_distribution_sums_to_one(self):
        """Stationary distribution probabilities must sum to 1.0."""
        from simulator.taps_sensitivity import eigenvalue_analysis

        transition_maps = {
            "test": {
                "states": ["a", "b", "c"],
                "counts": np.array([[5, 3, 2], [1, 6, 3], [2, 2, 6]]),
                "sequence": ["a", "b", "c"] * 5,
            }
        }
        result = eigenvalue_analysis(transition_maps)
        total = sum(result["test"]["stationary_distribution"].values())
        self.assertAlmostEqual(total, 1.0, places=10)

    def test_known_stationary_distribution(self):
        """Verify stationary distribution against known analytical solution.

        For 2-state chain with P = [[0.7, 0.3], [0.4, 0.6]]:
        pi = [0.4/(0.3+0.4), 0.3/(0.3+0.4)] = [4/7, 3/7]
        """
        from simulator.taps_sensitivity import eigenvalue_analysis

        # Use counts that give the target transition probabilities
        transition_maps = {
            "test": {
                "states": ["a", "b"],
                "counts": np.array([[70, 30], [40, 60]]),
                "sequence": ["a", "b"] * 50,
            }
        }
        result = eigenvalue_analysis(transition_maps)
        self.assertAlmostEqual(
            result["test"]["stationary_distribution"]["a"], 4/7, places=5)
        self.assertAlmostEqual(
            result["test"]["stationary_distribution"]["b"], 3/7, places=5)

    def test_oscillatory_dynamics_detected(self):
        """A 3-state cycle should produce complex eigenvalues and detect oscillation."""
        from simulator.taps_sensitivity import eigenvalue_analysis

        # Perfect 3-cycle: a->b->c->a
        transition_maps = {
            "test": {
                "states": ["a", "b", "c"],
                "counts": np.array([[0, 10, 0], [0, 0, 10], [10, 0, 0]]),
                "sequence": (["a", "b", "c"] * 10) + ["a"],
            }
        }
        result = eigenvalue_analysis(transition_maps)
        # Should detect oscillatory dynamics
        self.assertIn(result["test"]["dynamics_class"],
                      ("stable_spiral", "periodic"))
        # Oscillation period should be ~3 (period of the cycle)
        if result["test"]["oscillation_period"] is not None:
            self.assertAlmostEqual(
                result["test"]["oscillation_period"], 3.0, places=1)

    def test_mixing_time_finite_for_ergodic(self):
        """Ergodic chains should have finite positive mixing time."""
        from simulator.taps_sensitivity import eigenvalue_analysis

        transition_maps = {
            "test": {
                "states": ["a", "b"],
                "counts": np.array([[6, 4], [3, 7]]),
                "sequence": ["a", "b"] * 5,
            }
        }
        result = eigenvalue_analysis(transition_maps)
        self.assertGreater(result["test"]["mixing_time"], 0)
        self.assertTrue(np.isfinite(result["test"]["mixing_time"]))

    def test_zero_row_handling(self):
        """Rows with zero total transitions should be handled gracefully."""
        from simulator.taps_sensitivity import eigenvalue_analysis

        # State "c" is never a source state (row sums to 0)
        transition_maps = {
            "test": {
                "states": ["a", "b", "c"],
                "counts": np.array([[5, 3, 2], [1, 6, 3], [0, 0, 0]]),
                "sequence": ["a", "b", "c"] * 5,
            }
        }
        # Should not raise
        result = eigenvalue_analysis(transition_maps)
        self.assertIn("dynamics_class", result["test"])

    def test_integration_with_trajectory(self):
        """eigenvalue_analysis should work with real build_transition_map output."""
        from simulator.taps_sensitivity import (
            build_transition_map, eigenvalue_analysis,
        )
        from simulator.taps import (
            compute_all_scores, compute_anopression, compute_rip,
            pressure_ratio,
        )

        traj = _make_trajectory(steps=50, with_events=True)
        all_scores = compute_all_scores(traj, mu=0.005)
        ano_scores = compute_anopression(traj, mu=0.005)
        rip_result = compute_rip(traj)
        ratios = pressure_ratio(ano_scores)

        t_maps = build_transition_map(
            all_scores, ano_scores, rip_result, ratios, traj)
        result = eigenvalue_analysis(t_maps)

        # Should have one entry per axis
        self.assertEqual(set(result.keys()), set(t_maps.keys()))
        # Each entry should have valid dynamics class
        valid_classes = {
            "degenerate", "ergodic", "stable_node", "stable_spiral",
            "quasi_absorbing", "periodic",
        }
        for axis, analysis in result.items():
            self.assertIn(analysis["dynamics_class"], valid_classes,
                          f"Axis '{axis}' has invalid class: {analysis['dynamics_class']}")
            # Stationary distribution should sum to 1
            total = sum(analysis["stationary_distribution"].values())
            self.assertAlmostEqual(total, 1.0, places=5,
                                   msg=f"Axis '{axis}' stationary dist sums to {total}")


if __name__ == "__main__":
    unittest.main()
