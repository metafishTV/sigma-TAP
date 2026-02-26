"""Mode Transition Map — categorical state classification and counting.

CLAIM POLICY LABEL: exploratory

Classifies system state at each simulation step across six categorical axes
(rip_dominance, pressure_regime, ano_dominant, syntegration_phase,
transvolution_dir, texture_type) and builds transition-count matrices that
summarise how the system moves between modes over time.

These diagnostics are post-hoc descriptive tools; they do not alter the
simulation dynamics.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from itertools import product as _product

from simulator.taps import (
    Trajectory,
    compute_all_scores,
    compute_anopression,
    compute_rip,
    pressure_ratio,
)
from simulator.metathetic import MetatheticEnsemble


# ---------------------------------------------------------------------------
# Function 1: classify_step
# ---------------------------------------------------------------------------

def classify_step(
    all_scores: dict[str, list[float]],
    ano_scores: dict[str, list[float]],
    rip_result: dict[str, list],
    ratios: list[float],
    step: int,
) -> dict[str, str]:
    """Classify system state at one simulation step across 6 categorical axes.

    Parameters
    ----------
    all_scores : dict
        Output of ``compute_all_scores`` — flat dict of mode-score lists.
    ano_scores : dict
        Output of ``compute_anopression`` — anopressive / anapressive lists.
    rip_result : dict
        Output of ``compute_rip`` — includes ``"dominance"`` label list.
    ratios : list[float]
        Output of ``pressure_ratio`` — one float per step.
    step : int
        Index of the step to classify.

    Returns
    -------
    dict[str, str]
        Keys are the six axis names; values are categorical labels.
    """
    result: dict[str, str] = {}

    # --- rip_dominance ---
    result["rip_dominance"] = rip_result["dominance"][step]

    # --- pressure_regime ---
    r = ratios[step]
    if r > 1.2:
        result["pressure_regime"] = "entropy"
    elif r < 0.8:
        result["pressure_regime"] = "extropy"
    else:
        result["pressure_regime"] = "equilibrium"

    # --- ano_dominant (argmax of expression / impression / adpression) ---
    ano_keys = ["expression", "impression", "adpression"]
    ano_vals = [ano_scores[k][step] for k in ano_keys]
    result["ano_dominant"] = ano_keys[int(np.argmax(ano_vals))]

    # --- syntegration_phase (argmax of S modes) ---
    s_keys = ["disintegration", "preservation", "integration", "synthesis"]
    s_vals = [all_scores[k][step] for k in s_keys]
    result["syntegration_phase"] = s_keys[int(np.argmax(s_vals))]

    # --- transvolution_dir ---
    inv = all_scores["involution"][step]
    evo = all_scores["evolution"][step]
    if abs(inv - evo) < 0.1:
        result["transvolution_dir"] = "balanced"
    elif inv > evo:
        result["transvolution_dir"] = "involution"
    else:
        result["transvolution_dir"] = "evolution"

    # texture_type: proxy using pressure_regime until WI-3 (Task 4)
    # implements real dM-variance-based texture classification.
    result["texture_type"] = result["pressure_regime"]

    return result


# ---------------------------------------------------------------------------
# Function 2: build_transition_map
# ---------------------------------------------------------------------------

def build_transition_map(
    all_scores: dict[str, list[float]],
    ano_scores: dict[str, list[float]],
    rip_result: dict[str, list],
    ratios: list[float],
    trajectory: Trajectory,
) -> dict[str, dict]:
    """Build transition-count matrices for every classification axis.

    For each of the six axes produced by ``classify_step``:
    1. Classify every step.
    2. Extract the label sequence.
    3. Find unique states (sorted for deterministic ordering).
    4. Build a numpy transition count matrix: ``counts[i, j]`` = number of
       times state *i* was followed by state *j*.

    Returns
    -------
    dict[str, dict]
        Keyed by axis name.  Each value is::

            {"states": list[str], "counts": np.ndarray, "sequence": list[str]}

        Total transitions for each axis equals ``len(trajectory) - 1``.
    """
    n_steps = len(trajectory)
    if n_steps < 2:
        return {}

    # Classify every step
    classifications: list[dict[str, str]] = [
        classify_step(all_scores, ano_scores, rip_result, ratios, step=t)
        for t in range(n_steps)
    ]

    axes = list(classifications[0].keys())
    result: dict[str, dict] = {}

    for axis in axes:
        sequence = [c[axis] for c in classifications]
        states = sorted(set(sequence))
        state_idx = {s: i for i, s in enumerate(states)}
        n_states = len(states)

        counts = np.zeros((n_states, n_states), dtype=int)
        for t in range(n_steps - 1):
            i = state_idx[sequence[t]]
            j = state_idx[sequence[t + 1]]
            counts[i, j] += 1

        result[axis] = {
            "states": states,
            "counts": counts,
            "sequence": sequence,
        }

    return result


# ---------------------------------------------------------------------------
# Function 3: transition_summary
# ---------------------------------------------------------------------------

def transition_summary(transition_maps: dict[str, dict]) -> dict:
    """Extract structural features from transition maps.

    Parameters
    ----------
    transition_maps : dict
        Output of ``build_transition_map``.

    Returns
    -------
    dict with keys:
        absorbing_states : dict[str, list[str]]
            Per axis, states where self-transition > 50% of outgoing.
        common_pathways : dict[str, list[tuple[str, str, int]]]
            Per axis, top-3 off-diagonal transitions as (from, to, count).
        path_entropy : dict[str, float]
            Per axis, Shannon entropy of flattened transition distribution
            (log base 2).
    """
    absorbing_states: dict[str, list[str]] = {}
    common_pathways: dict[str, list[tuple[str, str, int]]] = {}
    path_entropy: dict[str, float] = {}

    for axis, data in transition_maps.items():
        states = data["states"]
        counts = data["counts"]
        n = len(states)

        # --- absorbing_states ---
        absorbing: list[str] = []
        for i in range(n):
            row_total = counts[i].sum()
            if row_total > 0 and counts[i, i] > 0.5 * row_total:
                absorbing.append(states[i])
        absorbing_states[axis] = absorbing

        # --- common_pathways (top-3 off-diagonal) ---
        off_diag: list[tuple[str, str, int]] = []
        for i in range(n):
            for j in range(n):
                if i != j and counts[i, j] > 0:
                    off_diag.append((states[i], states[j], int(counts[i, j])))
        off_diag.sort(key=lambda x: x[2], reverse=True)
        common_pathways[axis] = off_diag[:3]

        # --- path_entropy (Shannon, base 2, over flattened distribution) ---
        flat = counts.flatten().astype(float)
        total = flat.sum()
        if total > 0:
            probs = flat / total
            # Only include non-zero entries to avoid log(0)
            nonzero = probs[probs > 0]
            entropy = -float(np.sum(nonzero * np.log2(nonzero)))
            path_entropy[axis] = entropy
        else:
            path_entropy[axis] = 0.0

    return {
        "absorbing_states": absorbing_states,
        "common_pathways": common_pathways,
        "path_entropy": path_entropy,
    }


# ---------------------------------------------------------------------------
# Function 4: _compute_sensitivity (helper)
# ---------------------------------------------------------------------------

def _compute_sensitivity(
    grid_points: list[dict],
    mode_summaries: dict[str, dict[str, list[float]]],
    param_grid: dict[str, list[float]],
) -> dict[str, dict[str, float]]:
    """Compute normalized sensitivity range per mode per swept parameter.

    For each parameter that was swept (has >1 unique value in *param_grid*),
    group grid points by that parameter's value, compute the mean of the
    per-grid-point mode means at each value, and report::

        normalized_range = (max_mean - min_mean) / max(eps, overall_mean)

    Parameters
    ----------
    grid_points : list[dict]
        One dict per grid point with keys ``mu``, ``alpha``, ``a``.
    mode_summaries : dict
        ``{mode_name: {"mean": [float, ...], "std": [...], "final": [...]}}``
    param_grid : dict
        The original parameter grid passed to :func:`sweep_taps_modes`.

    Returns
    -------
    dict[str, dict[str, float]]
        ``{mode_name: {param_name: normalized_range}}``
    """
    eps = 1e-12

    # Identify swept parameters (those with >1 value).
    swept_params: list[str] = [
        name for name, vals in param_grid.items() if len(vals) > 1
    ]

    result: dict[str, dict[str, float]] = {}

    for mode_name, summary in mode_summaries.items():
        means = summary["mean"]
        param_sens: dict[str, float] = {}

        for param_name in swept_params:
            # Group grid-point indices by the value of this parameter.
            value_to_means: dict[float, list[float]] = {}
            for idx, gp in enumerate(grid_points):
                val = gp[param_name]
                value_to_means.setdefault(val, []).append(means[idx])

            # Mean of mode means at each parameter value.
            per_value_means = [
                np.mean(m_list) for m_list in value_to_means.values()
            ]

            overall_mean = abs(np.mean(means))
            max_pv = max(per_value_means)
            min_pv = min(per_value_means)
            normalized_range = (max_pv - min_pv) / max(eps, overall_mean)
            param_sens[param_name] = float(normalized_range)

        result[mode_name] = param_sens

    return result


# ---------------------------------------------------------------------------
# Function 5: sweep_taps_modes
# ---------------------------------------------------------------------------

def sweep_taps_modes(
    param_grid: dict[str, list],
    n_agents: int = 20,
    steps: int = 120,
    seed: int = 42,
    variant: str = "logistic",
    carrying_capacity: float = 2e5,
) -> dict:
    """Run an ensemble at each point in a parameter grid and summarise TAPS scores.

    Parameters
    ----------
    param_grid : dict
        Keys ``mu``, ``alpha``, ``a`` each mapping to a list of values.
        Missing keys default to ``[0.005]``, ``[5e-3]``, ``[3.0]``.
    n_agents : int
        Number of agents per ensemble run.
    steps : int
        Simulation steps per run.
    seed : int
        Base random seed; incremented per grid point.
    variant : str
        TAP growth variant passed to :class:`MetatheticEnsemble`.
    carrying_capacity : float
        Carrying capacity passed to :class:`MetatheticEnsemble`.

    Returns
    -------
    dict with keys:
        grid : list[dict]
            One dict per grid point with ``mu``, ``alpha``, ``a``.
        mode_summaries : dict[str, dict[str, list[float]]]
            Per mode: ``mean``, ``std``, ``final`` lists (one entry per grid point).
        sensitivity : dict
            Output of :func:`_compute_sensitivity`.
        transition_maps : list[dict]
            One transition map dict per grid point.
    """
    mu_vals = param_grid.get("mu", [0.005])
    alpha_vals = param_grid.get("alpha", [5e-3])
    a_vals = param_grid.get("a", [3.0])

    grid_points: list[dict] = []
    mode_summaries: dict[str, dict[str, list[float]]] = {}
    transition_maps: list[dict] = []

    run_idx = 0
    for mu, alpha, a in _product(mu_vals, alpha_vals, a_vals):
        gp = {"mu": mu, "alpha": alpha, "a": a}
        grid_points.append(gp)

        # Create and run ensemble.
        ensemble = MetatheticEnsemble(
            n_agents=n_agents,
            initial_M=1.0,
            alpha=alpha,
            a=a,
            mu=mu,
            variant=variant,
            carrying_capacity=carrying_capacity,
            seed=seed + run_idx,
        )
        trajectory = ensemble.run(steps)

        # Compute TAPS scores.
        all_scores = compute_all_scores(trajectory, mu=mu)
        ano_scores = compute_anopression(trajectory, mu=mu)
        rip_result = compute_rip(trajectory)
        ratios = pressure_ratio(ano_scores)

        # Per-mode summaries.
        for mode_name, scores_list in all_scores.items():
            arr = np.array(scores_list, dtype=float)
            if mode_name not in mode_summaries:
                mode_summaries[mode_name] = {"mean": [], "std": [], "final": []}
            mode_summaries[mode_name]["mean"].append(float(np.mean(arr)))
            mode_summaries[mode_name]["std"].append(float(np.std(arr)))
            mode_summaries[mode_name]["final"].append(float(arr[-1]) if len(arr) > 0 else 0.0)

        # Pressure ratio summary.
        pr_arr = np.array(ratios, dtype=float)
        pr_key = "pressure_ratio"
        if pr_key not in mode_summaries:
            mode_summaries[pr_key] = {"mean": [], "std": [], "final": []}
        mode_summaries[pr_key]["mean"].append(float(np.mean(pr_arr)))
        mode_summaries[pr_key]["std"].append(float(np.std(pr_arr)))
        mode_summaries[pr_key]["final"].append(float(pr_arr[-1]) if len(pr_arr) > 0 else 0.0)

        # Build transition map for this grid point.
        t_map = build_transition_map(all_scores, ano_scores, rip_result, ratios, trajectory)
        transition_maps.append(t_map)

        run_idx += 1

    # Compute sensitivity metrics.
    sensitivity = _compute_sensitivity(grid_points, mode_summaries, param_grid)

    return {
        "grid": grid_points,
        "mode_summaries": mode_summaries,
        "sensitivity": sensitivity,
        "transition_maps": transition_maps,
    }


# ---------------------------------------------------------------------------
# Function 6: compute_divergence
# ---------------------------------------------------------------------------

def compute_divergence(
    param_grid: dict[str, list],
    n_agents: int = 20,
    steps: int = 120,
    seed: int = 42,
    variant: str = "logistic",
    carrying_capacity: float = 2e5,
    gated_cluster: int = 2,
    ungated_cluster: int = 0,
) -> dict:
    """Compare gated vs ungated ensemble behaviour across a parameter grid.

    At each grid point runs BOTH a gated (``affordance_min_cluster=gated_cluster``)
    and ungated (``affordance_min_cluster=ungated_cluster``) ensemble with the
    SAME seed, then computes per-mode mean absolute differences and transition
    map Frobenius-norm divergences.

    Parameters
    ----------
    param_grid : dict
        Keys ``mu``, ``alpha``, ``a`` each mapping to a list of values.
        Missing keys default to ``[0.005]``, ``[5e-3]``, ``[3.0]``.
    n_agents, steps, seed, variant, carrying_capacity :
        Passed through to :class:`MetatheticEnsemble`.
    gated_cluster : int
        ``affordance_min_cluster`` for the gated run (default 2).
    ungated_cluster : int
        ``affordance_min_cluster`` for the ungated run (default 0).

    Returns
    -------
    dict with keys:
        grid : list[dict]
            One dict per grid point with ``mu``, ``alpha``, ``a``.
        mode_divergence : dict[str, list[float]]
            Per mode: list of mean-absolute-difference values (one per grid point).
        transition_divergence : dict[str, list[float]]
            Per axis: list of Frobenius norms of aligned transition-count
            matrix differences (one per grid point).
        significant_regimes : list[dict]
            Grid points where max divergence across all modes exceeds 0.1.
    """
    mu_vals = param_grid.get("mu", [0.005])
    alpha_vals = param_grid.get("alpha", [5e-3])
    a_vals = param_grid.get("a", [3.0])

    grid_points: list[dict] = []
    mode_divergence: dict[str, list[float]] = {}
    transition_divergence: dict[str, list[float]] = {}
    significant_regimes: list[dict] = []

    run_idx = 0
    for mu, alpha, a in _product(mu_vals, alpha_vals, a_vals):
        gp = {"mu": mu, "alpha": alpha, "a": a}
        grid_points.append(gp)

        shared_seed = seed + run_idx
        run_idx += 1

        # --- Run gated ensemble ---
        ens_gated = MetatheticEnsemble(
            n_agents=n_agents,
            initial_M=1.0,
            alpha=alpha,
            a=a,
            mu=mu,
            variant=variant,
            carrying_capacity=carrying_capacity,
            affordance_min_cluster=gated_cluster,
            seed=shared_seed,
        )
        traj_gated = ens_gated.run(steps)

        # --- Run ungated ensemble ---
        ens_ungated = MetatheticEnsemble(
            n_agents=n_agents,
            initial_M=1.0,
            alpha=alpha,
            a=a,
            mu=mu,
            variant=variant,
            carrying_capacity=carrying_capacity,
            affordance_min_cluster=ungated_cluster,
            seed=shared_seed,
        )
        traj_ungated = ens_ungated.run(steps)

        # --- Compute TAPS scores for both ---
        scores_g = compute_all_scores(traj_gated, mu=mu)
        scores_u = compute_all_scores(traj_ungated, mu=mu)

        # --- Per-mode mean absolute difference ---
        all_mode_names = sorted(set(scores_g.keys()) | set(scores_u.keys()))
        max_div = 0.0
        for mode_name in all_mode_names:
            arr_g = np.array(scores_g.get(mode_name, []), dtype=float)
            arr_u = np.array(scores_u.get(mode_name, []), dtype=float)
            min_len = min(len(arr_g), len(arr_u))
            if min_len > 0:
                diff = float(np.mean(np.abs(arr_g[:min_len] - arr_u[:min_len])))
            else:
                diff = 0.0

            if mode_name not in mode_divergence:
                mode_divergence[mode_name] = []
            mode_divergence[mode_name].append(diff)
            max_div = max(max_div, diff)

        # --- Transition map divergence (Frobenius norm of aligned matrices) ---
        ano_g = compute_anopression(traj_gated, mu=mu)
        ano_u = compute_anopression(traj_ungated, mu=mu)
        rip_g = compute_rip(traj_gated)
        rip_u = compute_rip(traj_ungated)
        ratios_g = pressure_ratio(ano_g)
        ratios_u = pressure_ratio(ano_u)

        tmap_g = build_transition_map(scores_g, ano_g, rip_g, ratios_g, traj_gated)
        tmap_u = build_transition_map(scores_u, ano_u, rip_u, ratios_u, traj_ungated)

        all_axes = sorted(set(tmap_g.keys()) | set(tmap_u.keys()))
        for axis in all_axes:
            data_g = tmap_g.get(axis)
            data_u = tmap_u.get(axis)

            if data_g is None or data_u is None:
                frob = 0.0
            else:
                # Build unified state list and aligned matrices
                states_g = data_g["states"]
                states_u = data_u["states"]
                unified = sorted(set(states_g) | set(states_u))
                n_unified = len(unified)
                idx_map = {s: i for i, s in enumerate(unified)}

                mat_g = np.zeros((n_unified, n_unified), dtype=float)
                mat_u = np.zeros((n_unified, n_unified), dtype=float)

                # Fill gated matrix
                for ri, rs in enumerate(states_g):
                    for ci, cs in enumerate(states_g):
                        mat_g[idx_map[rs], idx_map[cs]] = data_g["counts"][ri, ci]

                # Fill ungated matrix
                for ri, rs in enumerate(states_u):
                    for ci, cs in enumerate(states_u):
                        mat_u[idx_map[rs], idx_map[cs]] = data_u["counts"][ri, ci]

                frob = float(np.linalg.norm(mat_g - mat_u))

            if axis not in transition_divergence:
                transition_divergence[axis] = []
            transition_divergence[axis].append(frob)

        # --- Check significance ---
        if max_div > 0.1:
            significant_regimes.append(gp)

    return {
        "grid": grid_points,
        "mode_divergence": mode_divergence,
        "transition_divergence": transition_divergence,
        "significant_regimes": significant_regimes,
    }
