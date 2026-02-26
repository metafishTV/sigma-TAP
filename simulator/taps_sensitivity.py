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

from simulator.taps import Trajectory


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
