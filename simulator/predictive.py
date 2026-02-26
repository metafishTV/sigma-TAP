"""Predictive Orientation Diagnostic — step-ahead Markov prediction.

CLAIM POLICY LABEL: exploratory

Predicts system mode transitions using post-hoc transition matrices,
computing parallel matching rates (usual case) and surprisal values
(improbable case). Detects candidate adpressive events when surprisal
exceeds an adaptive statistical threshold, and annihilation/extinction
events when axis state diversity collapses.

This is a post-hoc diagnostic tool; it does not alter simulation dynamics.
Online/incremental mode is documented for future work.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from simulator.taps_sensitivity import build_transition_map


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class StepPrediction:
    """One prediction per axis per step."""
    step: int
    axis: str
    predicted_distribution: dict[str, float]
    top_predictions: list[tuple[str, float]]
    actual_state: str
    parallel_match: bool
    surprisal: float


@dataclass
class AdpressionEvent:
    """Flagged when surprisal exceeds adaptive threshold or state collapses."""
    step: int
    axis: str
    surprisal: float
    threshold: float
    predicted_top: str
    actual_state: str
    probability_of_actual: float
    event_type: str  # "transition_surprisal" or "state_collapse"


@dataclass
class PredictiveDiagnosticResult:
    """Full diagnostic output."""
    predictions: list[StepPrediction]
    parallel_matching_rate: dict[str, float]
    mean_surprisal: dict[str, float]
    adpression_events: list[AdpressionEvent]
    state_collapse_events: list[AdpressionEvent]
    horizon: int
    grain: str
    step_count: int
    axes_analyzed: list[str]


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def predict_step(
    current_state: str,
    transition_matrix: np.ndarray,
    states: list[str],
    horizon: int = 1,
) -> dict[str, float]:
    """Predict probability distribution over next states.

    Parameters
    ----------
    current_state : str
        Current state label (e.g. "entropy").
    transition_matrix : np.ndarray
        Row-stochastic transition matrix (n x n).
    states : list[str]
        State labels aligned with matrix indices.
    horizon : int
        Steps ahead to predict. 1 = next step, k = k steps ahead via P^k.

    Returns
    -------
    dict[str, float]
        Mapping from state label to predicted probability.
        If current_state is unknown, returns uniform distribution.
    """
    n = len(states)
    if n == 0:
        return {}

    # Find index of current state
    try:
        idx = states.index(current_state)
    except ValueError:
        # Unknown state — return uniform
        return {s: 1.0 / n for s in states}

    # Apply matrix power for multi-step look-ahead
    if horizon > 1:
        P_k = np.linalg.matrix_power(transition_matrix, horizon)
    else:
        P_k = transition_matrix

    row = P_k[idx]
    return {states[i]: float(row[i]) for i in range(n)}


def compute_surprisal(
    predicted_dist: dict[str, float],
    actual_state: str,
    max_surprisal: float = 10.0,
) -> float:
    """Compute information-theoretic surprisal for an observed transition.

    Parameters
    ----------
    predicted_dist : dict[str, float]
        Predicted probability distribution {state: probability}.
    actual_state : str
        The state that actually occurred.
    max_surprisal : float
        Cap for zero-probability events (default 10.0 bits = P < 1/1024).

    Returns
    -------
    float
        Surprisal in bits: -log2(P(actual_state)).
        Capped at max_surprisal if probability is 0 or state is unknown.
    """
    prob = predicted_dist.get(actual_state, 0.0)
    if prob <= 0.0:
        return max_surprisal
    return -math.log2(prob)


def detect_adpression(
    surprisals: list[float],
    threshold_sd: float = 2.0,
    burn_in: int = 5,
) -> list[tuple[int, float, float]]:
    """Detect candidate adpressive events via adaptive surprisal threshold.

    Uses expanding-window running mean and standard deviation. Flags steps
    where surprisal > running_mean + threshold_sd * running_sd.

    Parameters
    ----------
    surprisals : list[float]
        Surprisal values per step.
    threshold_sd : float
        Number of standard deviations above mean for flagging (default 2.0).
    burn_in : int
        Minimum steps before flagging begins (default 5).

    Returns
    -------
    list[tuple[int, float, float]]
        List of (step_index, surprisal_value, threshold_at_step).
    """
    if len(surprisals) <= burn_in:
        return []

    events: list[tuple[int, float, float]] = []

    # Welford's online algorithm — initialize with burn-in window
    count = 0
    mean = 0.0
    M2 = 0.0
    for k in range(burn_in):
        count += 1
        delta = surprisals[k] - mean
        mean += delta / count
        delta2 = surprisals[k] - mean
        M2 += delta * delta2

    for i in range(burn_in, len(surprisals)):
        # Stats from surprisals[:i] — exactly matches original expanding window
        variance = M2 / count if count > 0 else 0.0
        sd = variance ** 0.5
        threshold = mean + threshold_sd * sd
        if sd > 0 and surprisals[i] > threshold:
            events.append((i, surprisals[i], threshold))
        elif sd == 0 and surprisals[i] > mean:
            events.append((i, surprisals[i], mean))

        # Update running stats to include surprisals[i] for next iteration
        count += 1
        delta = surprisals[i] - mean
        mean += delta / count
        delta2 = surprisals[i] - mean
        M2 += delta * delta2

    return events


def detect_state_collapse(
    trajectory_states: list[str],
    axis: str,
    window: int = 10,
) -> list[AdpressionEvent]:
    """Detect annihilation/extinction events via state diversity collapse.

    Tracks per-axis state diversity using a sliding window. When diversity
    drops from > 1 to exactly 1 (and the axis was not always single-state),
    flags a potential annihilation/extinction adpressive event.

    An extinction is a *created* event --- the dynamics produced the
    annihilation; it did not happen passively.

    Parameters
    ----------
    trajectory_states : list[str]
        Sequence of state labels for one axis.
    axis : str
        Axis name for event labeling.
    window : int
        Sliding window size for diversity measurement.

    Returns
    -------
    list[AdpressionEvent]
        Collapse events with event_type="state_collapse".
    """
    n = len(trajectory_states)
    if n < window:
        return []

    # Check if axis was always single-state
    all_states = set(trajectory_states)
    if len(all_states) <= 1:
        return []

    events: list[AdpressionEvent] = []
    prev_diversity = len(set(trajectory_states[:window]))

    for i in range(window, n):
        current_window = trajectory_states[i - window + 1 : i + 1]
        current_diversity = len(set(current_window))
        # Collapse: was diverse, now single-state
        if prev_diversity > 1 and current_diversity == 1:
            events.append(AdpressionEvent(
                step=i,
                axis=axis,
                surprisal=0.0,
                threshold=0.0,
                predicted_top="(multiple)",
                actual_state=current_window[0],
                probability_of_actual=0.0,
                event_type="state_collapse",
            ))
        prev_diversity = current_diversity

    return events


# ---------------------------------------------------------------------------
# Axis groups for grain levels
# ---------------------------------------------------------------------------
COARSE_AXES = ["syntegration_phase", "rip_dominance", "ano_dominant"]
FINE_AXES = COARSE_AXES + ["pressure_regime", "transvolution_dir"]


def run_predictive_diagnostic(
    trajectory: list[dict],
    scores: dict,
    ano_scores: dict,
    rip: dict,
    ratios: list[float],
    horizon: int = 1,
    grain: str = "coarse",
    top_k: int = 3,
    threshold_sd: float = 2.0,
) -> PredictiveDiagnosticResult:
    """Run full predictive orientation diagnostic on a simulation trajectory.

    Parameters
    ----------
    trajectory : list[dict]
        Simulation trajectory (list of step dicts).
    scores : dict
        Output of compute_all_scores().
    ano_scores : dict
        Output of compute_anopression().
    rip : dict
        Output of compute_rip().
    ratios : list[float]
        Output of pressure_ratio().
    horizon : int
        Steps ahead for prediction (1 = next step, k = k steps via P^k).
    grain : str
        "coarse" (3 axes) or "fine" (5 axes).
    top_k : int
        Number of top predictions to record per step per axis.
    threshold_sd : float
        Standard deviations above mean for adpression detection.

    Returns
    -------
    PredictiveDiagnosticResult
    """
    # Build transition maps
    t_maps = build_transition_map(scores, ano_scores, rip, ratios, trajectory)

    # Select axes
    axes = COARSE_AXES if grain == "coarse" else FINE_AXES
    axes = [a for a in axes if a in t_maps]

    # Build row-stochastic matrices
    stochastic: dict[str, tuple[np.ndarray, list[str]]] = {}
    for axis in axes:
        data = t_maps[axis]
        states = data["states"]
        counts = data["counts"].astype(float).copy()
        n = len(states)
        row_sums = counts.sum(axis=1)
        for i in range(n):
            if row_sums[i] > 0:
                counts[i] /= row_sums[i]
            else:
                counts[i, i] = 1.0
        stochastic[axis] = (counts, states)

    # Walk through trajectory step by step
    n_steps = len(trajectory)
    predictions: list[StepPrediction] = []
    surprisals_by_axis: dict[str, list[float]] = {a: [] for a in axes}

    for step_idx in range(1, n_steps):
        for axis in axes:
            P, states = stochastic[axis]
            sequence = t_maps[axis]["sequence"]

            prev_state = sequence[step_idx - 1]
            actual_state = sequence[step_idx]

            # Predict
            pred_dist = predict_step(prev_state, P, states, horizon=horizon)

            # Top-k predictions
            sorted_preds = sorted(pred_dist.items(), key=lambda x: -x[1])
            top_preds = sorted_preds[:top_k]

            # Parallel match
            parallel_match = (top_preds[0][0] == actual_state) if top_preds else False

            # Surprisal
            surp = compute_surprisal(pred_dist, actual_state)
            surprisals_by_axis[axis].append(surp)

            predictions.append(StepPrediction(
                step=step_idx,
                axis=axis,
                predicted_distribution=pred_dist,
                top_predictions=top_preds,
                actual_state=actual_state,
                parallel_match=parallel_match,
                surprisal=surp,
            ))

    # Detect adpression events
    adpression_events: list[AdpressionEvent] = []
    for axis in axes:
        surps = surprisals_by_axis[axis]
        raw_events = detect_adpression(surps, threshold_sd=threshold_sd)
        for (sidx, sval, thresh) in raw_events:
            step_num = sidx + 1  # surprisals list starts at step 1
            # Find the prediction for this step
            pred_dist = {}
            actual = ""
            for p in predictions:
                if p.step == step_num and p.axis == axis:
                    pred_dist = p.predicted_distribution
                    actual = p.actual_state
                    break
            top_pred = max(pred_dist, key=pred_dist.get) if pred_dist else ""
            prob_actual = pred_dist.get(actual, 0.0)
            adpression_events.append(AdpressionEvent(
                step=step_num,
                axis=axis,
                surprisal=sval,
                threshold=thresh,
                predicted_top=top_pred,
                actual_state=actual,
                probability_of_actual=prob_actual,
                event_type="transition_surprisal",
            ))

    # Detect state collapse events
    state_collapse_events: list[AdpressionEvent] = []
    for axis in axes:
        sequence = t_maps[axis]["sequence"]
        collapses = detect_state_collapse(sequence, axis)
        state_collapse_events.extend(collapses)

    # Aggregate statistics
    parallel_matching_rate: dict[str, float] = {}
    mean_surprisal: dict[str, float] = {}
    for axis in axes:
        axis_preds = [p for p in predictions if p.axis == axis]
        if axis_preds:
            matches = sum(1 for p in axis_preds if p.parallel_match)
            parallel_matching_rate[axis] = matches / len(axis_preds)
            mean_surprisal[axis] = (
                sum(p.surprisal for p in axis_preds) / len(axis_preds)
            )
        else:
            parallel_matching_rate[axis] = 0.0
            mean_surprisal[axis] = 0.0

    return PredictiveDiagnosticResult(
        predictions=predictions,
        parallel_matching_rate=parallel_matching_rate,
        mean_surprisal=mean_surprisal,
        adpression_events=adpression_events,
        state_collapse_events=state_collapse_events,
        horizon=horizon,
        grain=grain,
        step_count=n_steps - 1,
        axes_analyzed=axes,
    )
