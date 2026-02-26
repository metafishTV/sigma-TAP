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
from typing import Any

import numpy as np


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
    for i in range(burn_in, len(surprisals)):
        window = surprisals[:i]
        mean = sum(window) / len(window)
        variance = sum((x - mean) ** 2 for x in window) / len(window)
        sd = variance ** 0.5
        threshold = mean + threshold_sd * sd
        if sd > 0 and surprisals[i] > threshold:
            events.append((i, surprisals[i], threshold))
        elif sd == 0 and surprisals[i] > mean:
            # Zero variance — any deviation above mean is notable
            events.append((i, surprisals[i], mean))
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
