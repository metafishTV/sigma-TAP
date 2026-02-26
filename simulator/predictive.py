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
