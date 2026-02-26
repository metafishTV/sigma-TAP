"""TAPS/RIP diagnostic overlay for metathetic ensemble trajectories.

CLAIM POLICY LABEL: exploratory
Computes Transvolution (T), Anopression/Anapression (A), Praxis (P),
Syntegration (S), and RIP mode scores post-hoc from trajectory data.

TAPS (Transanoprasyn) and RIP (Recursively Iterative Praxis/Preservation)
are an unpublished dynamic dispositional model by the project author.
See docs/plans/2026-02-26-taps-stage1-design.md for full theoretical
framework, formulas, and literature connections.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np


# Type alias for trajectory data
Trajectory = list[dict[str, Any]]


def _event_deltas(trajectory: Trajectory) -> list[dict[str, int]]:
    """Compute per-step event deltas from cumulative snapshot counters.

    Returns a list of dicts, one per step, with keys: self, absorptive,
    novel, env, disintegration. Step 0 uses the raw cumulative values.
    """
    keys = [
        ("n_self_metatheses", "self"),
        ("n_absorptive_cross", "absorptive"),
        ("n_novel_cross", "novel"),
        ("n_env_transitions", "env"),
        ("n_disintegration_redistributions", "disintegration"),
    ]
    deltas = []
    for i, snap in enumerate(trajectory):
        d = {}
        for snap_key, delta_key in keys:
            curr = snap.get(snap_key, 0)
            prev = trajectory[i - 1].get(snap_key, 0) if i > 0 else 0
            d[delta_key] = curr - prev
        d["total"] = sum(d.values())
        deltas.append(d)
    return deltas


# ---------------------------------------------------------------------------
# T — Transvolution
# ---------------------------------------------------------------------------

def compute_transvolution(trajectory: Trajectory) -> dict[str, list[float]]:
    """Compute involution, evolution, and condensation scores per step.

    Involution = (self + absorptive) / total_events  [inward folding]
    Evolution  = (novel + disintegration + env) / total_events  [outward unfolding]
    Condensation = involution * evolution  [coupled product]
    """
    deltas = _event_deltas(trajectory)
    involution = []
    evolution = []
    condensation = []

    for d in deltas:
        total = max(1, d["total"])
        inv = (d["self"] + d["absorptive"]) / total
        evo = (d["novel"] + d["disintegration"] + d["env"]) / total
        involution.append(inv)
        evolution.append(evo)
        condensation.append(inv * evo)

    return {
        "involution": involution,
        "evolution": evolution,
        "condensation": condensation,
    }


# ---------------------------------------------------------------------------
# A — Anopression / Anapression
# ---------------------------------------------------------------------------

def compute_anopression(
    trajectory: Trajectory,
    mu: float = 0.005,
) -> dict[str, list[float]]:
    """Compute anopressive and anapressive pressure scores per step.

    Anopressive (normalized to sum=1):
      expression  = agents_with_dM>0 / n_active
      impression  = delta_absorptive / total_events
      adpression  = (delta_self + delta_disint) / total_events

    Anapressive (NOT normalized, can exceed 1.0):
      oppression  = 1 - (mean_dM / max_observed_dM)
      suppression = 1 - affordance_mean
      depression  = mu * mean_M / max(eps, |mean_dM|)
      compression = mean_M / K
    """
    deltas = _event_deltas(trajectory)

    # Pre-compute dM series for oppression
    dM_series = []
    for i, snap in enumerate(trajectory):
        if i == 0:
            dM_series.append(0.0)
        else:
            dM_series.append(snap["total_M"] - trajectory[i - 1]["total_M"])

    max_abs_dM = max(abs(dm) for dm in dM_series) if dM_series else 1.0
    if max_abs_dM == 0:
        max_abs_dM = 1.0

    expression = []
    impression = []
    adpression = []
    oppression = []
    suppression = []
    depression = []
    compression = []

    for i, (snap, d) in enumerate(zip(trajectory, deltas)):
        n_active = max(1, snap.get("n_active", 1))
        total_events = max(1, d["total"])
        mean_M = snap["total_M"] / n_active
        dM = dM_series[i]
        K = snap.get("K_env", None)
        aff = snap.get("affordance_mean", 0.0)

        # --- Anopressive (raw, then normalize) ---
        # Expression: fraction of M growing (proxy: dM > 0 at ensemble level)
        expr_raw = 1.0 if dM > 0 else (0.5 if dM == 0 else 0.0)
        # Impression: absorptive events
        impr_raw = d["absorptive"] / total_events if d["total"] > 0 else 0.0
        # Adpression: punctuated events (self-metathesis + disintegration)
        adpr_raw = (d["self"] + d["disintegration"]) / total_events if d["total"] > 0 else 0.0

        ano_total = expr_raw + impr_raw + adpr_raw
        if ano_total > 0:
            expression.append(expr_raw / ano_total)
            impression.append(impr_raw / ano_total)
            adpression.append(adpr_raw / ano_total)
        else:
            expression.append(0.0)
            impression.append(0.0)
            adpression.append(0.0)

        # --- Anapressive (not normalized) ---
        oppression.append(1.0 - (dM / max_abs_dM) if max_abs_dM > 0 else 1.0)
        suppression.append(1.0 - aff)
        eps = 1e-10
        depression.append(mu * mean_M / max(eps, abs(dM)) if abs(dM) > eps else 0.0)
        compression.append(mean_M / K if K and K > 0 else 0.0)

    return {
        "expression": expression,
        "impression": impression,
        "adpression": adpression,
        "oppression": oppression,
        "suppression": suppression,
        "depression": depression,
        "compression": compression,
    }


def pressure_ratio(ano_scores: dict[str, list[float]]) -> list[float]:
    """Compute pressure ratio per step: sum(anapressive) / 1.0.

    > 1.0 = net entropy (breaking exceeds building capacity)
    < 1.0 = net extropy (building exceeds breaking)
    = 1.0 = syntropic equilibrium
    """
    n = len(ano_scores["oppression"])
    ratios = []
    for t in range(n):
        ana_total = (
            ano_scores["oppression"][t]
            + ano_scores["suppression"][t]
            + ano_scores["depression"][t]
            + ano_scores["compression"][t]
        )
        ratios.append(ana_total)
    return ratios


# ---------------------------------------------------------------------------
# P — Praxis
# ---------------------------------------------------------------------------

def compute_praxis(trajectory: Trajectory) -> dict[str, list[float]]:
    """Compute projection, reflection, and action scores per step."""
    deltas = _event_deltas(trajectory)
    projection = [snap.get("innovation_potential", 0.0) for snap in trajectory]
    reflection = [snap.get("affordance_mean", 0.0) for snap in trajectory]
    action = [float(d["total"]) for d in deltas]
    return {"projection": projection, "reflection": reflection, "action": action}


# ---------------------------------------------------------------------------
# S — Syntegration
# ---------------------------------------------------------------------------

def compute_syntegration(trajectory: Trajectory) -> dict[str, list[float]]:
    """Compute disintegration, preservation, integration, synthesis per step."""
    deltas = _event_deltas(trajectory)
    disintegration = [float(d["disintegration"]) for d in deltas]
    preservation = []
    integration = [float(d["absorptive"]) for d in deltas]
    synthesis = [float(d["self"] + d["novel"]) for d in deltas]

    for snap in trajectory:
        n_act = snap.get("n_active", 1)
        n_dor = snap.get("n_dormant", 0)
        total = max(1, n_act + n_dor)
        preservation.append(n_dor / total)

    return {
        "disintegration": disintegration,
        "preservation": preservation,
        "integration": integration,
        "synthesis": synthesis,
    }


# ---------------------------------------------------------------------------
# RIP — Recursion / Iteration / Praxis dominance
# ---------------------------------------------------------------------------

def compute_rip(trajectory: Trajectory) -> dict[str, list]:
    """Compute RIP dominance classification per step.

    Recursion: |dM| when no metathesis events (pure TAP tick).
    Iteration: |affordance change| + |dormancy change| (state accumulation).
    Praxis: total metathesis events (agentic action).
    """
    deltas = _event_deltas(trajectory)
    recursion_scores = []
    iteration_scores = []
    praxis_scores = []
    dominance = []

    for i, (snap, d) in enumerate(zip(trajectory, deltas)):
        # Recursion: TAP tick magnitude (only counts when no events)
        if i == 0:
            dM = 0.0
        else:
            dM = abs(snap["total_M"] - trajectory[i - 1]["total_M"])
        rec = dM if d["total"] == 0 else 0.0

        # Iteration: state accumulation that changes future recursions
        if i == 0:
            d_aff = 0.0
            d_dorm = 0.0
        else:
            d_aff = abs(snap.get("affordance_mean", 0) - trajectory[i - 1].get("affordance_mean", 0))
            d_dorm = abs(snap.get("n_dormant", 0) - trajectory[i - 1].get("n_dormant", 0))
        ite = d_aff + d_dorm

        # Praxis: agentic events
        pra = float(d["total"])

        recursion_scores.append(rec)
        iteration_scores.append(ite)
        praxis_scores.append(pra)

        # Dominance
        scores = {"recursion": rec, "iteration": ite, "praxis": pra}
        dom = max(scores, key=scores.get)
        dominance.append(dom)

    return {
        "recursion": recursion_scores,
        "iteration": iteration_scores,
        "praxis": praxis_scores,
        "dominance": dominance,
    }


# ---------------------------------------------------------------------------
# Aggregate + Correlation
# ---------------------------------------------------------------------------

def compute_all_scores(
    trajectory: Trajectory,
    mu: float = 0.005,
) -> dict[str, list[float]]:
    """Compute all TAPS + RIP mode scores, returning a flat dict of arrays.

    Keys are mode names (involution, evolution, condensation, expression,
    impression, adpression, oppression, suppression, depression, compression,
    projection, reflection, action, disintegration, preservation, integration,
    synthesis). All arrays have length == len(trajectory).
    """
    t_scores = compute_transvolution(trajectory)
    a_scores = compute_anopression(trajectory, mu=mu)
    p_scores = compute_praxis(trajectory)
    s_scores = compute_syntegration(trajectory)

    all_scores = {}
    for d in [t_scores, a_scores, p_scores, s_scores]:
        all_scores.update(d)
    return all_scores


def correlation_matrix(
    scores: dict[str, list[float]],
) -> dict[str, Any]:
    """Compute pairwise Pearson correlation between all mode score series.

    Returns:
        dict with keys:
          matrix: 2D list of r values
          labels: ordered list of mode names
          highly_correlated: list of (mode_a, mode_b, r) where |r| > 0.85
          independent_count: modes with no |r| > 0.85 partner
    """
    labels = sorted(scores.keys())
    n = len(labels)
    data = np.array([scores[label] for label in labels])  # shape (n_modes, n_steps)

    # Handle constant series (std=0) gracefully
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = np.corrcoef(data)
    corr = np.nan_to_num(corr, nan=0.0)

    matrix = corr.tolist()

    highly_correlated = []
    correlated_modes = set()
    for i in range(n):
        for j in range(i + 1, n):
            r = corr[i, j]
            if abs(r) > 0.85:
                highly_correlated.append((labels[i], labels[j], round(float(r), 4)))
                correlated_modes.add(labels[i])
                correlated_modes.add(labels[j])

    independent_count = n - len(correlated_modes)

    return {
        "matrix": matrix,
        "labels": labels,
        "highly_correlated": highly_correlated,
        "independent_count": independent_count,
    }
