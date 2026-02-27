"""Post-hoc turbulence diagnostics for TAP / sigma-TAP trajectories.

LAYER: Interpretive — these diagnostics compute derived quantities that
aid physical interpretation but do not feed back into the simulation.

Decision Bandwidth B(t) = sigma(Xi) * tau / f'(M)
  B > 1: laminar (all affordances evaluable)
  B < 1: turbulent (affordance overflow)

Praxitive Reynolds Re_prax = f'(M) * M / (sigma(Xi) + alpha)
  Analogous to fluid Reynolds number: ratio of innovation pressure
  to evaluative capacity.  The denominator sigma(Xi) + alpha combines
  the agent's learned evaluation efficiency with the structural
  innovation rate alpha, ensuring Re_prax remains finite when sigma -> 0
  and capturing the minimum evaluative baseline from the system itself.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .sigma_tap import sigma_linear
from .state import ModelParams
from .tap import innovation_kernel_derivative


@dataclass
class TurbulenceDiagnostics:
    """Turbulence diagnostic outputs (Interpretive layer)."""

    t: np.ndarray
    B_decision: np.ndarray
    Re_prax: np.ndarray
    laminar_fraction: float
    transition_time: float | None


def compute_turbulence_diagnostics(
    result,
    params: ModelParams,
    sigma0: float,
    gamma: float,
    tau_decision: float = 1.0,
) -> TurbulenceDiagnostics:
    """Compute turbulence diagnostics from a completed trajectory.

    Parameters
    ----------
    result : ContinuousResult (or any object with .t, .M, .Xi arrays).
    params : ModelParams used for the run.
    sigma0, gamma : Sigma parameters used for the run.
    tau_decision : Decision horizon (time available for evaluation).

    Returns
    -------
    TurbulenceDiagnostics with B(t), Re_prax(t), and summary statistics.
    """
    t = np.asarray(result.t)
    M = np.asarray(result.M)
    Xi = np.asarray(result.Xi)

    n = len(t)
    B = np.zeros(n)
    Re = np.zeros(n)

    for i in range(n):
        sig = sigma_linear(Xi[i], sigma0, gamma)
        fprime = innovation_kernel_derivative(M[i], params.alpha, params.a)

        if fprime > 0 and np.isfinite(fprime):
            B[i] = sig * tau_decision / fprime
        else:
            B[i] = float("inf") if fprime == 0 else 0.0

        denom = sig + params.alpha
        if denom > 0 and np.isfinite(fprime):
            Re[i] = fprime * M[i] / denom
        else:
            Re[i] = 0.0

    # Summary statistics.
    finite_B = B[np.isfinite(B)]
    if len(finite_B) > 0:
        laminar_fraction = float(np.mean(finite_B > 1.0))
    else:
        laminar_fraction = 1.0

    transition_time = None
    for i in range(n):
        if np.isfinite(B[i]) and B[i] < 1.0:
            transition_time = float(t[i])
            break

    return TurbulenceDiagnostics(
        t=t,
        B_decision=B,
        Re_prax=Re,
        laminar_fraction=laminar_fraction,
        transition_time=transition_time,
    )
