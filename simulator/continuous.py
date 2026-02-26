"""Continuous-time ODE solver for TAP / sigma-TAP dynamics.

Wraps scipy.integrate.solve_ivp around the existing discrete-step
infrastructure (compute_birth_term, sigma_linear) to provide smooth
integration at arbitrary time points.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

from .sigma_tap import sigma_linear
from .state import ModelParams
from .tap import compute_birth_term


@dataclass
class ContinuousResult:
    """Output from continuous-time TAP integration."""

    t: np.ndarray
    M: np.ndarray
    Xi: np.ndarray
    sigma: np.ndarray
    f: np.ndarray
    terminated_by_overflow: bool
    blowup_time: float | None


def _build_rhs(params: ModelParams, sigma0: float, gamma: float):
    """Build the right-hand-side function for solve_ivp."""

    def rhs(t, y):
        M, Xi = y[0], y[1]
        if M < 0:
            M = 0.0

        sig = sigma_linear(Xi, sigma0, gamma)
        f_val = compute_birth_term(
            M,
            alpha=params.alpha,
            a=params.a,
            variant=params.tap_variant,
            alpha1=params.alpha1,
            carrying_capacity=params.carrying_capacity,
        )

        # Clamp infinite/nan birth terms to prevent solver stalling.
        if not np.isfinite(f_val):
            f_val = 1e15

        B = sig * f_val          # Birth term: sigma-scaled innovation rate
        D = params.mu * M         # Death/extinction term
        H = max(0.0, params.h_decay * Xi)  # Compression feedback

        dM = B - D
        # Xi tracks cumulative affordance pressure (total innovation exposure),
        # not net surviving objects.  Driven by B (not B-D) because even
        # innovations that go extinct still expand the evaluative landscape
        # an agent has encountered.  See Kauffman & Steel (2017) on the
        # distinction between realized M and explored affordance space.
        dXi = params.beta * B + params.eta * H

        return [dM, dXi]

    return rhs


def _overflow_event(m_cap: float):
    """Event function: triggers when M >= m_cap."""

    def event(t, y):
        return m_cap - y[0]

    event.terminal = True
    event.direction = -1
    return event


def run_continuous(
    initial_M: float,
    t_span: tuple[float, float],
    params: ModelParams,
    sigma0: float = 1.0,
    gamma: float = 0.0,
    t_eval: np.ndarray | None = None,
    method: str = "RK45",
    max_step: float = 1.0,
    m_cap: float = 1e9,
) -> ContinuousResult:
    """Integrate TAP/sigma-TAP dynamics in continuous time.

    Parameters
    ----------
    initial_M : Starting realized-object count.
    t_span : (t_start, t_end) integration window.
    params : ModelParams with variant, alpha, a, mu, etc.
    sigma0 : Baseline efficiency (default 1.0).
    gamma : Feedback strength (0 = pure TAP).
    t_eval : Optional array of times at which to report solution.
    method : ODE solver method (default RK45).
    max_step : Maximum internal step size.
    m_cap : Overflow cap — integration terminates when M >= m_cap.

    Returns
    -------
    ContinuousResult with time-series arrays.
    """
    rhs = _build_rhs(params, sigma0, gamma)
    overflow_ev = _overflow_event(m_cap)

    y0 = [initial_M, 0.0]

    sol = solve_ivp(
        rhs,
        t_span,
        y0,
        method=method,
        t_eval=t_eval,
        max_step=max_step,
        events=[overflow_ev],
        dense_output=False,
        rtol=1e-8,
        atol=1e-10,
    )

    t_arr = sol.t
    M_arr = sol.y[0]
    Xi_arr = sol.y[1]

    # Recompute sigma and f at each output point for diagnostics.
    sigma_arr = np.maximum(0.0, sigma0 * (1.0 + gamma * Xi_arr))
    f_arr = np.array([
        compute_birth_term(
            m, alpha=params.alpha, a=params.a,
            variant=params.tap_variant, alpha1=params.alpha1,
            carrying_capacity=params.carrying_capacity,
        )
        for m in M_arr
    ])

    terminated = sol.status == 1  # event triggered
    blowup_time = None
    if terminated and sol.t_events and len(sol.t_events[0]) > 0:
        blowup_time = float(sol.t_events[0][0])

    return ContinuousResult(
        t=t_arr,
        M=M_arr,
        Xi=Xi_arr,
        sigma=sigma_arr,
        f=f_arr,
        terminated_by_overflow=terminated,
        blowup_time=blowup_time,
    )
