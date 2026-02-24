from __future__ import annotations

from dataclasses import asdict
import math
from typing import Callable

from .sigma_tap import sigma_linear, xi_update_two_channel
from .state import ModelParams, ModelState
from .tap import innovation_kernel_closed


HFunc = Callable[[ModelState], float]


def run_sigma_tap(
    initial_M: float,
    steps: int,
    params: ModelParams,
    sigma0: float,
    gamma: float,
    h_func: HFunc | None = None,
    append_terminal_state: bool = True,
    m_cap: float = 1e9,
    xi_cap: float = 1e9,
) -> list[dict]:
    """Simple forward simulator over closure equations.

    Row schema is transition-indexed:
    - `M`, `Xi` are state values at time t
    - `M_t1`, `Xi_t1` are state values at time t+1
    """
    rows: list[dict] = []
    state = ModelState(t=0, M=initial_M, Xi=0.0)
    h_func = h_func or (lambda _s: 0.0)
    blowup_step: int | None = None

    for _ in range(steps):
        sigma = sigma_linear(state.Xi, sigma0, gamma)
        f = innovation_kernel_closed(state.M, params.alpha, params.a)
        B = sigma * f
        D = params.mu * state.M
        H = float(h_func(state))

        if (not math.isfinite(state.M)) or (not math.isfinite(f)):
            raw_next_M = float("inf")
            raw_next_Xi = float("inf")
        else:
            raw_next_M = state.M + (B - D)
            raw_next_Xi = xi_update_two_channel(state.Xi, params.beta, B, params.eta, H)

        overflow_now = (not math.isfinite(raw_next_M)) or (not math.isfinite(raw_next_Xi)) or (raw_next_M >= m_cap) or (raw_next_Xi >= xi_cap)
        if overflow_now and blowup_step is None:
            blowup_step = state.t + 1

        next_M = min(raw_next_M, m_cap) if math.isfinite(raw_next_M) else m_cap
        next_Xi = min(raw_next_Xi, xi_cap) if math.isfinite(raw_next_Xi) else xi_cap

        row = asdict(state)
        row.update(
            {
                "sigma": sigma,
                "f": f,
                "H": H,
                "t_next": state.t + 1,
                "M_t1": next_M,
                "Xi_t1": next_Xi,
                # backward-compatible aliases
                "next_M": next_M,
                "next_Xi": next_Xi,
                "overflow_detected": overflow_now,
                "blowup_step": blowup_step,
            }
        )
        rows.append(row)

        state = ModelState(t=state.t + 1, M=next_M, Xi=next_Xi, B=B, D=D)

    if append_terminal_state:
        terminal = asdict(state)
        terminal.update({"is_terminal": True, "blowup_step": blowup_step})
        rows.append(terminal)

    return rows
