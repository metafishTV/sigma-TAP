from __future__ import annotations

from .state import ModelState


def h_compression(state: ModelState, decay: float = 0.01) -> float:
    """
    Minimal H-functional proxy for implicate densification.

    Interprets accumulated consummation as compressible surplus and applies
    a proportional decay-to-compression channel. Non-negative by construction.
    """
    return max(0.0, state.Xi * decay)
