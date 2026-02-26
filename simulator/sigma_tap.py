"""Sigma-TAP feedback functions.

The sigma-TAP extension adds a learning feedback loop to Kauffman's TAP
dynamics.  The core idea: as an agent explores the adjacent possible,
its evaluation efficiency sigma evolves with accumulated affordance
exposure Xi.

sigma(Xi) = sigma0 * (1 + gamma * Xi)

This is the simplest monotone feedback consistent with the TAP framework:
higher cumulative exposure -> faster evaluation of new affordances.
gamma = 0 recovers the standard (non-learning) TAP.

Xi itself accumulates via two channels:
  dXi = beta * B + eta * H
where B is the innovation birth term and H is a compression/memory
feedback from the current Xi state.  The two-channel form separates
direct innovation exposure (beta * B) from endogenous consolidation
(eta * H), analogous to exploration vs exploitation in adjacent-
possible search.
"""
from __future__ import annotations


def sigma_linear(xi: float, sigma0: float, gamma: float) -> float:
    """Linear sigma feedback: sigma = sigma0 * (1 + gamma * Xi), floored at 0."""
    return max(0.0, sigma0 * (1.0 + gamma * xi))


def xi_update_two_channel(xi: float, beta: float, B: float, eta: float, H: float) -> float:
    """Discrete Xi update: Xi_{t+1} = Xi_t + beta * B + eta * H."""
    return xi + beta * B + eta * H
