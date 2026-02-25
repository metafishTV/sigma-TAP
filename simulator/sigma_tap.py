from __future__ import annotations


def sigma_linear(xi: float, sigma0: float, gamma: float) -> float:
    return max(0.0, sigma0 * (1.0 + gamma * xi))


def xi_update_two_channel(xi: float, beta: float, B: float, eta: float, H: float) -> float:
    return xi + beta * B + eta * H
