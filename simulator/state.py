from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelParams:
    alpha: float
    a: float
    mu: float
    beta: float = 0.0
    eta: float = 0.0
    tap_variant: str = "baseline"
    alpha1: float = 0.0
    carrying_capacity: float | None = None
    h_decay: float = 0.02


@dataclass
class ModelState:
    t: int
    M: float
    Xi: float = 0.0
    B: Optional[float] = None
    D: Optional[float] = None
