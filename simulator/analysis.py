from __future__ import annotations

import math
from statistics import median


def identifiability_gate(observed_fields: set[str]) -> dict:
    """Pre-B3 gate: what claims are admissible given observability."""
    has_M = "M_t" in observed_fields
    has_BD = "B_t" in observed_fields and "D_t" in observed_fields
    has_Xi_proxy = "Xi_proxy" in observed_fields or "Xi_t" in observed_fields

    return {
        "can_fit_reduced_form": has_M,
        "can_test_layered_falsification": has_M and (has_BD or has_Xi_proxy),
        "needs_proxy_for_structural_claims": has_M and not (has_BD or has_Xi_proxy),
    }


def adaptive_xi_plateau_threshold(
    xi_series: list[float],
    dt: float = 1.0,
    warmup_fraction: float = 0.2,
    multiplier: float = 3.0,
    rel_floor: float = 1e-6,
) -> float:
    """
    Adaptive ξ-plateau threshold from early-run ΔXi/dt dynamics.

    threshold = max(multiplier * median(|ΔXi/dt| over warmup window), rel_floor * max(Xi_warmup)).
    """
    if len(xi_series) < 3:
        return 0.0
    n = len(xi_series)
    warmup_n = max(2, int(n * warmup_fraction))
    warmup = xi_series[:warmup_n]
    rates = [abs((warmup[i + 1] - warmup[i]) / dt) for i in range(len(warmup) - 1)]
    if not rates:
        return 0.0
    return max(multiplier * median(rates), rel_floor * max(warmup))


def precursor_guard_active(dxi_dt: float, xi_plateau: float) -> bool:
    """True when consummation-rate is above adaptive plateau threshold."""
    return dxi_dt >= xi_plateau




def classify_regime(
    xi_traj: list[float],
    m_traj: list[float],
    xi_plateau_thr: float,
    m_blow: float = 1e9,
    slope_ratio_exp: float = 1.2,
) -> str:
    """
    Coarse regime label from one trajectory.

    Returns one of:
    - "extinction"
    - "explosive"
    - "precursor-active"
    - "exponential"
    - "plateau"
    """
    if len(m_traj) < 3 or len(xi_traj) < 3:
        return "plateau"

    if m_traj[-1] <= 0:
        return "extinction"
    if m_traj[-1] >= m_blow or any((not math.isfinite(v)) or v == float("inf") for v in m_traj):
        return "explosive"

    dxi = xi_traj[-1] - xi_traj[-2]
    if precursor_guard_active(dxi, xi_plateau_thr):
        return "precursor-active"

    # Growth-shape heuristic from stepwise M increments.
    dm = [m_traj[i + 1] - m_traj[i] for i in range(len(m_traj) - 1)]
    if not dm:
        return "plateau"

    early = max(1, len(dm) // 3)
    late = max(1, len(dm) // 3)
    early_mean = sum(dm[:early]) / early
    late_mean = sum(dm[-late:]) / late

    if early_mean <= 0 or abs(early_mean) < 1e-15:
        return "plateau"
    if late_mean / early_mean >= slope_ratio_exp:
        return "exponential"
    return "plateau"


def find_fixed_point(alpha: float, mu: float, a: float, m_lo: float = 1e-3, m_hi: float = 1e4, grid_n: int = 200) -> float | None:
    """
    Numerically find positive M* solving f(M*) = mu*M* using the closed TAP kernel.

    Returns None if no positive bracketed root is found in [m_lo, m_hi].
    """
    from .tap import innovation_kernel_closed

    def g(m: float) -> float:
        return innovation_kernel_closed(m, alpha, a) - mu * m

    # Log grid bracket search for sign changes.
    logs = [math.log(m_lo) + i * (math.log(m_hi) - math.log(m_lo)) / (grid_n - 1) for i in range(grid_n)]
    ms = [math.exp(x) for x in logs]
    vals = [g(m) for m in ms]

    bracket = None
    for i in range(len(ms) - 1):
        v1, v2 = vals[i], vals[i + 1]
        if not (math.isfinite(v1) and math.isfinite(v2)):
            continue
        if v1 == 0:
            return ms[i]
        if v1 * v2 < 0:
            bracket = (ms[i], ms[i + 1])
            break

    if bracket is None:
        return None

    lo, hi = bracket
    vlo = g(lo)
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        vm = g(mid)
        if not math.isfinite(vm):
            hi = mid
            continue
        if abs(vm) < 1e-10:
            return mid
        if vlo * vm <= 0:
            hi = mid
        else:
            lo, vlo = mid, vm
    return 0.5 * (lo + hi)


def fit_explosive_logistic_boundary(
    records: list[dict],
    explosive_labels: set[str] | None = None,
    lr: float = 0.05,
    epochs: int = 4000,
    l2: float = 1e-4,
) -> dict:
    """
    Fit a simple logistic boundary on (log(alpha), log(mu)).

    y=1 for explosive-like labels, y=0 otherwise.
    Returns fitted coefficients and a compact training summary.
    """

    explosive_labels = explosive_labels or {"explosive"}

    X: list[tuple[float, float]] = []
    y: list[float] = []
    for r in records:
        try:
            a = float(r["alpha"])
            m = float(r["mu"])
            label = str(r["regime"])
        except Exception:
            continue
        if a <= 0 or m <= 0:
            continue
        X.append((math.log(a), math.log(m)))
        y.append(1.0 if label in explosive_labels else 0.0)

    if not X:
        return {"ok": False, "reason": "no_valid_rows"}

    # Standardize features for stable optimization.
    xs1 = [v[0] for v in X]
    xs2 = [v[1] for v in X]
    m1 = sum(xs1) / len(xs1)
    m2 = sum(xs2) / len(xs2)
    s1 = (sum((z - m1) ** 2 for z in xs1) / max(1, len(xs1) - 1)) ** 0.5 or 1.0
    s2 = (sum((z - m2) ** 2 for z in xs2) / max(1, len(xs2) - 1)) ** 0.5 or 1.0
    Xs = [((u - m1) / s1, (v - m2) / s2) for u, v in X]

    b0 = 0.0
    b1 = 0.0
    b2 = 0.0

    def sig(z: float) -> float:
        if z >= 0:
            ez = math.exp(-z)
            return 1.0 / (1.0 + ez)
        ez = math.exp(z)
        return ez / (1.0 + ez)

    n = float(len(Xs))
    for _ in range(epochs):
        g0 = 0.0
        g1 = 0.0
        g2 = 0.0
        for (x1, x2), yi in zip(Xs, y):
            p = sig(b0 + b1 * x1 + b2 * x2)
            d = (p - yi)
            g0 += d
            g1 += d * x1
            g2 += d * x2
        # mean gradient + L2
        g0 = g0 / n
        g1 = g1 / n + l2 * b1
        g2 = g2 / n + l2 * b2
        b0 -= lr * g0
        b1 -= lr * g1
        b2 -= lr * g2

    # train accuracy summary
    correct = 0
    probs: list[float] = []
    for (x1, x2), yi in zip(Xs, y):
        p = sig(b0 + b1 * x1 + b2 * x2)
        probs.append(p)
        pred = 1.0 if p >= 0.5 else 0.0
        correct += int(pred == yi)

    return {
        "ok": True,
        "n": len(Xs),
        "labels_positive": sum(int(v) for v in y),
        "coef": {"intercept": b0, "log_alpha": b1, "log_mu": b2},
        "feature_standardization": {"log_alpha_mean": m1, "log_alpha_std": s1, "log_mu_mean": m2, "log_mu_std": s2},
        "train_accuracy": correct / len(Xs),
        "mean_pred_prob": sum(probs) / len(probs),
    }


def fit_explosive_logistic_boundary_3d(
    records: list[dict],
    explosive_labels: set[str] | None = None,
    lr: float = 0.05,
    epochs: int = 5000,
    l2: float = 1e-4,
) -> dict:
    """
    Fit logistic boundary on (log(alpha), log(mu), log(M0)).

    This estimates initialization sensitivity explicitly via coefficient c3 on log(M0).
    """

    explosive_labels = explosive_labels or {"explosive"}

    X: list[tuple[float, float, float]] = []
    y: list[float] = []
    for r in records:
        try:
            a = float(r["alpha"])
            m = float(r["mu"])
            m0 = float(r["m0"])
            label = str(r["regime"])
        except Exception:
            continue
        if a <= 0 or m <= 0 or m0 <= 0:
            continue
        X.append((math.log(a), math.log(m), math.log(m0)))
        y.append(1.0 if label in explosive_labels else 0.0)

    if not X:
        return {"ok": False, "reason": "no_valid_rows"}

    xs1 = [v[0] for v in X]
    xs2 = [v[1] for v in X]
    xs3 = [v[2] for v in X]
    m1 = sum(xs1) / len(xs1)
    m2 = sum(xs2) / len(xs2)
    m3 = sum(xs3) / len(xs3)
    s1 = (sum((z - m1) ** 2 for z in xs1) / max(1, len(xs1) - 1)) ** 0.5 or 1.0
    s2 = (sum((z - m2) ** 2 for z in xs2) / max(1, len(xs2) - 1)) ** 0.5 or 1.0
    s3 = (sum((z - m3) ** 2 for z in xs3) / max(1, len(xs3) - 1)) ** 0.5 or 1.0
    Xs = [((u - m1) / s1, (v - m2) / s2, (w - m3) / s3) for u, v, w in X]

    b0 = b1 = b2 = b3 = 0.0

    def sig(z: float) -> float:
        if z >= 0:
            ez = math.exp(-z)
            return 1.0 / (1.0 + ez)
        ez = math.exp(z)
        return ez / (1.0 + ez)

    n = float(len(Xs))
    for _ in range(epochs):
        g0 = g1 = g2 = g3 = 0.0
        for (x1, x2, x3), yi in zip(Xs, y):
            p = sig(b0 + b1 * x1 + b2 * x2 + b3 * x3)
            d = p - yi
            g0 += d
            g1 += d * x1
            g2 += d * x2
            g3 += d * x3
        g0 = g0 / n
        g1 = g1 / n + l2 * b1
        g2 = g2 / n + l2 * b2
        g3 = g3 / n + l2 * b3
        b0 -= lr * g0
        b1 -= lr * g1
        b2 -= lr * g2
        b3 -= lr * g3

    correct = 0
    probs: list[float] = []
    for (x1, x2, x3), yi in zip(Xs, y):
        p = sig(b0 + b1 * x1 + b2 * x2 + b3 * x3)
        probs.append(p)
        pred = 1.0 if p >= 0.5 else 0.0
        correct += int(pred == yi)

    return {
        "ok": True,
        "n": len(Xs),
        "labels_positive": sum(int(v) for v in y),
        "coef": {"intercept": b0, "log_alpha": b1, "log_mu": b2, "log_m0": b3},
        "feature_standardization": {
            "log_alpha_mean": m1,
            "log_alpha_std": s1,
            "log_mu_mean": m2,
            "log_mu_std": s2,
            "log_m0_mean": m3,
            "log_m0_std": s3,
        },
        "train_accuracy": correct / len(Xs),
        "mean_pred_prob": sum(probs) / len(probs),
    }


def mstar_isocurve_mu(alpha: float, a: float, m_star: float) -> float:
    """Analytical Mode C curve: mu(alpha | M*=const)."""
    from .tap import innovation_kernel_closed

    return innovation_kernel_closed(m_star, alpha, a) / m_star


def innovation_rate_scaling(
    m_traj: list[float],
    dt: float = 1.0,
) -> dict:
    """Fit dk/dt ~ k^sigma using log-log OLS on finite differences.

    Inspired by Taalbi (2025): sigma ~ 1 indicates resource-constrained
    linear-in-k dynamics. sigma > 1 indicates unconstrained super-linear TAP.

    Returns dict with 'exponent', 'r_squared', 'n_points'.
    """

    if len(m_traj) < 3:
        return {"exponent": 1.0, "r_squared": 0.0, "n_points": len(m_traj)}

    from scipy.stats import linregress

    rates = [(m_traj[i + 1] - m_traj[i]) / dt for i in range(len(m_traj) - 1)]
    midpoints = [0.5 * (m_traj[i] + m_traj[i + 1]) for i in range(len(m_traj) - 1)]

    log_k = []
    log_rate = []
    for k, r in zip(midpoints, rates):
        if k > 0 and r > 0:
            log_k.append(math.log(k))
            log_rate.append(math.log(r))

    n = len(log_k)
    if n < 2:
        return {"exponent": 1.0, "r_squared": 0.0, "n_points": n}

    result = linregress(log_k, log_rate)
    return {"exponent": result.slope, "r_squared": result.rvalue ** 2, "n_points": n}


def constraint_tag(
    m_traj: list[float],
    carrying_capacity: float | None,
    dt: float = 1.0,
) -> str:
    """Tag observed dynamics as adjacency-limited, resource-limited, or mixed.

    Uses Taalbi's framing: if carrying capacity is absent or M is far from it,
    dynamics are adjacency-limited. If M approaches K, resource-limited.
    """
    if carrying_capacity is None or carrying_capacity <= 0:
        return "adjacency-limited"

    if len(m_traj) < 2:
        return "adjacency-limited"

    m_final = m_traj[-1]
    ratio = m_final / carrying_capacity

    if ratio > 0.8:
        return "resource-limited"
    elif ratio > 0.3:
        return "mixed"
    else:
        return "adjacency-limited"
