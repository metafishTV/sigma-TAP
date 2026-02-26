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


def _fit_logistic_boundary_general(
    X: list[list[float]],
    y: list[float],
    lr: float = 0.05,
    epochs: int = 4000,
    l2: float = 1e-4,
) -> tuple[list[float], list[list[float]], list[float], list[float]]:
    """General N-dimensional logistic boundary via gradient descent.

    Returns (coefficients, X_standardized, means, stds).
    coefficients[0] = intercept, coefficients[1:] = feature weights.
    """
    if not X:
        return [], [], [], []

    n_features = len(X[0])
    n_samples = len(X)

    # Standardize
    means = []
    stds = []
    for f in range(n_features):
        vals = [x[f] for x in X]
        m = sum(vals) / len(vals)
        s = (sum((v - m) ** 2 for v in vals) / max(1, len(vals) - 1)) ** 0.5 or 1.0
        means.append(m)
        stds.append(s)

    Xs = [[(x[f] - means[f]) / stds[f] for f in range(n_features)] for x in X]

    b = [0.0] * (n_features + 1)

    def sig(z: float) -> float:
        if z >= 0:
            ez = math.exp(-z)
            return 1.0 / (1.0 + ez)
        ez = math.exp(z)
        return ez / (1.0 + ez)

    n_f = float(n_samples)
    for _ in range(epochs):
        g = [0.0] * (n_features + 1)
        for xs, yi in zip(Xs, y):
            z = b[0] + sum(b[f + 1] * xs[f] for f in range(n_features))
            d = sig(z) - yi
            g[0] += d
            for f in range(n_features):
                g[f + 1] += d * xs[f]
        g[0] /= n_f
        for f in range(n_features):
            g[f + 1] = g[f + 1] / n_f + l2 * b[f + 1]
        for k in range(n_features + 1):
            b[k] -= lr * g[k]

    return b, Xs, means, stds


def fit_explosive_logistic_boundary(
    records: list[dict],
    explosive_labels: set[str] | None = None,
    lr: float = 0.05,
    epochs: int = 4000,
    l2: float = 1e-4,
) -> dict:
    """Fit logistic boundary on (log(alpha), log(mu))."""
    explosive_labels = explosive_labels or {"explosive"}
    X = []
    y = []
    for r in records:
        try:
            a, m, label = float(r["alpha"]), float(r["mu"]), str(r["regime"])
        except Exception:
            continue
        if a <= 0 or m <= 0:
            continue
        X.append([math.log(a), math.log(m)])
        y.append(1.0 if label in explosive_labels else 0.0)

    if not X:
        return {"ok": False, "reason": "no_valid_rows"}

    b, Xs, means, stds = _fit_logistic_boundary_general(X, y, lr, epochs, l2)

    def sig(z):
        if z >= 0:
            return 1.0 / (1.0 + math.exp(-z))
        ez = math.exp(z)
        return ez / (1.0 + ez)

    correct = 0
    probs = []
    for xs, yi in zip(Xs, y):
        p = sig(b[0] + b[1] * xs[0] + b[2] * xs[1])
        probs.append(p)
        correct += int((1.0 if p >= 0.5 else 0.0) == yi)

    return {
        "ok": True, "n": len(Xs),
        "labels_positive": sum(int(v) for v in y),
        "coef": {"intercept": b[0], "log_alpha": b[1], "log_mu": b[2]},
        "feature_standardization": {
            "log_alpha_mean": means[0], "log_alpha_std": stds[0],
            "log_mu_mean": means[1], "log_mu_std": stds[1],
        },
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
    """Fit logistic boundary on (log(alpha), log(mu), log(M0))."""
    explosive_labels = explosive_labels or {"explosive"}
    X = []
    y = []
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
        X.append([math.log(a), math.log(m), math.log(m0)])
        y.append(1.0 if label in explosive_labels else 0.0)

    if not X:
        return {"ok": False, "reason": "no_valid_rows"}

    b, Xs, means, stds = _fit_logistic_boundary_general(X, y, lr, epochs, l2)

    def sig(z):
        if z >= 0:
            return 1.0 / (1.0 + math.exp(-z))
        ez = math.exp(z)
        return ez / (1.0 + ez)

    correct = 0
    probs = []
    for xs, yi in zip(Xs, y):
        p = sig(b[0] + b[1] * xs[0] + b[2] * xs[1] + b[3] * xs[2])
        probs.append(p)
        correct += int((1.0 if p >= 0.5 else 0.0) == yi)

    return {
        "ok": True, "n": len(Xs),
        "labels_positive": sum(int(v) for v in y),
        "coef": {"intercept": b[0], "log_alpha": b[1], "log_mu": b[2], "log_m0": b[3]},
        "feature_standardization": {
            "log_alpha_mean": means[0], "log_alpha_std": stds[0],
            "log_mu_mean": means[1], "log_mu_std": stds[1],
            "log_m0_mean": means[2], "log_m0_std": stds[2],
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
