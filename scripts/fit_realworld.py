"""Fit sigma-TAP to real-world datasets of combinatorial growth.

Strategy (hierarchical):
  Run 1: Core TAP with 3 parameters (s, p, mu) -- the physics backbone
         dM/dt = s * M^p - mu * M
         This is the *tamed* TAP regime: super-linear but sub-exponential.
  Run 2: Lock (s, p, mu) from Run 1, fit 2 learning parameters (gamma, beta)
  Run 3: Compare against null models (exponential, logistic, power-law)

Key insight: Real-world systems live in the TAMED regime of TAP --
growth is super-linear but sub-exponential because real constraints
prevent the full combinatorial explosion.  We model: f(M) = s * M^p, p > 1.

Theoretical basis: Kauffman's combinatorial kernel f(M) = alpha*a*(exp(M*ln(1+1/a))-1-M/a)
behaves as ~ M^p for moderate M before exploding for large M.  In real systems, resource
constraints, institutional friction, and finite search horizons tame the full combinatorial
explosion into an effective power-law regime (see Taalbi 2025, "Long-run patterns in the
discovery of the adjacent possible").  The power-law kernel is thus a reduced-form
approximation to TAP in the empirically relevant range, NOT a competing model.

Usage:
  python scripts/fit_realworld.py
  python scripts/fit_realworld.py --grid-size 10
"""
from __future__ import annotations

import csv
import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from scipy.optimize import minimize

from simulator.analysis import innovation_rate_scaling, constraint_tag

ROOT = Path(__file__).resolve().parents[1]


def load_datasets() -> dict:
    """Load datasets from config/realworld_datasets.json."""
    path = ROOT / "config" / "realworld_datasets.json"
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data["datasets"]


def _cost_logspace(predicted: np.ndarray, observed: np.ndarray) -> float:
    """Log-space MSE cost for scale-invariant fitting."""
    pred = np.clip(predicted, 1.0, None)
    obs = np.clip(observed, 1.0, None)
    return float(np.mean((np.log10(pred) - np.log10(obs)) ** 2))


def _euler_tap(
    t_data: np.ndarray,
    M0: float,
    s: float,
    p: float,
    mu: float,
    gamma: float = 0.0,
    beta: float = 0.0,
    K: float | None = None,
    substeps: int = 20,
) -> np.ndarray | None:
    """Forward-Euler integration of dM/dt = sigma * s * M^p - mu * M.

    The power-law kernel f(M) = s * M^p is the tamed TAP approximation:
    the full combinatorial kernel behaves like M^p in the large-M regime.
    With sigma-feedback: sigma(Xi) = 1 + gamma * Xi, dXi/dt = beta * B.
    With logistic cap: f(M) *= max(0, 1 - M/K).
    """
    dt_intervals = np.diff(t_data)
    M = M0
    Xi = 0.0
    m_cap = M0 * 1e6
    predicted = [M]

    for dt_full in dt_intervals:
        dt_sub = dt_full / substeps
        for _ in range(substeps):
            if M <= 0:
                M = 0.0
                break
            f_val = s * (M ** p)
            if K is not None and K > 0:
                f_val *= max(0.0, 1.0 - M / K)
            if not math.isfinite(f_val) or f_val > 1e18:
                return None
            sig = max(0.0, 1.0 + gamma * Xi)
            B = sig * f_val
            D = mu * M
            M += (B - D) * dt_sub
            Xi += beta * B * dt_sub
            if M > m_cap or not math.isfinite(M):
                return None
        predicted.append(M)

    return np.array(predicted)


def fit_single_variant(
    years: list[float],
    counts: list[float],
    variant: str = "baseline",
    grid_size: int = 5,
) -> dict:
    """Fit a TAP variant to one dataset using power-law TAP kernel.

    Variants:
      baseline: dM/dt = s*M^p - mu*M
      logistic: dM/dt = s*M^p*(1-M/K) - mu*M
      learning: dM/dt = sigma(Xi)*s*M^p - mu*M, dXi/dt = beta*B

    Returns dict with 'rmse', 'params', 'variant', 'cost'.
    """
    t_data = np.array(years, dtype=float) - years[0]
    obs = np.array(counts, dtype=float)
    M0 = obs[0]

    # Grid search over (log_s, p, log_mu).
    log_s_range = np.linspace(-10, -1, grid_size)
    p_range = np.linspace(1.01, 2.5, grid_size)
    log_mu_range = np.linspace(-4, -1, grid_size)

    best_cost = 1e6
    best_x = None

    K = max(counts) * 2.0 if variant == "logistic" else None

    for ls in log_s_range:
        for pp in p_range:
            for lm in log_mu_range:
                s = 10.0 ** ls
                mu = 10.0 ** lm
                pred = _euler_tap(t_data, M0, s, pp, mu, K=K)
                if pred is None or len(pred) != len(obs):
                    continue
                cost = _cost_logspace(pred, obs)
                if cost < best_cost:
                    best_cost = cost
                    best_x = [ls, pp, lm]

    # Nelder-Mead refinement.
    if best_x is not None:
        def objective(x):
            s = 10.0 ** x[0]
            pp = x[1]
            mu = 10.0 ** x[2]
            if pp < 0.5 or pp > 4.0:
                return 1e6
            pred = _euler_tap(t_data, M0, s, pp, mu, K=K)
            if pred is None or len(pred) != len(obs):
                return 1e6
            return _cost_logspace(pred, obs)

        try:
            res = minimize(objective, best_x, method="Nelder-Mead",
                          options={"maxiter": 500, "xatol": 0.005, "fatol": 1e-6})
            if res.fun < best_cost:
                best_cost = res.fun
                best_x = list(res.x)
        except Exception:
            pass

    if best_x is None:
        return {"variant": variant, "cost": 1e6, "rmse": float("inf"), "params": {}}

    s_fit = 10.0 ** best_x[0]
    p_fit = best_x[1]
    mu_fit = 10.0 ** best_x[2]

    # Pass 2: fit learning parameters (gamma, beta) with locked core.
    gamma_fit = 0.0
    beta_fit = 0.0
    if variant == "learning":
        best_learn_cost = best_cost
        for lg in np.linspace(-3, 0, 5):
            for lb in np.linspace(-3, 0, 5):
                gamma = 10.0 ** lg
                beta = 10.0 ** lb
                pred = _euler_tap(t_data, M0, s_fit, p_fit, mu_fit,
                                  gamma=gamma, beta=beta)
                if pred is None or len(pred) != len(obs):
                    continue
                cost = _cost_logspace(pred, obs)
                if cost < best_learn_cost:
                    best_learn_cost = cost
                    gamma_fit = gamma
                    beta_fit = beta
        best_cost = best_learn_cost

    rmse_log = math.sqrt(best_cost) if best_cost < 1e6 else float("inf")

    return {
        "variant": variant,
        "cost": best_cost,
        "rmse": rmse_log,
        "params": {
            "s": s_fit, "p": p_fit, "mu": mu_fit,
            "K": K, "gamma": gamma_fit, "beta": beta_fit,
        },
    }


def fit_null_models(years: list[float], counts: list[float]) -> dict:
    """Fit simple null models for comparison."""
    t = np.array(years, dtype=float) - years[0]
    obs = np.array(counts, dtype=float)

    results = {}

    # Exponential: M(t) = M0 * exp(r*t)
    try:
        def exp_cost(x):
            r = x[0]
            pred = obs[0] * np.exp(np.clip(r * t, -50, 50))
            return _cost_logspace(pred, obs)
        res = minimize(exp_cost, [0.1], method="Nelder-Mead")
        results["exponential"] = {"cost": res.fun, "rmse": math.sqrt(res.fun), "r": float(res.x[0])}
    except Exception:
        results["exponential"] = {"cost": 1e6, "rmse": float("inf")}

    # Logistic growth: M(t) = K / (1 + ((K-M0)/M0)*exp(-r*t))
    try:
        def logistic_cost(x):
            r, K = x[0], 10 ** x[1]
            if K <= obs[0]:
                return 1e6
            pred = K / (1 + ((K - obs[0]) / obs[0]) * np.exp(-r * t))
            return _cost_logspace(pred, obs)
        res = minimize(logistic_cost, [0.1, math.log10(max(counts) * 2)], method="Nelder-Mead")
        results["logistic_growth"] = {"cost": res.fun, "rmse": math.sqrt(res.fun)}
    except Exception:
        results["logistic_growth"] = {"cost": 1e6, "rmse": float("inf")}

    # Power law: M(t) = M0 * (1 + t)^p
    try:
        def power_cost(x):
            p = x[0]
            pred = obs[0] * (1 + t) ** p
            return _cost_logspace(pred, obs)
        res = minimize(power_cost, [1.5], method="Nelder-Mead")
        results["power_law"] = {"cost": res.fun, "rmse": math.sqrt(res.fun), "p": float(res.x[0])}
    except Exception:
        results["power_law"] = {"cost": 1e6, "rmse": float("inf")}

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Fit TAP variants to real-world data")
    parser.add_argument("--grid-size", type=int, default=8)
    args = parser.parse_args()

    datasets = load_datasets()
    variants = ["baseline", "logistic", "learning"]

    all_results = []

    for ds_name, ds in datasets.items():
        print(f"\n{'='*60}")
        print(f"{ds_name}: {ds['description']}")
        print(f"{'='*60}")

        # Null models.
        nulls = fit_null_models(ds["years"], ds["counts"])
        for model_name, nr in nulls.items():
            print(f"  {model_name:20s}: RMSE(log) = {nr['rmse']:.4f}")
            all_results.append({
                "dataset": ds_name, "model": model_name,
                "rmse_log": f"{nr['rmse']:.6f}", "variant": "",
            })

        # TAP variants.
        for variant in variants:
            result = fit_single_variant(
                ds["years"], ds["counts"],
                variant=variant,
                grid_size=args.grid_size,
            )
            p = result["params"]
            if p:
                print(f"  TAP-{variant:15s}: RMSE(log) = {result['rmse']:.4f}  "
                      f"s={p['s']:.2e} p={p['p']:.3f} mu={p['mu']:.2e}")
            else:
                print(f"  TAP-{variant:15s}: RMSE(log) = {result['rmse']:.4f}  (no fit)")

            # Long-run scaling on best-fit trajectory.
            if p:
                t_data = np.array(ds["years"], dtype=float) - ds["years"][0]
                pred = _euler_tap(t_data, ds["counts"][0],
                                  p["s"], p["p"], p["mu"],
                                  gamma=p.get("gamma", 0),
                                  beta=p.get("beta", 0),
                                  K=p.get("K"))
                if pred is not None:
                    scaling = innovation_rate_scaling(list(pred))
                    tag = constraint_tag(list(pred), p.get("K"))
                    print(f"    scaling exp: {scaling['exponent']:.3f} "
                          f"(R2={scaling['r_squared']:.3f}), constraint: {tag}")

            all_results.append({
                "dataset": ds_name, "model": f"TAP-{variant}",
                "rmse_log": f"{result['rmse']:.6f}", "variant": variant,
            })

    # Write CSV output.
    out_csv = ROOT / "outputs" / "realworld_fit.csv"
    out_csv.parent.mkdir(exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "model", "variant", "rmse_log"])
        w.writeheader()
        w.writerows(all_results)
    print(f"\nWrote {out_csv}")


if __name__ == "__main__":
    main()
