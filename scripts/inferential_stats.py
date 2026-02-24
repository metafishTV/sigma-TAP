import argparse
import csv
import json
import math
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from simulator.analysis import fit_explosive_logistic_boundary, fit_explosive_logistic_boundary_3d


def percentile(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return float("nan")
    pos = (len(sorted_vals) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_vals[lo]
    w = pos - lo
    return sorted_vals[lo] * (1 - w) + sorted_vals[hi] * w


def bootstrap_ci(values: list[float], alpha: float = 0.05) -> dict:
    s = sorted(values)
    return {
        "lower": percentile(s, alpha / 2),
        "upper": percentile(s, 1 - alpha / 2),
    }


def load_csv(path: str) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def paired_panel(rows: list[dict], a_value: str = "8", g0: str = "0", g1: str = "0.2"):
    panel = [r for r in rows if r.get("a") == a_value and r.get("gamma") in {g0, g1}]
    by = {}
    for r in panel:
        key = (r["alpha"], r["mu"], r["m0"])
        by[(key, r["gamma"])] = r
    keys = sorted({k for k, _ in by.keys()})
    pairs = []
    for k in keys:
        r0 = by.get((k, g0))
        r1 = by.get((k, g1))
        if r0 and r1:
            pairs.append((k, r0, r1))
    return pairs


def run_matched_timing(rows: list[dict], rng: random.Random, n_perm: int = 5000) -> dict:
    pairs = paired_panel(rows)
    # retain matched blowup-step panel used in manuscript timing claim
    matched = []
    for key, r0, r1 in pairs:
        if r0.get("blowup_step", "") and r1.get("blowup_step", ""):
            matched.append((key, int(r0["blowup_step"]), int(r1["blowup_step"])))

    diffs = [a - b for _, a, b in matched]  # >0 means faster at gamma=0.2
    obs_mean = sum(diffs) / len(diffs)

    # sign-flip permutation test under paired null
    ge = 0
    for _ in range(n_perm):
        val = 0.0
        for d in diffs:
            val += d if rng.random() < 0.5 else -d
        val /= len(diffs)
        if abs(val) >= abs(obs_mean):
            ge += 1
    p_two = (ge + 1) / (n_perm + 1)

    return {
        "matched_blowup_panel_n": len(matched),
        "mean_blowup_step_gamma_0": sum(a for _, a, _ in matched) / len(matched),
        "mean_blowup_step_gamma_0.2": sum(b for _, _, b in matched) / len(matched),
        "mean_difference_gamma0_minus_gamma0.2": obs_mean,
        "paired_signflip_permutation_p_two_sided": p_two,
    }


def run_occupancy(rows: list[dict], rng: random.Random, n_boot: int = 1000, n_perm: int = 2000) -> dict:
    pairs = paired_panel(rows)
    # binary explosive indicator on full paired 300-cell panel
    y0 = [1 if r0["regime"] in {"explosive", "precursor-active"} else 0 for _, r0, _ in pairs]
    y1 = [1 if r1["regime"] in {"explosive", "precursor-active"} else 0 for _, _, r1 in pairs]
    n = len(y0)
    p0 = sum(y0) / n
    p1 = sum(y1) / n
    obs_diff = p1 - p0

    # paired permutation on binary differences
    d = [b - a for a, b in zip(y0, y1)]
    ge = 0
    for _ in range(n_perm):
        val = 0.0
        for di in d:
            val += di if rng.random() < 0.5 else -di
        val /= n
        if abs(val) >= abs(obs_diff):
            ge += 1
    p_two = (ge + 1) / (n_perm + 1)

    # bootstrap CI on overall proportion difference (resample paired cells)
    boot = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        b0 = sum(y0[i] for i in idx) / n
        b1 = sum(y1[i] for i in idx) / n
        boot.append(b1 - b0)

    return {
        "paired_cells_n": n,
        "explosive_or_precursor_fraction_gamma_0": p0,
        "explosive_or_precursor_fraction_gamma_0.2": p1,
        "difference_gamma0.2_minus_gamma0": obs_diff,
        "difference_bootstrap_95ci": bootstrap_ci(boot),
        "paired_signflip_permutation_p_two_sided": p_two,
    }


def boot_coef(records: list[dict], fit_fn, coef_keys: list[str], rng: random.Random, n_boot: int = 500, fit_kwargs: dict | None = None) -> dict:
    n = len(records)
    samples = {k: [] for k in coef_keys}
    for _ in range(n_boot):
        rec = [records[rng.randrange(n)] for _ in range(n)]
        out = fit_fn(rec, explosive_labels={"explosive", "precursor-active"}, **(fit_kwargs or {}))
        if not out.get("ok"):
            continue
        coef = out["coef"]
        for k in coef_keys:
            samples[k].append(coef[k])
    return {k: {"mean": sum(v) / len(v), "ci95": bootstrap_ci(v)} for k, v in samples.items() if v}


def run_coefficients(mode_a_rows: list[dict], mode_b_rows: list[dict], rng: random.Random, n_boot_coef: int = 500) -> dict:
    # side-by-side 2D for sign reversal claim
    a2 = boot_coef(mode_a_rows, fit_explosive_logistic_boundary, ["log_alpha", "log_mu"], rng, n_boot=n_boot_coef, fit_kwargs={"epochs": 300})
    b2 = boot_coef(mode_b_rows, fit_explosive_logistic_boundary, ["log_alpha", "log_mu"], rng, n_boot=n_boot_coef, fit_kwargs={"epochs": 300})
    # mode B 3D for init sensitivity
    b3 = boot_coef(mode_b_rows, fit_explosive_logistic_boundary_3d, ["log_alpha", "log_mu", "log_m0"], rng, n_boot=n_boot_coef, fit_kwargs={"epochs": 300})

    def disjoint(ci1, ci2):
        return (ci1["upper"] < ci2["lower"]) or (ci2["upper"] < ci1["lower"])

    return {
        "mode_a_2d_bootstrap": a2,
        "mode_b_2d_bootstrap": b2,
        "mode_b_3d_bootstrap": b3,
        "sign_reversal_ci_nonoverlap": {
            "log_alpha": disjoint(a2["log_alpha"]["ci95"], b2["log_alpha"]["ci95"]),
            "log_mu": disjoint(a2["log_mu"]["ci95"], b2["log_mu"]["ci95"]),
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sigma-feedback", default="outputs/sweep_sigma_feedback.csv")
    ap.add_argument("--mode-a", default="outputs/sweep_alpha_mu.csv")
    ap.add_argument("--mode-b", default="outputs/sweep_mode_b.csv")
    ap.add_argument("--out", default="outputs/inferential_stats.json")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mode", choices=["all","matched","occupancy","coefficients"], default="all")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--n-perm", type=int, default=2000)
    ap.add_argument("--n-boot-coef", type=int, default=500)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    sigma_rows = load_csv(args.sigma_feedback)
    mode_a_rows = load_csv(args.mode_a)
    mode_b_rows = load_csv(args.mode_b)

    if args.mode == "matched":
        out = {"matched_timing": run_matched_timing(sigma_rows, rng, n_perm=args.n_perm)}
    elif args.mode == "occupancy":
        out = {"occupancy_shift": run_occupancy(sigma_rows, rng, n_boot=args.n_boot, n_perm=args.n_perm)}
    elif args.mode == "coefficients":
        out = {"boundary_coefficients": run_coefficients(mode_a_rows, mode_b_rows, rng, n_boot_coef=args.n_boot_coef)}
    else:
        out = {
            "matched_timing": run_matched_timing(sigma_rows, rng, n_perm=args.n_perm),
            "occupancy_shift": run_occupancy(sigma_rows, rng, n_boot=args.n_boot, n_perm=args.n_perm),
            "boundary_coefficients": run_coefficients(mode_a_rows, mode_b_rows, rng, n_boot_coef=args.n_boot_coef),
        }

    Path(args.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
