import csv
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from simulator.analysis import adaptive_xi_plateau_threshold, classify_regime
from simulator.hfuncs import h_compression
from simulator.simulate import run_sigma_tap
from simulator.state import ModelParams


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: python scripts/followup_precursor_longitudinal.py <sweep_csv>")

    src_path = sys.argv[1]
    with open(src_path, newline="") as f:
        rows = list(csv.DictReader(f))

    precursor_rows = [r for r in rows if r.get("regime") == "precursor-active" and float(r.get("gamma", 0.0)) > 0.0]

    writer = csv.DictWriter(
        sys.stdout,
        fieldnames=[
            "alpha",
            "mu",
            "a",
            "gamma",
            "m0",
            "steps_short",
            "steps_long",
            "regime_short",
            "regime_long",
            "final_M_short",
            "final_M_long",
            "final_Xi_short",
            "final_Xi_long",
            "under_resolved_short",
            "under_resolved_long",
        ],
    )
    writer.writeheader()

    traj_path = os.environ.get("PRECURSOR_TRAJ_CSV")
    traj_writer = None
    traj_file = None
    if traj_path:
        traj_file = open(traj_path, "w", newline="")
        traj_writer = csv.DictWriter(
            traj_file,
            fieldnames=[
                "cell_id",
                "alpha",
                "mu",
                "a",
                "gamma",
                "m0",
                "t",
                "M_t",
                "Xi_t",
            ],
        )
        traj_writer.writeheader()

    for idx, r in enumerate(precursor_rows):
        alpha = float(r["alpha"])
        mu = float(r["mu"])
        a = float(r["a"])
        gamma = float(r["gamma"])
        m0 = float(r["m0"])
        steps_short = int(r["steps"])
        steps_long = max(600, steps_short * 4)

        params = ModelParams(alpha=alpha, a=a, mu=mu, beta=0.05, eta=0.02)
        long_rows = run_sigma_tap(
            initial_M=m0,
            steps=steps_long,
            params=params,
            sigma0=1.0,
            gamma=gamma,
            h_func=lambda s: h_compression(s, decay=0.02),
            append_terminal_state=True,
        )
        tr = [x for x in long_rows if "M_t1" in x]
        xi = [tr[0]["Xi"]] + [x["Xi_t1"] for x in tr]
        m = [tr[0]["M"]] + [x["M_t1"] for x in tr]
        thr = adaptive_xi_plateau_threshold(xi)
        regime_long = classify_regime(xi, m, thr)
        under_resolved_long = m[-1] <= (m[0] * 1.05)

        writer.writerow(
            {
                "alpha": r["alpha"],
                "mu": r["mu"],
                "a": r["a"],
                "gamma": r["gamma"],
                "m0": r["m0"],
                "steps_short": steps_short,
                "steps_long": steps_long,
                "regime_short": r["regime"],
                "regime_long": regime_long,
                "final_M_short": r["final_M"],
                "final_M_long": f"{m[-1]:.8g}",
                "final_Xi_short": r["final_Xi"],
                "final_Xi_long": f"{xi[-1]:.8g}",
                "under_resolved_short": r["under_resolved"],
                "under_resolved_long": str(under_resolved_long),
            }
        )

        if traj_writer is not None:
            cell_id = f"cell_{idx:03d}"
            for t, (mt, xit) in enumerate(zip(m, xi)):
                traj_writer.writerow(
                    {
                        "cell_id": cell_id,
                        "alpha": r["alpha"],
                        "mu": r["mu"],
                        "a": r["a"],
                        "gamma": r["gamma"],
                        "m0": r["m0"],
                        "t": t,
                        "M_t": f"{mt:.8g}",
                        "Xi_t": f"{xit:.8g}",
                    }
                )

    if traj_file is not None:
        traj_file.close()


if __name__ == "__main__":
    main()
