import csv
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from simulator.analysis import adaptive_xi_plateau_threshold, classify_regime
from simulator.hfuncs import h_compression
from simulator.simulate import run_sigma_tap
from simulator.state import ModelParams


def logspace(lo: float, hi: float, n: int) -> list[float]:
    import math
    a = math.log10(lo)
    b = math.log10(hi)
    if n == 1:
        return [10 ** a]
    return [10 ** (a + i * (b - a) / (n - 1)) for i in range(n)]


def main() -> None:
    alphas = logspace(1e-5, 1e-2, 10)
    mus = logspace(1e-3, 1e-1, 10)
    gamma_values = [0.5, 1.0, 2.0, 5.0]

    writer = csv.DictWriter(
        sys.stdout,
        fieldnames=[
            "mode",
            "alpha",
            "mu",
            "a",
            "gamma",
            "m0",
            "steps",
            "final_M",
            "final_Xi",
            "regime",
            "blowup_step",
            "under_resolved",
        ],
    )
    writer.writeheader()

    for gamma in gamma_values:
        for alpha in alphas:
            for mu in mus:
                params = ModelParams(alpha=alpha, a=8.0, mu=mu, beta=0.05, eta=0.02)
                steps = 300 if (alpha >= 1e-3 and mu <= 1e-2) else 120
                rows = run_sigma_tap(
                    initial_M=20.0,
                    steps=steps,
                    params=params,
                    sigma0=1.0,
                    gamma=gamma,
                    h_func=lambda s: h_compression(s, decay=0.02),
                    append_terminal_state=True,
                    m_cap=1e9,
                    xi_cap=1e9,
                )
                tr = [r for r in rows if "M_t1" in r]
                xi = [tr[0]["Xi"]] + [r["Xi_t1"] for r in tr]
                m = [tr[0]["M"]] + [r["M_t1"] for r in tr]
                thr = adaptive_xi_plateau_threshold(xi)
                regime = classify_regime(xi, m, thr, m_blow=1e9)
                under_resolved = m[-1] <= (m[0] * 1.05)
                blowup_step = next((r.get("blowup_step") for r in tr if r.get("overflow_detected")), None)

                writer.writerow(
                    {
                        "mode": "gamma_threshold_mode_b",
                        "alpha": f"{alpha:.8g}",
                        "mu": f"{mu:.8g}",
                        "a": "8",
                        "gamma": f"{gamma:.8g}",
                        "m0": "20",
                        "steps": steps,
                        "final_M": f"{m[-1]:.8g}",
                        "final_Xi": f"{xi[-1]:.8g}",
                        "regime": regime,
                        "blowup_step": "" if blowup_step is None else int(blowup_step),
                        "under_resolved": str(under_resolved),
                    }
                )


if __name__ == "__main__":
    main()
