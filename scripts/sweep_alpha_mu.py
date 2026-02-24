import csv
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from simulator.analysis import adaptive_xi_plateau_threshold, classify_regime, find_fixed_point
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

    writer = csv.DictWriter(
        sys.stdout,
        fieldnames=[
            "alpha",
            "mu",
            "m_star",
            "m0",
            "steps",
            "final_M",
            "final_Xi",
            "regime",
            "xi_plateau",
            "under_resolved",
        ],
    )
    writer.writeheader()

    for alpha in alphas:
        for mu in mus:
            m_star = find_fixed_point(alpha=alpha, mu=mu, a=8.0)
            m0 = (1.05 * m_star) if m_star is not None else 3.0

            params = ModelParams(alpha=alpha, a=8.0, mu=mu, beta=0.05, eta=0.02)
            # Longer horizon near expected boundary: high alpha, low mu.
            steps = 120 if (alpha >= 1e-3 and mu <= 1e-2) else 40
            rows = run_sigma_tap(
                initial_M=m0,
                steps=steps,
                params=params,
                sigma0=1.0,
                gamma=0.0,
                h_func=lambda s: h_compression(s, decay=0.02),
                append_terminal_state=True,
            )
            tr = [r for r in rows if "M_t1" in r]
            xi = [tr[0]["Xi"]] + [r["Xi_t1"] for r in tr]
            m = [tr[0]["M"]] + [r["M_t1"] for r in tr]
            thr = adaptive_xi_plateau_threshold(xi)
            regime = classify_regime(xi, m, thr)

            under_resolved = m[-1] <= (m[0] * 1.05)
            writer.writerow(
                {
                    "alpha": f"{alpha:.8g}",
                    "mu": f"{mu:.8g}",
                    "m_star": "" if m_star is None else f"{m_star:.8g}",
                    "m0": f"{m0:.8g}",
                    "steps": steps,
                    "final_M": f"{m[-1]:.8g}",
                    "final_Xi": f"{xi[-1]:.8g}",
                    "regime": regime,
                    "xi_plateau": f"{thr:.8g}",
                    "under_resolved": str(under_resolved),
                }
            )


if __name__ == "__main__":
    main()
