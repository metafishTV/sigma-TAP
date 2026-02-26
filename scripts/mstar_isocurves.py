import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from simulator.analysis import mstar_isocurve_mu


def main() -> None:
    alphas = np.logspace(np.log10(1e-5), np.log10(1e-2), 30).tolist()
    m_stars = [5.0, 10.0, 20.0, 50.0, 100.0]
    writer = csv.DictWriter(sys.stdout, fieldnames=["m_star", "alpha", "mu"])
    writer.writeheader()
    for m in m_stars:
        for a in alphas:
            writer.writerow({"m_star": f"{m:g}", "alpha": f"{a:.8g}", "mu": f"{mstar_isocurve_mu(a, 8.0, m):.8g}"})


if __name__ == '__main__':
    main()
