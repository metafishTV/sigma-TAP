import csv
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from simulator.analysis import fit_explosive_logistic_boundary_3d


def main(path: str = 'outputs/sweep_alpha_mu.csv') -> None:
    rows = []
    with open(path) as f:
        rows.extend(csv.DictReader(f))

    fit = fit_explosive_logistic_boundary_3d(rows, explosive_labels={"explosive", "precursor-active"})
    print(json.dumps(fit, indent=2, sort_keys=True))


if __name__ == '__main__':
    p = sys.argv[1] if len(sys.argv) > 1 else 'outputs/sweep_alpha_mu.csv'
    main(p)
