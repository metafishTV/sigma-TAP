import csv
import json
import sys


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit('usage: python scripts/summarize_recruited_cells.py <recruited_cells_a8.csv> [out.json]')

    src = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else 'outputs/recruited_cells_a8_summary.json'

    with open(src, newline='') as f:
        rows = list(csv.DictReader(f))

    near = sum(1 for r in rows if r.get('mechanism') == 'near_boundary_amplification')
    comp = sum(1 for r in rows if r.get('mechanism') == 'discrete_threshold_compensation')

    summary = {
        'panel': {'a': 8, 'gamma_low': 0.0, 'gamma_high': 0.2},
        'recruited_count': len(rows),
        'mechanism_counts': {
            'near_boundary_amplification': near,
            'discrete_threshold_compensation': comp,
        },
    }

    with open(out, 'w') as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
