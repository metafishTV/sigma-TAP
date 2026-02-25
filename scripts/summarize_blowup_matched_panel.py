import csv
import json
import sys
from statistics import mean


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit('usage: python scripts/summarize_blowup_matched_panel.py <blowup_matched_panel_a8.csv> [out.json]')

    src = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else 'outputs/blowup_matched_panel_a8_summary.json'

    with open(src, newline='') as f:
        rows = list(csv.DictReader(f))

    g0 = [int(r['blowup_step_gamma_0']) for r in rows]
    g05 = [int(r['blowup_step_gamma_0.05']) for r in rows]
    g02 = [int(r['blowup_step_gamma_0.2']) for r in rows]

    summary = {
        'a': '8',
        'gamma_slices': ['0', '0.05', '0.2'],
        'matched_cells_all_three': len(rows),
        'matched_mean_blowup_step': {
            '0': mean(g0) if g0 else None,
            '0.05': mean(g05) if g05 else None,
            '0.2': mean(g02) if g02 else None,
        },
        'matched_monotone_nonincreasing': (mean(g0) >= mean(g05) >= mean(g02)) if rows else None,
    }

    with open(out, 'w') as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
