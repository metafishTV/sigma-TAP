import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from statistics import mean


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('src')
    ap.add_argument('--variant', default='baseline')
    args = ap.parse_args()

    by_ag = defaultdict(list)
    totals = defaultdict(int)
    with open(args.src, newline='') as f:
        for r in csv.DictReader(f):
            if 'variant' in r and r.get('variant', 'baseline') != args.variant:
                continue
            a = r.get('a', '')
            g = r.get('gamma', '')
            key = (a, g)
            totals[key] += 1
            bs = r.get('blowup_step', '')
            if bs != '':
                by_ag[key].append(int(bs))

    rows = []
    for (a, g), n in sorted(totals.items(), key=lambda kv: (float(kv[0][0]), float(kv[0][1]))):
        vals = by_ag.get((a, g), [])
        rows.append(
            {
                'variant': args.variant,
                'a': a,
                'gamma': g,
                'rows_total': n,
                'rows_with_blowup': len(vals),
                'mean_blowup_step': None if not vals else mean(vals),
                'min_blowup_step': None if not vals else min(vals),
                'max_blowup_step': None if not vals else max(vals),
            }
        )

    writer = csv.DictWriter(
        sys.stdout,
        fieldnames=[
            'variant',
            'a',
            'gamma',
            'rows_total',
            'rows_with_blowup',
            'mean_blowup_step',
            'min_blowup_step',
            'max_blowup_step',
        ],
    )
    writer.writeheader()
    for r in rows:
        writer.writerow(r)

    out_json = {
        'variant': args.variant,
        'rows': rows,
        'note': 'mean_blowup_step computed over rows where blowup_step is present',
    }
    json_path = os.environ.get('BLOWUP_INTERACTION_JSON')
    if json_path:
        with open(json_path, 'w') as jf:
            json.dump(out_json, jf, indent=2)


if __name__ == '__main__':
    main()
