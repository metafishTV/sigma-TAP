import argparse
import csv
import json
import os
import sys
from statistics import mean


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('src')
    ap.add_argument('--variant', default='baseline')
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.src, newline='')))
    if rows and 'variant' in rows[0]:
        rows = [r for r in rows if r.get('variant', 'baseline') == args.variant]

    target_gamma = ['0', '0.05', '0.2']
    panel = [r for r in rows if r.get('a') == '8' and r.get('gamma') in target_gamma]

    by_key_gamma = {}
    for r in panel:
        key = (r['alpha'], r['mu'], r['m0'])
        by_key_gamma[(key, r['gamma'])] = r

    keys = sorted({k for k, _ in by_key_gamma.keys()})

    matched = []
    for key in keys:
        triplet = []
        ok = True
        for g in target_gamma:
            rr = by_key_gamma.get((key, g))
            if rr is None or rr.get('blowup_step', '') == '':
                ok = False
                break
            triplet.append(rr)
        if ok:
            matched.append((key, triplet))

    out_rows = []
    for key, triplet in matched:
        alpha, mu, m0 = key
        rec = {'variant': args.variant, 'alpha': alpha, 'mu': mu, 'm0': m0}
        for r in triplet:
            g = r['gamma']
            rec[f'blowup_step_gamma_{g}'] = int(r['blowup_step'])
        out_rows.append(rec)

    means = {}
    for g in target_gamma:
        vals = [int(r[f'blowup_step_gamma_{g}']) for r in out_rows]
        means[g] = mean(vals) if vals else None

    monotone_nonincreasing = None
    if all(means[g] is not None for g in target_gamma):
        monotone_nonincreasing = means['0'] >= means['0.05'] >= means['0.2']

    blow0 = {k for k in keys if (by_key_gamma.get((k, '0'), {}).get('blowup_step', '') != '')}
    blow2 = {k for k in keys if (by_key_gamma.get((k, '0.2'), {}).get('blowup_step', '') != '')}

    summary = {
        'variant': args.variant,
        'a': '8',
        'gamma_slices': target_gamma,
        'candidate_cells': len(keys),
        'blowup_cells_gamma_0': len(blow0),
        'blowup_cells_gamma_0.2': len(blow2),
        'recruited_at_gamma_0.2': len(blow2 - blow0),
        'retained_from_gamma_0_to_0.2': len(blow0 & blow2),
        'matched_cells_all_three': len(out_rows),
        'matched_mean_blowup_step': means,
        'matched_monotone_nonincreasing': monotone_nonincreasing,
    }

    writer = csv.DictWriter(
        sys.stdout,
        fieldnames=['variant', 'alpha', 'mu', 'm0', 'blowup_step_gamma_0', 'blowup_step_gamma_0.05', 'blowup_step_gamma_0.2'],
    )
    writer.writeheader()
    for r in out_rows:
        writer.writerow(r)

    out_json = os.environ.get('MATCHED_PANEL_JSON')
    if out_json:
        with open(out_json, 'w') as f:
            json.dump(summary, f, indent=2)


if __name__ == '__main__':
    main()
