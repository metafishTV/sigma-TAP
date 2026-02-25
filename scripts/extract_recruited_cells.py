import argparse
import csv
import json
import math
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from simulator.analysis import find_fixed_point, mstar_isocurve_mu


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('src')
    ap.add_argument('--variant', default='baseline')
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.src, newline='')))
    if rows and 'variant' in rows[0]:
        rows = [r for r in rows if r.get('variant', 'baseline') == args.variant]

    panel = [r for r in rows if r.get('a') == '8' and r.get('gamma') in {'0', '0.2'}]
    by_key_gamma = {}
    for r in panel:
        key = (r['alpha'], r['mu'], r['m0'])
        by_key_gamma[(key, r['gamma'])] = r

    keys = sorted({k for k, _ in by_key_gamma.keys()})

    recruited = []
    for key in keys:
        r0 = by_key_gamma.get((key, '0'))
        r2 = by_key_gamma.get((key, '0.2'))
        if r0 is None or r2 is None:
            continue
        blow0 = r0.get('blowup_step', '') != ''
        blow2 = r2.get('blowup_step', '') != ''
        if (not blow0) and blow2:
            alpha = float(r2['alpha'])
            mu = float(r2['mu'])
            m0 = float(r2['m0'])
            a = float(r2['a'])
            mu_iso = mstar_isocurve_mu(alpha=alpha, a=a, m_star=m0)
            ratio = mu / mu_iso if mu_iso > 0 else float('nan')
            m_star = find_fixed_point(alpha=alpha, mu=mu, a=a)
            m0_over_mstar = (m0 / m_star) if (m_star is not None and m_star > 0) else float('nan')
            abs_log_ratio = abs(math.log(ratio)) if ratio > 0 else float('nan')
            mechanism = 'near_boundary_amplification' if (math.isfinite(abs_log_ratio) and abs_log_ratio < 0.5) else 'discrete_threshold_compensation'
            recruited.append(
                {
                    'variant': args.variant,
                    'alpha': r2['alpha'],
                    'mu': r2['mu'],
                    'm0': r2['m0'],
                    'a': r2['a'],
                    'gamma_cross': '0_to_0.2',
                    'mechanism': mechanism,
                    'blowup_step_gamma_0': '',
                    'blowup_step_gamma_0.2': r2.get('blowup_step', ''),
                    'mu_iso_mstar_eq_m0': f'{mu_iso:.8g}',
                    'mu_over_mu_iso': f'{ratio:.8g}',
                    'abs_log_mu_ratio': f'{abs_log_ratio:.8g}' if math.isfinite(abs_log_ratio) else '',
                    'm_star': '' if m_star is None else f'{m_star:.8g}',
                    'm0_over_m_star': f'{m0_over_mstar:.8g}' if math.isfinite(m0_over_mstar) else '',
                }
            )

    writer = csv.DictWriter(
        sys.stdout,
        fieldnames=[
            'variant', 'alpha', 'mu', 'm0', 'a', 'gamma_cross', 'mechanism',
            'blowup_step_gamma_0', 'blowup_step_gamma_0.2', 'mu_iso_mstar_eq_m0',
            'mu_over_mu_iso', 'abs_log_mu_ratio', 'm_star', 'm0_over_m_star',
        ],
    )
    writer.writeheader()
    for r in recruited:
        writer.writerow(r)

    summary = {
        'variant': args.variant,
        'recruited_n': len(recruited),
        'mechanism_counts': {
            'near_boundary_amplification': sum(1 for r in recruited if r['mechanism'] == 'near_boundary_amplification'),
            'discrete_threshold_compensation': sum(1 for r in recruited if r['mechanism'] == 'discrete_threshold_compensation'),
        },
        'panel': {'a': 8, 'gamma_low': 0.0, 'gamma_high': 0.2},
    }
    out_json = os.environ.get('RECRUITED_JSON')
    if out_json:
        with open(out_json, 'w') as f:
            json.dump(summary, f, indent=2)


if __name__ == '__main__':
    main()
