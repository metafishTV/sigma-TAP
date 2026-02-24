import csv
import json
from pathlib import Path

OUT_MD = Path('outputs/report_tables.md')
OUT_CSV = Path('outputs/report_tables.csv')
OUT_TEX = Path('outputs/report_tables.tex')


def load_json(path: str):
    p = Path(path)
    if not p.exists():
        return None
    return json.loads(p.read_text())


def fmt(x, nd=3):
    if isinstance(x, (int, float)):
        return f"{x:.{nd}f}"
    return str(x)


def missing_token(path: str) -> str:
    return f"[artifact missing: {path}]"


def main():
    inf = load_json('outputs/inferential_stats.json')
    if not inf:
        raise SystemExit('Missing required artifact: outputs/inferential_stats.json')

    precursor_path = 'outputs/precursor_longitudinal_summary.json'
    recruited_path = 'outputs/recruited_cells_a8_summary.json'
    matched_path = 'outputs/blowup_matched_panel_a8_summary.json'

    precursor = load_json(precursor_path)
    recruited = load_json(recruited_path)
    matched = load_json(matched_path)

    mt = inf['matched_timing']
    occ = inf['occupancy_shift']
    coef = inf['boundary_coefficients']

    mt_p_text = 'perm p<0.001' if mt['paired_signflip_permutation_p_two_sided'] < 0.001 else f"perm p={fmt(mt['paired_signflip_permutation_p_two_sided'], 4)}"
    occ_p_text = 'perm p<0.001' if occ['paired_signflip_permutation_p_two_sided'] < 0.001 else f"perm p={fmt(occ['paired_signflip_permutation_p_two_sided'], 4)}"

    rows = [
        ['Claim', 'Estimate', 'Uncertainty/Test', 'n'],
        [
            'Matched timing reduction (gamma 0 to 0.2)',
            f"{fmt(mt['mean_difference_gamma0_minus_gamma0.2'], 2)} steps",
            mt_p_text,
            str(mt['matched_blowup_panel_n']),
        ],
        [
            'Occupancy shift (gamma 0.2 minus 0)',
            fmt(occ['difference_gamma0.2_minus_gamma0'], 3),
            f"95% CI [{fmt(occ['difference_bootstrap_95ci']['lower'], 3)}, {fmt(occ['difference_bootstrap_95ci']['upper'], 3)}], {occ_p_text}",
            str(occ['paired_cells_n']),
        ],
        [
            'Occupancy fractions by gamma (0, 0.2)',
            f"{fmt(occ['explosive_or_precursor_fraction_gamma_0'], 3)}, {fmt(occ['explosive_or_precursor_fraction_gamma_0.2'], 3)}",
            'paired panel proportion',
            str(occ['paired_cells_n']),
        ],
        [
            'Mode A vs B sign reversal non-overlap',
            f"log_alpha={coef['sign_reversal_ci_nonoverlap']['log_alpha']}, log_mu={coef['sign_reversal_ci_nonoverlap']['log_mu']}",
            'bootstrap CI separation',
            '-',
        ],
    ]

    if precursor:
        rows.append([
            'Precursor-active long-horizon resolution',
            f"{precursor.get('long_counts', {}).get('explosive', 'NA')}/{precursor.get('rows', 'NA')}",
            f"under-resolved={precursor.get('under_resolved_long_true', 'NA')}",
            str(precursor.get('rows', '-')),
        ])
    else:
        rows.append([
            'Precursor-active long-horizon resolution',
            missing_token(precursor_path),
            'source artifact unavailable',
            '-',
        ])

    if recruited:
        mech = recruited.get('mechanism_counts', {})
        rows.append([
            'Recruited-cell mechanism split',
            f"near-boundary={mech.get('near_boundary_amplification', 'NA')}, compensation={mech.get('discrete_threshold_compensation', 'NA')}",
            'analytic mechanism assignment',
            str(recruited.get('recruited_count', '-')),
        ])
    else:
        rows.append([
            'Recruited-cell mechanism split',
            missing_token(recruited_path),
            'source artifact unavailable',
            '-',
        ])

    if matched:
        mm = matched.get('matched_mean_blowup_step', {})
        rows.append([
            'Matched means by gamma (a=8)',
            f"g0={fmt(mm.get('0'), 2)}, g0.05={fmt(mm.get('0.05'), 2)}, g0.2={fmt(mm.get('0.2'), 2)}",
            f"monotone_nonincreasing={matched.get('matched_monotone_nonincreasing')}",
            str(matched.get('matched_cells_all_three', '-')),
        ])
    else:
        rows.append([
            'Matched means by gamma (a=8)',
            missing_token(matched_path),
            'source artifact unavailable',
            '-',
        ])

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)

    with OUT_MD.open('w') as f:
        f.write('# Dynamic Manuscript Tables\n\n')
        f.write('| ' + ' | '.join(rows[0]) + ' |\n')
        f.write('|---|---|---|---|\n')
        for r in rows[1:]:
            f.write('| ' + ' | '.join(map(str, r)) + ' |\n')

    with OUT_CSV.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    # Minimal LaTeX tabular
    with OUT_TEX.open('w') as f:
        f.write('\\begin{tabular}{p{4cm}p{3cm}p{6cm}p{1cm}}\n\\hline\n')
        f.write(' & '.join(rows[0]) + ' \\\\ \\hline\n')
        for r in rows[1:]:
            escaped = [str(x).replace('_', '\\_') for x in r]
            f.write(' & '.join(escaped) + ' \\\\ \n')
        f.write('\\hline\n\\end{tabular}\n')

    print(f'Wrote {OUT_MD}, {OUT_CSV}, {OUT_TEX}')


if __name__ == '__main__':
    main()
