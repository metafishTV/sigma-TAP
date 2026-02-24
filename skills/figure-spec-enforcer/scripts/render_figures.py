import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_spec(path: Path):
    return json.loads(path.read_text())


def load_json(path: Path):
    return json.loads(path.read_text())


def load_csv(path: Path):
    with path.open(newline='') as f:
        return list(csv.DictReader(f))


def required_path(paths: dict, key: str) -> Path:
    p = Path(paths[key])
    if not p.exists():
        raise SystemExit(f'Missing required artifact: {p}')
    return p


def optional_path(paths: dict, key: str, warnings: list[str]) -> Path | None:
    raw = paths.get(key)
    if not raw:
        warnings.append(f'Skipping dependent figure(s): path key "{key}" not present in spec.')
        return None
    p = Path(raw)
    if not p.exists():
        warnings.append(f'Skipping dependent figure(s): artifact missing at {p}.')
        return None
    return p


def validate(spec):
    required = ['style', 'paths']
    for k in required:
        if k not in spec:
            raise SystemExit(f'Missing spec key: {k}')
    _ = required_path(spec['paths'], 'inferential')


def figure_matched_vs_unmatched(inferential: dict, matched_summary: dict, sweep_sigma: list[dict], out_path: Path, style: dict, palette: list[str]):
    target_gamma = ['0', '0.05', '0.2']
    unmatched = {}
    for g in target_gamma:
        vals = [
            float(r['blowup_step'])
            for r in sweep_sigma
            if r.get('a') == '8' and r.get('gamma') == g and r.get('blowup_step', '') != ''
        ]
        unmatched[g] = (sum(vals) / len(vals)) if vals else None

    mm = matched_summary['matched_mean_blowup_step']
    matched = {'0': float(inferential['matched_timing']['mean_blowup_step_gamma_0']), '0.05': float(mm['0.05']), '0.2': float(inferential['matched_timing']['mean_blowup_step_gamma_0.2'])}

    x = [0.0, 0.05, 0.2]
    y_unmatched = [unmatched['0'], unmatched['0.05'], unmatched['0.2']]
    y_matched = [matched['0'], matched['0.05'], matched['0.2']]

    plt.figure(figsize=(5, 3.4), dpi=style['dpi'])
    plt.plot(x, y_unmatched, marker='o', color=palette[1], label='Unmatched means')
    plt.plot(x, y_matched, marker='o', color=palette[0], label='Matched means')
    plt.xticks(x, ['0', '0.05', '0.2'])
    plt.xlabel('gamma')
    plt.ylabel('mean blow-up step')
    plt.title('Composition artifact: unmatched vs matched means (a=8)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def figure_a_gating(sweep_sigma: list[dict], out_path: Path, style: dict, palette: list[str]):
    a_vals = [2, 4, 8, 16]
    counts = []
    total_per_a = []
    for a in a_vals:
        aa = str(a)
        rows = [r for r in sweep_sigma if r.get('a') == aa]
        total = len(rows)
        # manuscript positive class is {explosive, precursor-active}
        explosive_or_precursor = sum(1 for r in rows if r.get('regime') in {'explosive', 'precursor-active'})
        counts.append(explosive_or_precursor)
        total_per_a.append(total)

    frac = [(c / t) if t else 0 for c, t in zip(counts, total_per_a)]

    fig, ax1 = plt.subplots(figsize=(5.2, 3.4), dpi=style['dpi'])
    ax1.bar([str(a) for a in a_vals], counts, color=palette[2], alpha=0.8)
    ax1.set_xlabel('a')
    ax1.set_ylabel('explosive-or-precursor count', color=palette[2])
    ax1.tick_params(axis='y', labelcolor=palette[2])

    ax2 = ax1.twinx()
    ax2.plot([str(a) for a in a_vals], frac, color=palette[0], marker='o')
    ax2.set_ylabel('explosive-or-precursor fraction', color=palette[0])
    ax2.tick_params(axis='y', labelcolor=palette[0])

    plt.title('a-gating of positive-class occupancy (sigma panel aggregate)')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)


def figure_high_gamma_dual(sweep_gamma: list[dict], out_path: Path, style: dict, palette: list[str]):
    gammas = [0.5, 1.0, 2.0, 5.0]
    counts = []
    means = []
    for g in gammas:
        rows = [
            r for r in sweep_gamma
            if abs(float(r.get('a', 'nan')) - 8.0) < 1e-12
            and abs(float(r.get('m0', 'nan')) - 20.0) < 1e-12
            and abs(float(r.get('gamma', 'nan')) - g) < 1e-12
        ]
        explosive_rows = [r for r in rows if r.get('blowup_step', '') != '']
        if not rows:
            raise SystemExit(f'No rows found for high-gamma panel at gamma={g}; check sweep_gamma_threshold artifact formatting.')
        counts.append(len(explosive_rows))
        vals = [float(r['blowup_step']) for r in explosive_rows]
        means.append((sum(vals) / len(vals)) if vals else float('nan'))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.0, 3.2), dpi=style['dpi'])
    xlabels = [str(g) for g in gammas]

    ax1.plot(xlabels, counts, color=palette[1], marker='o')
    ax1.set_title('Explosive count')
    ax1.set_xlabel('gamma')
    ax1.set_ylabel('count (out of 100)')

    ax2.plot(xlabels, means, color=palette[0], marker='o')
    ax2.set_title('Mean blow-up step')
    ax2.set_xlabel('gamma')
    ax2.set_ylabel('steps')

    fig.suptitle('High-gamma probe: occupancy vs timing')
    fig.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)


def figure_precursor_trajectory(precursor_traj: list[dict], out_path: Path, style: dict, palette: list[str]):
    if not precursor_traj:
        raise SystemExit('Precursor trajectory artifact is empty.')

    # use first cell_id in artifact, no live simulation rerun
    first_cell = precursor_traj[0]['cell_id']
    rows = [r for r in precursor_traj if r.get('cell_id') == first_cell]
    rows.sort(key=lambda r: int(r['t']))

    t = [int(r['t']) for r in rows]
    m = [float(r['M_t']) for r in rows]
    xi = [float(r['Xi_t']) for r in rows]

    fig, ax1 = plt.subplots(figsize=(5.6, 3.4), dpi=style['dpi'])
    ax1.plot(t, m, color=palette[0], label='M_t')
    ax1.set_xlabel('step')
    ax1.set_ylabel('M_t', color=palette[0])
    ax1.tick_params(axis='y', labelcolor=palette[0])

    ax2 = ax1.twinx()
    ax2.plot(t, xi, color=palette[1], label='Xi_t')
    ax2.set_ylabel('Xi_t', color=palette[1])
    ax2.tick_params(axis='y', labelcolor=palette[1])

    plt.title('Representative precursor-active trajectory')
    fig.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)


def figure_recruited_scatter(recruited_rows: list[dict], out_path: Path, style: dict, palette: list[str]):
    x = [float(r['m0_over_m_star']) for r in recruited_rows]
    y = [abs(float(r['abs_log_mu_ratio'])) for r in recruited_rows]
    mech = [r['mechanism'] for r in recruited_rows]

    colors = [palette[0] if m == 'near_boundary_amplification' else palette[1] for m in mech]
    plt.figure(figsize=(4.6, 3.4), dpi=style['dpi'])
    plt.scatter(x, y, c=colors)
    for i in range(len(recruited_rows)):
        plt.annotate(str(i + 1), (x[i], y[i]), fontsize=7)
    plt.axvline(1.0, color='#888888', linewidth=0.8, linestyle='--')
    plt.axhline(0.2, color='#888888', linewidth=0.8, linestyle='--')
    plt.xlabel('M0 / M*')
    plt.ylabel('|log(mu / mu_iso)|')
    plt.title('Recruited-cell mechanism geometry')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def figure_modec_overlay(mode_b_rows: list[dict], isocurves_rows: list[dict], recruited_rows: list[dict], out_path: Path, style: dict, palette: list[str]):
    subset = [r for r in mode_b_rows if r.get('m0') == '20']
    x_exp = [float(r['alpha']) for r in subset if r.get('regime') in {'explosive', 'precursor-active'}]
    y_exp = [float(r['mu']) for r in subset if r.get('regime') in {'explosive', 'precursor-active'}]
    x_plt = [float(r['alpha']) for r in subset if r.get('regime') == 'plateau']
    y_plt = [float(r['mu']) for r in subset if r.get('regime') == 'plateau']

    iso20 = [(float(r['alpha']), float(r['mu'])) for r in isocurves_rows if r.get('m_star') in {'20', '20.0'}]
    iso20.sort(key=lambda z: z[0])

    plt.figure(figsize=(5.0, 4.0), dpi=style['dpi'])
    plt.scatter(x_plt, y_plt, s=12, alpha=0.5, color='#888888', label='plateau')
    plt.scatter(x_exp, y_exp, s=12, alpha=0.5, color=palette[0], label='explosive/precursor')
    if iso20:
        plt.plot([a for a, _ in iso20], [m for _, m in iso20], color=palette[1], linewidth=1.5, label='Mode C isocurve (M*=20)')

    comp = [r for r in recruited_rows if r.get('mechanism') == 'discrete_threshold_compensation']
    if comp:
        plt.scatter([float(r['alpha']) for r in comp], [float(r['mu']) for r in comp], color='red', s=35, marker='x', label='compensation cells')

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('alpha')
    plt.ylabel('mu')
    plt.title('Mode C baseline vs Mode B outcomes (m0=20)')
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def figure_occupancy_shift_bar(inferential: dict, out_path: Path, style: dict, palette: list[str]):
    occ = inferential['occupancy_shift']
    diff = occ['difference_gamma0.2_minus_gamma0']
    lower = occ['difference_bootstrap_95ci']['lower']
    upper = occ['difference_bootstrap_95ci']['upper']

    plt.figure(figsize=(4, 3), dpi=style['dpi'])
    plt.bar(['Δ occupancy'], [diff], color=palette[1])
    plt.errorbar([0], [diff], yerr=[[diff - lower], [upper - diff]], fmt='none', ecolor=style['ci_color'], capsize=4)
    plt.ylabel('difference')
    plt.title('Occupancy shift (gamma 0.2 - 0)')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--spec', default='skills/figure-spec-enforcer/assets/figure_spec.json')
    ap.add_argument('--validate-only', action='store_true')
    args = ap.parse_args()

    spec = load_spec(Path(args.spec))
    validate(spec)
    if args.validate_only:
        print('Spec/artifacts validation passed')
        return

    style = spec['style']
    paths = spec['paths']
    inferential = load_json(required_path(paths, 'inferential'))

    fig_dir = Path(paths['figure_dir'])
    fig_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({'font.size': style['font_size']})
    palette = style['palette']

    warnings: list[str] = []

    matched_summary_path = optional_path(paths, 'matched_summary', warnings)
    sweep_sigma_path = optional_path(paths, 'sweep_sigma_feedback', warnings)
    sweep_gamma_path = optional_path(paths, 'sweep_gamma_threshold', warnings)
    precursor_traj_path = optional_path(paths, 'precursor_trajectory', warnings)
    recruited_cells_path = optional_path(paths, 'recruited_cells', warnings)
    mode_b_path = optional_path(paths, 'sweep_mode_b', warnings)
    isocurves_path = optional_path(paths, 'mstar_isocurves', warnings)

    # always-run figures
    figure_occupancy_shift_bar(inferential, fig_dir / 'occupancy_shift_bar.png', style, palette)

    a = inferential['boundary_coefficients']['mode_a_2d_bootstrap']
    b = inferential['boundary_coefficients']['mode_b_2d_bootstrap']
    labels = ['log_alpha', 'log_mu']
    a_mean = [a[k]['mean'] for k in labels]
    b_mean = [b[k]['mean'] for k in labels]
    a_err = [[a[k]['mean'] - a[k]['ci95']['lower'] for k in labels], [a[k]['ci95']['upper'] - a[k]['mean'] for k in labels]]
    b_err = [[b[k]['mean'] - b[k]['ci95']['lower'] for k in labels], [b[k]['ci95']['upper'] - b[k]['mean'] for k in labels]]

    x = range(len(labels))
    plt.figure(figsize=(5, 3), dpi=style['dpi'])
    plt.errorbar([i - 0.1 for i in x], a_mean, yerr=a_err, fmt='o', color=palette[2], label='Mode A')
    plt.errorbar([i + 0.1 for i in x], b_mean, yerr=b_err, fmt='o', color=palette[0], label='Mode B')
    plt.xticks(list(x), labels)
    plt.axhline(0, color='#888888', linewidth=0.8)
    plt.title('Boundary coefficient CIs (2D)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / 'coefficient_ci_modes.png')
    plt.close()

    b3 = inferential['boundary_coefficients']['mode_b_3d_bootstrap']
    plt.figure(figsize=(5.6, 3.2), dpi=style['dpi'])
    x2 = [0, 1, 2]
    mode_b3_labels = ['log_alpha', 'log_mu', 'log_m0']
    means = [b3[k]['mean'] for k in mode_b3_labels]
    lows = [b3[k]['ci95']['lower'] for k in mode_b3_labels]
    highs = [b3[k]['ci95']['upper'] for k in mode_b3_labels]
    yerr = [[m - l for m, l in zip(means, lows)], [h - m for m, h in zip(means, highs)]]
    plt.errorbar(x2, means, yerr=yerr, fmt='o', color=palette[1], label='Mode B 3D')
    plt.axhline(0, color='#888888', linewidth=0.8)
    plt.xticks(x2, mode_b3_labels)
    plt.title('Initialization sensitivity (Mode B 3D)')
    plt.tight_layout()
    plt.savefig(fig_dir / 'coefficient_ci_mode_b_3d.png')
    plt.close()

    if matched_summary_path and sweep_sigma_path:
        ms = load_json(matched_summary_path)
        sweep_sigma = load_csv(sweep_sigma_path)
        figure_matched_vs_unmatched(inferential, ms, sweep_sigma, fig_dir / 'composition_artifact_matched_vs_unmatched.png', style, palette)
        mm = ms['matched_mean_blowup_step']
        plt.figure(figsize=(4, 3), dpi=style['dpi'])
        plt.plot([0.0, 0.05, 0.2], [float(mm['0']), float(mm['0.05']), float(mm['0.2'])], marker='o', color=palette[0])
        plt.xticks([0.0, 0.05, 0.2], ['0', '0.05', '0.2'])
        plt.xlabel('gamma')
        plt.ylabel('mean blow-up step')
        plt.title('Matched panel timing')
        plt.tight_layout()
        plt.savefig(fig_dir / 'matched_blowup_by_gamma.png')
        plt.close()
    else:
        warnings.append('Skipped matched/composition figures due to missing matched_summary or sweep_sigma_feedback artifact.')

    if sweep_sigma_path:
        figure_a_gating(load_csv(sweep_sigma_path), fig_dir / 'a_gating_explosive_structure.png', style, palette)
    else:
        warnings.append('Skipped a-gating figure due to missing sweep_sigma_feedback artifact.')

    if sweep_gamma_path:
        figure_high_gamma_dual(load_csv(sweep_gamma_path), fig_dir / 'high_gamma_timing_vs_occupancy.png', style, palette)
    else:
        warnings.append('Skipped high-gamma dual figure due to missing sweep_gamma_threshold artifact.')

    if precursor_traj_path:
        figure_precursor_trajectory(load_csv(precursor_traj_path), fig_dir / 'precursor_active_representative_trajectory.png', style, palette)
    else:
        warnings.append('Skipped precursor trajectory figure due to missing precursor_trajectory artifact.')

    recruited_rows = None
    if recruited_cells_path:
        recruited_rows = load_csv(recruited_cells_path)
        figure_recruited_scatter(recruited_rows, fig_dir / 'recruited_cells_mechanism_scatter.png', style, palette)
    else:
        warnings.append('Skipped recruited-cell mechanism scatter due to missing recruited_cells artifact.')

    if mode_b_path and isocurves_path and recruited_rows is not None:
        figure_modec_overlay(load_csv(mode_b_path), load_csv(isocurves_path), recruited_rows, fig_dir / 'modec_overlay_modeb_phase_scatter.png', style, palette)
    else:
        warnings.append('Skipped Mode C overlay figure due to missing sweep_mode_b, mstar_isocurves, or recruited_cells artifact.')

    if warnings:
        print('Figure generation completed with warnings:')
        for w in warnings:
            print('-', w)
    else:
        print('Figure generation completed with all figures.')

    print(f'Wrote figures to {fig_dir}')


if __name__ == '__main__':
    main()
