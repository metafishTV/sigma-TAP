import argparse
import json
import re
from pathlib import Path

DEFAULT_COEF_TOL = 0.07


def fmt(x: float, nd: int) -> str:
    return f"{x:.{nd}f}"


def contains_with_anchor(text: str, anchor: str, needle: str) -> bool:
    pattern = re.compile(re.escape(anchor) + r".*?" + re.escape(needle), re.S)
    return bool(pattern.search(text))


def within_tol(observed: float, stated: float, tol: float) -> bool:
    return abs(observed - stated) <= tol


def build_checks(text: str, stats: dict, declared: dict, coef_tol: float) -> list[dict]:
    checks: list[dict] = []

    mt = stats['matched_timing']
    occ = stats['occupancy_shift']
    bc = stats['boundary_coefficients']

    matched_p = mt['paired_signflip_permutation_p_two_sided']
    checks.append({
        'name': 'matched_p_threshold_claim',
        'artifact_value': matched_p,
        'expected': 'p < 0.001 when matched p < 0.001',
        'pass': (matched_p < 0.001) and contains_with_anchor(text, 'Mean paired reduction', 'p < 0.001'),
    })
    checks.append({
        'name': 'matched_mean_difference_text_claim',
        'artifact_value': mt['mean_difference_gamma0_minus_gamma0.2'],
        'expected': fmt(mt['mean_difference_gamma0_minus_gamma0.2'], 2),
        'pass': contains_with_anchor(text, 'Mean paired reduction from', fmt(mt['mean_difference_gamma0_minus_gamma0.2'], 2)),
    })

    occ_p = occ['paired_signflip_permutation_p_two_sided']
    occ_p_rounded = fmt(occ_p, 3)
    checks.append({
        'name': 'occupancy_p_value_claim',
        'artifact_value': occ_p,
        'expected': f'p = {occ_p_rounded} (3dp)',
        'pass': contains_with_anchor(text, 'paired sign-flip permutation on binary outcomes yields', f'p = {occ_p_rounded}'),
    })
    checks.append({
        'name': 'occupancy_difference_text_claim',
        'artifact_value': occ['difference_gamma0.2_minus_gamma0'],
        'expected': fmt(occ['difference_gamma0.2_minus_gamma0'], 3),
        'pass': contains_with_anchor(text, 'difference $=', fmt(occ['difference_gamma0.2_minus_gamma0'], 3)),
    })

    ci_l = fmt(occ['difference_bootstrap_95ci']['lower'], 3)
    ci_u = fmt(occ['difference_bootstrap_95ci']['upper'], 3)
    checks.append({
        'name': 'occupancy_ci_claim',
        'artifact_value': occ['difference_bootstrap_95ci'],
        'expected': f'[{ci_l}, {ci_u}]',
        'pass': contains_with_anchor(text, 'bootstrap 95% CI is', f'[{ci_l}, {ci_u}]'),
    })

    g0 = fmt(occ['explosive_or_precursor_fraction_gamma_0'], 3)
    g2 = fmt(occ['explosive_or_precursor_fraction_gamma_0.2'], 3)
    checks.append({
        'name': 'occupancy_fraction_gamma0_claim',
        'artifact_value': occ['explosive_or_precursor_fraction_gamma_0'],
        'expected': g0,
        'pass': contains_with_anchor(text, 'explosive-or-precursor fraction rises from', g0),
    })
    checks.append({
        'name': 'occupancy_fraction_gamma0.2_claim',
        'artifact_value': occ['explosive_or_precursor_fraction_gamma_0.2'],
        'expected': g2,
        'pass': contains_with_anchor(text, 'explosive-or-precursor fraction rises from', g2),
    })

    checks.append({
        'name': 'matched_panel_n_claim',
        'artifact_value': mt['matched_blowup_panel_n'],
        'expected': f"n = {mt['matched_blowup_panel_n']}",
        'pass': contains_with_anchor(text, 'sign-flip permutation test', f"n = {mt['matched_blowup_panel_n']}"),
    })

    checks.append({
        'name': 'matched_mean_gamma0_claim',
        'artifact_value': mt['mean_blowup_step_gamma_0'],
        'expected': fmt(mt['mean_blowup_step_gamma_0'], 2),
        'pass': contains_with_anchor(text, 'mean blow-up step is', fmt(mt['mean_blowup_step_gamma_0'], 2)),
    })
    checks.append({
        'name': 'matched_mean_gamma0.2_claim',
        'artifact_value': mt['mean_blowup_step_gamma_0.2'],
        'expected': fmt(mt['mean_blowup_step_gamma_0.2'], 2),
        'pass': contains_with_anchor(text, 'mean blow-up step is', fmt(mt['mean_blowup_step_gamma_0.2'], 2)),
    })

    nonoverlap = bc['sign_reversal_ci_nonoverlap']
    checks.append({
        'name': 'sign_reversal_nonoverlap_log_alpha_flag',
        'artifact_value': nonoverlap['log_alpha'],
        'expected': True,
        'pass': bool(nonoverlap['log_alpha']),
    })
    checks.append({
        'name': 'sign_reversal_nonoverlap_log_mu_flag',
        'artifact_value': nonoverlap['log_mu'],
        'expected': True,
        'pass': bool(nonoverlap['log_mu']),
    })

    mode_a = bc['mode_a_2d_bootstrap']
    mode_b = bc['mode_b_2d_bootstrap']
    checks.append({
        'name': 'sign_reversal_nonoverlap_log_alpha_geometry',
        'artifact_value': {
            'mode_a_upper': mode_a['log_alpha']['ci95']['upper'],
            'mode_b_lower': mode_b['log_alpha']['ci95']['lower'],
        },
        'expected': 'mode_a_upper < mode_b_lower',
        'pass': mode_a['log_alpha']['ci95']['upper'] < mode_b['log_alpha']['ci95']['lower'],
    })
    checks.append({
        'name': 'sign_reversal_nonoverlap_log_mu_geometry',
        'artifact_value': {
            'mode_a_lower': mode_a['log_mu']['ci95']['lower'],
            'mode_b_upper': mode_b['log_mu']['ci95']['upper'],
        },
        'expected': 'mode_a_lower > mode_b_upper',
        'pass': mode_a['log_mu']['ci95']['lower'] > mode_b['log_mu']['ci95']['upper'],
    })

    checks.append({
        'name': 'manuscript_nonoverlap_phrase_present',
        'artifact_value': 'non-overlapping confidence intervals',
        'expected': 'phrase present',
        'pass': 'non-overlapping confidence intervals' in text,
    })

    # Ensure manuscript text was actually updated for coefficient CI prose.
    mode_a_2d_anchor = 'Under Mode A, the 2D logistic fit yields'
    for coef in ['log_alpha', 'log_mu']:
        lower = fmt(declared['mode_a_2d_bootstrap'][coef]['lower'], 2)
        upper = fmt(declared['mode_a_2d_bootstrap'][coef]['upper'], 2)
        checks.append({
            'name': f'manuscript_mode_a_2d_{coef}_ci_lower_text',
            'expected': lower,
            'pass': contains_with_anchor(text, mode_a_2d_anchor, lower),
        })
        checks.append({
            'name': f'manuscript_mode_a_2d_{coef}_ci_upper_text',
            'expected': upper,
            'pass': contains_with_anchor(text, mode_a_2d_anchor, upper),
        })

    mode_b_2d_anchor = 'Under Mode B, the 2D fit yields'
    for coef in ['log_alpha', 'log_mu']:
        lower = fmt(declared['mode_b_2d_bootstrap'][coef]['lower'], 2)
        upper = fmt(declared['mode_b_2d_bootstrap'][coef]['upper'], 2)
        checks.append({
            'name': f'manuscript_mode_b_2d_{coef}_ci_lower_text',
            'expected': lower,
            'pass': contains_with_anchor(text, mode_b_2d_anchor, lower),
        })
        checks.append({
            'name': f'manuscript_mode_b_2d_{coef}_ci_upper_text',
            'expected': upper,
            'pass': contains_with_anchor(text, mode_b_2d_anchor, upper),
        })

    mode_b_3d_anchor = 'The Mode B 3D fit adding $\\log M_0$ yields'
    for coef in ['log_alpha', 'log_mu', 'log_m0']:
        lower = fmt(declared['mode_b_3d_bootstrap'][coef]['lower'], 2)
        upper = fmt(declared['mode_b_3d_bootstrap'][coef]['upper'], 2)
        checks.append({
            'name': f'manuscript_mode_b_3d_{coef}_ci_lower_text',
            'expected': lower,
            'pass': contains_with_anchor(text, mode_b_3d_anchor, lower),
        })
        checks.append({
            'name': f'manuscript_mode_b_3d_{coef}_ci_upper_text',
            'expected': upper,
            'pass': contains_with_anchor(text, mode_b_3d_anchor, upper),
        })

    for mode, fields in declared.items():
        for key, target in fields.items():
            obs = bc[mode][key]
            checks.append({
                'name': f'{mode}_{key}_mean_tol',
                'artifact_value': obs['mean'],
                'manuscript_value': target['mean'],
                'tolerance': coef_tol,
                'pass': within_tol(obs['mean'], target['mean'], coef_tol),
            })
            checks.append({
                'name': f'{mode}_{key}_ci_lower_tol',
                'artifact_value': obs['ci95']['lower'],
                'manuscript_value': target['lower'],
                'tolerance': coef_tol,
                'pass': within_tol(obs['ci95']['lower'], target['lower'], coef_tol),
            })
            checks.append({
                'name': f'{mode}_{key}_ci_upper_tol',
                'artifact_value': obs['ci95']['upper'],
                'manuscript_value': target['upper'],
                'tolerance': coef_tol,
                'pass': within_tol(obs['ci95']['upper'], target['upper'], coef_tol),
            })

    return checks


def main() -> None:
    ap = argparse.ArgumentParser(description='Audit manuscript quantitative claims against inferential artifact.')
    ap.add_argument('--manuscript', default='MANUSCRIPT_SYNTHESIS_FINAL.md')
    ap.add_argument('--inferential', default='outputs/inferential_stats.json')
    ap.add_argument('--declared', default='config/manuscript_declared_values.json')
    ap.add_argument('--config', default='config/audit_config.json')
    ap.add_argument('--out', default='outputs/claim_audit_report.json')
    args = ap.parse_args()

    manuscript = Path(args.manuscript)
    inferential = Path(args.inferential)
    declared_path = Path(args.declared)
    config_path = Path(args.config)
    out = Path(args.out)

    if not manuscript.exists():
        raise SystemExit(f'Manuscript file not found: {manuscript}')
    if not inferential.exists():
        raise SystemExit(f'Inferential artifact not found: {inferential}')
    if not declared_path.exists():
        raise SystemExit(f'Declared values file not found: {declared_path}')
    if not config_path.exists():
        raise SystemExit(f'Audit config file not found: {config_path}')

    text = manuscript.read_text()
    stats = json.loads(inferential.read_text())
    declared = json.loads(declared_path.read_text())
    config = json.loads(config_path.read_text())
    coef_tol = float(config.get('coef_tolerance', DEFAULT_COEF_TOL))

    checks = build_checks(text, stats, declared, coef_tol)
    result = {
        'manuscript': str(manuscript),
        'inferential': str(inferential),
        'declared': str(declared_path),
        'config': str(config_path),
        'coef_tolerance': coef_tol,
        'check_count': len(checks),
        'all_pass': all(c['pass'] for c in checks),
        'checks': checks,
    }

    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
