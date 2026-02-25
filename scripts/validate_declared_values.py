import argparse
import json
from pathlib import Path

DEFAULT_COEF_TOL = 0.07


def within_tol(observed: float, declared: float, tol: float) -> bool:
    return abs(observed - declared) <= tol


def main() -> None:
    ap = argparse.ArgumentParser(description='Validate declared manuscript values against inferential artifact coefficients.')
    ap.add_argument('--inferential', default='outputs/inferential_stats.json')
    ap.add_argument('--declared', default='config/manuscript_declared_values.json')
    ap.add_argument('--config', default='config/audit_config.json')
    ap.add_argument('--out', default='outputs/declared_values_validation_report.json')
    args = ap.parse_args()

    inferential = Path(args.inferential)
    declared_path = Path(args.declared)
    config_path = Path(args.config)
    out = Path(args.out)

    if not inferential.exists():
        raise SystemExit(f'Inferential artifact not found: {inferential}')
    if not declared_path.exists():
        raise SystemExit(f'Declared values file not found: {declared_path}')
    if not config_path.exists():
        raise SystemExit(f'Audit config file not found: {config_path}')

    stats = json.loads(inferential.read_text())
    declared = json.loads(declared_path.read_text())
    config = json.loads(config_path.read_text())
    tol = float(config.get('coef_tolerance', DEFAULT_COEF_TOL))
    coeffs = stats['boundary_coefficients']

    checks: list[dict] = []
    for mode, fields in declared.items():
        if mode not in coeffs:
            checks.append({
                'name': f'{mode}._mode_present',
                'pass': False,
                'error': f'Missing mode in inferential stats: {mode}',
            })
            continue
        for coef, target in fields.items():
            if coef not in coeffs[mode]:
                checks.append({
                    'name': f'{mode}.{coef}._coef_present',
                    'pass': False,
                    'error': f'Missing coefficient in inferential stats: {mode}.{coef}',
                })
                continue
            obs = coeffs[mode][coef]
            triples = [
                ('mean', obs['mean'], target['mean']),
                ('ci95.lower', obs['ci95']['lower'], target['lower']),
                ('ci95.upper', obs['ci95']['upper'], target['upper']),
            ]
            for label, observed, stated in triples:
                checks.append({
                    'name': f'{mode}.{coef}.{label}',
                    'artifact_value': observed,
                    'declared_value': stated,
                    'tolerance': tol,
                    'pass': within_tol(observed, stated, tol),
                })

    result = {
        'inferential': str(inferential),
        'declared': str(declared_path),
        'config': str(config_path),
        'coef_tolerance': tol,
        'check_count': len(checks),
        'all_pass': all(c.get('pass', False) for c in checks),
        'checks': checks,
    }
    out.write_text(json.dumps(result, indent=2))

    if not result['all_pass']:
        failed = [c for c in checks if not c.get('pass', False)]
        msg_lines = ['Declared-values validation failed:']
        for f in failed:
            if 'error' in f:
                msg_lines.append(f"- {f['error']}")
            else:
                msg_lines.append(
                    f"- {f['name']} drift: observed={f['artifact_value']:.6f}, declared={f['declared_value']:.6f}, tol={tol:.3f}"
                )
        raise SystemExit('\n'.join(msg_lines))

    print(f'Declared-values validation passed ({len(checks)} checks, tol={tol:.3f}).')


if __name__ == '__main__':
    main()
