import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str], env: dict | None = None, stdout_path: Path | None = None) -> None:
    print('+', ' '.join(cmd))
    if stdout_path is not None:
        with stdout_path.open('w') as f:
            subprocess.run(cmd, check=True, env=env, stdout=f)
    else:
        subprocess.run(cmd, check=True, env=env)



def main() -> None:
    ap = argparse.ArgumentParser(description='Run inferential -> report tables -> figures -> claim audit')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--n-boot', type=int, default=200)
    ap.add_argument('--n-perm', type=int, default=500)
    ap.add_argument('--n-boot-coef', type=int, default=50)
    ap.add_argument('--camera-ready', action='store_true', help='Use camera-ready resampling defaults (n_boot=1000, n_perm=2000, n_boot_coef=500)')
    args = ap.parse_args()

    if args.camera_ready:
        args.n_boot = 1000
        args.n_perm = 2000
        args.n_boot_coef = 500

    run([
        sys.executable,
        'scripts/inferential_stats.py',
        '--mode', 'all',
        '--seed', str(args.seed),
        '--n-boot', str(args.n_boot),
        '--n-perm', str(args.n_perm),
        '--n-boot-coef', str(args.n_boot_coef),
    ])

    run([
        sys.executable,
        'scripts/validate_declared_values.py',
        '--inferential', 'outputs/inferential_stats.json',
        '--declared', 'config/manuscript_declared_values.json',
        '--config', 'config/audit_config.json',
        '--out', 'outputs/declared_values_validation_report.json',
    ])

    # Regenerate precursor longitudinal summary + trajectory artifact (for figure rendering without live rerun).
    env = os.environ.copy()
    env['PRECURSOR_TRAJ_CSV'] = 'outputs/precursor_longitudinal_trajectory.csv'
    run(
        [sys.executable, 'scripts/followup_precursor_longitudinal.py', 'outputs/sweep_sigma_feedback.csv'],
        env=env,
        stdout_path=Path('outputs/precursor_longitudinal.csv'),
    )

    run([sys.executable, 'scripts/summarize_precursor_longitudinal.py', 'outputs/precursor_longitudinal.csv', 'outputs/precursor_longitudinal_summary.json'])

    run([sys.executable, 'scripts/summarize_recruited_cells.py', 'outputs/recruited_cells_a8.csv', 'outputs/recruited_cells_a8_summary.json'])
    run([sys.executable, 'scripts/summarize_blowup_matched_panel.py', 'outputs/blowup_matched_panel_a8.csv', 'outputs/blowup_matched_panel_a8_summary.json'])

    # Sensitivity analysis + publication figures.
    run([sys.executable, 'scripts/sensitivity_analysis.py'])
    run(
        [sys.executable, 'scripts/sweep_variants.py'],
        stdout_path=Path('outputs/variant_comparison.csv'),
    )
    run([sys.executable, 'scripts/generate_figures.py'])

    # Metathetic ensemble long-run diagnostics (exploratory).
    run([sys.executable, 'scripts/longrun_diagnostics.py', '--seed', str(args.seed)])

    # Evidence report: mechanistic/functional interpretation + tier validation (T3.1 + T3.2).
    run([sys.executable, 'scripts/build_evidence_report.py'])

    # Claim audit (T4.2): cross-reference CLAIMS.md <-> annotations <-> disk.
    run([sys.executable, 'scripts/audit_claims.py'])

    report_path = Path('outputs/claim_audit_report.json')
    if report_path.exists():
        report = json.loads(report_path.read_text())
        if not report.get('all_pass', False):
            raise SystemExit('Claim audit failed; see outputs/claim_audit_report.json')
        print('Claim audit passed (all_pass=true)')

    # Run manifest (T4.1): record git state, configs, artifact inventory.
    run([
        sys.executable, 'scripts/generate_manifest.py',
        '--seed', str(args.seed),
        '--n-boot', str(args.n_boot),
        '--n-perm', str(args.n_perm),
        '--n-boot-coef', str(args.n_boot_coef),
    ])
    print('Pipeline completed successfully.')


if __name__ == '__main__':
    main()
