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


def maybe_run(cmd: list[str], *, required: bool, reason: str) -> bool:
    target = Path(cmd[1]) if len(cmd) > 1 else None
    if target is not None and not target.exists():
        msg = f"Skipping missing tool: {target} ({reason})"
        if required:
            raise SystemExit(msg)
        print('!', msg)
        return False
    run(cmd)
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description='Run inferential -> report tables -> figures -> claim audit')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--n-boot', type=int, default=200)
    ap.add_argument('--n-perm', type=int, default=500)
    ap.add_argument('--n-boot-coef', type=int, default=50)
    ap.add_argument('--strict-tools', action='store_true', help='Fail if optional builder/figure/audit scripts are missing')

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
    run([sys.executable, 'scripts/longrun_diagnostics.py'])

    report_builder_ok = maybe_run([sys.executable, 'skills/manuscript-report-builder/scripts/build_report_tables.py'], required=args.strict_tools, reason='report table builder')
    figure_builder_ok = maybe_run([sys.executable, 'skills/figure-spec-enforcer/scripts/render_figures.py'], required=args.strict_tools, reason='figure renderer')
    claim_audit_ok = maybe_run([sys.executable, 'skills/claim-to-artifact-auditor/scripts/audit_claims.py'], required=args.strict_tools, reason='claim auditor')

    if figure_builder_ok:
        expected_figures = [
            'outputs/figures/matched_blowup_by_gamma.png',
            'outputs/figures/occupancy_shift_bar.png',
            'outputs/figures/coefficient_ci_modes.png',
            'outputs/figures/coefficient_ci_mode_b_3d.png',
            'outputs/figures/composition_artifact_matched_vs_unmatched.png',
            'outputs/figures/a_gating_explosive_structure.png',
            'outputs/figures/high_gamma_timing_vs_occupancy.png',
            'outputs/figures/precursor_active_representative_trajectory.png',
            'outputs/figures/recruited_cells_mechanism_scatter.png',
            'outputs/figures/modec_overlay_modeb_phase_scatter.png',
            'outputs/figures/trajectory_variants.png',
            'outputs/figures/extinction_sensitivity.png',
            'outputs/figures/adjacency_sensitivity.png',
            'outputs/figures/turbulence_bandwidth.png',
            'outputs/figures/scaling_exponents.png',
            'outputs/figures/heaps_law.png',
            'outputs/figures/concentration_gini.png',
        ]
        for fp in expected_figures:
            p = Path(fp)
            if (not p.exists()) or p.stat().st_size == 0:
                raise SystemExit(f'Missing or empty expected figure: {fp}')

    report_path = Path('outputs/claim_audit_report.json')
    if claim_audit_ok:
        report = json.loads(report_path.read_text())
        if not report.get('all_pass', False):
            raise SystemExit('Claim audit failed; see outputs/claim_audit_report.json')
        print('Pipeline completed with all_pass=true')
    else:
        print('Pipeline completed without claim audit (skills tool missing, non-strict mode).')

    if not report_builder_ok:
        print('! Report table builder not run; outputs/report_tables.* may be stale or missing.')


if __name__ == '__main__':
    main()
