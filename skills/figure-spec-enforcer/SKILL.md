---
name: figure-spec-enforcer
description: Enforce consistent chart styles/scales and generate publication-ready sigma-TAP figures from output artifacts using a fixed figure specification.
---

# Figure Spec Enforcer

Use this skill when the user requests consistent high-quality figures across reruns.

## Workflow

1. Edit spec template if needed:
   - `skills/figure-spec-enforcer/assets/figure_spec.json`
2. Run:
   - `python skills/figure-spec-enforcer/scripts/render_figures.py`
3. Validate:
   - `python skills/figure-spec-enforcer/scripts/render_figures.py --validate-only`

## Outputs

- `outputs/figures/matched_blowup_by_gamma.png`
- `outputs/figures/occupancy_shift_bar.png`
- `outputs/figures/coefficient_ci_modes.png`
- `outputs/figures/coefficient_ci_mode_b_3d.png`
- `outputs/figures/composition_artifact_matched_vs_unmatched.png`
- `outputs/figures/a_gating_explosive_structure.png`
- `outputs/figures/high_gamma_timing_vs_occupancy.png`
- `outputs/figures/precursor_active_representative_trajectory.png`
- `outputs/figures/recruited_cells_mechanism_scatter.png`
- `outputs/figures/modec_overlay_modeb_phase_scatter.png`

## Guardrails

- Fixed palette, DPI, and deterministic layout from spec.
- Fails fast if required fields/artifacts are missing.

## Coordinated entrypoint

- Full ordered pipeline (inferential -> tables -> figures -> audit): `python run_reporting_pipeline.py`


## Data inputs

Figure generation reads inferential + sweep artifacts declared in `assets/figure_spec.json` (including sigma feedback sweep, high-gamma sweep, recruited cells, Mode B sweep, isocurves, and precursor trajectory artifacts). Missing optional artifacts are reported with explicit skip warnings.

- Camera-ready full pipeline defaults: `python run_reporting_pipeline.py --camera-ready`
