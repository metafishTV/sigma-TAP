---
name: manuscript-report-builder
description: Build manuscript-ready dynamic tables from sigma-TAP output artifacts (inferential_stats.json and sweep summaries), exporting markdown/csv/latex snapshots after each run.
---

# Manuscript Report Builder

Use this skill when the user asks for run-updating tables, printable report summaries, or manuscript-ready quantitative tables.

## Workflow

1. Ensure inferential artifacts exist (`outputs/inferential_stats.json` at minimum).
2. Run:
   - `python skills/manuscript-report-builder/scripts/build_report_tables.py`
3. Generated outputs:
   - `outputs/report_tables.md`
   - `outputs/report_tables.csv`
   - `outputs/report_tables.tex`

## Inputs

- `outputs/inferential_stats.json` (required)
- Optional enrichments if present:
  - `outputs/precursor_longitudinal_summary.json`
  - `outputs/recruited_cells_a8_summary.json`
  - `outputs/blowup_matched_panel_a8_summary.json`

## Notes

- Keeps one canonical tabulation per run.
- Uses rounded manuscript-friendly formatting while preserving full precision in source artifacts.

## Coordinated entrypoint

- Preferred full pipeline run: `python run_reporting_pipeline.py`

- Camera-ready full pipeline defaults: `python run_reporting_pipeline.py --camera-ready`
