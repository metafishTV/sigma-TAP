# Reporting & Audit Pipeline

This repository uses one coordinated entrypoint for reproducible reporting artifacts:

- `python run_reporting_pipeline.py`
- Camera-ready settings: `python run_reporting_pipeline.py --camera-ready`

## Defaults and camera-ready mode

`run_reporting_pipeline.py` defaults are tuned for faster dev iteration:

- `--n-boot 200`
- `--n-perm 500`
- `--n-boot-coef 50`

`--camera-ready` enforces submission settings:

- `--n-boot 1000`
- `--n-perm 2000`
- `--n-boot-coef 500`

## Step order (fail-fast)

1. **Inferential rerun**
   - Script: `scripts/inferential_stats.py`
   - Output: `outputs/inferential_stats.json`
2. **Declared-values validation**
   - Script: `scripts/validate_declared_values.py`
   - Inputs:
     - `outputs/inferential_stats.json`
     - `config/manuscript_declared_values.json`
     - `config/audit_config.json`
   - Output:
     - `outputs/declared_values_validation_report.json`
   - Pipeline aborts if `all_pass` is false.
3. **Precursor followup + summaries**
   - Scripts:
     - `scripts/followup_precursor_longitudinal.py`
     - `scripts/summarize_precursor_longitudinal.py`
     - `scripts/summarize_recruited_cells.py`
     - `scripts/summarize_blowup_matched_panel.py`
4. **Report tables**
   - Script: `skills/manuscript-report-builder/scripts/build_report_tables.py`
   - Outputs: `outputs/report_tables.md|csv|tex`
5. **Figures**
   - Script: `skills/figure-spec-enforcer/scripts/render_figures.py`
   - Outputs: `outputs/figures/*.png`
   - Pipeline verifies expected figure files exist and are non-zero bytes.
6. **Claim audit**
   - Script: `skills/claim-to-artifact-auditor/scripts/audit_claims.py`
   - Output: `outputs/claim_audit_report.json`
   - Pipeline aborts if `all_pass` is false.

## Configuration triangle (authoritative files)

The manuscript-coefficient validation and claim audit rely on three source-controlled files:

1. `config/audit_config.json`
   - Shared tolerance (e.g., `coef_tolerance`) used by both validator and auditor.
2. `config/manuscript_declared_values.json`
   - Declared manuscript coefficient means/CI bounds used for tolerance checks.
3. `MANUSCRIPT_SYNTHESIS_FINAL.md`
   - Narrative claim text checked with anchor-aware manuscript string checks.

## Source-controlled config vs generated artifacts

- **Source-controlled configuration / declarations**:
  - `config/audit_config.json`
  - `config/manuscript_declared_values.json`
- **Generated each pipeline run**:
  - `outputs/inferential_stats.json`
  - `outputs/declared_values_validation_report.json`
  - `outputs/report_tables.*`
  - `outputs/figures/*.png`
  - `outputs/claim_audit_report.json`
  - precursor/recruited/matched summary artifacts.

## Repro guidance

If clearing output artifacts for a clean rerun, remove only generated files under `outputs/` and keep `config/manuscript_declared_values.json` intact, since it is a declared-values input rather than a derived statistic.
