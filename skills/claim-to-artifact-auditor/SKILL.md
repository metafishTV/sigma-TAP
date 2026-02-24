---
name: claim-to-artifact-auditor
description: Audit quantitative manuscript claims against current output artifacts and emit a pass/fail report to prevent drift after reruns.
---

# Claim-to-Artifact Auditor

Use this skill when manuscript numbers may have drifted from regenerated outputs.

## Workflow

1. Run auditor:
   - `python skills/claim-to-artifact-auditor/scripts/audit_claims.py`
2. Optional path overrides:
   - `python skills/claim-to-artifact-auditor/scripts/audit_claims.py --manuscript MANUSCRIPT_SYNTHESIS_FINAL.md --declared config/manuscript_declared_values.json --config config/audit_config.json`
3. (Optional) pre-audit declared-value validation against fresh inferential output:
   - `python scripts/validate_declared_values.py --inferential outputs/inferential_stats.json --declared config/manuscript_declared_values.json --config config/audit_config.json`
4. Review reports:
   - `outputs/declared_values_validation_report.json`
   - `outputs/claim_audit_report.json`

## Current checks

- Matched timing p-value threshold, mean paired reduction text, and matched means/n claims.
- Occupancy difference, p-value, CI, and fraction claims.
- Sign-reversal non-overlap flags and direct CI-geometry non-overlap checks.
- Mode A/Mode B (2D + 3D) coefficient mean/CI drift checks (tolerance-based) using `config/manuscript_declared_values.json`.
- Mode A/Mode B 2D/3D CI-bound manuscript string checks (anchor-aware) to ensure prose updates accompany declared-value updates.

## Extension

Add new checks in `build_checks(...)` in the script to expand coverage (anchor-aware and tolerance-aware patterns).

## Coordinated entrypoint

- Full ordered pipeline with fail-fast on audit: `python run_reporting_pipeline.py`
- Camera-ready full pipeline defaults: `python run_reporting_pipeline.py --camera-ready`
