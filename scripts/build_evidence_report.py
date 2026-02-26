"""Build and validate the evidence report from claim annotations.

Reads config/claim_annotations.json, validates completeness and
tier consistency, cross-references artifacts, and produces
outputs/evidence_report.json.

Usage:
  python scripts/build_evidence_report.py
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

REPO = Path(__file__).resolve().parents[1]
ANNOTATIONS_PATH = REPO / "config" / "claim_annotations.json"
CLAIMS_MD_PATH = REPO / "CLAIMS.md"
OUTPUT_PATH = REPO / "outputs" / "evidence_report.json"


def parse_claim_ids_from_claims_md(path: Path) -> list[str]:
    """Extract claim IDs (C1, C2, ...) from CLAIMS.md table."""
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8")
    return re.findall(r'\|\s*(C\d+)\s*\|', text)


def validate_annotations(
    data: dict,
    claim_ids: list[str],
) -> list[str]:
    """Validate claim annotations for completeness and consistency.

    Returns list of error strings. Empty list = valid.
    """
    errors: list[str] = []

    for cid in claim_ids:
        if cid not in data:
            errors.append(f"{cid}: missing from annotations")

    for cid, entry in data.items():
        for field in ("mechanistic", "functional", "evidence_tier",
                      "tier_justification", "artifacts", "claim_policy_label"):
            if field not in entry:
                errors.append(f"{cid}: missing required field '{field}'")

        for field in ("mechanistic", "functional"):
            val = entry.get(field, "")
            if len(val) < 20:
                errors.append(f"{cid}: {field} block too short ({len(val)} chars, need >= 20)")

        tier = entry.get("evidence_tier", "")
        if tier not in ("A", "B", "C"):
            errors.append(f"{cid}: invalid evidence_tier '{tier}' (must be A, B, or C)")

        tj = entry.get("tier_justification", "")
        if len(tj) < 10:
            errors.append(f"{cid}: tier_justification too short")

        if entry.get("claim_policy_label") == "exploratory" and tier != "C":
            errors.append(f"{cid}: exploratory claim must be tier C, got '{tier}'")

        arts = entry.get("artifacts", [])
        if not isinstance(arts, list) or len(arts) == 0:
            errors.append(f"{cid}: artifacts must be a non-empty list")

    return errors


def check_artifacts_exist(data: dict, repo: Path) -> list[str]:
    """Warn about missing artifact files."""
    warnings: list[str] = []
    for cid, entry in data.items():
        for art in entry.get("artifacts", []):
            p = repo / art
            if not p.exists():
                warnings.append(f"{cid}: artifact not found: {art}")
    return warnings


def build_report(data: dict) -> dict:
    """Build the enriched evidence report from validated annotations."""
    tier_counts = {"A": 0, "B": 0, "C": 0}
    claims = {}
    for cid, entry in sorted(data.items()):
        tier = entry.get("evidence_tier", "C")
        tier_counts[tier] = tier_counts.get(tier, 0) + 1
        claims[cid] = {
            "mechanistic": entry["mechanistic"],
            "functional": entry["functional"],
            "evidence_tier": tier,
            "tier_justification": entry["tier_justification"],
            "artifacts": entry["artifacts"],
            "claim_policy_label": entry["claim_policy_label"],
        }
    return {
        "claims": claims,
        "summary": {
            "total_claims": len(claims),
            "tier_distribution": tier_counts,
            "all_validated": True,
        },
    }


def main() -> None:
    if not ANNOTATIONS_PATH.exists():
        raise SystemExit(f"Missing: {ANNOTATIONS_PATH}")

    data = json.loads(ANNOTATIONS_PATH.read_text(encoding="utf-8"))
    claim_ids = parse_claim_ids_from_claims_md(CLAIMS_MD_PATH)
    if not claim_ids:
        claim_ids = list(data.keys())
        print(f"Warning: could not parse CLAIMS.md, using annotation keys: {claim_ids}")

    errors = validate_annotations(data, claim_ids)
    if errors:
        print("VALIDATION ERRORS:")
        for e in errors:
            print(f"  - {e}")
        raise SystemExit(f"Evidence report validation failed with {len(errors)} error(s)")

    warnings = check_artifacts_exist(data, REPO)
    if warnings:
        print("Artifact warnings (non-fatal):")
        for w in warnings:
            print(f"  - {w}")

    report = build_report(data)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH}")

    td = report["summary"]["tier_distribution"]
    print(f"  Tier A (robust):      {td['A']}")
    print(f"  Tier B (supported):   {td['B']}")
    print(f"  Tier C (exploratory): {td['C']}")


if __name__ == "__main__":
    main()
