"""Artifact-to-claim auditor: cross-references CLAIMS.md, annotations, and disk.

Cross-references three sources:
  1. CLAIMS.md  (claim ID -> status)
  2. config/claim_annotations.json  (claim ID -> tier + artifacts)
  3. Actual files on disk

Produces outputs/claim_audit_report.json and exits non-zero on failures.

Usage:
  python scripts/audit_claims.py
"""
from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

ROOT = Path(__file__).resolve().parents[1]
CLAIMS_MD = ROOT / "CLAIMS.md"
ANNOTATIONS_PATH = ROOT / "config" / "claim_annotations.json"
REPORT_PATH = ROOT / "outputs" / "claim_audit_report.json"


@dataclass
class AuditResult:
    """Container for audit results."""

    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    all_pass: bool = True

    def to_dict(self) -> dict:
        """Return a serialisable dictionary."""
        return {
            "errors": self.errors,
            "warnings": self.warnings,
            "all_pass": self.all_pass,
        }


def parse_claims_md_statuses(path: Path) -> dict[str, str]:
    """Parse CLAIMS.md table rows, returning {claim_id: status}.

    Status values: supported, partial, exploratory.
    Returns {} for missing file or no matches.
    """
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8")
    # Match table rows like: | C1 | ... | supported |
    pattern = r"\|\s*(C\d+)\s*\|.*\|\s*(supported|partial|exploratory)\s*\|"
    matches = re.findall(pattern, text)
    return {cid: status for cid, status in matches}


def cross_reference(
    statuses: dict[str, str],
    annotations: dict[str, dict],
    repo: Path,
) -> AuditResult:
    """Cross-reference CLAIMS.md statuses against annotations and disk artifacts.

    Rules:
      1. Every claim ID in CLAIMS.md must have a matching annotation (error if missing).
      2. Orphan annotations (in annotations but not CLAIMS.md) produce warnings.
      3. supported or partial claims with tier C -> error.
      4. exploratory claims with tier != C -> error.
      5. Every listed artifact should exist on disk (warning if missing).
    """
    result = AuditResult()

    # Rule 1: every claim in statuses must have an annotation
    for cid in statuses:
        if cid not in annotations:
            result.errors.append(f"{cid}: missing annotation entry")
            result.all_pass = False

    # Rule 2: orphan annotations
    for cid in annotations:
        if cid not in statuses:
            result.warnings.append(f"{cid}: orphan annotation (not in CLAIMS.md)")

    # Rules 3 & 4: tier/status consistency
    for cid, status in statuses.items():
        if cid not in annotations:
            continue  # already flagged above
        tier = annotations[cid].get("evidence_tier", "")
        if status in ("supported", "partial") and tier == "C":
            result.errors.append(
                f"{cid}: {status} claim has tier C (expected A or B)"
            )
            result.all_pass = False
        if status == "exploratory" and tier != "C":
            result.errors.append(
                f"{cid}: exploratory claim has tier {tier} (expected C)"
            )
            result.all_pass = False

    # Rule 5: artifact existence
    for cid, entry in annotations.items():
        for art in entry.get("artifacts", []):
            p = repo / art
            if not p.exists():
                result.warnings.append(f"{cid}: artifact not found: {art}")

    return result


def main() -> None:
    """Read files, run cross-reference, write report, exit non-zero on errors."""
    # Parse CLAIMS.md
    statuses = parse_claims_md_statuses(CLAIMS_MD)
    if not statuses:
        print(f"Warning: could not parse CLAIMS.md statuses, using annotation keys")
        annotations_raw = json.loads(ANNOTATIONS_PATH.read_text(encoding="utf-8")) if ANNOTATIONS_PATH.exists() else {}
        statuses = {k: "unknown" for k in annotations_raw}

    # Load annotations
    if not ANNOTATIONS_PATH.exists():
        print(f"ERROR: {ANNOTATIONS_PATH} not found")
        sys.exit(1)
    annotations = json.loads(ANNOTATIONS_PATH.read_text(encoding="utf-8"))

    # Cross-reference
    result = cross_reference(statuses, annotations, ROOT)

    # Print summary
    if result.errors:
        print("ERRORS:")
        for e in result.errors:
            print(f"  - {e}")
    if result.warnings:
        print("WARNINGS:")
        for w in result.warnings:
            print(f"  - {w}")

    # Write report
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(result.to_dict(), indent=2), encoding="utf-8"
    )
    print(f"Wrote {REPORT_PATH}")

    if not result.all_pass:
        sys.exit(1)

    print("Audit PASSED.")


if __name__ == "__main__":
    main()
