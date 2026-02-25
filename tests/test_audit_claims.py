"""Tests for claim-to-artifact auditor (T4.2)."""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.audit_claims import AuditResult, cross_reference, parse_claims_md_statuses

SAMPLE_CLAIMS_MD = """\
# Claims

| ID | Claim | Source paper(s) | Supporting artifact(s) | Status |
|----|-------|----------------|----------------------|--------|
| C1 | First claim | Paper A | `outputs/a.csv` | supported |
| C2 | Second claim | (exploratory) | `outputs/b.csv` | exploratory |
| C3 | Third claim | Paper B | `outputs/c.csv` | partial |
"""


class TestParseClaimsMdStatuses(unittest.TestCase):
    def test_parses_all_statuses(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", delete=False, encoding="utf-8"
        ) as f:
            f.write(SAMPLE_CLAIMS_MD)
            f.flush()
            path = Path(f.name)
        try:
            result = parse_claims_md_statuses(path)
            self.assertEqual(result, {"C1": "supported", "C2": "exploratory", "C3": "partial"})
        finally:
            path.unlink()

    def test_empty_file(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", delete=False, encoding="utf-8"
        ) as f:
            f.write("# No table here\n")
            f.flush()
            path = Path(f.name)
        try:
            result = parse_claims_md_statuses(path)
            self.assertEqual(result, {})
        finally:
            path.unlink()

    def test_missing_file(self):
        result = parse_claims_md_statuses(Path("/nonexistent.md"))
        self.assertEqual(result, {})


class TestCrossReference(unittest.TestCase):
    def _make_annotation(self, tier: str = "A", artifacts: list[str] | None = None):
        return {
            "evidence_tier": tier,
            "artifacts": artifacts or [],
        }

    def test_all_good(self):
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            (repo / "outputs").mkdir()
            (repo / "outputs" / "a.csv").write_text("data")
            statuses = {"C1": "supported"}
            annotations = {"C1": self._make_annotation("A", ["outputs/a.csv"])}
            result = cross_reference(statuses, annotations, repo)
            self.assertTrue(result.all_pass)
            self.assertEqual(result.errors, [])
            self.assertEqual(result.warnings, [])

    def test_supported_claim_tier_c_fails(self):
        statuses = {"C1": "supported"}
        annotations = {"C1": self._make_annotation("C")}
        result = cross_reference(statuses, annotations, Path("."))
        self.assertFalse(result.all_pass)
        self.assertTrue(any("C1" in e and "tier" in e.lower() for e in result.errors))

    def test_missing_annotation_fails(self):
        statuses = {"C1": "supported", "C2": "partial"}
        annotations = {"C1": self._make_annotation("A")}
        result = cross_reference(statuses, annotations, Path("."))
        self.assertFalse(result.all_pass)
        self.assertTrue(any("C2" in e for e in result.errors))

    def test_missing_artifact_warns(self):
        statuses = {"C1": "supported"}
        annotations = {"C1": self._make_annotation("A", ["outputs/nonexistent.csv"])}
        result = cross_reference(statuses, annotations, Path("."))
        self.assertTrue(any("nonexistent" in w for w in result.warnings))

    def test_exploratory_tier_a_fails(self):
        statuses = {"C1": "exploratory"}
        annotations = {"C1": self._make_annotation("A")}
        result = cross_reference(statuses, annotations, Path("."))
        self.assertFalse(result.all_pass)
        self.assertTrue(any("C1" in e for e in result.errors))

    def test_orphan_annotation_warned(self):
        statuses = {"C1": "supported"}
        annotations = {
            "C1": self._make_annotation("A"),
            "C99": self._make_annotation("B"),
        }
        result = cross_reference(statuses, annotations, Path("."))
        self.assertTrue(any("C99" in w for w in result.warnings))

    def test_partial_claim_tier_c_fails(self):
        statuses = {"C1": "partial"}
        annotations = {"C1": self._make_annotation("C")}
        result = cross_reference(statuses, annotations, Path("."))
        self.assertFalse(result.all_pass)
        self.assertTrue(any("C1" in e and "tier" in e.lower() for e in result.errors))


class TestAuditResult(unittest.TestCase):
    def test_to_dict(self):
        ar = AuditResult(errors=["e1"], warnings=["w1"], all_pass=False)
        d = ar.to_dict()
        self.assertEqual(d["errors"], ["e1"])
        self.assertEqual(d["warnings"], ["w1"])
        self.assertFalse(d["all_pass"])

    def test_empty_passes(self):
        ar = AuditResult()
        self.assertTrue(ar.all_pass)
        self.assertEqual(ar.errors, [])
        self.assertEqual(ar.warnings, [])


if __name__ == "__main__":
    unittest.main()
