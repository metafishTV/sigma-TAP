"""Integration tests for pipeline reproducibility features (T4.1 + T4.2)."""
import json
import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

REPO = Path(__file__).resolve().parents[1]


class TestManifestScript(unittest.TestCase):
    """Verify manifest generator is importable and produces valid output."""

    def test_importable(self):
        from scripts.generate_manifest import build_manifest
        self.assertTrue(callable(build_manifest))

    def test_manifest_contains_git(self):
        from scripts.generate_manifest import build_manifest
        m = build_manifest(
            outputs_dir=REPO / "outputs",
            config_dir=REPO / "config",
            pipeline_args={},
        )
        self.assertIn("git", m)
        self.assertIn("sha", m["git"])


class TestAuditScript(unittest.TestCase):
    """Verify auditor runs against real repo state."""

    def test_importable(self):
        from scripts.audit_claims import cross_reference, parse_claims_md_statuses
        self.assertTrue(callable(cross_reference))

    def test_real_repo_audit(self):
        from scripts.audit_claims import cross_reference, parse_claims_md_statuses
        statuses = parse_claims_md_statuses(REPO / "CLAIMS.md")
        annotations = json.loads(
            (REPO / "config" / "claim_annotations.json").read_text(encoding="utf-8")
        )
        result = cross_reference(statuses, annotations, REPO)
        # The real repo should pass audit.
        self.assertTrue(result.all_pass, f"Audit errors: {result.errors}")


class TestPipelineReferences(unittest.TestCase):
    """Verify pipeline no longer references non-existent skills/ paths."""

    def test_no_skills_path_references(self):
        pipeline = (REPO / "run_reporting_pipeline.py").read_text(encoding="utf-8")
        self.assertNotIn("skills/claim-to-artifact-auditor", pipeline)
        self.assertNotIn("skills/manuscript-report-builder", pipeline)
        self.assertNotIn("skills/figure-spec-enforcer", pipeline)

    def test_manifest_call_present(self):
        pipeline = (REPO / "run_reporting_pipeline.py").read_text(encoding="utf-8")
        self.assertIn("generate_manifest", pipeline)

    def test_audit_call_present(self):
        pipeline = (REPO / "run_reporting_pipeline.py").read_text(encoding="utf-8")
        self.assertIn("audit_claims", pipeline)

    def test_no_maybe_run(self):
        """maybe_run function should be removed."""
        pipeline = (REPO / "run_reporting_pipeline.py").read_text(encoding="utf-8")
        self.assertNotIn("maybe_run", pipeline)

    def test_no_strict_tools_arg(self):
        """--strict-tools arg should be removed."""
        pipeline = (REPO / "run_reporting_pipeline.py").read_text(encoding="utf-8")
        self.assertNotIn("strict-tools", pipeline)
        self.assertNotIn("strict_tools", pipeline)
