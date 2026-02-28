# P4 Reproducibility & Auditability Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make the full sigma-TAP pipeline reproducible from clean checkout with a run manifest, and add a standalone claim-to-artifact auditor that fails when evidence links are broken.

**Architecture:** Two new standalone scripts — `scripts/generate_manifest.py` (captures git state, config hashes, artifact inventory with SHA256) and `scripts/audit_claims.py` (cross-references CLAIMS.md, claim_annotations.json, and disk artifacts). Both are wired into the existing `run_reporting_pipeline.py` as final stages, replacing the placeholder `skills/` path references. All outputs are JSON for machine readability.

**Tech Stack:** Python 3.12, hashlib (SHA256), json, subprocess (git commands), pathlib, unittest

---

### Task 1: Run manifest generator

**Files:**
- Create: `scripts/generate_manifest.py`
- Test: `tests/test_manifest.py`

**Context:**
The pipeline (`run_reporting_pipeline.py`) already runs all stages end-to-end. This task adds a manifest at the very end that records exactly what was produced, so any future run can be compared against it. The manifest captures: git SHA + branch + dirty flag, config file hashes, artifact inventory with sizes + SHA256, Python version, pipeline arguments, and ISO 8601 timestamp.

**Step 1: Write failing tests**

File: `tests/test_manifest.py`

```python
"""Tests for run manifest generator (T4.1)."""
import hashlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.generate_manifest import (
    get_git_info,
    hash_file,
    hash_configs,
    inventory_artifacts,
    build_manifest,
)


class TestHashFile(unittest.TestCase):
    def test_hash_known_content(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("hello")
            f.flush()
            h = hash_file(Path(f.name))
        os.unlink(f.name)
        expected = hashlib.sha256(b"hello").hexdigest()
        self.assertEqual(h, expected)

    def test_hash_missing_file(self):
        h = hash_file(Path("/nonexistent/file.txt"))
        self.assertIsNone(h)


class TestHashConfigs(unittest.TestCase):
    def test_hashes_json_files(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "a.json"
            p.write_text('{"key": "val"}')
            result = hash_configs(Path(td))
        self.assertIn("a.json", result)
        self.assertEqual(len(result["a.json"]), 64)  # SHA256 hex length

    def test_empty_dir(self):
        with tempfile.TemporaryDirectory() as td:
            result = hash_configs(Path(td))
        self.assertEqual(result, {})


class TestInventoryArtifacts(unittest.TestCase):
    def test_lists_files_with_size_and_hash(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "data.csv"
            p.write_text("a,b\n1,2")
            inv = inventory_artifacts(Path(td))
        self.assertEqual(len(inv), 1)
        self.assertEqual(inv[0]["file"], "data.csv")
        self.assertIn("size_bytes", inv[0])
        self.assertIn("sha256", inv[0])

    def test_includes_subdirectory_files(self):
        with tempfile.TemporaryDirectory() as td:
            sub = Path(td) / "figures"
            sub.mkdir()
            (sub / "plot.png").write_bytes(b"\x89PNG")
            inv = inventory_artifacts(Path(td))
        self.assertEqual(len(inv), 1)
        self.assertEqual(inv[0]["file"], "figures/plot.png")

    def test_empty_dir_returns_empty(self):
        with tempfile.TemporaryDirectory() as td:
            inv = inventory_artifacts(Path(td))
        self.assertEqual(inv, [])


class TestGetGitInfo(unittest.TestCase):
    def test_returns_sha_and_branch(self):
        info = get_git_info()
        # We are in a git repo, so this should work.
        self.assertIn("sha", info)
        self.assertIn("branch", info)
        self.assertIn("dirty", info)
        self.assertEqual(len(info["sha"]), 40)
        self.assertIsInstance(info["dirty"], bool)


class TestBuildManifest(unittest.TestCase):
    def test_manifest_has_required_keys(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "outputs"
            out.mkdir()
            cfg = Path(td) / "config"
            cfg.mkdir()
            (out / "data.csv").write_text("x")
            (cfg / "a.json").write_text("{}")
            manifest = build_manifest(
                outputs_dir=out,
                config_dir=cfg,
                pipeline_args={"seed": 42},
            )
        for key in ("git", "python_version", "timestamp", "config_hashes",
                     "artifacts", "pipeline_args"):
            self.assertIn(key, manifest, f"Missing key: {key}")
        self.assertEqual(manifest["pipeline_args"]["seed"], 42)
        self.assertEqual(len(manifest["artifacts"]), 1)

    def test_manifest_timestamp_is_iso(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "outputs"
            out.mkdir()
            cfg = Path(td) / "config"
            cfg.mkdir()
            manifest = build_manifest(outputs_dir=out, config_dir=cfg)
        from datetime import datetime
        # Should parse without error.
        datetime.fromisoformat(manifest["timestamp"])
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_manifest.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.generate_manifest'`

**Step 3: Write the implementation**

File: `scripts/generate_manifest.py`

```python
"""Generate a run manifest recording git state, configs, and artifacts.

Produces outputs/run_manifest.json capturing everything needed to verify
reproducibility of a pipeline run.

Usage:
  python scripts/generate_manifest.py
  python scripts/generate_manifest.py --seed 42 --n-boot 200
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = ROOT / "outputs"
CONFIG_DIR = ROOT / "config"
MANIFEST_PATH = OUTPUTS_DIR / "run_manifest.json"


def hash_file(path: Path) -> str | None:
    """SHA256 hex digest of file contents. Returns None if file missing."""
    if not path.exists():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_configs(config_dir: Path) -> dict[str, str]:
    """Return {filename: sha256} for all .json files in config_dir."""
    result = {}
    if not config_dir.exists():
        return result
    for p in sorted(config_dir.glob("*.json")):
        h = hash_file(p)
        if h is not None:
            result[p.name] = h
    return result


def inventory_artifacts(outputs_dir: Path) -> list[dict]:
    """List all files in outputs_dir with relative path, size, and SHA256."""
    items = []
    if not outputs_dir.exists():
        return items
    for p in sorted(outputs_dir.rglob("*")):
        if not p.is_file():
            continue
        # Skip the manifest itself to avoid circular reference.
        if p.name == "run_manifest.json":
            continue
        rel = p.relative_to(outputs_dir).as_posix()
        items.append({
            "file": rel,
            "size_bytes": p.stat().st_size,
            "sha256": hash_file(p),
        })
    return items


def get_git_info() -> dict:
    """Capture git SHA, branch, and dirty flag."""
    def _run(cmd: list[str]) -> str:
        try:
            return subprocess.check_output(
                cmd, stderr=subprocess.DEVNULL, text=True
            ).strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return ""

    sha = _run(["git", "rev-parse", "HEAD"])
    branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    status = _run(["git", "status", "--porcelain"])
    return {
        "sha": sha,
        "branch": branch,
        "dirty": len(status) > 0,
    }


def build_manifest(
    outputs_dir: Path | None = None,
    config_dir: Path | None = None,
    pipeline_args: dict | None = None,
) -> dict:
    """Build the complete run manifest."""
    outputs_dir = outputs_dir or OUTPUTS_DIR
    config_dir = config_dir or CONFIG_DIR
    return {
        "git": get_git_info(),
        "python_version": sys.version,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config_hashes": hash_configs(config_dir),
        "artifacts": inventory_artifacts(outputs_dir),
        "pipeline_args": pipeline_args or {},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate run manifest")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-boot", type=int, default=200)
    parser.add_argument("--n-perm", type=int, default=500)
    parser.add_argument("--n-boot-coef", type=int, default=50)
    args = parser.parse_args()

    pipeline_args = {
        "seed": args.seed,
        "n_boot": args.n_boot,
        "n_perm": args.n_perm,
        "n_boot_coef": args.n_boot_coef,
    }

    manifest = build_manifest(pipeline_args=pipeline_args)
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {MANIFEST_PATH}")
    print(f"  Git SHA:    {manifest['git']['sha'][:12]}")
    print(f"  Branch:     {manifest['git']['branch']}")
    print(f"  Dirty:      {manifest['git']['dirty']}")
    print(f"  Artifacts:  {len(manifest['artifacts'])} files")
    print(f"  Timestamp:  {manifest['timestamp']}")


if __name__ == "__main__":
    main()
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_manifest.py -v`
Expected: All 10 tests PASS.

**Step 5: Commit**

```bash
git add scripts/generate_manifest.py tests/test_manifest.py
git commit -m "feat: add run manifest generator with git state + artifact inventory (T4.1)"
```

---

### Task 2: Artifact-to-claim auditor

**Files:**
- Create: `scripts/audit_claims.py`
- Test: `tests/test_audit_claims.py`

**Context:**
The pipeline currently references a non-existent `skills/claim-to-artifact-auditor/scripts/audit_claims.py` via `maybe_run`. This task creates a proper standalone auditor that cross-references three sources: `CLAIMS.md` (status column), `config/claim_annotations.json` (tier + artifacts), and actual files on disk. It produces `outputs/claim_audit_report.json` and exits non-zero on failures.

Rules enforced:
1. Every claim ID in CLAIMS.md must have a matching annotation entry.
2. Every annotation's listed artifacts must exist on disk.
3. `supported` claims must have evidence_tier A or B (not C).
4. `exploratory` claims must have evidence_tier C.
5. `paper-aligned` claims must list at least one artifact that actually exists.
6. Report any annotation claim IDs not present in CLAIMS.md (orphan check).

**Step 1: Write failing tests**

File: `tests/test_audit_claims.py`

```python
"""Tests for claim-to-artifact auditor (T4.2)."""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.audit_claims import (
    parse_claims_md_statuses,
    cross_reference,
    AuditResult,
)


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
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write(SAMPLE_CLAIMS_MD)
            f.flush()
            result = parse_claims_md_statuses(Path(f.name))
        os.unlink(f.name)
        self.assertEqual(result, {"C1": "supported", "C2": "exploratory", "C3": "partial"})

    def test_empty_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write("# No table\n")
            f.flush()
            result = parse_claims_md_statuses(Path(f.name))
        os.unlink(f.name)
        self.assertEqual(result, {})

    def test_missing_file(self):
        result = parse_claims_md_statuses(Path("/nonexistent.md"))
        self.assertEqual(result, {})


class TestCrossReference(unittest.TestCase):
    def test_all_good(self):
        """Supported claim with tier A and existing artifact passes."""
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            (repo / "outputs").mkdir()
            (repo / "outputs" / "a.csv").write_text("data")
            statuses = {"C1": "supported"}
            annotations = {
                "C1": {
                    "mechanistic": "x" * 25,
                    "functional": "y" * 25,
                    "evidence_tier": "A",
                    "tier_justification": "z" * 15,
                    "artifacts": ["outputs/a.csv"],
                    "claim_policy_label": "paper-aligned",
                },
            }
            result = cross_reference(statuses, annotations, repo)
        self.assertTrue(result.all_pass)
        self.assertEqual(result.errors, [])

    def test_supported_claim_tier_c_fails(self):
        """Supported claim with tier C is an error."""
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            (repo / "outputs").mkdir()
            (repo / "outputs" / "a.csv").write_text("data")
            statuses = {"C1": "supported"}
            annotations = {
                "C1": {
                    "mechanistic": "x" * 25, "functional": "y" * 25,
                    "evidence_tier": "C", "tier_justification": "z" * 15,
                    "artifacts": ["outputs/a.csv"],
                    "claim_policy_label": "paper-aligned",
                },
            }
            result = cross_reference(statuses, annotations, repo)
        self.assertFalse(result.all_pass)
        self.assertTrue(any("tier" in e.lower() for e in result.errors))

    def test_missing_annotation_fails(self):
        """Claim in CLAIMS.md but not in annotations is an error."""
        with tempfile.TemporaryDirectory() as td:
            statuses = {"C1": "supported", "C2": "supported"}
            annotations = {
                "C1": {
                    "mechanistic": "x" * 25, "functional": "y" * 25,
                    "evidence_tier": "A", "tier_justification": "z" * 15,
                    "artifacts": ["outputs/a.csv"],
                    "claim_policy_label": "paper-aligned",
                },
            }
            repo = Path(td)
            (repo / "outputs").mkdir()
            (repo / "outputs" / "a.csv").write_text("data")
            result = cross_reference(statuses, annotations, repo)
        self.assertFalse(result.all_pass)
        self.assertTrue(any("C2" in e for e in result.errors))

    def test_missing_artifact_warns(self):
        """Missing artifact file generates a warning."""
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            statuses = {"C1": "supported"}
            annotations = {
                "C1": {
                    "mechanistic": "x" * 25, "functional": "y" * 25,
                    "evidence_tier": "A", "tier_justification": "z" * 15,
                    "artifacts": ["outputs/missing.csv"],
                    "claim_policy_label": "paper-aligned",
                },
            }
            result = cross_reference(statuses, annotations, repo)
        self.assertGreater(len(result.warnings), 0)
        self.assertTrue(any("missing.csv" in w for w in result.warnings))

    def test_exploratory_tier_a_fails(self):
        """Exploratory claim with tier A is an error."""
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            statuses = {"C1": "exploratory"}
            annotations = {
                "C1": {
                    "mechanistic": "x" * 25, "functional": "y" * 25,
                    "evidence_tier": "A", "tier_justification": "z" * 15,
                    "artifacts": ["outputs/a.csv"],
                    "claim_policy_label": "exploratory",
                },
            }
            result = cross_reference(statuses, annotations, repo)
        self.assertFalse(result.all_pass)

    def test_orphan_annotation_warned(self):
        """Annotation not in CLAIMS.md produces a warning."""
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            (repo / "outputs").mkdir()
            (repo / "outputs" / "a.csv").write_text("data")
            statuses = {"C1": "supported"}
            annotations = {
                "C1": {
                    "mechanistic": "x" * 25, "functional": "y" * 25,
                    "evidence_tier": "A", "tier_justification": "z" * 15,
                    "artifacts": ["outputs/a.csv"],
                    "claim_policy_label": "paper-aligned",
                },
                "C99": {
                    "mechanistic": "x" * 25, "functional": "y" * 25,
                    "evidence_tier": "C", "tier_justification": "z" * 15,
                    "artifacts": ["outputs/a.csv"],
                    "claim_policy_label": "exploratory",
                },
            }
            result = cross_reference(statuses, annotations, repo)
        self.assertTrue(any("C99" in w for w in result.warnings))

    def test_partial_claim_tier_c_fails(self):
        """Partial claim should not be tier C."""
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            (repo / "outputs").mkdir()
            (repo / "outputs" / "a.csv").write_text("data")
            statuses = {"C1": "partial"}
            annotations = {
                "C1": {
                    "mechanistic": "x" * 25, "functional": "y" * 25,
                    "evidence_tier": "C", "tier_justification": "z" * 15,
                    "artifacts": ["outputs/a.csv"],
                    "claim_policy_label": "exploratory",
                },
            }
            result = cross_reference(statuses, annotations, repo)
        self.assertFalse(result.all_pass)


class TestAuditResult(unittest.TestCase):
    def test_to_dict(self):
        r = AuditResult(errors=["e1"], warnings=["w1"], all_pass=False)
        d = r.to_dict()
        self.assertEqual(d["errors"], ["e1"])
        self.assertEqual(d["warnings"], ["w1"])
        self.assertFalse(d["all_pass"])

    def test_empty_passes(self):
        r = AuditResult(errors=[], warnings=[], all_pass=True)
        self.assertTrue(r.all_pass)
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_audit_claims.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.audit_claims'`

**Step 3: Write the implementation**

File: `scripts/audit_claims.py`

```python
"""Cross-reference claims, annotations, and artifacts for auditability.

Reads CLAIMS.md (status column), config/claim_annotations.json (tiers +
artifacts), and checks actual files on disk. Produces
outputs/claim_audit_report.json and exits non-zero on failures.

Enforced rules:
  - Every CLAIMS.md claim ID must have a matching annotation.
  - Supported/partial claims must have evidence_tier A or B.
  - Exploratory claims must have evidence_tier C.
  - Every listed artifact should exist on disk (warning if missing).
  - Orphan annotations (not in CLAIMS.md) produce warnings.

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
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    all_pass: bool = True

    def to_dict(self) -> dict:
        return {
            "errors": self.errors,
            "warnings": self.warnings,
            "all_pass": self.all_pass,
        }


def parse_claims_md_statuses(path: Path) -> dict[str, str]:
    """Extract {claim_id: status} from CLAIMS.md table rows.

    Expects rows like: | C1 | ... | supported |
    """
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8")
    result = {}
    for match in re.finditer(
        r'\|\s*(C\d+)\s*\|.*\|\s*(supported|partial|exploratory)\s*\|',
        text,
    ):
        result[match.group(1)] = match.group(2)
    return result


def cross_reference(
    statuses: dict[str, str],
    annotations: dict[str, dict],
    repo: Path,
) -> AuditResult:
    """Cross-reference claims, annotations, and disk artifacts."""
    result = AuditResult()

    # 1. Every CLAIMS.md entry must have an annotation.
    for cid in statuses:
        if cid not in annotations:
            result.errors.append(f"{cid}: in CLAIMS.md but missing from annotations")
            result.all_pass = False

    # 2. Orphan annotations (in annotations but not in CLAIMS.md).
    for cid in annotations:
        if cid not in statuses:
            result.warnings.append(f"{cid}: in annotations but not in CLAIMS.md")

    # 3. Tier consistency with status.
    for cid, status in statuses.items():
        entry = annotations.get(cid)
        if entry is None:
            continue
        tier = entry.get("evidence_tier", "")

        if status in ("supported", "partial") and tier == "C":
            result.errors.append(
                f"{cid}: status '{status}' requires tier A or B, got C"
            )
            result.all_pass = False

        if status == "exploratory" and tier != "C":
            result.errors.append(
                f"{cid}: status 'exploratory' requires tier C, got {tier}"
            )
            result.all_pass = False

    # 4. Artifact existence.
    for cid, entry in annotations.items():
        for art in entry.get("artifacts", []):
            p = repo / art
            if not p.exists():
                result.warnings.append(f"{cid}: artifact not found: {art}")

    return result


def main() -> None:
    if not ANNOTATIONS_PATH.exists():
        raise SystemExit(f"Missing: {ANNOTATIONS_PATH}")

    annotations = json.loads(ANNOTATIONS_PATH.read_text(encoding="utf-8"))
    statuses = parse_claims_md_statuses(CLAIMS_MD)
    if not statuses:
        print("Warning: could not parse CLAIMS.md statuses, using annotation keys")
        statuses = {k: "unknown" for k in annotations}

    result = cross_reference(statuses, annotations, ROOT)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report = result.to_dict()
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {REPORT_PATH}")

    if result.errors:
        print(f"\nAUDIT ERRORS ({len(result.errors)}):")
        for e in result.errors:
            print(f"  ERROR: {e}")

    if result.warnings:
        print(f"\nAudit warnings ({len(result.warnings)}):")
        for w in result.warnings:
            print(f"  WARN:  {w}")

    if result.all_pass:
        print("\nClaim audit: PASS (all_pass=true)")
    else:
        raise SystemExit(f"Claim audit: FAIL ({len(result.errors)} error(s))")


if __name__ == "__main__":
    main()
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_audit_claims.py -v`
Expected: All 10 tests PASS.

**Step 5: Commit**

```bash
git add scripts/audit_claims.py tests/test_audit_claims.py
git commit -m "feat: add claim-to-artifact auditor with cross-reference checks (T4.2)"
```

---

### Task 3: Pipeline integration

**Files:**
- Modify: `run_reporting_pipeline.py`
- Test: `tests/test_pipeline_integration.py`

**Context:**
Wire the manifest generator and claim auditor into the pipeline. Replace the three `maybe_run` calls that reference non-existent `skills/` paths with direct calls to the new scripts. The manifest is generated last (after all artifacts exist). The auditor runs before the manifest (so the audit report is included in the manifest inventory).

**Step 1: Write failing integration test**

File: `tests/test_pipeline_integration.py`

```python
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
```

**Step 2: Run tests to verify some fail**

Run: `python -m pytest tests/test_pipeline_integration.py -v`
Expected: `TestPipelineReferences` tests FAIL (pipeline still has `skills/` references).

**Step 3: Modify run_reporting_pipeline.py**

Replace the `maybe_run` calls and figure check block (lines 93-129) with direct calls to the new scripts:

```python
# In main(), replace the three maybe_run + figure check + audit check blocks with:

    # Claim audit (T4.2): cross-reference CLAIMS.md <-> annotations <-> disk.
    run([sys.executable, 'scripts/audit_claims.py'])

    report_path = Path('outputs/claim_audit_report.json')
    if report_path.exists():
        report = json.loads(report_path.read_text())
        if not report.get('all_pass', False):
            raise SystemExit('Claim audit failed; see outputs/claim_audit_report.json')
        print('Claim audit passed (all_pass=true)')

    # Run manifest (T4.1): record git state, configs, artifact inventory.
    manifest_cmd = [
        sys.executable, 'scripts/generate_manifest.py',
        '--seed', str(args.seed),
        '--n-boot', str(args.n_boot),
        '--n-perm', str(args.n_perm),
        '--n-boot-coef', str(args.n_boot_coef),
    ]
    run(manifest_cmd)
    print('Pipeline completed successfully.')
```

Also remove the `maybe_run` function if no longer used, and remove the expected_figures block (artifact existence is now handled by the auditor + manifest).

**Step 4: Run all tests**

Run: `python -m pytest tests/ -v`
Expected: All tests PASS (existing 120 + new ~25 = ~145).

**Step 5: Commit**

```bash
git add run_reporting_pipeline.py tests/test_pipeline_integration.py
git commit -m "feat: wire manifest + auditor into pipeline, remove skills/ placeholders (T4.1+T4.2)"
```

---

### Task 4: Verify full pipeline and push

**Step 1: Run the auditor standalone to verify repo state**

Run: `python scripts/audit_claims.py`
Expected: `Claim audit: PASS (all_pass=true)`

**Step 2: Run the manifest generator standalone**

Run: `python scripts/generate_manifest.py --seed 42`
Expected: Writes `outputs/run_manifest.json` with git SHA, artifacts, etc.

**Step 3: Run full test suite**

Run: `python -m pytest tests/ -v`
Expected: All tests PASS.

**Step 4: Push**

```bash
git push origin unified-integration-20260225
```
