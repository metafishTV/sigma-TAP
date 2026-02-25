"""Run manifest generator for reproducibility tracking.

Captures git state, config hashes, artifact inventory, Python version,
pipeline arguments, and timestamp.  Writes outputs/run_manifest.json.

Usage:
  python scripts/generate_manifest.py [--seed N] [--n-boot N] [--n-perm N] [--n-boot-coef N]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = REPO_ROOT / "outputs" / "run_manifest.json"
DEFAULT_OUTPUTS_DIR = REPO_ROOT / "outputs"
DEFAULT_CONFIG_DIR = REPO_ROOT / "config"

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def hash_file(path: Path) -> str | None:
    """Return the SHA256 hex digest of *path*, or None if the file is missing."""
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()
    except (FileNotFoundError, OSError):
        return None


def hash_configs(config_dir: Path) -> dict[str, str]:
    """Return ``{filename: sha256}`` for every ``.json`` file in *config_dir*."""
    result: dict[str, str] = {}
    if not config_dir.is_dir():
        return result
    for p in sorted(config_dir.iterdir()):
        if p.suffix == ".json" and p.is_file():
            digest = hash_file(p)
            if digest is not None:
                result[p.name] = digest
    return result


def inventory_artifacts(outputs_dir: Path) -> list[dict]:
    """Inventory all files under *outputs_dir* with size and SHA256.

    Returns a list of dicts, each with keys ``file``, ``size_bytes``, and
    ``sha256``.  The ``file`` value is a POSIX-style relative path from
    *outputs_dir*.  ``run_manifest.json`` itself is excluded.
    """
    items: list[dict] = []
    if not outputs_dir.is_dir():
        return items
    for p in sorted(outputs_dir.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(outputs_dir)
        if rel == Path("run_manifest.json"):
            continue
        digest = hash_file(p)
        items.append({
            "file": rel.as_posix(),
            "size_bytes": p.stat().st_size,
            "sha256": digest,
        })
    return items


def get_git_info() -> dict:
    """Return ``{sha, branch, dirty}`` from the current git repository."""
    def _run(args: list[str]) -> str:
        try:
            return subprocess.check_output(
                args, cwd=str(REPO_ROOT), stderr=subprocess.DEVNULL, text=True,
            ).strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return ""

    sha = _run(["git", "rev-parse", "HEAD"])
    branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    status = _run(["git", "status", "--porcelain"])
    dirty = len(status) > 0
    return {"sha": sha, "branch": branch, "dirty": dirty}


def build_manifest(
    outputs_dir: Path,
    config_dir: Path,
    pipeline_args: dict,
) -> dict:
    """Assemble the full run manifest dictionary.

    Keys: ``git``, ``python_version``, ``timestamp``, ``config_hashes``,
    ``artifacts``, ``pipeline_args``.
    """
    return {
        "git": get_git_info(),
        "python_version": platform.python_version(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config_hashes": hash_configs(config_dir),
        "artifacts": inventory_artifacts(outputs_dir),
        "pipeline_args": pipeline_args,
    }


def main() -> None:
    """CLI entry point -- parse args, build manifest, write JSON."""
    parser = argparse.ArgumentParser(description="Generate a run manifest.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--n-boot", type=int, default=None)
    parser.add_argument("--n-perm", type=int, default=None)
    parser.add_argument("--n-boot-coef", type=int, default=None)
    args = parser.parse_args()

    pipeline_args = {k: v for k, v in vars(args).items() if v is not None}

    manifest = build_manifest(
        outputs_dir=DEFAULT_OUTPUTS_DIR,
        config_dir=DEFAULT_CONFIG_DIR,
        pipeline_args=pipeline_args,
    )

    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest written to {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
