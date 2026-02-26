"""Tests for the run manifest generator."""
import hashlib
import os
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.generate_manifest import (
    build_manifest,
    get_git_info,
    hash_configs,
    hash_file,
    inventory_artifacts,
)


class TestHashFile(unittest.TestCase):
    def test_hash_known_content(self):
        """SHA256 of a file with known content 'hello' matches expected digest."""
        expected = hashlib.sha256(b"hello").hexdigest()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as f:
            f.write(b"hello")
            tmp = Path(f.name)
        try:
            result = hash_file(tmp)
            self.assertEqual(result, expected)
        finally:
            tmp.unlink()

    def test_hash_missing_file(self):
        """hash_file returns None for a non-existent path."""
        result = hash_file(Path("/nonexistent/file/abc123.txt"))
        self.assertIsNone(result)


class TestHashConfigs(unittest.TestCase):
    def test_hashes_json_files(self):
        """hash_configs returns a dict mapping JSON filenames to SHA256 digests."""
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "a.json"
            p.write_text('{"key": "value"}', encoding="utf-8")
            result = hash_configs(Path(td))
            self.assertIn("a.json", result)
            expected = hashlib.sha256(b'{"key": "value"}').hexdigest()
            self.assertEqual(result["a.json"], expected)

    def test_empty_dir(self):
        """hash_configs returns an empty dict for a directory with no .json files."""
        with tempfile.TemporaryDirectory() as td:
            result = hash_configs(Path(td))
            self.assertEqual(result, {})


class TestInventoryArtifacts(unittest.TestCase):
    def test_lists_files_with_size_and_hash(self):
        """inventory_artifacts returns dicts with file, size_bytes, sha256 keys."""
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "output.csv"
            p.write_bytes(b"a,b,c\n1,2,3\n")
            items = inventory_artifacts(Path(td))
            self.assertEqual(len(items), 1)
            item = items[0]
            self.assertIn("file", item)
            self.assertIn("size_bytes", item)
            self.assertIn("sha256", item)
            self.assertEqual(item["file"], "output.csv")
            self.assertEqual(item["size_bytes"], p.stat().st_size)
            self.assertEqual(
                item["sha256"], hashlib.sha256(b"a,b,c\n1,2,3\n").hexdigest()
            )

    def test_includes_subdirectory_files(self):
        """inventory_artifacts includes files in subdirectories with posix paths."""
        with tempfile.TemporaryDirectory() as td:
            sub = Path(td) / "figures"
            sub.mkdir()
            img = sub / "plot.png"
            img.write_bytes(b"\x89PNG fake")
            items = inventory_artifacts(Path(td))
            self.assertEqual(len(items), 1)
            self.assertEqual(items[0]["file"], "figures/plot.png")

    def test_empty_dir_returns_empty(self):
        """inventory_artifacts returns an empty list for an empty directory."""
        with tempfile.TemporaryDirectory() as td:
            items = inventory_artifacts(Path(td))
            self.assertEqual(items, [])


class TestGetGitInfo(unittest.TestCase):
    def test_returns_sha_and_branch(self):
        """get_git_info returns sha (40 hex chars), branch (str), dirty (bool)."""
        info = get_git_info()
        self.assertIn("sha", info)
        self.assertIn("branch", info)
        self.assertIn("dirty", info)
        self.assertEqual(len(info["sha"]), 40)
        self.assertRegex(info["sha"], r"^[0-9a-f]{40}$")
        self.assertIsInstance(info["branch"], str)
        self.assertIsInstance(info["dirty"], bool)


class TestBuildManifest(unittest.TestCase):
    def test_manifest_has_required_keys(self):
        """build_manifest returns a dict with all required top-level keys."""
        with tempfile.TemporaryDirectory() as out_dir, tempfile.TemporaryDirectory() as cfg_dir:
            manifest = build_manifest(
                outputs_dir=Path(out_dir),
                config_dir=Path(cfg_dir),
                pipeline_args={"seed": 42},
            )
            for key in ("git", "python_version", "timestamp", "config_hashes",
                        "artifacts", "pipeline_args"):
                self.assertIn(key, manifest, f"Missing key: {key}")

    def test_manifest_timestamp_is_iso(self):
        """The timestamp field is a valid ISO 8601 string."""
        with tempfile.TemporaryDirectory() as out_dir, tempfile.TemporaryDirectory() as cfg_dir:
            manifest = build_manifest(
                outputs_dir=Path(out_dir),
                config_dir=Path(cfg_dir),
                pipeline_args={},
            )
            # Should not raise
            dt = datetime.fromisoformat(manifest["timestamp"])
            self.assertIsInstance(dt, datetime)


if __name__ == "__main__":
    unittest.main()
