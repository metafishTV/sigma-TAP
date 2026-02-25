"""Tests for evidence report builder (T3.1 + T3.2)."""
import json
import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

REPO = Path(__file__).resolve().parents[1]
ANNOTATIONS_PATH = REPO / "config" / "claim_annotations.json"


class TestAnnotationsFileExists(unittest.TestCase):
    def test_file_exists(self):
        self.assertTrue(ANNOTATIONS_PATH.exists(), "config/claim_annotations.json missing")

    def test_valid_json(self):
        data = json.loads(ANNOTATIONS_PATH.read_text(encoding="utf-8"))
        self.assertIsInstance(data, dict)


class TestAnnotationsCompleteness(unittest.TestCase):
    def setUp(self):
        self.data = json.loads(ANNOTATIONS_PATH.read_text(encoding="utf-8"))

    def test_all_claims_present(self):
        for cid in [f"C{i}" for i in range(1, 9)]:
            self.assertIn(cid, self.data, f"Missing annotation for {cid}")

    def test_mechanistic_non_empty(self):
        for cid, entry in self.data.items():
            self.assertIn("mechanistic", entry, f"{cid} missing mechanistic block")
            self.assertGreater(len(entry["mechanistic"]), 20, f"{cid} mechanistic too short")

    def test_functional_non_empty(self):
        for cid, entry in self.data.items():
            self.assertIn("functional", entry, f"{cid} missing functional block")
            self.assertGreater(len(entry["functional"]), 20, f"{cid} functional too short")

    def test_evidence_tier_valid(self):
        for cid, entry in self.data.items():
            self.assertIn("evidence_tier", entry, f"{cid} missing evidence_tier")
            self.assertIn(entry["evidence_tier"], {"A", "B", "C"})

    def test_tier_justification_present(self):
        for cid, entry in self.data.items():
            self.assertIn("tier_justification", entry)
            self.assertGreater(len(entry["tier_justification"]), 10)

    def test_exploratory_claims_are_tier_c(self):
        for cid, entry in self.data.items():
            if entry.get("claim_policy_label") == "exploratory":
                self.assertEqual(entry["evidence_tier"], "C", f"{cid} is exploratory but not tier C")

    def test_artifacts_listed(self):
        for cid, entry in self.data.items():
            self.assertIn("artifacts", entry)
            self.assertIsInstance(entry["artifacts"], list)
            self.assertGreater(len(entry["artifacts"]), 0)
