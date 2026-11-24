"""Input-integrity checks for joining judge scores."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class InputChecksTest(unittest.TestCase):
    def test_judge_merge_preserves_intersection(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            a = path / "a.jsonl"
            b = path / "b.jsonl"
            out = path / "merged.jsonl"
            a.write_text(json.dumps({"story_id": "s1", "query_id": "q1",
                                     "culture": "c", "culture_score_A": 3}) + "\n")
            b.write_text("\n".join(json.dumps(row) for row in (
                {"story_id": "s1", "query_id": "q1", "culture": "c", "culture_score_B": 4},
                {"story_id": "s2", "query_id": "q2", "culture": "c", "culture_score_B": 2},
            )) + "\n")
            cmd = [sys.executable, str(ROOT / "scripts/merge_judges.py"),
                   "--judge_a", str(a), "--judge_b", str(b), "--out", str(out)]
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            rows = [json.loads(line) for line in out.read_text().splitlines()]
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["story_id"], "s1")
            result = subprocess.run(cmd, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)  # Existing output is protected.

    def test_duplicate_judge_ids_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            a, b, out = path/"a.jsonl", path/"b.jsonl", path/"out.jsonl"
            record = {"story_id": "s1", "query_id": "q1"}
            a.write_text((json.dumps(record) + "\n") * 2)
            b.write_text(json.dumps(record) + "\n")
            result = subprocess.run([sys.executable, str(ROOT/"scripts/merge_judges.py"),
                "--judge_a", str(a), "--judge_b", str(b), "--out", str(out)],
                capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(out.exists())

    def test_conflicting_query_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            a, b, out = path/"a.jsonl", path/"b.jsonl", path/"out.jsonl"
            a.write_text(json.dumps({"story_id": "s1", "query_id": "q1"}) + "\n")
            b.write_text(json.dumps({"story_id": "s1", "query_id": "q2"}) + "\n")
            result = subprocess.run([sys.executable, str(ROOT/"scripts/merge_judges.py"),
                "--judge_a", str(a), "--judge_b", str(b), "--out", str(out)],
                capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(out.exists())


if __name__ == "__main__":
    unittest.main()
