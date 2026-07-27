from __future__ import annotations

import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from gta_helper.reporting import build_report_archive, is_unresolved_session


class ReportingTests(unittest.TestCase):
    def test_missing_result_and_low_confidence_are_unresolved(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            session = Path(directory)
            metadata = session / "session.json"
            metadata.write_text(json.dumps({"label": "pending"}), encoding="utf-8")
            self.assertTrue(is_unresolved_session(session, 0.68))

            metadata.write_text(json.dumps({"result_summary": "candidate", "result_confidence": 0.5}), encoding="utf-8")
            self.assertTrue(is_unresolved_session(session, 0.68))

            metadata.write_text(json.dumps({"result_summary": "answer", "result_confidence": 0.9}), encoding="utf-8")
            self.assertFalse(is_unresolved_session(session, 0.68))

    def test_report_contains_only_metadata_and_session_jpegs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            session = root / "attempt"
            session.mkdir()
            (session / "session.json").write_text("{}", encoding="utf-8")
            (session / "frame_0000.jpg").write_bytes(b"jpeg")
            (session / "unrelated.png").write_bytes(b"private")
            archive = build_report_archive(session, root / "report.zip")

            with zipfile.ZipFile(archive) as report:
                self.assertEqual(sorted(report.namelist()), ["frame_0000.jpg", "session.json"])


if __name__ == "__main__":
    unittest.main()
