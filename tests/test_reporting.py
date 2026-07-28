from __future__ import annotations

import json
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np

from gta_helper.capture import DxCapture
from gta_helper.reporting import DiagnosticReporter, build_report_archive, is_unresolved_session, session_outcome
from server.receiver import validate_report_archive


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
            self.assertEqual(session_outcome(session, 0.68), "success")

            metadata.write_text(json.dumps({"answer_outcome": "failure", "result_summary": "answer", "result_confidence": 0.9}), encoding="utf-8")
            self.assertEqual(session_outcome(session, 0.68), "failure")

    def test_reporter_queues_success_and_failure_sessions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reporter = DiagnosticReporter("https://example.invalid/v1/reports", 0.68)
            for name, metadata in (
                ("success", {"answer_outcome": "success"}),
                ("failure", {"answer_outcome": "failure"}),
            ):
                session = root / name
                session.mkdir()
                (session / "session.json").write_text(json.dumps(metadata), encoding="utf-8")
                self.assertTrue(reporter.submit(session))
            self.assertEqual(reporter._queue.qsize(), 2)

    def test_reporter_queues_manual_session_for_automatic_upload(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            session = Path(directory)
            (session / "session.json").write_text(
                json.dumps({"capture_trigger": "manual", "answer_outcome": "failure"}),
                encoding="utf-8",
            )
            reporter = DiagnosticReporter("https://example.invalid/v1/reports", 0.68)

            self.assertTrue(reporter.submit(session))
            queued_session, report_kind = reporter._queue.get_nowait()
            self.assertEqual(queued_session, session)
            self.assertEqual(report_kind, "manual")

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

    def test_manual_diagnostic_archive_passes_receiver_validation(self) -> None:
        frame = np.random.default_rng(7710).integers(0, 256, (90, 160, 3), dtype=np.uint8)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            session = DxCapture.save_diagnostic(frame, root, "gta")
            archive = build_report_archive(session, root / "manual-report.zip")

            metadata = validate_report_archive(archive.read_bytes())
            self.assertEqual(metadata["capture_trigger"], "manual")


if __name__ == "__main__":
    unittest.main()
