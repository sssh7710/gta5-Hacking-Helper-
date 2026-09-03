from __future__ import annotations

import json
import tempfile
import unittest
import urllib.error
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from gta_helper.capture import DxCapture
from gta_helper.reporting import (
    DiagnosticReporter,
    build_report_archive,
    is_unresolved_session,
    session_outcome,
    upload_diagnostic_frame,
    upload_report,
)
from server.receiver import validate_report_archive


class ReportingTests(unittest.TestCase):
    def test_upload_retries_rate_limit_with_same_report_id(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            session = Path(directory)
            (session / "session.json").write_text("{}", encoding="utf-8")
            (session / "frame_0000.jpg").write_bytes(b"jpeg")
            response = MagicMock()
            response.status = 200
            response.__enter__.return_value = response
            rate_limited = urllib.error.HTTPError(
                "https://example.invalid/v1/reports", 429, "Too Many Requests", {}, None
            )

            with patch("gta_helper.reporting.urllib.request.urlopen", side_effect=[rate_limited, response]) as urlopen:
                with patch("gta_helper.reporting.time.sleep") as sleep:
                    report_id = upload_report(session, "https://example.invalid/v1/reports")

            self.assertEqual(urlopen.call_count, 2)
            self.assertEqual(sleep.call_args_list[0].args, (5,))
            request_ids = [call.args[0].get_header("X-report-id") for call in urlopen.call_args_list]
            self.assertEqual(request_ids, [report_id, report_id])

    def test_upload_diagnostic_frame_sends_archive_without_local_session(self) -> None:
        frame = np.random.default_rng(7710).integers(0, 256, (90, 160, 3), dtype=np.uint8)
        response = MagicMock()
        response.status = 200
        response.__enter__.return_value = response

        with patch("gta_helper.reporting.urllib.request.urlopen", return_value=response) as urlopen:
            report_id = upload_diagnostic_frame(frame, "https://example.invalid/v1/reports", "gta")

        request = urlopen.call_args.args[0]
        metadata = validate_report_archive(request.data)
        self.assertEqual(metadata["capture_trigger"], "manual")
        self.assertEqual(metadata["frame_count"], 1)
        self.assertTrue(report_id)

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
