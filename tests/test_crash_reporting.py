from __future__ import annotations

import io
import json
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import Mock, patch

from gta_helper.crash_reporting import CrashReporter
from server.receiver import ValidationError, StorageFullError, classify_report, store_report, validate_report_archive


class CrashReportingTests(unittest.TestCase):
    def test_unhandled_startup_exception_is_saved_before_exit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text('{"diagnostic_upload_enabled":false}')
            script = (
                "from pathlib import Path; from gta_helper.crash_reporting import CrashReporter; "
                "import sys; reporter=CrashReporter(Path(sys.argv[1])); reporter.install(); "
                "raise RuntimeError('sensitive-message')"
            )
            result = subprocess.run([sys.executable, "-B", "-c", script, directory], capture_output=True, timeout=10)
            self.assertNotEqual(result.returncode, 0)
            files = list((root / "diagnostics" / "app-errors").glob("*.json"))
            self.assertEqual(len(files), 1)
            self.assertEqual(json.loads(files[0].read_text())["source"], "main")
            self.assertNotIn("sensitive-message", files[0].read_text())

    def test_repeated_error_is_throttled_and_local_storage_is_bounded(self):
        with tempfile.TemporaryDirectory() as directory:
            reporter = CrashReporter(Path(directory))
            self.assertIsNotNone(self.record(reporter))
            self.assertIsNone(self.record(reporter))
            for index in range(110):
                reporter.record(RuntimeError, RuntimeError(), None, "ui" if index % 2 else "thread")
            self.assertEqual(len(list(reporter.directory.glob("*.json"))), 100)

    def test_thread_and_ui_exceptions_are_recorded(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text('{"diagnostic_upload_enabled":false}')
            script = """
import sys, threading
from pathlib import Path
from gta_helper.crash_reporting import CrashReporter
reporter = CrashReporter(Path(sys.argv[1]))
reporter.install()
def fail():
    raise ValueError('synthetic')
worker = threading.Thread(target=fail)
worker.start()
worker.join()
try:
    fail()
except ValueError as exc:
    reporter.tk_exception(type(exc), exc, exc.__traceback__)
"""
            result = subprocess.run([sys.executable, "-B", "-c", script, directory], capture_output=True, timeout=10)
            self.assertEqual(result.returncode, 0)
            sources = {json.loads(path.read_text())["source"] for path in (root / "diagnostics/app-errors").glob("*.json")}
            self.assertEqual(sources, {"thread", "ui"})

    def record(self, reporter):
        try:
            raise RuntimeError("private-token-and-user-path")
        except RuntimeError as exc:
            return reporter.record(type(exc), exc, exc.__traceback__, "main")

    def test_private_message_and_paths_are_omitted_and_archive_is_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            reporter = CrashReporter(Path(directory))
            path = self.record(reporter)
            text = path.read_text(encoding="utf-8")
            self.assertNotIn("private-token", text)
            metadata = json.loads(text)
            self.assertEqual(metadata["stack"][0]["file"], "test_crash_reporting.py")
            response = Mock(status=200)
            response.__enter__ = Mock(return_value=response)
            response.__exit__ = Mock(return_value=False)
            with patch("gta_helper.crash_reporting.urllib.request.urlopen", return_value=response) as upload:
                self.assertTrue(reporter.send(path))
            request = upload.call_args.args[0]
            received = validate_report_archive(request.data)
            self.assertEqual(classify_report(received), "app_error")
            self.assertFalse(path.exists())

    def test_disabled_upload_and_network_failure_keep_pending_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reporter = CrashReporter(root)
            path = self.record(reporter)
            (root / "config.json").write_text('{"diagnostic_upload_enabled":false}')
            with patch("gta_helper.crash_reporting.urllib.request.urlopen") as upload:
                self.assertFalse(reporter.send(path))
                upload.assert_not_called()
            (root / "config.json").write_text('{"diagnostic_upload_enabled":true}')
            with patch("gta_helper.crash_reporting.urllib.request.urlopen", side_effect=OSError("offline")):
                self.assertFalse(reporter.send(path))
            self.assertTrue(path.exists())

    def test_independent_half_quotas_and_idempotency(self):
        with tempfile.TemporaryDirectory() as directory, patch("server.receiver.MAX_STORAGE_BYTES", 20):
            path, created = store_report(b"12345678", "a"*32, directory, "success")
            self.assertTrue(created)
            with self.assertRaises(StorageFullError):
                store_report(b"123", "b"*32, directory, "failure")
            error, created = store_report(b"12345678", "c"*32, directory, "app_error")
            self.assertTrue(created)
            self.assertEqual(error.parent.parent.name, "app_error")
            self.assertFalse(store_report(b"12345678", "c"*32, directory, "app_error")[1])
            with self.assertRaises(StorageFullError):
                store_report(b"123", "d"*32, directory, "app_error")

    def test_rejects_missing_frames_and_invalid_error_metadata(self):
        for metadata in ({}, {"report_type": "app_error", "source": []}):
            buffer = io.BytesIO()
            with zipfile.ZipFile(buffer, "w") as archive:
                archive.writestr("session.json", json.dumps(metadata))
            with self.assertRaises(ValidationError):
                validate_report_archive(buffer.getvalue())
