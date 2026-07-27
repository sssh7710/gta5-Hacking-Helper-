from __future__ import annotations

import io
import json
import tempfile
import threading
import unittest
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from server.receiver import ReportServer, ValidationError, cleanup_reports, store_report, validate_report_archive


def report_zip(metadata: dict[str, object] | None = None, image: bytes = b"\xff\xd8jpeg\xff\xd9") -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("session.json", json.dumps(metadata or {"label": "pending"}))
        archive.writestr("frame_0000_00000ms.jpg", image)
    return output.getvalue()


class ReceiverTests(unittest.TestCase):
    def test_accepts_expected_session_archive(self) -> None:
        metadata = validate_report_archive(report_zip({"label": "fragment_fingerprint_pending"}))
        self.assertEqual(metadata["label"], "fragment_fingerprint_pending")

    def test_rejects_non_jpeg_and_nested_paths(self) -> None:
        with self.assertRaises(ValidationError):
            validate_report_archive(report_zip(image=b"not-jpeg"))

        output = io.BytesIO()
        with zipfile.ZipFile(output, "w") as archive:
            archive.writestr("session.json", "{}")
            archive.writestr("nested/frame_0000_00000ms.jpg", b"\xff\xd8x\xff\xd9")
        with self.assertRaises(ValidationError):
            validate_report_archive(output.getvalue())

    def test_store_is_idempotent_and_cleanup_removes_expired_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            now = datetime(2026, 7, 27, tzinfo=timezone.utc)
            data = report_zip()
            path, created = store_report(data, "a" * 32, directory, now=now)
            duplicate, created_again = store_report(data, "a" * 32, directory, now=now)
            self.assertTrue(created)
            self.assertFalse(created_again)
            self.assertEqual(path, duplicate)
            self.assertEqual(path.read_bytes(), data)

            removed = cleanup_reports(directory, 30, now=datetime(2026, 9, 1, tzinfo=timezone.utc))
            self.assertEqual(removed, 1)
            self.assertFalse(path.exists())

    def test_http_receiver_accepts_declared_content_length(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            server = ReportServer(("127.0.0.1", 0), directory)
            worker = threading.Thread(target=server.serve_forever, daemon=True)
            worker.start()
            data = report_zip()
            report_id = "b" * 32
            request = urllib.request.Request(
                f"http://127.0.0.1:{server.server_port}/v1/reports",
                data=data,
                method="POST",
                headers={
                    "Content-Type": "application/zip",
                    "Content-Length": str(len(data)),
                    "X-Report-Id": report_id,
                    "X-App-Version": "1.0.0-beta.9",
                },
            )
            try:
                with urllib.request.urlopen(request, timeout=3) as response:
                    self.assertEqual(response.status, 200)
            finally:
                server.shutdown()
                server.server_close()
                worker.join(timeout=3)
            self.assertEqual(len(list(Path(directory).glob(f"????-??-??/{report_id}.zip"))), 1)


if __name__ == "__main__":
    unittest.main()
