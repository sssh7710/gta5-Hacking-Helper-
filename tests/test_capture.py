from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from gta_helper.capture import DiagnosticFrameRecorder


class DiagnosticFrameRecorderTests(unittest.TestCase):
    def test_stores_seven_second_attempt_as_photos_and_metadata(self) -> None:
        now = [100.0]
        frame = np.zeros((90, 160, 3), dtype=np.uint8)

        with tempfile.TemporaryDirectory() as directory:
            recorder = DiagnosticFrameRecorder(directory, duration_seconds=7.0, fps=2.0, clock=lambda: now[0])
            session_dir = recorder.start(frame, "keypad_5x4", {"grid_columns": 5})
            self.assertTrue(recorder.active)
            self.assertEqual(recorder.start(frame, "ignored"), session_dir)
            recorder.annotate(result_confidence=np.float32(0.91))

            now[0] = 100.2
            self.assertIsNone(recorder.add(frame))
            now[0] = 100.5
            self.assertIsNone(recorder.add(frame))
            now[0] = 107.0
            completed = recorder.add(frame)

            self.assertEqual(completed, session_dir)
            self.assertFalse(recorder.active)
            self.assertEqual(len(list(session_dir.glob("frame_*.jpg"))), 3)
            metadata = json.loads((session_dir / "session.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["frame_count"], 3)
            self.assertEqual(metadata["grid_columns"], 5)
            self.assertAlmostEqual(metadata["result_confidence"], 0.91, places=5)

    def test_prunes_oldest_attempt_folder_when_total_limit_is_exceeded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            oldest = root / "attempt_20260101_000000_old"
            newer = root / "attempt_20260102_000000_new"
            oldest.mkdir()
            newer.mkdir()
            (oldest / "frame.jpg").write_bytes(b"a" * 80)
            (newer / "frame.jpg").write_bytes(b"b" * 80)

            recorder = DiagnosticFrameRecorder(root, max_total_bytes=100)
            removed = recorder.prune_old_sessions()

            self.assertEqual(removed, [oldest])
            self.assertFalse(oldest.exists())
            self.assertTrue(newer.exists())


if __name__ == "__main__":
    unittest.main()
