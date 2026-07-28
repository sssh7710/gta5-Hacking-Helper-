from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from gta_helper.capture import DiagnosticFrameRecorder, DxCapture, _pixel_standard_deviation


class DiagnosticFrameRecorderTests(unittest.TestCase):
    def test_fast_pixel_standard_deviation_matches_numpy(self) -> None:
        frames = [
            np.zeros((90, 160, 3), dtype=np.uint8),
            np.full((90, 160, 3), (0, 0, 255), dtype=np.uint8),
            np.random.default_rng(7710).integers(0, 256, (90, 160, 3), dtype=np.uint8),
        ]

        for frame in frames:
            with self.subTest(standard_deviation=float(frame.std())):
                self.assertAlmostEqual(_pixel_standard_deviation(frame), float(frame.std()), places=10)

    def test_stores_seven_second_attempt_as_photos_and_metadata(self) -> None:
        now = [100.0]
        frame = np.zeros((90, 160, 3), dtype=np.uint8)

        with tempfile.TemporaryDirectory() as directory:
            recorder = DiagnosticFrameRecorder(directory, duration_seconds=7.0, fps=2.0, clock=lambda: now[0])
            session_dir = recorder.start(frame, "keypad_5x4", {"grid_columns": 5})
            self.assertTrue(recorder.active)
            self.assertEqual(recorder.start(frame, "ignored"), session_dir)
            recorder.annotate(result_summary="정답", result_confidence=np.float32(0.91))

            now[0] = 100.2
            self.assertIsNone(recorder.add(frame))
            now[0] = 100.5
            self.assertIsNone(recorder.add(frame))
            now[0] = 107.0
            completed = recorder.add(frame)

            self.assertIsNotNone(completed)
            assert completed is not None
            self.assertEqual(completed.parent.name, "success")
            self.assertEqual(completed.name, session_dir.name)
            self.assertFalse(recorder.active)
            self.assertFalse(session_dir.exists())
            self.assertEqual(len(list(completed.glob("frame_*.jpg"))), 3)
            metadata = json.loads((completed / "session.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["frame_count"], 3)
            self.assertEqual(metadata["grid_columns"], 5)
            self.assertAlmostEqual(metadata["result_confidence"], 0.91, places=5)
            self.assertEqual(metadata["answer_outcome"], "success")
            self.assertTrue(metadata["answer_provided"])

    def test_stores_missing_answer_in_failure_folder(self) -> None:
        frame = np.zeros((90, 160, 3), dtype=np.uint8)

        with tempfile.TemporaryDirectory() as directory:
            recorder = DiagnosticFrameRecorder(directory)
            started = recorder.start(frame, "unresolved")
            completed = recorder.finish()

            self.assertIsNotNone(completed)
            assert completed is not None
            self.assertEqual(completed.parent.name, "failure")
            self.assertEqual(completed.name, started.name)
            metadata = json.loads((completed / "session.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["answer_outcome"], "failure")
            self.assertFalse(metadata["answer_provided"])

    def test_result_from_another_puzzle_does_not_overwrite_active_session(self) -> None:
        frame = np.zeros((90, 160, 3), dtype=np.uint8)

        with tempfile.TemporaryDirectory() as directory:
            recorder = DiagnosticFrameRecorder(directory)
            recorder.start(frame, "cayo_fingerprint", {"puzzle": "CAYO_FINGERPRINT"})
            self.assertTrue(recorder.annotate(
                expected_puzzle="CAYO_FINGERPRINT",
                result_summary="카요 정답",
                result_confidence=0.72,
            ))
            self.assertFalse(recorder.annotate(
                expected_puzzle="DOT_MEMORY",
                result_summary="점멸 원 정답 위치",
                result_confidence=0.90,
            ))
            completed = recorder.finish()

            self.assertIsNotNone(completed)
            assert completed is not None
            metadata = json.loads((completed / "session.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["puzzle"], "CAYO_FINGERPRINT")
            self.assertEqual(metadata["result_summary"], "카요 정답")
            self.assertAlmostEqual(metadata["result_confidence"], 0.72)

    def test_prunes_oldest_attempt_folder_when_total_limit_is_exceeded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            oldest = root / "attempt_20260101_000000_old"
            newer = root / "success" / "attempt_20260102_000000_new"
            oldest.mkdir()
            newer.mkdir(parents=True)
            (oldest / "frame.jpg").write_bytes(b"a" * 80)
            (newer / "frame.jpg").write_bytes(b"b" * 80)

            recorder = DiagnosticFrameRecorder(root, max_total_bytes=100)
            removed = recorder.prune_old_sessions()

            self.assertEqual(removed, [oldest])
            self.assertFalse(oldest.exists())
            self.assertTrue(newer.exists())

    def test_manual_diagnostic_is_saved_as_uploadable_session(self) -> None:
        frame = np.random.default_rng(7710).integers(0, 256, (90, 160, 3), dtype=np.uint8)

        with tempfile.TemporaryDirectory() as directory:
            session = DxCapture.save_diagnostic(frame, directory, "gta")

            self.assertEqual(session.parent.name, "manual")
            self.assertEqual(len(list(session.glob("frame_*.jpg"))), 1)
            metadata = json.loads((session / "session.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["capture_trigger"], "manual")
            self.assertEqual(metadata["frame_count"], 1)
            self.assertEqual(metadata["answer_outcome"], "failure")

    def test_manual_diagnostics_over_limit_are_reduced_to_half_limit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manual = root / "manual"
            manual.mkdir()
            sessions = []
            for index in range(4):
                session = manual / f"attempt_2026010{index + 1}_000000_000000_gta"
                session.mkdir()
                (session / "frame_0000_00000ms.jpg").write_bytes(bytes([index]) * 30)
                sessions.append(session)

            removed = DxCapture.prune_manual_diagnostics(root, max_total_bytes=100)

            self.assertEqual(removed, sessions[:3])
            self.assertFalse(any(session.exists() for session in sessions[:3]))
            self.assertTrue(sessions[3].exists())


if __name__ == "__main__":
    unittest.main()
