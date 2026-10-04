from __future__ import annotations

import queue
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from app import Scanner
from gta_helper.capture import DiagnosticFrameRecorder
from gta_helper.config import AppConfig
from gta_helper.models import PuzzleType, SolveResult


class ScannerDiagnosticTests(unittest.TestCase):
    def run_frames(self, directory, *, result=None, content_visible=True, new_session=False):
        config = AppConfig()
        config.diagnostic_dir = directory
        config.target_fps = 1000
        scanner = Scanner(config, queue.Queue())
        analyzer = Mock()
        analyzer.dot.grid_visible = False
        analyzer.dot.input_visible = False
        analyzer.dot.current_pattern = ()
        analyzer.casino_layout_checked = True
        analyzer.casino_screen_visible = True
        analyzer.casino_selection_visible = False
        analyzer.casino_content_visible = content_visible
        analyzer.update.return_value = result
        recorder = DiagnosticFrameRecorder(directory)
        recorder.annotate = Mock(wraps=recorder.annotate)
        if new_session:
            recorder.add = Mock(side_effect=lambda _frame: recorder.finish())
        capture = Mock(backend='test')
        calls = 0

        def grab(_game):
            nonlocal calls
            calls += 1
            if new_session:
                analyzer.casino_selection_visible = calls == 1
            if calls == 2:
                scanner.stop_event.set()
            return np.zeros((48, 64, 3), dtype=np.uint8)

        capture.grab.side_effect = grab
        with (
            patch('app.PuzzleAnalyzer', return_value=analyzer),
            patch('app.DiagnosticFrameRecorder', return_value=recorder),
            patch('app.DxCapture', return_value=capture),
            patch('app.find_game_window', return_value=object()),
        ):
            scanner.run()
        return scanner, recorder

    def test_duplicate_display_answer_still_annotates_active_diagnostic(self):
        answer = SolveResult(PuzzleType.FRAGMENT_FINGERPRINT, .86, 'synthetic answer')
        with tempfile.TemporaryDirectory() as directory:
            scanner, recorder = self.run_frames(directory, result=answer)
            self.assertEqual(recorder.annotate.call_count, 2)
            events = list(scanner.events.queue)
            self.assertEqual(sum(kind == 'result' for kind, _ in events), 1)
            self.assertEqual(len(list((Path(directory) / 'success').glob('*'))), 1)

    def test_empty_transition_panel_does_not_start_failed_diagnostic(self):
        with tempfile.TemporaryDirectory() as directory:
            self.run_frames(directory, content_visible=False)
            self.assertEqual(list(Path(directory).rglob('session.json')), [])

    def test_same_answer_is_recorded_in_new_session_after_selection(self):
        answer = SolveResult(PuzzleType.FRAGMENT_FINGERPRINT, .86, 'synthetic answer')
        with tempfile.TemporaryDirectory() as directory:
            self.run_frames(directory, result=answer, new_session=True)
            self.assertEqual(len(list((Path(directory) / 'success').glob('*'))), 2)

    def test_visible_unresolved_fingerprint_still_records_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            self.run_frames(directory, content_visible=True)
            self.assertEqual(len(list((Path(directory) / 'failure').glob('*'))), 1)


if __name__ == '__main__':
    unittest.main()
