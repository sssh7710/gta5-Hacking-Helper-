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
from gta_helper.models import GridPoint, PuzzleType, SolveResult


class ScannerDiagnosticTests(unittest.TestCase):
    def test_keypad_detection_gap_does_not_erase_confirmed_answer(self):
        config = AppConfig()
        config.target_fps = 1000
        config.diagnostic_capture_enabled = False
        scanner = Scanner(config, queue.Queue())
        answer = SolveResult(PuzzleType.DOT_MEMORY, .89, 'synthetic answer')
        analyzer = Mock()
        analyzer.dot.input_visible = False
        analyzer.dot.current_grid_shape = (5, 6)
        analyzer.casino_layout_checked = False
        calls = 0

        def analyze(_frame):
            nonlocal calls
            calls += 1
            analyzer.dot.grid_visible = calls in (1, 17)
            analyzer.dot.current_pattern = (GridPoint(1, 1),) if analyzer.dot.grid_visible else ()
            if calls == 17:
                scanner.stop_event.set()
            return answer if calls == 1 else None

        analyzer.update.side_effect = analyze
        capture = Mock(backend='test')
        capture.grab.return_value = np.zeros((48, 64, 3), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as directory:
            config.diagnostic_dir = directory
            with (
                patch('app.PuzzleAnalyzer', return_value=analyzer),
                patch('app.DxCapture', return_value=capture),
                patch('app.find_game_window', return_value=object()),
            ):
                scanner.run()
        events = list(scanner.events.queue)
        answer_index = next(i for i, (kind, _) in enumerate(events) if kind == 'result')
        self.assertFalse(any(kind == 'keypad_start' for kind, _ in events[answer_index + 1:]))
        self.assertTrue(any(kind == 'status' and '재감지' in payload
                            for kind, payload in events[answer_index + 1:]))

    def test_transient_keypad_pattern_does_not_erase_confirmed_answer(self):
        config = AppConfig()
        config.target_fps = 1000
        config.diagnostic_capture_enabled = False
        scanner = Scanner(config, queue.Queue())
        answer = SolveResult(PuzzleType.DOT_MEMORY, .89, 'synthetic answer')
        analyzer = Mock()
        analyzer.dot.grid_visible = True
        analyzer.dot.input_visible = False
        analyzer.dot.current_grid_shape = (5, 6)
        analyzer.casino_layout_checked = False
        calls = 0

        def analyze(_frame):
            nonlocal calls
            calls += 1
            analyzer.dot.current_pattern = (GridPoint(1 if calls == 1 else 2, 1),)
            if calls == 2:
                scanner.stop_event.set()
            return answer if calls == 1 else None

        analyzer.update.side_effect = analyze
        capture = Mock(backend='test')
        capture.grab.return_value = np.zeros((48, 64, 3), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as directory:
            config.diagnostic_dir = directory
            with (
                patch('app.PuzzleAnalyzer', return_value=analyzer),
                patch('app.DxCapture', return_value=capture),
                patch('app.find_game_window', return_value=object()),
            ):
                scanner.run()
        events = list(scanner.events.queue)
        answer_index = next(i for i, (kind, _) in enumerate(events) if kind == 'result')
        self.assertFalse(any(kind == 'keypad_start' for kind, _ in events[answer_index + 1:]))
        self.assertTrue(any(kind == 'status' and '이전 정답' in payload
                            for kind, payload in events[answer_index + 1:]))

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
