from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from gta_helper.analyzer import PuzzleAnalyzer
from gta_helper.models import PuzzleType, SolveResult


class AnalyzerTests(unittest.TestCase):
    def test_high_resolution_frame_is_downscaled_for_analysis_only(self) -> None:
        frame = np.zeros((1800, 2880, 3), dtype=np.uint8)
        analyzer = PuzzleAnalyzer()
        analyzer.dot.update = Mock(return_value=None)

        analyzer.update(frame)

        analyzed = analyzer.dot.update.call_args.args[0]
        self.assertEqual(analyzed.shape, (1080, 1728, 3))
        self.assertEqual(frame.shape, (1800, 2880, 3))

    def test_recent_keypad_grid_is_not_treated_as_fingerprint(self) -> None:
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        analyzer = PuzzleAnalyzer()
        updates = 0

        def detect_keypad(_frame: np.ndarray) -> None:
            nonlocal updates
            updates += 1
            analyzer.dot._grid_visible = updates == 1
            return None

        analyzer.dot.update = Mock(side_effect=detect_keypad)
        analyzer._frame_number = 1

        with (
            patch("gta_helper.analyzer.casino_fingerprint_layout") as casino_layout,
            patch("gta_helper.analyzer.cayo_layout") as cayo_layout,
        ):
            results = [analyzer.update(frame) for _ in range(3)]

        self.assertTrue(all(result is None for result in results))
        casino_layout.assert_not_called()
        cayo_layout.assert_not_called()
        self.assertFalse(analyzer.casino_layout_checked)
        self.assertEqual(analyzer._keypad_guard_frames, 13)

    def test_generic_rectangles_do_not_trigger_fingerprint_answer(self) -> None:
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        for row in range(2):
            for column in range(4):
                left = 80 + column * 120
                top = 100 + row * 140
                cv2.rectangle(frame, (left, top), (left + 90, top + 100), (255, 255, 255), 3)
                cv2.line(frame, (left + 10, top + 20), (left + 75, top + 80), (255, 255, 255), 3)
        cv2.rectangle(frame, (40, 390), (590, 680), (255, 255, 255), 4)

        analyzer = PuzzleAnalyzer()
        results = [analyzer.update(frame) for _ in range(8)]

        self.assertTrue(all(result is None for result in results))

    def test_selected_casino_components_are_not_reanalyzed_or_treated_as_cayo(self) -> None:
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        target = np.zeros((300, 220, 3), dtype=np.uint8)
        candidates = [np.full((80, 80, 3), 10, dtype=np.uint8) for _ in range(8)]
        for index in (2, 4, 5, 7):
            candidates[index - 1][:] = 90
        analyzer = PuzzleAnalyzer()
        analyzer.dot.update = Mock(return_value=None)
        analyzer.fragment.solve_regions = Mock()

        with (
            patch("gta_helper.analyzer.casino_fingerprint_layout", return_value=(target, candidates)),
            patch("gta_helper.analyzer.cayo_layout") as cayo_layout,
        ):
            results = [analyzer.update(frame) for _ in range(4)]

        self.assertTrue(all(result is None for result in results))
        analyzer.fragment.solve_regions.assert_not_called()
        cayo_layout.assert_not_called()
        self.assertTrue(analyzer.casino_layout_checked)
        self.assertTrue(analyzer.casino_screen_visible)
        self.assertTrue(analyzer.casino_selection_visible)

    def test_fingerprint_candidate_is_rechecked_on_the_next_frame(self) -> None:
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        target = np.zeros((300, 220, 3), dtype=np.uint8)
        candidates = [np.full((80, 80, 3), 10, dtype=np.uint8) for _ in range(8)]
        answer = SolveResult(
            PuzzleType.FRAGMENT_FINGERPRINT,
            .90,
            "지문 조각 정답",
            details=["선택: 1번 · 3번 · 5번 · 7번"],
        )
        analyzer = PuzzleAnalyzer()
        analyzer.dot.update = Mock(return_value=None)
        analyzer.fragment.solve_regions = Mock(return_value=answer)

        with patch("gta_helper.analyzer.casino_fingerprint_layout", return_value=(target, candidates)):
            results = [analyzer.update(frame) for _ in range(3)]

        self.assertIsNone(results[0])
        self.assertIsNone(results[1])
        self.assertEqual(results[2], answer)
        self.assertEqual(analyzer.dot.update.call_count, 2)
        self.assertEqual(analyzer.fragment.solve_regions.call_count, 2)


if __name__ == "__main__":
    unittest.main()
