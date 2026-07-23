from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from gta_helper.analyzer import PuzzleAnalyzer


class AnalyzerTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
