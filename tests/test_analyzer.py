from __future__ import annotations

import unittest

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


if __name__ == "__main__":
    unittest.main()
