from __future__ import annotations

import unittest

import cv2
import numpy as np

from gta_helper.solvers import CayoFingerprintSolver, DotMemorySolver, FragmentFingerprintSolver, VoltLabSolver


def dot_frame(active: set[tuple[int, int]]) -> np.ndarray:
    image = np.zeros((540, 900, 3), dtype=np.uint8)
    for row in range(3):
        for col in range(4):
            color = (255, 245, 90) if (row, col) in active else (120, 65, 25)
            cv2.circle(image, (230 + col * 120, 150 + row * 115), 22, color, -1)
    return image


class SolverTests(unittest.TestCase):
    def test_dot_solver_confirms_repeated_pattern(self) -> None:
        solver = DotMemorySolver(repeats_needed=3)
        pattern = {(0, 1), (1, 3), (2, 0)}
        result = None
        for _ in range(3):
            result = solver.update(dot_frame(pattern))
            solver.update(dot_frame(set()))
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual([(point.row, point.column) for point in result.locations], [(1, 2), (2, 4), (3, 1)])
        self.assertGreaterEqual(result.confidence, 0.68)

    def test_fragment_solver_selects_four_matching_pieces(self) -> None:
        target = np.zeros((240, 240, 3), dtype=np.uint8)
        cv2.ellipse(target, (120, 120), (88, 104), 15, 0, 360, (255, 255, 255), 3)
        for radius in range(20, 100, 14):
            cv2.ellipse(target, (120, 120), (radius, radius + 8), 15, 20, 330, (180, 180, 180), 2)
        correct = [target[10:90, 10:90], target[20:100, 130:210], target[120:200, 20:100], target[130:210, 130:210]]
        noise = [np.random.default_rng(seed).integers(0, 20, (80, 80, 3), dtype=np.uint8) for seed in range(4)]
        result = FragmentFingerprintSolver().solve_regions(target, [correct[0], noise[0], correct[1], noise[1], correct[2], noise[2], correct[3], noise[3]])
        self.assertIsNotNone(result)
        assert result is not None
        self.assertIn("1번", result.details[0])
        self.assertIn("3번", result.details[0])
        self.assertIn("5번", result.details[0])
        self.assertIn("7번", result.details[0])

    def test_cayo_solver_reports_minimum_turn_direction(self) -> None:
        bands = []
        for index in range(5):
            band = np.zeros((40, 160, 3), dtype=np.uint8)
            cv2.putText(band, str(index), (55, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            bands.append(band)
        target = np.vstack(bands)
        current = [bands[2], bands[2], bands[4], bands[0], bands[3]]
        result = CayoFingerprintSolver().solve_regions(target, current)
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(len(result.details), 5)
        self.assertIn("1번 줄", result.details[0])

    def test_voltlab_solver_finds_unique_multiplier_mapping(self) -> None:
        result = VoltLabSolver().solve_values(95, [1, 4, 9], [1, 1, 10])
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.debug["multipliers"], (1, 1, 10))
        self.assertIn("9 → ×10", result.details)
