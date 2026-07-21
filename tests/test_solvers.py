from __future__ import annotations

import unittest

import cv2
import numpy as np

from gta_helper.solvers import CayoFingerprintSolver, DotMemorySolver, FragmentFingerprintSolver, VoltLabSolver


def dot_frame(active: set[tuple[int, int]], red_active: set[tuple[int, int]] | None = None) -> np.ndarray:
    red_active = red_active or set()
    image = np.zeros((540, 1400, 3), dtype=np.uint8)
    for row in range(5):
        for col in range(6):
            center = (320 + col * 95, 130 + row * 72)
            cv2.circle(image, center, 22, (95, 95, 95), 2)
            if (row, col) in active:
                cv2.circle(image, center, 13, (255, 245, 90), -1)
            if (row, col) in red_active:
                cv2.circle(image, center, 13, (40, 40, 220), -1)
    return image


class SolverTests(unittest.TestCase):
    def test_dot_solver_accepts_second_matching_complete_pattern_by_default(self) -> None:
        solver = DotMemorySolver()
        pattern = {(0, 0), (0, 2), (0, 3), (0, 4), (2, 5), (3, 1)}
        self.assertIsNone(solver.update(dot_frame(pattern)))
        solver.update(dot_frame(set()))
        result = solver.update(dot_frame(pattern))
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(
            [(point.row, point.column) for point in result.locations],
            [(1, 1), (1, 3), (1, 4), (1, 5), (3, 6), (4, 2)],
        )
        display = result.display_text()
        self.assertIn("1번째 줄: 1번, 3번, 4번, 5번", display)
        self.assertIn("2번째 줄: 없음", display)
        self.assertIn("4번째 줄: 2번", display)
        self.assertIn("5번째 줄: 없음", display)
        self.assertNotIn("위에서", display)
        self.assertEqual(display.count("1번째 줄"), 1)

    def test_dot_solver_accepts_consecutive_complete_frames(self) -> None:
        solver = DotMemorySolver()
        pattern = {(0, 4), (2, 3), (2, 5), (3, 2), (4, 0), (4, 1)}
        self.assertIsNone(solver.update(dot_frame(pattern)))
        result = solver.update(dot_frame(pattern))
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(
            [(point.row, point.column) for point in result.locations],
            [(1, 5), (3, 4), (3, 6), (4, 3), (5, 1), (5, 2)],
        )

    def test_dot_solver_confirms_repeated_pattern(self) -> None:
        solver = DotMemorySolver(repeats_needed=3)
        pattern = {(0, 1), (0, 5), (1, 3), (2, 4), (3, 2), (4, 0)}
        result = None
        for _ in range(3):
            result = solver.update(dot_frame(pattern))
            solver.update(dot_frame(set()))
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual([(point.row, point.column) for point in result.locations], [(1, 2), (1, 6), (2, 4), (3, 5), (4, 3), (5, 1)])
        self.assertGreaterEqual(result.confidence, 0.68)

    def test_dot_solver_keeps_first_answer_until_grid_disappears(self) -> None:
        solver = DotMemorySolver(repeats_needed=2)
        first = {(0, 0), (0, 5), (1, 1), (2, 2), (3, 3), (4, 4)}
        second = {(0, 0), (0, 5), (1, 4), (2, 3), (3, 2), (4, 1)}
        result = None
        for _ in range(2):
            result = solver.update(dot_frame(first))
            solver.update(dot_frame(set()))
        self.assertIsNotNone(result)
        for _ in range(3):
            self.assertIsNone(solver.update(dot_frame(second)))
            solver.update(dot_frame(set()))

    def test_dot_solver_ignores_incomplete_animation(self) -> None:
        solver = DotMemorySolver(repeats_needed=2)
        for _ in range(4):
            self.assertIsNone(solver.update(dot_frame({(0, 0)})))
            solver.update(dot_frame(set()))
        five_points = {(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)}
        self.assertIsNone(solver.update(dot_frame(five_points)))
        self.assertIsNone(solver.update(dot_frame(five_points)))

    def test_dot_solver_rearms_for_second_pattern_after_red_input(self) -> None:
        solver = DotMemorySolver(repeats_needed=2)
        first = {(0, 4), (2, 3), (2, 5), (3, 2), (4, 0), (4, 1)}
        second = {(1, 3), (1, 5), (2, 0), (2, 1), (3, 2), (3, 4)}
        result = None
        for _ in range(2):
            result = solver.update(dot_frame(first))
            solver.update(dot_frame(set()))
        self.assertIsNotNone(result)

        self.assertIsNone(solver.update(dot_frame(set(), {(0, 0)})))
        for _ in range(2):
            result = solver.update(dot_frame(second))
            solver.update(dot_frame(set()))

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(
            [(point.row, point.column) for point in result.locations],
            [(2, 4), (2, 6), (3, 1), (3, 2), (4, 3), (4, 5)],
        )

    def test_dot_solver_rearms_after_answer_disappears(self) -> None:
        solver = DotMemorySolver(repeats_needed=2)
        first = {(0, 0), (0, 5), (1, 1), (2, 2), (3, 3), (4, 4)}
        second = {(0, 0), (0, 5), (1, 4), (2, 3), (3, 2), (4, 1)}
        result = None
        for _ in range(2):
            result = solver.update(dot_frame(first))
            solver.update(dot_frame(set()))
        self.assertIsNotNone(result)
        for _ in range(15):
            solver.update(dot_frame(set()))
        result = None
        for _ in range(2):
            result = solver.update(dot_frame(second))
            solver.update(dot_frame(set()))
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual([(point.row, point.column) for point in result.locations], [(1, 1), (1, 6), (2, 5), (3, 4), (4, 3), (5, 2)])

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
        self.assertGreaterEqual(result.confidence, .68)

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
