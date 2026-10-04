from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

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


def kortz_frame(active: set[tuple[int, int]]) -> np.ndarray:
    image = np.zeros((540, 1400, 3), dtype=np.uint8)
    for row in range(4):
        for col in range(5):
            center = (320 + col * 95, 130 + row * 72)
            cv2.circle(image, center, 22, (95, 95, 95), 2)
            if (row, col) in active:
                cv2.circle(image, center, 13, (255, 245, 90), -1)
    return image


class SolverTests(unittest.TestCase):
    def test_arcade_faint_dotted_grid_and_complete_pattern(self) -> None:
        # 실제 FHD 화면에서 격자만 남기고 나머지를 검게 가린 자료다.
        data = np.fromfile(Path(__file__).with_name('arcade_keypad_grid.png'), dtype=np.uint8)
        frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
        self.assertIsNotNone(frame)
        solver = DotMemorySolver()
        self.assertIsNone(solver.update(frame))
        self.assertTrue(solver.grid_visible)
        self.assertEqual(solver.current_grid_shape, (5, 6))

        # 실제 격자에 알려진 배열을 합성해 좌표 매핑도 확인한다.
        rows = (1, 3, 2, 4, 0, 2)
        for column, row in enumerate(rows):
            cv2.circle(frame, (499 + column * 108, 343 + row * 108), 30, (240, 200, 50), -1)
        self.assertIsNone(solver.update(frame))
        self.assertIsNone(solver.update(frame))
        self.assertIsNone(solver.update(frame))
        self.assertIsNone(solver.update(cv2.imdecode(data, cv2.IMREAD_COLOR)))
        result = solver.update(frame)
        self.assertIsNotNone(result)
        self.assertEqual(sorted((p.column, p.row) for p in result.locations),
                         list(enumerate((row + 1 for row in rows), start=1)))

    def test_arcade_partially_bright_columns_keep_six_by_five_grid(self) -> None:
        data = np.fromfile(Path(__file__).with_name('arcade_keypad_grid.png'), dtype=np.uint8)
        frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
        for row in range(5):
            for column in range(4):
                cv2.circle(frame, (499 + column * 108, 343 + row * 108), 43, (150, 150, 150), 2)
        solver = DotMemorySolver()
        self.assertIsNone(solver.update(frame))
        self.assertTrue(solver.grid_visible)
        self.assertEqual(solver.current_grid_shape, (5, 6))

    def test_known_six_by_five_grid_does_not_shrink_to_detected_inner_subset(self) -> None:
        solver = DotMemorySolver()
        pattern = {(0, 0), (0, 5), (1, 1), (2, 2), (3, 3), (4, 4)}
        frame = dot_frame(pattern)
        self.assertIsNone(solver.update(frame))
        self.assertEqual(solver.current_grid_shape, (5, 6))
        subset_circles = np.array([[
            (320 + col * 95, 130 + row * 72, 22)
            for row in range(4) for col in range(5)
        ]], dtype=np.float32)
        with patch('gta_helper.solvers.cv2.HoughCircles', return_value=subset_circles):
            self.assertIsNone(solver.update(frame))
            result = solver.update(frame)
            self.assertIsNotNone(result)
            self.assertEqual(result.debug['grid_columns'], 6)
            self.assertEqual(len(result.locations), 6)
            solver.reset()
            self.assertIsNone(solver._grid_geometry)
            solver.update(frame)
            self.assertEqual(solver.current_grid_shape, (4, 5))

    def test_confirmation_blinks_do_not_seed_next_round_repeat_counts(self) -> None:
        solver = DotMemorySolver()
        first = {(0, 0), (0, 5), (1, 1), (2, 2), (3, 3), (4, 4)}
        middle = {(0, 4), (1, 0), (1, 2), (2, 5), (3, 1), (3, 3)}
        solver.update(dot_frame(first))
        solver.update(dot_frame(set()))
        self.assertIsNotNone(solver.update(dot_frame(first)))
        for _ in range(3):
            solver.update(dot_frame(set()))
            self.assertIsNone(solver.update(dot_frame(first)))
        solver.update(dot_frame(set()))
        self.assertIsNone(solver.update(dot_frame(middle)))
        solver.update(dot_frame(set()))
        self.assertIsNone(solver.update(dot_frame(first)))
        solver.update(dot_frame(set()))
        self.assertIsNotNone(solver.update(dot_frame(first)))

    def test_known_grid_recovers_when_only_eighteen_rings_are_detected(self) -> None:
        solver = DotMemorySolver()
        pattern = {(0, 0), (0, 5), (1, 1), (2, 2), (3, 3), (4, 4)}
        frame = dot_frame(pattern)
        self.assertIsNone(solver.update(frame))
        solver.update(dot_frame(set()))
        sparse_circles = np.array([[
            (320 + col * 95, 130 + row * 72, 22)
            for row in range(3) for col in range(6)
        ]], dtype=np.float32)
        with patch('gta_helper.solvers.cv2.HoughCircles', return_value=sparse_circles):
            result = solver.update(frame)
            self.assertTrue(solver.grid_visible)
            self.assertEqual(solver.current_grid_shape, (5, 6))
            self.assertIsNotNone(result)
            self.assertEqual(len(result.locations), 6)

            fresh_solver = DotMemorySolver()
            self.assertIsNone(fresh_solver.update(frame))
            self.assertFalse(fresh_solver.grid_visible)
        self.assertIsNone(solver.update(np.zeros_like(frame)))
        self.assertFalse(solver.grid_visible)

    def test_fingerprint_scores_reuse_only_identical_preprocessed_inputs(self) -> None:
        target = np.zeros((100, 100, 3), dtype=np.uint8)
        target[20:60, 20:60] = 255
        candidates = [np.zeros((40, 40, 3), dtype=np.uint8) for _ in range(8)]
        solver = FragmentFingerprintSolver()
        with patch('gta_helper.solvers._score_prepared_fingerprint_piece', return_value=.8) as score:
            solver.solve_regions(target, candidates)
            solver.solve_regions(target.copy(), [piece.copy() for piece in candidates])
            self.assertEqual(score.call_count, 8)
            candidates[0][10:25, 10:25] = 255
            solver.solve_regions(target, candidates)
            self.assertEqual(score.call_count, 16)
            target[60:80, 60:80] = 255
            solver.solve_regions(target, candidates)
            self.assertEqual(score.call_count, 24)
            solver.solve_regions(target, candidates[:-1])
            self.assertEqual(score.call_count, 31)

    def test_dot_solver_selects_supported_regular_axis_over_spurious_circles(self) -> None:
        values = [499] * 4 + [607] * 4 + [715] * 4 + [823] * 4 + [930] * 4 + [442, 550, 877]
        self.assertEqual(DotMemorySolver._regular_axis(values, tolerance=19, size=5), [499, 607, 715, 823, 930])

    def test_dot_solver_detects_kortz_center_hard_six_by_five_pattern(self) -> None:
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
        self.assertIn("1번 신호: 1번째 칸", display)
        self.assertIn("2번 신호: 4번째 칸", display)
        self.assertIn("3번 신호: 1번째 칸", display)
        self.assertIn("4번 신호: 1번째 칸", display)
        self.assertIn("5번 신호: 1번째 칸", display)
        self.assertIn("6번 신호: 3번째 칸", display)
        self.assertNotIn("위에서", display)
        self.assertNotIn("번째 줄", display)

    def test_dot_solver_does_not_treat_partial_six_by_five_grid_as_five_by_four(self) -> None:
        solver = DotMemorySolver(repeats_needed=2)
        pattern = {(4, 0), (1, 1), (2, 2), (3, 3), (4, 4), (1, 5)}
        frame = dot_frame(pattern)
        cv2.circle(frame, (320, 130), 30, (0, 0, 0), -1)

        self.assertIsNone(solver.update(frame))
        self.assertTrue(solver.grid_visible)
        self.assertEqual(solver.current_grid_shape, (5, 6))
        solver.update(dot_frame(set()))
        result = solver.update(frame)

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(
            [(point.row, point.column) for point in result.locations],
            [(2, 2), (2, 6), (3, 3), (4, 4), (5, 1), (5, 5)],
        )

    def test_dot_solver_waits_for_repeated_final_pattern(self) -> None:
        solver = DotMemorySolver()
        first = {(0, 3), (2, 0), (2, 2), (2, 4), (3, 1), (3, 5)}
        middle = {(0, 2), (1, 1), (1, 5), (2, 0), (2, 3), (3, 4)}
        final = {(0, 2), (0, 4), (2, 0), (2, 3), (2, 5), (3, 1)}

        for pattern in (first, first, middle, middle, final, final):
            self.assertIsNone(solver.update(dot_frame(pattern)))
        solver.update(dot_frame(set()))
        result = solver.update(dot_frame(final))
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(
            [(point.row, point.column) for point in result.locations],
            [(1, 3), (1, 5), (3, 1), (3, 4), (3, 6), (4, 2)],
        )

    def test_dot_solver_confirms_complete_pattern_held_for_three_frames(self) -> None:
        solver = DotMemorySolver()
        pattern = {(0, 0), (0, 5), (1, 2), (2, 3), (3, 1), (4, 4)}

        self.assertIsNone(solver.update(dot_frame(pattern)))
        self.assertIsNone(solver.update(dot_frame(pattern)))
        result = solver.update(dot_frame(pattern))

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.debug["completion"], "stable")
        self.assertEqual(result.debug["stable_frames"], 3)

    def test_dot_solver_uses_last_complete_pattern_when_final_repeat_is_dropped(self) -> None:
        solver = DotMemorySolver(final_blank_frames=6)
        first = {(0, 0), (0, 5), (1, 1), (2, 2), (3, 3), (4, 4)}
        middle = {(0, 4), (1, 0), (1, 2), (2, 5), (3, 1), (3, 3)}
        final = {(0, 2), (1, 0), (1, 4), (2, 3), (3, 1), (3, 5)}

        for pattern in (first, middle):
            self.assertIsNone(solver.update(dot_frame(pattern)))
            self.assertIsNone(solver.update(dot_frame(set())))
        self.assertIsNone(solver.update(dot_frame(final)))
        for _ in range(5):
            self.assertIsNone(solver.update(dot_frame(set())))

        result = solver.update(dot_frame(set()))

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(
            [(point.row, point.column) for point in result.locations],
            [(1, 3), (2, 1), (2, 5), (3, 4), (4, 2), (4, 6)],
        )
        self.assertEqual(result.debug["completion"], "animation_end")
        self.assertGreaterEqual(result.confidence, 0.68)

    def test_dot_solver_does_not_fallback_after_only_two_patterns(self) -> None:
        solver = DotMemorySolver(final_blank_frames=3)
        first = {(0, 0), (0, 5), (1, 1), (2, 2), (3, 3), (4, 4)}
        second = {(0, 4), (1, 0), (1, 2), (2, 5), (3, 1), (3, 3)}

        for pattern in (first, second):
            self.assertIsNone(solver.update(dot_frame(pattern)))
            self.assertIsNone(solver.update(dot_frame(set())))
        for _ in range(4):
            self.assertIsNone(solver.update(dot_frame(set())))

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

    def test_dot_solver_emits_next_round_without_grid_disappearing(self) -> None:
        solver = DotMemorySolver(repeats_needed=2)
        first = {(0, 0), (0, 5), (1, 1), (2, 2), (3, 3), (4, 4)}
        second = {(0, 0), (0, 5), (1, 4), (2, 3), (3, 2), (4, 1)}
        result = None
        for _ in range(2):
            result = solver.update(dot_frame(first))
            solver.update(dot_frame(set()))
        self.assertIsNotNone(result)
        result = None
        for _ in range(2):
            result = solver.update(dot_frame(second))
            solver.update(dot_frame(set()))
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(
            [(point.row, point.column) for point in result.locations],
            [(1, 1), (1, 6), (2, 5), (3, 4), (4, 3), (5, 2)],
        )

    def test_dot_solver_does_not_reuse_previous_round_animation_count(self) -> None:
        solver = DotMemorySolver(repeats_needed=2)
        intermediate = {(0, 0), (0, 5), (1, 4), (2, 3), (3, 2), (4, 1)}
        final = {(0, 4), (2, 3), (2, 5), (3, 2), (4, 0), (4, 1)}

        self.assertIsNone(solver.update(dot_frame(intermediate)))
        solver.update(dot_frame(set()))
        self.assertIsNone(solver.update(dot_frame(final)))
        solver.update(dot_frame(set()))
        self.assertIsNotNone(solver.update(dot_frame(final)))
        solver.update(dot_frame(set()))

        self.assertIsNone(solver.update(dot_frame(intermediate)))

    def test_dot_solver_ignores_incomplete_animation(self) -> None:
        solver = DotMemorySolver(repeats_needed=2)
        for _ in range(4):
            self.assertIsNone(solver.update(dot_frame({(0, 0)})))
            solver.update(dot_frame(set()))
        five_points = {(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)}
        self.assertIsNone(solver.update(dot_frame(five_points)))
        self.assertIsNone(solver.update(dot_frame(five_points)))

    def test_dot_solver_detects_kortz_center_normal_five_by_four_pattern(self) -> None:
        solver = DotMemorySolver(repeats_needed=2)
        pattern = {(0, 0), (2, 1), (1, 2), (3, 3), (0, 4)}

        self.assertIsNone(solver.update(kortz_frame(pattern)))
        self.assertTrue(solver.grid_visible)
        self.assertFalse(solver.input_visible)
        self.assertEqual(solver.current_grid_shape, (4, 5))
        solver.update(kortz_frame(set()))
        result = solver.update(kortz_frame(pattern))

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(
            [(point.row, point.column) for point in result.locations],
            [(1, 1), (1, 5), (2, 3), (3, 2), (4, 4)],
        )
        display = result.display_text()
        self.assertIn("1번 신호: 1번째 칸", display)
        self.assertIn("2번 신호: 3번째 칸", display)
        self.assertIn("3번 신호: 2번째 칸", display)
        self.assertIn("4번 신호: 4번째 칸", display)
        self.assertIn("5번 신호: 1번째 칸", display)
        self.assertNotIn("번째 줄", display)

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
        self.assertTrue(solver.grid_visible)
        self.assertTrue(solver.input_visible)
        for _ in range(2):
            result = solver.update(dot_frame(second))
            solver.update(dot_frame(set()))

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(
            [(point.row, point.column) for point in result.locations],
            [(2, 4), (2, 6), (3, 1), (3, 2), (4, 3), (4, 5)],
        )

    def test_dot_solver_handles_four_kortz_center_heist_rounds_with_same_pattern(self) -> None:
        solver = DotMemorySolver(repeats_needed=2)
        pattern = {(0, 4), (2, 3), (2, 5), (3, 2), (4, 0), (4, 1)}

        for _round in range(4):
            self.assertIsNone(solver.update(dot_frame(pattern)))
            solver.update(dot_frame(set()))
            result = solver.update(dot_frame(pattern))
            self.assertIsNotNone(result)
            assert result is not None
            self.assertEqual(
                [(point.row, point.column) for point in result.locations],
                [(1, 5), (3, 4), (3, 6), (4, 3), (5, 1), (5, 2)],
            )
            self.assertIsNone(solver.update(dot_frame(set(), {(0, 0)})))

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

    def test_dot_solver_keeps_answer_during_casino_input_and_confirmation_blinks(self) -> None:
        solver = DotMemorySolver()
        pattern = {(0, 0), (0, 5), (1, 1), (2, 2), (3, 3), (4, 4)}
        for _ in range(2):
            self.assertIsNone(solver.update(dot_frame(pattern)))
        self.assertIsNotNone(solver.update(dot_frame(pattern)))

        partial = dot_frame({(0, 0), (1, 1)})
        cv2.circle(partial, (510, 274), 4, (40, 40, 220), -1)
        for _ in range(20):
            self.assertIsNone(solver.update(partial))
            self.assertFalse(solver.input_visible)
        for _ in range(4):
            self.assertIsNone(solver.update(dot_frame(set())))
            self.assertIsNone(solver.update(partial))
        for _ in range(4):
            self.assertIsNone(solver.update(dot_frame(pattern)))
            self.assertIsNone(solver.update(dot_frame(set())))
        for _ in range(4):
            self.assertIsNone(solver.update(dot_frame(pattern)))

        next_pattern = {(0, 0), (0, 5), (1, 4), (2, 3), (3, 2), (4, 1)}
        self.assertIsNone(solver.update(dot_frame(next_pattern)))
        self.assertIsNone(solver.update(dot_frame(next_pattern)))
        self.assertIsNotNone(solver.update(dot_frame(next_pattern)))

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

    def test_fragment_solver_accepts_fourth_practice_fingerprint_scores(self) -> None:
        target = np.zeros((240, 240, 3), dtype=np.uint8)
        candidates = [np.zeros((80, 80, 3), dtype=np.uint8) for _ in range(8)]
        measured_scores = iter((0.515, 0.726, 0.379, 0.388, 0.612, 0.311, 0.888, 0.335))

        with patch("gta_helper.solvers._score_prepared_fingerprint_piece", side_effect=measured_scores):
            result = FragmentFingerprintSolver().solve_regions(target, candidates)

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.details[0], "선택: 1번 · 2번 · 5번 · 7번")
        self.assertGreaterEqual(result.debug["margin"], .12)

    def test_fragment_solver_uses_clear_margin_in_confidence(self) -> None:
        target = np.zeros((240, 240, 3), dtype=np.uint8)
        candidates = [np.zeros((80, 80, 3), dtype=np.uint8) for _ in range(8)]
        measured_scores = iter((0.299, 0.391, 0.759, 0.363, 0.531, 0.785, 0.571, 0.329))

        with patch("gta_helper.solvers._score_prepared_fingerprint_piece", side_effect=measured_scores):
            result = FragmentFingerprintSolver().solve_regions(target, candidates)

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.details[0], "선택: 3번 · 5번 · 6번 · 7번")
        self.assertLess(result.debug["mean_score"], .68)
        self.assertGreaterEqual(result.confidence, .68)

    def test_fragment_solver_resolves_third_practice_fingerprint_lines(self) -> None:
        target = np.zeros((240, 240, 3), dtype=np.uint8)
        candidates = [np.zeros((80, 80, 3), dtype=np.uint8) for _ in range(8)]
        measured_scores = iter((0.499, 0.449, 0.282, 0.267, 0.290, 0.828, 0.920, 0.872))

        with patch("gta_helper.solvers._score_prepared_fingerprint_piece", side_effect=measured_scores):
            result = FragmentFingerprintSolver().solve_regions(target, candidates)

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.details[0], "선택: 1번 · 6번 · 7번 · 8번")
        self.assertGreaterEqual(result.debug["margin"], .04)

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

    def test_cayo_solver_rejects_low_confidence_environment_match(self) -> None:
        target = np.zeros((200, 160, 3), dtype=np.uint8)
        rows = [np.zeros((40, 160, 3), dtype=np.uint8) for _ in range(5)]

        with patch("gta_helper.solvers._score_template", return_value=0.30):
            self.assertIsNone(CayoFingerprintSolver().solve_regions(target, rows))

    def test_voltlab_solver_finds_unique_multiplier_mapping(self) -> None:
        result = VoltLabSolver().solve_values(95, [1, 4, 9], [1, 1, 10])
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.debug["multipliers"], (1, 1, 10))
        self.assertIn("9 → ×10", result.details)
