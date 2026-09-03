from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from itertools import combinations, permutations
from typing import Iterable

import cv2
import numpy as np

from .models import GridPoint, PuzzleType, SolveResult


_FINGERPRINT_PROCESSING_SCALE = .65
_FINGERPRINT_SCALES = tuple(float(scale) for scale in np.arange(.50, 1.71, .05))
_MIN_CAYO_CONFIDENCE = .68


def _edge(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    return cv2.Canny(gray, 40, 130)


def _score_template(target: np.ndarray, piece: np.ndarray) -> float:
    target_edge, piece_edge = _edge(target), _edge(piece)
    if piece_edge.shape[0] > target_edge.shape[0] or piece_edge.shape[1] > target_edge.shape[1]:
        scale = min(target_edge.shape[0] / piece_edge.shape[0], target_edge.shape[1] / piece_edge.shape[1]) * 0.98
        piece_edge = cv2.resize(piece_edge, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    if min(piece_edge.shape) < 8 or min(target_edge.shape) < 8:
        return -1.0
    # 거의 빈 조각은 TM_CCOEFF_NORMED에서 잘못 높은 점수를 낼 수 있다.
    if cv2.countNonZero(piece_edge) < max(12, piece_edge.size * 0.006):
        return -1.0
    return float(cv2.minMaxLoc(cv2.matchTemplate(target_edge, piece_edge, cv2.TM_CCOEFF_NORMED))[1])


def _prepare_fingerprint_target(target: np.ndarray) -> np.ndarray | None:
    target_gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY) if target.ndim == 3 else target
    if min(target_gray.shape) < 24:
        return None
    target_gray = cv2.threshold(target_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    return cv2.resize(
        target_gray,
        None,
        fx=_FINGERPRINT_PROCESSING_SCALE,
        fy=_FINGERPRINT_PROCESSING_SCALE,
        interpolation=cv2.INTER_AREA,
    )


def _score_prepared_fingerprint_piece(target_gray: np.ndarray, piece: np.ndarray) -> float:
    """전처리된 전체 지문 안에서 후보 조각을 다중 크기로 찾는다."""
    piece_gray = cv2.cvtColor(piece, cv2.COLOR_BGR2GRAY) if piece.ndim == 3 else piece
    height, width = piece_gray.shape
    margin_y, margin_x = round(height * .12), round(width * .12)
    piece_gray = piece_gray[margin_y:height - margin_y, margin_x:width - margin_x]
    if min(piece_gray.shape) < 12:
        return -1.0

    # 카지노 UI의 점무늬 배경은 후보마다 위치가 달라 명암 상관계수에
    # 잘못 기여한다. 밝은 지문선만 이진화해 실제 선 모양을 비교한다.
    piece_gray = cv2.threshold(piece_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    if cv2.countNonZero(piece_gray) < max(12, piece_gray.size * .006):
        return -1.0

    # 1920x1080 실전 캡처에서는 원본 크기 그대로 다중 크기 비교를 하면
    # 후보 8개 판정에 약 0.5~0.9초가 걸린다. 지문선 모양과 기존 임계값을
    # 유지하는 범위에서 비교 영상만 축소해 matchTemplate 연산량을 줄인다.
    piece_gray = cv2.resize(
        piece_gray,
        None,
        fx=_FINGERPRINT_PROCESSING_SCALE,
        fy=_FINGERPRINT_PROCESSING_SCALE,
        interpolation=cv2.INTER_AREA,
    )
    best = -1.0
    for scale in _FINGERPRINT_SCALES:
        resized = cv2.resize(
            piece_gray,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC,
        )
        if resized.shape[0] >= target_gray.shape[0] or resized.shape[1] >= target_gray.shape[1]:
            continue
        score = float(cv2.minMaxLoc(cv2.matchTemplate(target_gray, resized, cv2.TM_CCOEFF_NORMED))[1])
        best = max(best, score)
    return best


def _score_fingerprint_piece(target: np.ndarray, piece: np.ndarray) -> float:
    """후보 테두리를 제외하고 전체 지문 안에서 다중 크기로 조각을 찾는다."""
    prepared_target = _prepare_fingerprint_target(target)
    if prepared_target is None:
        return -1.0
    return _score_prepared_fingerprint_piece(prepared_target, piece)


class DotMemorySolver:
    """점멸 패턴을 프레임 전환 단위로 모아 마지막 반복 패턴을 확정한다."""

    def __init__(self, repeats_needed: int = 2, final_blank_frames: int = 6) -> None:
        self.repeats_needed = repeats_needed
        self.final_blank_frames = max(1, int(final_blank_frames))
        self._previous: tuple[GridPoint, ...] | None = None
        self._counts: Counter[tuple[GridPoint, ...]] = Counter()
        self._last_result: tuple[GridPoint, ...] | None = None
        self._pending_pattern: tuple[GridPoint, ...] | None = None
        self._pending_regularity = 0.0
        self._pending_grid_shape = (0, 0)
        self._blank_frames_after_pattern = 0
        self._grid_visible = False
        self._red_input_visible = False
        self._missing_grid_frames = 0
        self._inactive_grid_frames = 0
        self._different_pattern_seen = False
        self.current_pattern: tuple[GridPoint, ...] = ()
        self.current_grid_shape = (0, 0)

    def reset(self) -> None:
        self._previous = None
        self._counts.clear()
        self._last_result = None
        self._pending_pattern = None
        self._pending_regularity = 0.0
        self._pending_grid_shape = (0, 0)
        self._blank_frames_after_pattern = 0
        self._grid_visible = False
        self._red_input_visible = False
        self._missing_grid_frames = 0
        self._inactive_grid_frames = 0
        self._different_pattern_seen = False
        self.current_pattern = ()
        self.current_grid_shape = (0, 0)

    @property
    def grid_visible(self) -> bool:
        """지원하는 점멸 키패드 격자가 현재 화면에 보이는지 반환한다."""
        return self._grid_visible

    @property
    def input_visible(self) -> bool:
        """사용자가 정답을 입력하는 빨간 표시 단계인지 반환한다."""
        return self._red_input_visible

    @staticmethod
    def _cluster(values: list[int], tolerance: int) -> list[int]:
        groups: list[list[int]] = []
        for value in sorted(values):
            if not groups or value - groups[-1][-1] > tolerance:
                groups.append([value])
            else:
                groups[-1].append(value)
        return [round(float(np.median(group))) for group in groups]

    @staticmethod
    def _regular_axis(values: list[int], tolerance: int, size: int) -> list[int]:
        groups: list[list[int]] = []
        for value in sorted(values):
            if not groups or value - groups[-1][-1] > tolerance:
                groups.append([value])
            else:
                groups[-1].append(value)
        if len(groups) < size:
            return []

        centers = [round(float(np.median(group))) for group in groups]
        best: tuple[tuple[int, float], list[int]] | None = None
        for indices in combinations(range(len(groups)), size):
            selected = [centers[index] for index in indices]
            steps = np.diff(selected)
            if min(steps, default=0) <= 0:
                continue
            step_ratio = float(max(steps) / min(steps))
            if step_ratio > 1.35:
                continue
            support = sum(len(groups[index]) for index in indices)
            score = (support, -step_ratio)
            if best is None or score > best[0]:
                best = (score, selected)
        return [] if best is None else best[1]

    def _detect(self, frame: np.ndarray) -> tuple[tuple[GridPoint, ...], float] | None:
        self._grid_visible = False
        self._red_input_visible = False
        self.current_pattern = ()
        self.current_grid_shape = (0, 0)
        # 점멸 키패드는 6×5 또는 5×4 격자다. 켜진 점만으로
        # 행/열을 만들면 순간 색상과 UI 글자를 좌표로 오인하므로 모든 원의
        # 테두리를 먼저 찾고, 완전한 격자일 때만 판정한다.
        height, width = frame.shape[:2]
        scan = cv2.cvtColor(
            frame[:round(height * .86), :round(width * .70)],
            cv2.COLOR_BGR2GRAY,
        )
        circles = cv2.HoughCircles(cv2.medianBlur(scan, 5), cv2.HOUGH_GRADIENT, 1.2, max(34, frame.shape[0] // 15), param1=80, param2=25, minRadius=max(10, frame.shape[0] // 38), maxRadius=max(16, frame.shape[0] // 13))
        if circles is None:
            return None

        raw = [
            (round(x), round(y), round(radius))
            for x, y, radius in circles[0]
            if width * .20 < x < width * .65
            and height * .18 < y < height * .82
            and height * .025 < radius < height * .065
        ]
        if len(raw) < 20:
            return None

        median_radius = int(np.median([radius for _, _, radius in raw]))
        raw = [item for item in raw if .65 <= item[2] / max(median_radius, 1) <= 1.35]
        axis_tolerance = max(8, round(median_radius * .45))
        grids: list[tuple[float, list[int], list[int]]] = []
        for column_count, row_count in ((6, 5), (5, 4)):
            candidate_xs = self._regular_axis([x for x, _, _ in raw], axis_tolerance, column_count)
            candidate_ys = self._regular_axis([y for _, y, _ in raw], axis_tolerance, row_count)
            if not candidate_xs or not candidate_ys:
                continue
            occupied = sum(
                any(abs(x - grid_x) <= median_radius and abs(y - grid_y) <= median_radius for x, y, _ in raw)
                for grid_y in candidate_ys
                for grid_x in candidate_xs
            )
            occupancy = occupied / (column_count * row_count)
            if occupancy >= .90:
                grids.append((occupancy, candidate_xs, candidate_ys))
        if not grids:
            return None
        # A partly obscured 6x5 grid can leave a perfect 5x4 inner subset.
        # Prefer the larger valid grid so that subset is not reported as a
        # shifted normal-mode answer.
        regularity, xs, ys = max(grids, key=lambda item: (len(item[1]) * len(item[2]), item[0]))
        self._grid_visible = True
        self.current_grid_shape = (len(ys), len(xs))

        # 점은 청록색으로 켜진 뒤 빨간 표시로 남을 수 있다. 두 색을 모두 읽되,
        # 흰색 선택 테두리와 어두운 격자 무늬는 제외한다.
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        cyan = cv2.inRange(hsv, (70, 80, 70), (135, 255, 255))
        red = cv2.bitwise_or(cv2.inRange(hsv, (0, 90, 80), (15, 255, 255)), cv2.inRange(hsv, (165, 90, 80), (179, 255, 255)))
        values: list[tuple[GridPoint, float, float]] = []
        sample_radius = max(6, round(median_radius * .48))
        yy, xx = np.ogrid[-sample_radius:sample_radius + 1, -sample_radius:sample_radius + 1]
        disk = xx ** 2 + yy ** 2 <= sample_radius ** 2
        for row, y in enumerate(ys, start=1):
            for column, x in enumerate(xs, start=1):
                top, bottom = max(0, y - sample_radius), min(height, y + sample_radius + 1)
                left, right = max(0, x - sample_radius), min(width, x + sample_radius + 1)
                cyan_patch = cyan[top:bottom, left:right]
                red_patch = red[top:bottom, left:right]
                cyan_coverage = float(np.mean(cyan_patch[disk])) / 255.0
                red_coverage = float(np.mean(red_patch[disk])) / 255.0
                values.append((GridPoint(row, column), cyan_coverage, red_coverage))

        red_coverage = np.array([red_value for _, _, red_value in values])
        red_threshold = max(.035, float(np.median(red_coverage)) + .025)
        self._red_input_visible = bool(np.any(red_coverage >= red_threshold))
        if self._red_input_visible:
            # 빨간 원은 사용자가 답을 입력하는 단계다. 이전 답의 잠금을 풀되
            # 빨간 선택 이력을 다음 정답 패턴으로 세지 않는다.
            return None

        coverage = np.array([cyan_value for _, cyan_value, _ in values])
        threshold = max(.035, float(np.median(coverage)) + .025)
        active = tuple(point for point, cyan_value, _ in values if cyan_value >= threshold)
        # 어려움 6×5 패턴은 완성 시 6개, 보통 5×4 패턴은 5개가 켜진다.
        # 두 화면 모두 신호 열마다 세로 위치가 하나씩 있어야 완성 패턴이다.
        # 완성 전 중간 프레임을 정답으로 고정하지 않는다.
        expected_active = len(xs)
        if len(active) != expected_active:
            return None
        if {point.column for point in active} != set(range(1, len(xs) + 1)):
            return None
        self.current_pattern = active
        return active, regularity

    def update(self, frame: np.ndarray) -> SolveResult | None:
        detected = self._detect(frame)
        if detected is None:
            self._previous = None
            if self._grid_visible:
                self._missing_grid_frames = 0
                if self._red_input_visible:
                    fallback = self._fallback_result("input_visible")
                    if fallback is not None:
                        return fallback
                    self._counts.clear()
                    self._last_result = None
                    self._pending_pattern = None
                    self._blank_frames_after_pattern = 0
                    self._inactive_grid_frames = 0
                elif self._pending_pattern is not None:
                    self._blank_frames_after_pattern += 1
                    if self._blank_frames_after_pattern >= self.final_blank_frames:
                        fallback = self._fallback_result("animation_end")
                        if fallback is not None:
                            return fallback
                elif self._last_result is not None:
                    self._inactive_grid_frames += 1
                    if self._inactive_grid_frames >= 15:
                        self._counts.clear()
                        self._last_result = None
                        self._inactive_grid_frames = 0
            else:
                self._inactive_grid_frames = 0
                self._missing_grid_frames += 1
                if self._missing_grid_frames >= 15:
                    self.reset()
            return None
        self._missing_grid_frames = 0
        self._inactive_grid_frames = 0
        pattern, regularity = detected
        self._pending_pattern = pattern
        self._pending_regularity = regularity
        self._pending_grid_shape = self.current_grid_shape
        self._blank_frames_after_pattern = 0
        if self._last_result is not None and pattern != self._last_result:
            # 실전 코르츠 센터 습격 화면은 숫자를 입력해도 격자가 사라지거나 빨간 점으로 바뀌지 않는다.
            # 이전 정답과 다른 완성 배열이 보이면 다음 판이 시작된 것으로 간주한다.
            self._different_pattern_seen = True
        # 움직이는 중간 배열은 여러 프레임 유지될 수 있다. 같은 프레임을
        # 반복해서 센 것이 아니라, 다른 배열/암전 뒤 다시 나타난 배열만
        # 반복 표시로 인정해 마지막 패턴을 확정한다.
        if pattern != self._previous:
            self._counts[pattern] += 1
            self._previous = pattern
        count = self._counts[pattern]
        can_emit = pattern != self._last_result or self._different_pattern_seen
        if count >= self.repeats_needed and can_emit:
            confidence = min(0.98, 0.58 + 0.10 * count + 0.12 * regularity)
            return self._result(pattern, confidence, {"repeats": count, "completion": "repeated"})
        return None

    def _fallback_result(self, completion: str) -> SolveResult | None:
        pattern = self._pending_pattern
        unique_patterns = len(self._counts)
        if pattern is None or unique_patterns < 3:
            return None
        if pattern == self._last_result and not self._different_pattern_seen:
            return None
        confidence = min(0.84, 0.56 + 0.04 * min(unique_patterns, 4) + 0.08 * self._pending_regularity)
        return self._result(
            pattern,
            confidence,
            {"repeats": self._counts[pattern], "completion": completion, "unique_patterns": unique_patterns},
        )

    def _result(
        self,
        pattern: tuple[GridPoint, ...],
        confidence: float,
        debug: dict[str, object],
    ) -> SolveResult:
        rows, columns = self._pending_grid_shape
        self._last_result = pattern
        # 다음 판의 중간 배열이 이전 판에서 얻은 횟수를 재사용하지 않게 판별 이력을 분리한다.
        self._counts.clear()
        self._different_pattern_seen = False
        self._pending_pattern = None
        self._blank_frames_after_pattern = 0
        return SolveResult(
            puzzle=PuzzleType.DOT_MEMORY,
            confidence=confidence,
            summary="점멸 원 정답 위치",
            locations=list(pattern),
            details=["1번 신호부터 순서대로 표시된 세로 칸을 선택하세요."],
            debug={**debug, "grid_rows": rows, "grid_columns": columns},
        )


class FragmentFingerprintSolver:
    def solve_regions(self, target: np.ndarray, candidates: Iterable[np.ndarray]) -> SolveResult | None:
        prepared_target = _prepare_fingerprint_target(target)
        if prepared_target is None:
            return None
        scored = [
            (index + 1, _score_prepared_fingerprint_piece(prepared_target, candidate))
            for index, candidate in enumerate(candidates)
        ]
        if len(scored) < 4:
            return None
        scored.sort(key=lambda item: item[1], reverse=True)
        selected = scored[:4]
        # 지문 선은 서로 닮아 최고 점수만으로 네 조각을 고르면 오답이 날 수 있다.
        # 4위 조각이 충분히 맞고, 5위와의 차이도 뚜렷할 때만 답을 낸다.
        fifth_score = scored[4][1] if len(scored) > 4 else -1.0
        margin = selected[-1][1] - fifth_score
        # 실제 1920x1080 연습 화면에서 확인된 가장 약한 정답 조각은
        # 이진 선 점수 약 0.50, 5위와의 최소 차이는 약 0.05다.
        if selected[-1][1] < 0.44 or margin < 0.04:
            return None
        mean_score = float(np.mean([score for _, score in selected]))
        # 정답 4개와 나머지가 뚜렷하게 갈리는지도 신뢰도에 반영한다.
        # 네 번째 연습 지문의 실측 평균은 0.66이지만 5위와 0.14 차이가 나므로
        # 단순 평균만 사용하면 신뢰도 표시 기준(0.68)에서 잘못 탈락한다.
        confidence = max(0.0, min(0.99, mean_score + margin * .25))
        return SolveResult(
            puzzle=PuzzleType.FRAGMENT_FINGERPRINT,
            confidence=confidence,
            summary="지문 조각 정답",
            details=["선택: " + " · ".join(f"{index}번" for index, _ in sorted(selected)), "정답 4개를 선택한 뒤 확인하세요."],
            debug={"scores": scored, "margin": margin, "mean_score": mean_score},
        )


class CayoFingerprintSolver:
    def solve_regions(self, target: np.ndarray, current_rows: Iterable[np.ndarray]) -> SolveResult | None:
        rows = list(current_rows)
        if len(rows) < 2:
            return None
        target_bands = np.array_split(target, len(rows), axis=0)
        details: list[str] = []
        scores: list[float] = []
        count = len(rows)
        for row_index, row in enumerate(rows):
            matches = [_score_template(band, row) for band in target_bands]
            current = int(np.argmax(matches))
            desired = row_index
            right = (desired - current) % count
            left = (current - desired) % count
            if min(left, right) == 0:
                movement = "현재 위치"
            elif right <= left:
                movement = f"오른쪽 {right}칸"
            else:
                movement = f"왼쪽 {left}칸"
            details.append(f"{row_index + 1}번 줄: 조각 {desired + 1} ({movement})")
            scores.append(matches[current])
        confidence = max(0.0, min(0.99, float(np.mean(scores))))
        # 앱에서 결과를 표시하는 기준과 동일하게 취급한다. 낮은 점수의
        # 환경 화면이나 작전실 UI가 카요 퍼즐로 오인되면 빈 결과가
        # 진단 세션으로 저장되므로, 충분히 일치할 때만 확정한다.
        if confidence < _MIN_CAYO_CONFIDENCE:
            return None
        return SolveResult(
            puzzle=PuzzleType.CAYO_FINGERPRINT,
            confidence=confidence,
            summary="카요 지문 조립 정답",
            details=details,
            debug={"row_scores": scores},
        )


class VoltLabSolver:
    """숫자 3개를 x1, x2, x10에 일대일 연결하는 카요 통신탑 퍼즐 풀이기."""

    multipliers = (1, 2, 10)

    def solve_values(self, target: int, values: Iterable[int], multipliers: Iterable[int] | None = None) -> SolveResult | None:
        numbers = list(values)
        available = list(multipliers) if multipliers is not None else list(self.multipliers)
        if len(numbers) != 3 or len(available) != 3 or any(number < 0 or number > 9 for number in numbers) or any(multiplier not in (1, 2, 10) for multiplier in available):
            return None
        matches = [assignment for assignment in set(permutations(available)) if sum(number * multiplier for number, multiplier in zip(numbers, assignment)) == target]
        if len(matches) != 1:
            # 여러 답 또는 해가 없는 경우 화면 인식 오차 가능성이 있어 추측하지 않는다.
            return None
        assignment = matches[0]
        details = [f"{number} → ×{multiplier}" for number, multiplier in zip(numbers, assignment)]
        return SolveResult(
            puzzle=PuzzleType.VOLTLAB,
            confidence=.99,
            summary=f"TARGET {target:03d} 연결 정답",
            details=details + ["왼쪽 숫자에서 해당 배율 기호로 차례로 연결하세요."],
            debug={"target": target, "values": numbers, "multipliers": assignment},
        )
