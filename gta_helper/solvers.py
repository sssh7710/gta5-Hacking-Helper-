from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from itertools import permutations
from typing import Iterable

import cv2
import numpy as np

from .models import GridPoint, PuzzleType, SolveResult


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


def _score_fingerprint_piece(target: np.ndarray, piece: np.ndarray) -> float:
    """후보 테두리를 제외하고 전체 지문 안에서 다중 크기로 조각을 찾는다."""
    target_gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY) if target.ndim == 3 else target
    piece_gray = cv2.cvtColor(piece, cv2.COLOR_BGR2GRAY) if piece.ndim == 3 else piece
    height, width = piece_gray.shape
    margin_y, margin_x = round(height * .12), round(width * .12)
    piece_gray = piece_gray[margin_y:height - margin_y, margin_x:width - margin_x]
    if min(target_gray.shape) < 24 or min(piece_gray.shape) < 12:
        return -1.0

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    target_gray = clahe.apply(target_gray)
    piece_gray = clahe.apply(piece_gray)
    best = -1.0
    for scale in np.arange(.50, 1.71, .05):
        resized = cv2.resize(
            piece_gray,
            None,
            fx=float(scale),
            fy=float(scale),
            interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC,
        )
        if resized.shape[0] >= target_gray.shape[0] or resized.shape[1] >= target_gray.shape[1]:
            continue
        score = float(cv2.minMaxLoc(cv2.matchTemplate(target_gray, resized, cv2.TM_CCOEFF_NORMED))[1])
        best = max(best, score)
    return best


class DotMemorySolver:
    """점멸 패턴을 프레임 전환 단위로 모아 마지막 반복 패턴을 확정한다."""

    def __init__(self, repeats_needed: int = 2) -> None:
        self.repeats_needed = repeats_needed
        self._previous: tuple[GridPoint, ...] | None = None
        self._counts: Counter[tuple[GridPoint, ...]] = Counter()
        self._last_result: tuple[GridPoint, ...] | None = None
        self._grid_visible = False
        self._red_input_visible = False
        self._missing_grid_frames = 0
        self._inactive_grid_frames = 0
        self.current_pattern: tuple[GridPoint, ...] = ()
        self.current_grid_shape = (0, 0)

    def reset(self) -> None:
        self._previous = None
        self._counts.clear()
        self._last_result = None
        self._grid_visible = False
        self._red_input_visible = False
        self._missing_grid_frames = 0
        self._inactive_grid_frames = 0
        self.current_pattern = ()
        self.current_grid_shape = (0, 0)

    @staticmethod
    def _cluster(values: list[int], tolerance: int) -> list[int]:
        groups: list[list[int]] = []
        for value in sorted(values):
            if not groups or value - groups[-1][-1] > tolerance:
                groups.append([value])
            else:
                groups[-1].append(value)
        return [round(float(np.median(group))) for group in groups]

    def _detect(self, frame: np.ndarray) -> tuple[tuple[GridPoint, ...], float] | None:
        self._grid_visible = False
        self._red_input_visible = False
        self.current_pattern = ()
        self.current_grid_shape = (0, 0)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # 실제 키패드는 6×5(카지노) 또는 5×4(코르츠) 격자다. 켜진 점만으로
        # 행/열을 만들면 순간 색상과 UI 글자를 좌표로 오인하므로 모든 원의
        # 테두리를 먼저 찾고, 완전한 격자일 때만 판정한다.
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        height, width = frame.shape[:2]
        scan = gray[:round(height * .86), :round(width * .70)]
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
        xs = self._cluster([x for x, _, _ in raw], max(14, median_radius))
        ys = self._cluster([y for _, y, _ in raw], max(14, median_radius))
        if (len(xs), len(ys)) not in {(6, 5), (5, 4)}:
            return None

        occupied = sum(
            any(abs(x - grid_x) <= median_radius and abs(y - grid_y) <= median_radius for x, y, _ in raw)
            for grid_y in ys
            for grid_x in xs
        )
        if occupied < len(xs) * len(ys) * .90:
            return None
        x_steps, y_steps = np.diff(xs), np.diff(ys)
        if min(x_steps, default=1) <= 0 or min(y_steps, default=1) <= 0:
            return None
        if max(x_steps, default=1) / min(x_steps, default=1) > 1.35 or max(y_steps, default=1) / min(y_steps, default=1) > 1.35:
            return None
        self._grid_visible = True
        self.current_grid_shape = (len(ys), len(xs))

        # 점은 청록색으로 켜진 뒤 빨간 표시로 남을 수 있다. 두 색을 모두 읽되,
        # 흰색 선택 테두리와 어두운 격자 무늬는 제외한다.
        cyan = cv2.inRange(hsv, (70, 80, 70), (135, 255, 255))
        red = cv2.bitwise_or(cv2.inRange(hsv, (0, 90, 80), (15, 255, 255)), cv2.inRange(hsv, (165, 90, 80), (179, 255, 255)))
        values: list[tuple[GridPoint, float, float]] = []
        sample_radius = max(6, round(median_radius * .48))
        for row, y in enumerate(ys, start=1):
            for column, x in enumerate(xs, start=1):
                top, bottom = max(0, y - sample_radius), min(height, y + sample_radius + 1)
                left, right = max(0, x - sample_radius), min(width, x + sample_radius + 1)
                cyan_patch = cyan[top:bottom, left:right]
                red_patch = red[top:bottom, left:right]
                yy, xx = np.ogrid[:cyan_patch.shape[0], :cyan_patch.shape[1]]
                disk = (xx - (x - left)) ** 2 + (yy - (y - top)) ** 2 <= sample_radius ** 2
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
        # 카지노 6×5 패턴은 완성 시 6개, 코르츠 5×4 패턴은 3개가 켜진다.
        # 완성 전 중간 프레임을 정답으로 고정하지 않는다.
        expected_active = 6 if (len(xs), len(ys)) == (6, 5) else 3
        if len(active) != expected_active:
            return None
        self.current_pattern = active
        regularity = min(1.0, occupied / (len(xs) * len(ys)))
        return active, regularity

    def update(self, frame: np.ndarray) -> SolveResult | None:
        detected = self._detect(frame)
        if detected is None:
            self._previous = None
            if self._grid_visible:
                self._missing_grid_frames = 0
                if self._red_input_visible:
                    self._counts.clear()
                    self._last_result = None
                    self._inactive_grid_frames = 0
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
        if self._last_result is not None:
            return None
        pattern, regularity = detected
        # 점멸 사이의 매우 짧은 암전 프레임을 캡처가 놓쳐도 같은 완성 패턴을
        # 연속 두 프레임 확인하면 반복 표시로 인정한다.
        self._counts[pattern] += 1
        self._previous = pattern
        count = self._counts[pattern]
        if count >= self.repeats_needed and pattern != self._last_result:
            self._last_result = pattern
            confidence = min(0.98, 0.58 + 0.10 * count + 0.12 * regularity)
            return SolveResult(
                puzzle=PuzzleType.DOT_MEMORY,
                confidence=confidence,
                summary="점멸 원 정답 위치",
                locations=list(pattern),
                details=["표시된 칸만 선택하세요."],
                debug={"repeats": count, "grid_rows": self.current_grid_shape[0], "grid_columns": self.current_grid_shape[1]},
            )
        return None


class FragmentFingerprintSolver:
    def solve_regions(self, target: np.ndarray, candidates: Iterable[np.ndarray]) -> SolveResult | None:
        scored = [(index + 1, _score_fingerprint_piece(target, candidate)) for index, candidate in enumerate(candidates)]
        if len(scored) < 4:
            return None
        scored.sort(key=lambda item: item[1], reverse=True)
        selected = scored[:4]
        # 지문 선은 서로 닮아 최고 점수만으로 네 조각을 고르면 오답이 날 수 있다.
        # 4위 조각이 충분히 맞고, 5위와의 차이도 뚜렷할 때만 답을 낸다.
        fifth_score = scored[4][1] if len(scored) > 4 else -1.0
        margin = selected[-1][1] - fifth_score
        if selected[-1][1] < 0.55 or margin < 0.08:
            return None
        confidence = max(0.0, min(0.99, float(np.mean([score for _, score in selected]))))
        return SolveResult(
            puzzle=PuzzleType.FRAGMENT_FINGERPRINT,
            confidence=confidence,
            summary="지문 조각 정답",
            details=["선택: " + " · ".join(f"{index}번" for index, _ in sorted(selected)), "정답 4개를 선택한 뒤 확인하세요."],
            debug={"scores": scored, "margin": margin},
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
        if confidence < 0.20:
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
