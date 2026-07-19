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


class DotMemorySolver:
    """점멸 패턴을 프레임 전환 단위로 모아 마지막 반복 패턴을 확정한다."""

    def __init__(self, repeats_needed: int = 3) -> None:
        self.repeats_needed = repeats_needed
        self._previous: tuple[GridPoint, ...] | None = None
        self._counts: Counter[tuple[GridPoint, ...]] = Counter()
        self._last_result: tuple[GridPoint, ...] | None = None

    def reset(self) -> None:
        self._previous = None
        self._counts.clear()
        self._last_result = None

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
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # 파랑/청록 계열 원을 찾는다. 색상은 힌트일 뿐, 밝기 비교로 활성화 여부를 판단한다.
        mask = cv2.inRange(hsv, (75, 35, 25), (130, 255, 255))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates: list[tuple[int, int, int, float]] = []
        minimum = max(12.0, frame.shape[0] * frame.shape[1] * 0.000008)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < minimum:
                continue
            (x, y), radius = cv2.minEnclosingCircle(contour)
            if radius < 3 or radius > min(frame.shape[:2]) * 0.09:
                continue
            circularity = 4 * np.pi * area / max(cv2.arcLength(contour, True) ** 2, 1)
            if circularity < 0.42:
                continue
            candidates.append((round(x), round(y), max(3, round(radius)), circularity))
        # 키패드 화면은 보통 3개 이하의 원만 켜져 있다. 이 경우에도 꺼진 원의
        # 테두리를 찾아 전체 격자를 고정해야 행/열 번호를 잃지 않는다.
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        scan = gray[:round(frame.shape[0] * .86), :round(frame.shape[1] * .70)]
        circles = cv2.HoughCircles(cv2.medianBlur(scan, 5), cv2.HOUGH_GRADIENT, 1.2, max(34, frame.shape[0] // 15), param1=80, param2=25, minRadius=max(10, frame.shape[0] // 38), maxRadius=max(16, frame.shape[0] // 13))
        grid: list[tuple[int, int, int]] = []
        if circles is not None:
            raw = [(round(x), round(y), round(radius)) for x, y, radius in circles[0] if frame.shape[1] * .07 < x < frame.shape[1] * .52 and frame.shape[0] * .18 < y < frame.shape[0] * .78 and frame.shape[0] * .035 < radius < frame.shape[0] * .060]
            if len(raw) >= 12:
                median_radius = int(np.median([radius for _, _, radius in raw]))
                raw = [item for item in raw if .65 <= item[2] / max(median_radius, 1) <= 1.35]
                xs = self._cluster([x for x, _, _ in raw], max(14, median_radius))
                ys = self._cluster([y for _, y, _ in raw], max(14, median_radius))
                occupied = sum(any(abs(x - grid_x) <= median_radius and abs(y - grid_y) <= median_radius for x, y, _ in raw) for grid_y in ys for grid_x in xs)
                if 3 <= len(xs) <= 8 and 3 <= len(ys) <= 6 and occupied >= len(xs) * len(ys) * .75:
                    grid = [(x, y, median_radius) for y in ys for x in xs]
        used_circle_grid = bool(grid)
        if grid:
            candidates = [(x, y, radius, 1.0) for x, y, radius in grid]
        if len(candidates) < 4:
            return None
        radius = int(np.median([item[2] for item in candidates]))
        xs = self._cluster([item[0] for item in candidates], max(8, radius * 2))
        ys = self._cluster([item[1] for item in candidates], max(8, radius * 2))
        if len(xs) < 2 or len(ys) < 2:
            return None
        values: list[tuple[GridPoint, float]] = []
        value_channel = hsv[:, :, 2]
        for x, y, _, _ in candidates:
            row = min(range(len(ys)), key=lambda i: abs(ys[i] - y)) + 1
            column = min(range(len(xs)), key=lambda i: abs(xs[i] - x)) + 1
            # 색상 자체보다 중앙의 밝기가 켜진 신호와 꺼진 청록 테두리를 잘 가른다.
            top, bottom = max(0, y - radius // 2), y + radius // 2 + 1
            left, right = max(0, x - radius // 2), x + radius // 2 + 1
            patch = value_channel[top:bottom, left:right]
            value = float(np.mean(patch))
            if used_circle_grid:
                # 중앙 팝업의 흰색은 밝지만 청록 신호가 아니므로 제외한다.
                cyan_coverage = float(np.mean(mask[top:bottom, left:right])) / 255.0
                value *= cyan_coverage
            values.append((GridPoint(row, column), value))
        brightness = np.array([value for _, value in values])
        threshold = float(np.median(brightness) + max(16, (brightness.max() - brightness.min()) * 0.30))
        active = tuple(sorted((point for point, value in values if value >= threshold), key=lambda p: (p.row, p.column)))
        if not active or len(active) == len(values):
            return None
        regularity = min(1.0, len(values) / max(4, len(xs) * len(ys)))
        return active, regularity

    def update(self, frame: np.ndarray) -> SolveResult | None:
        detected = self._detect(frame)
        if detected is None:
            self._previous = None
            return None
        pattern, regularity = detected
        if pattern != self._previous:
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
                debug={"repeats": count},
            )
        return None


class FragmentFingerprintSolver:
    def solve_regions(self, target: np.ndarray, candidates: Iterable[np.ndarray]) -> SolveResult | None:
        scored = [(index + 1, _score_template(target, candidate)) for index, candidate in enumerate(candidates)]
        if len(scored) < 4:
            return None
        scored.sort(key=lambda item: item[1], reverse=True)
        selected = scored[:4]
        # 지문 선은 서로 닮아 최고 점수만으로 네 조각을 고르면 오답이 날 수 있다.
        # 4위 조각이 충분히 맞고, 5위와의 차이도 뚜렷할 때만 답을 낸다.
        fifth_score = scored[4][1] if len(scored) > 4 else -1.0
        margin = selected[-1][1] - fifth_score
        if selected[-1][1] < 0.22 or margin < 0.035:
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
