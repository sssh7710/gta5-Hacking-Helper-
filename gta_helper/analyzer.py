from __future__ import annotations

from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from .models import PuzzleType, SolveResult
from .layout import casino_fingerprint_layout, cayo_layout
from .solvers import CayoFingerprintSolver, DotMemorySolver, FragmentFingerprintSolver
from .casino_reference import CasinoReferenceSolver
from .casino import selected_component_indices


ANALYSIS_MAX_HEIGHT = 1080
KEYPAD_GUARD_FRAMES = 15


def _analysis_frame(frame: np.ndarray) -> np.ndarray:
    """고해상도 캡처는 비율을 유지해 분석 비용만 줄인다."""
    height, width = frame.shape[:2]
    if height <= ANALYSIS_MAX_HEIGHT:
        return frame
    scale = ANALYSIS_MAX_HEIGHT / height
    return cv2.resize(
        frame,
        (max(1, round(width * scale)), ANALYSIS_MAX_HEIGHT),
        interpolation=cv2.INTER_AREA,
    )


class PuzzleAnalyzer:
    """공통 안정화 계층. 실제 UI 레이아웃은 진단 캡처를 추가해 확장한다."""

    def __init__(self) -> None:
        self.dot = DotMemorySolver()
        self.fragment = FragmentFingerprintSolver()
        self.casino_reference = CasinoReferenceSolver(Path(__file__).resolve().parents[1] / "assets" / "reference" / "casino_templates.json")
        self.cayo = CayoFingerprintSolver()
        self._seen: Counter[tuple] = Counter()
        self._frame_number = 0
        self.casino_layout_checked = False
        self.casino_screen_visible = False
        self.casino_selection_visible = False
        self._fingerprint_verification_pending = False
        self._keypad_guard_frames = 0

    def reset(self) -> None:
        self.dot.reset()
        self._seen.clear()
        self.casino_layout_checked = False
        self.casino_screen_visible = False
        self.casino_selection_visible = False
        self._fingerprint_verification_pending = False
        self._keypad_guard_frames = 0

    def update(self, frame: np.ndarray) -> SolveResult | None:
        frame = _analysis_frame(frame)
        # 지문 후보를 한 번 찾은 직후에는 점멸 퍼즐 전처리를 건너뛰고 다음
        # 프레임에서 곧바로 재확인한다. 평상시에는 점멸 퍼즐을 계속 우선한다.
        self._frame_number += 1
        self.casino_layout_checked = False
        fingerprint_active = self.casino_screen_visible or self._fingerprint_verification_pending
        result = None if fingerprint_active else self.dot.update(frame)
        if not fingerprint_active:
            if self.dot.grid_visible:
                self._keypad_guard_frames = KEYPAD_GUARD_FRAMES
            elif self._keypad_guard_frames > 0:
                self._keypad_guard_frames -= 1
        if result is None and self._keypad_guard_frames == 0 and (fingerprint_active or self._frame_number % 2 == 0):
            self.casino_layout_checked = True
            fragments = casino_fingerprint_layout(frame)
            self.casino_screen_visible = fragments is not None
            self.casino_selection_visible = False
            if fragments is not None:
                target, candidates = fragments
                # 선택한 조각은 흰색으로 밝아져 원래 무늬와 점수가 달라진다.
                # 첫 정답을 표시한 뒤 사용자가 입력하는 동안 재판정하지 않는다.
                self.casino_selection_visible = bool(selected_component_indices(candidates))
                if not self.casino_selection_visible:
                    result = self.fragment.solve_regions(target, candidates)
                    if result is None:
                        result = self.casino_reference.solve(target, candidates)
            elif result is None:
                # 카지노 지문 패널을 찾은 프레임을 카요 퍼즐로 다시 해석하면
                # 처리 중 화면에서 낮은 신뢰도의 오탐이 발생한다.
                cayo = cayo_layout(frame)
                if cayo is not None:
                    result = self.cayo.solve_regions(*cayo)
        if result is None:
            if self.casino_layout_checked:
                self._fingerprint_verification_pending = False
            return None
        self._seen[result.signature] += 1
        # 지문은 같은 답이 두 프레임 연속 확인될 때만 표시한다.
        if result.puzzle != PuzzleType.DOT_MEMORY:
            if self._seen[result.signature] < 2:
                self._fingerprint_verification_pending = True
                return None
            self._fingerprint_verification_pending = False
        return result
