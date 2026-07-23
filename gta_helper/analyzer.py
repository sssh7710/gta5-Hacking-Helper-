from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np

from .models import SolveResult
from .layout import casino_fingerprint_layout, cayo_layout
from .solvers import CayoFingerprintSolver, DotMemorySolver, FragmentFingerprintSolver
from .casino_reference import CasinoReferenceSolver
from .casino import selected_component_indices


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

    def reset(self) -> None:
        self.dot.reset()
        self._seen.clear()
        self.casino_layout_checked = False
        self.casino_screen_visible = False
        self.casino_selection_visible = False

    def update(self, frame: np.ndarray) -> SolveResult | None:
        # 점멸 원은 시간 정보가 필요하므로 매 프레임 처리한다.
        self._frame_number += 1
        self.casino_layout_checked = False
        result = self.dot.update(frame)
        if result is None and self._frame_number % 4 == 0:
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
            return None
        self._seen[result.signature] += 1
        # 지문은 같은 답이 두 프레임 연속 확인될 때만 표시한다.
        if result.puzzle.value != "점멸 원 기억" and self._seen[result.signature] < 2:
            return None
        return result
