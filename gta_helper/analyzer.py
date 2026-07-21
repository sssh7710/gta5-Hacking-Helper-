from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np

from .models import SolveResult
from .layout import casino_fingerprint_layout, cayo_layout
from .solvers import CayoFingerprintSolver, DotMemorySolver, FragmentFingerprintSolver
from .casino_reference import CasinoReferenceSolver


class PuzzleAnalyzer:
    """공통 안정화 계층. 실제 UI 레이아웃은 진단 캡처를 추가해 확장한다."""

    def __init__(self) -> None:
        self.dot = DotMemorySolver()
        self.fragment = FragmentFingerprintSolver()
        self.casino_reference = CasinoReferenceSolver(Path(__file__).resolve().parents[1] / "assets" / "reference" / "casino_templates.json")
        self.cayo = CayoFingerprintSolver()
        self._seen: Counter[tuple] = Counter()
        self._frame_number = 0

    def reset(self) -> None:
        self.dot.reset()
        self._seen.clear()

    def update(self, frame: np.ndarray) -> SolveResult | None:
        # 점멸 원은 시간 정보가 필요하므로 매 프레임 처리한다.
        self._frame_number += 1
        result = self.dot.update(frame)
        if result is None and self._frame_number % 4 == 0:
            fragments = casino_fingerprint_layout(frame)
            if fragments is not None:
                result = self.fragment.solve_regions(*fragments)
                if result is None:
                    target, candidates = fragments
                    result = self.casino_reference.solve(target, candidates)
            if result is None:
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
