from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class AppState(str, Enum):
    WAITING = "대기 중"
    ANALYZING = "화면 분석 중"
    SOLVED = "정답 표시"
    ERROR = "오류"


class PuzzleType(str, Enum):
    DOT_MEMORY = "점멸 원 기억"
    FRAGMENT_FINGERPRINT = "지문 조각 선택"
    CAYO_FINGERPRINT = "카요 지문 조립"
    VOLTLAB = "카요 Voltlab 숫자 연결"


class DisplayMode(str, Enum):
    CLICK_THROUGH = "클릭 통과 오버레이"
    WINDOW = "일반 작은 창"
    VOICE_ONLY = "음성 전용"


@dataclass(frozen=True)
class GridPoint:
    row: int
    column: int

    def label(self) -> str:
        return f"{self.row}번째 줄 {self.column}번"


@dataclass
class SolveResult:
    puzzle: PuzzleType
    confidence: float
    summary: str
    locations: list[GridPoint] = field(default_factory=list)
    details: list[str] = field(default_factory=list)
    debug: dict[str, Any] = field(default_factory=dict)

    @property
    def signature(self) -> tuple[Any, ...]:
        return (self.puzzle.value, self.summary, tuple(self.details), tuple(self.locations))

    def display_text(self) -> str:
        lines = [self.summary]
        if self.locations:
            grouped: dict[int, list[int]] = {}
            for item in sorted(self.locations, key=lambda point: (point.row, point.column)):
                grouped.setdefault(item.row, []).append(item.column)
            grid_rows = self.debug.get("grid_rows", max(grouped))
            if not isinstance(grid_rows, int) or grid_rows < max(grouped):
                grid_rows = max(grouped)
            lines.extend(
                f"{row}번째 줄: " + (", ".join(f"{column}번" for column in grouped.get(row, [])) or "없음")
                for row in range(1, grid_rows + 1)
            )
        lines.extend(self.details)
        return "\n".join(lines)
