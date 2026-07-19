from __future__ import annotations

from collections.abc import Iterable

import cv2
import numpy as np


def selected_component_indices(
    candidates: Iterable[np.ndarray], *, selected_brightness: float = 35.0
) -> tuple[int, ...]:
    """카지노 COMPONENTS 패널에서 밝게 표시된(사용자가 고른) 조각 번호를 읽는다.

    이 함수는 화면 상태를 기록하거나 회귀 자료를 만드는 용도이며, 게임에 입력을
    보내지 않는다. 화면 테마가 달라지면 실행 중 답으로 쓰지 않고 진단 자료로만 쓴다.
    """
    selected: list[int] = []
    for index, candidate in enumerate(candidates, start=1):
        if candidate.size == 0:
            continue
        gray = cv2.cvtColor(candidate, cv2.COLOR_BGR2GRAY) if candidate.ndim == 3 else candidate
        if float(np.mean(gray)) >= selected_brightness:
            selected.append(index)
    return tuple(selected)
