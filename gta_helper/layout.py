from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class Box:
    x: int
    y: int
    w: int
    h: int

    @property
    def area(self) -> int: return self.w * self.h

    def crop(self, frame: np.ndarray) -> np.ndarray:
        return frame[self.y:self.y + self.h, self.x:self.x + self.w].copy()


def _boxes(frame: np.ndarray) -> list[Box]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 45, 130)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    height, width = frame.shape[:2]
    answer: list[Box] = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        if w < width * .035 or h < height * .035 or area > width * height * .45:
            continue
        if x < 2 or y < 2 or x + w > width - 2 or y + h > height - 2:
            continue
        answer.append(Box(x, y, w, h))
    # 같은 테두리에서 나온 중복 상자는 제거한다.
    unique: list[Box] = []
    for box in sorted(answer, key=lambda item: item.area, reverse=True):
        if not any(abs(box.x - old.x) < 8 and abs(box.y - old.y) < 8 and abs(box.w - old.w) < 12 and abs(box.h - old.h) < 12 for old in unique):
            unique.append(box)
    return unique


def fragment_layout(frame: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]] | None:
    boxes = _boxes(frame)
    for seed in boxes:
        peers = [box for box in boxes if .65 <= box.w / seed.w <= 1.35 and .65 <= box.h / seed.h <= 1.35]
        if len(peers) < 8:
            continue
        candidates = sorted(peers, key=lambda item: (item.y, item.x))[:8]
        candidate_area = float(np.median([box.area for box in candidates]))
        targets = [box for box in boxes if box not in candidates and box.area > candidate_area * 1.6]
        if not targets:
            continue
        target = max(targets, key=lambda item: item.area)
        return target.crop(frame), [box.crop(frame) for box in candidates]
    return None


def casino_fingerprint_layout(frame: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]] | None:
    """실제 카지노 지문 UI의 큰 우측 패널과 좌측 2×4 후보 격자를 찾는다.

    화면 비율이 달라도 외곽 패널 안의 상대 위치를 이용하므로 고정 해상도 좌표를 쓰지 않는다.
    """
    boxes = _boxes(frame)
    area = frame.shape[0] * frame.shape[1]
    target_panels = [
        box for box in boxes
        if box.x > frame.shape[1] * .38 and .70 <= box.w / max(box.h, 1) <= 1.45 and box.area > area * .10
    ]
    if not target_panels:
        return None
    target_panel = max(target_panels, key=lambda box: box.area)
    inner = [
        box for box in boxes
        if box is not target_panel and target_panel.x + 8 <= box.x and target_panel.y + 8 <= box.y
        and box.x + box.w <= target_panel.x + target_panel.w - 8
        and box.y + box.h <= target_panel.y + target_panel.h - 8
        and box.area > target_panel.area * .08 and .40 <= box.w / max(box.h, 1) <= 1.05
    ]
    if not inner:
        return None
    target = max(inner, key=lambda box: box.area)
    component_panels = [
        box for box in boxes
        if box.x + box.w < target_panel.x and box.area > area * .07 and .40 <= box.w / max(box.h, 1) <= 1.0
    ]
    if not component_panels:
        return None
    components = max(component_panels, key=lambda box: box.area)
    # 제공된 연습 영상의 COMPONETS 패널: 두 열, 네 행의 상대 좌표.
    x_positions = (.196, .510)
    y_positions = (.028, .223, .419, .622)
    cell_width, cell_height = .300, .187
    candidates: list[np.ndarray] = []
    for y_factor in y_positions:
        for x_factor in x_positions:
            x = round(components.x + components.w * x_factor)
            y = round(components.y + components.h * y_factor)
            w = round(components.w * cell_width)
            h = round(components.h * cell_height)
            candidates.append(frame[y:y + h, x:x + w].copy())
    if any(min(candidate.shape[:2]) < 12 for candidate in candidates):
        return None
    return target.crop(frame), candidates


def cayo_layout(frame: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]] | None:
    boxes = _boxes(frame)
    wide = [box for box in boxes if box.w / max(box.h, 1) >= 1.5]
    for seed in wide:
        peers = [box for box in wide if .65 <= box.w / seed.w <= 1.35 and .55 <= box.h / seed.h <= 1.55]
        peers = sorted(peers, key=lambda item: item.y)
        if not 5 <= len(peers) <= 12:
            continue
        # 세로로 겹치지 않고 정렬된 줄만 채택한다.
        if any(peers[i + 1].y <= peers[i].y for i in range(len(peers) - 1)):
            continue
        strip_area = float(np.median([box.area for box in peers]))
        targets = [box for box in boxes if box not in peers and box.area > strip_area * 1.8]
        if targets:
            target = max(targets, key=lambda item: item.area)
            return target.crop(frame), [box.crop(frame) for box in peers]
    return None
