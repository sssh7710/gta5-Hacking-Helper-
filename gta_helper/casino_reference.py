from __future__ import annotations

import base64
import json
from pathlib import Path
import zlib

import cv2
import numpy as np

from .models import PuzzleType, SolveResult


def descriptor(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    gray = gray[round(height * .08):round(height * .92), round(width * .08):round(width * .92)]
    return cv2.resize(gray, size, interpolation=cv2.INTER_AREA).astype(np.float32)


def similarity(left: np.ndarray, right: np.ndarray) -> float:
    left = (left - left.mean()) / (left.std() + 1e-6)
    right = (right - right.mean()) / (right.std() + 1e-6)
    return float(np.mean(left * right))


class CasinoReferenceSolver:
    """연습 영상으로 수집한 템플릿을 쓰는 보수적 카지노 지문 판별기."""

    def __init__(self, path: Path) -> None:
        self.profiles: list[tuple[np.ndarray, list[np.ndarray]]] = []
        if not path.exists():
            return
        raw = json.loads(path.read_text(encoding="utf-8"))
        for item in raw.get("profiles", []):
            target = np.frombuffer(base64.b64decode(item["target"]), dtype=np.uint8).reshape(36, 24).astype(np.float32)
            pieces = [np.frombuffer(base64.b64decode(value), dtype=np.uint8).reshape(16, 16).astype(np.float32) for value in item["pieces"]]
            self.profiles.append((target, pieces))

    def solve(self, target: np.ndarray, candidates: list[np.ndarray]) -> SolveResult | None:
        if not self.profiles or len(candidates) != 8:
            return None
        target_scores = [similarity(descriptor(target, (24, 36)), saved) for saved, _ in self.profiles]
        best = int(np.argmax(target_scores))
        if target_scores[best] < .80 or (len(target_scores) > 1 and target_scores[best] - sorted(target_scores)[-2] < .15):
            return None
        pieces = self.profiles[best][1]
        scores = [max(similarity(descriptor(candidate, (16, 16)), piece) for piece in pieces) for candidate in candidates]
        order = sorted(range(8), key=lambda index: scores[index], reverse=True)
        if scores[order[3]] < .65 or scores[order[3]] - scores[order[4]] < .15:
            return None
        chosen = [index + 1 for index in sorted(order[:4])]
        return SolveResult(PuzzleType.FRAGMENT_FINGERPRINT, min(.99, target_scores[best]), "카지노 지문 조각 정답", details=["선택: " + " · ".join(f"{index}번" for index in chosen), "화면의 4개를 선택한 뒤 확인하세요."])
