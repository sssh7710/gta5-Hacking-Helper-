from __future__ import annotations

import unittest
from pathlib import Path

import cv2
import numpy as np

from gta_helper.layout import casino_fingerprint_layout
from gta_helper.casino import selected_component_indices
from gta_helper.casino_reference import CasinoReferenceSolver


def casino_screen(width: int, height: int, *, include_target_inner: bool = True) -> np.ndarray:
    """카지노 지문 화면의 상대 패널 위치를 가진 합성 캡처를 만든다."""
    image = np.zeros((height, width, 3), dtype=np.uint8)
    # 576x360 실습 영상에서 확인한 비율을 유지한 두 패널이다.
    components = (round(width * .12), round(height * .17), round(width * .24), round(height * .62))
    target_panel = (round(width * .43), round(height * .10), round(width * .42), round(height * .75))
    cv2.rectangle(image, components[:2], (components[0] + components[2], components[1] + components[3]), (220, 220, 220), 2)
    cv2.rectangle(image, target_panel[:2], (target_panel[0] + target_panel[2], target_panel[1] + target_panel[3]), (220, 220, 220), 2)
    if include_target_inner:
        tx, ty, tw, th = target_panel
        cv2.rectangle(image, (tx + round(tw * .28), ty + round(th * .18)), (tx + round(tw * .72), ty + round(th * .82)), (240, 240, 240), 2)
    return image


class CasinoLayoutTests(unittest.TestCase):
    def test_relative_layout_finds_target_and_eight_components(self) -> None:
        for width, height in ((1280, 720), (1920, 1080), (2560, 1080), (3840, 2160)):
            with self.subTest(size=(width, height)):
                layout = casino_fingerprint_layout(casino_screen(width, height))
                self.assertIsNotNone(layout)
                assert layout is not None
                target, candidates = layout
                self.assertEqual(len(candidates), 8)
                self.assertGreater(target.shape[0], candidates[0].shape[0])
                self.assertGreater(target.shape[1], candidates[0].shape[1])
                self.assertGreater(target.shape[0], height * .50)
                self.assertGreater(target.shape[1], width * .18)

    def test_selected_component_indices_reads_bright_tiles_only(self) -> None:
        candidates = [np.full((20, 20, 3), 10, dtype=np.uint8) for _ in range(8)]
        for index in (1, 4, 6, 7):
            candidates[index - 1][:] = 90
        self.assertEqual(selected_component_indices(candidates), (1, 4, 6, 7))

    def test_relative_layout_falls_back_when_target_inner_is_not_detected(self) -> None:
        frame = casino_screen(1920, 1080, include_target_inner=False)

        layout = casino_fingerprint_layout(frame)

        self.assertIsNotNone(layout)
        assert layout is not None
        target, candidates = layout
        self.assertEqual(len(candidates), 8)
        self.assertGreater(target.shape[0], candidates[0].shape[0])
        self.assertGreater(target.shape[1], candidates[0].shape[1])
        self.assertGreater(target.shape[0], 1080 * .35)
        self.assertLess(target.shape[0], 1080 * .50)

    def test_bundled_casino_templates_load(self) -> None:
        root = Path(__file__).resolve().parents[1]
        solver = CasinoReferenceSolver(root / "assets" / "reference" / "casino_templates.json")
        self.assertEqual(len(solver.profiles), 3)
