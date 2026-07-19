"""카지노 지문 연습 영상을 읽어, 화면에서 확인되는 선택 조합을 JSON으로 저장한다.

예시:
    .venv\\Scripts\\python.exe -B tools\\analyze_casino_video.py \\
      diagnostics\\fingerprint_reference.mp4 assets\\reference\\casino_video_labels.json \\
      --start 7 --end 34
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2

# `python tools\\...py`로 실행해도 프로젝트 패키지를 찾게 한다.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gta_helper.casino import selected_component_indices
from gta_helper.layout import casino_fingerprint_layout


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="카지노 지문 선택 변화 분석")
    parser.add_argument("video", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--end", type=float, default=float("inf"))
    parser.add_argument("--sample-fps", type=float, default=10.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise SystemExit(f"영상을 열 수 없습니다: {args.video}")
    source_fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    stride = max(1, round(source_fps / max(args.sample_fps, 1)))
    last: tuple[int, ...] | None = None
    completed: list[dict[str, object]] = []
    frame_number = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        seconds = frame_number / source_fps
        frame_number += 1
        if seconds < args.start or seconds > args.end or frame_number % stride:
            continue
        layout = casino_fingerprint_layout(frame)
        if layout is None:
            continue
        _, candidates = layout
        selected = selected_component_indices(candidates)
        if selected == last:
            continue
        last = selected
        print(f"{seconds:05.2f}s: {selected or '-'}")
        if len(selected) == 4:
            completed.append({"seconds": round(seconds, 2), "selected": list(selected)})
    capture.release()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"completed_selections": completed}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"저장: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
