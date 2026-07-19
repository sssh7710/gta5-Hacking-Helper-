r"""참조 영상을 일정 간격의 PNG 프레임으로 추출한다.

예시:
    .venv\Scripts\python.exe -B tools\extract_video_frames.py \
      diagnostics\fingerprint_reference.mp4 diagnostics\fingerprint_frames --start 7 --end 34 --interval 1
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OpenCV로 영상 프레임을 일정 간격 추출합니다.")
    parser.add_argument("video", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--start", type=float, default=0)
    parser.add_argument("--end", type=float, required=True)
    parser.add_argument("--interval", type=float, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.start < 0 or args.end < args.start or args.interval <= 0:
        raise SystemExit("start/end/interval 값이 올바르지 않습니다.")
    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise SystemExit(f"영상을 열 수 없습니다: {args.video}")
    args.output.mkdir(parents=True, exist_ok=True)
    current = args.start
    saved = 0
    while current <= args.end + 1e-9:
        capture.set(cv2.CAP_PROP_POS_MSEC, current * 1000)
        ok, frame = capture.read()
        if ok:
            destination = args.output / f"frame_{current:06.2f}s.png"
            if cv2.imwrite(str(destination), frame):
                saved += 1
        current = round(current + args.interval, 4)
    capture.release()
    print(f"{saved}개 프레임 저장: {args.output}")


if __name__ == "__main__":
    main()
