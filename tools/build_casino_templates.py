from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
import sys

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gta_helper.casino_reference import descriptor
from gta_helper.layout import casino_fingerprint_layout


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument("labels", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    labels = json.loads(args.labels.read_text(encoding="utf-8"))["completed_selections"]
    capture = cv2.VideoCapture(str(args.video))
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    profiles = []
    for label in labels:
        capture.set(cv2.CAP_PROP_POS_FRAMES, round(label["seconds"] * fps))
        ok, frame = capture.read()
        layout = casino_fingerprint_layout(frame) if ok else None
        if layout is None:
            continue
        target, candidates = layout
        encode = lambda image: base64.b64encode(image.astype("uint8").tobytes()).decode("ascii")
        profiles.append({"target": encode(descriptor(target, (24, 36))), "pieces": [encode(descriptor(candidates[index - 1], (16, 16))) for index in label["selected"]]})
    args.output.write_text(json.dumps({"profiles": profiles}, indent=2) + "\n", encoding="utf-8")
    print(f"saved {len(profiles)} profiles: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
