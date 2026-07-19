from __future__ import annotations

from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from .config import AppConfig
from .windowing import GameWindow, Rect


class CaptureError(RuntimeError):
    pass


class DxCapture:
    """DXcam 기반 외부 화면 캡처. 게임 창에 어떤 입력도 보내지 않는다."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.camera = None
        self.backend = ""
        self.output_bounds = Rect(0, 0, 0, 0)

    def open(self) -> None:
        try:
            import dxcam
        except ImportError as exc:
            raise CaptureError("DXcam이 없습니다. setup.bat을 실행하세요.") from exc
        backends = [self.config.capture_backend] if self.config.capture_backend != "auto" else ["dxgi", "winrt"]
        errors: list[str] = []
        for backend in backends:
            try:
                self.camera = dxcam.create(
                    output_idx=self.config.capture_output,
                    output_color="BGR",
                    backend=backend,
                )
                self.backend = backend
                return
            except Exception as exc:  # DXcam은 GPU/드라이버별 예외 형식이 다르다.
                errors.append(f"{backend}: {exc}")
        raise CaptureError("캡처 장치를 열 수 없습니다: " + " | ".join(errors))

    def close(self) -> None:
        if self.camera is not None:
            try:
                self.camera.release()
            except Exception:
                pass
        self.camera = None

    def grab(self, game: GameWindow | None = None) -> np.ndarray:
        if self.camera is None:
            self.open()
        assert self.camera is not None
        frame = self.camera.grab(new_frame_only=False)
        if frame is None or frame.size == 0:
            raise CaptureError("새 화면 프레임을 가져오지 못했습니다.")
        if frame.ndim != 3:
            raise CaptureError("지원하지 않는 캡처 프레임 형식입니다.")
        if float(frame.std()) < 1.0:
            raise CaptureError("검은 화면이 캡처되었습니다. 설정에서 WinRT를 선택해 보세요.")
        if game is None:
            return frame
        # 보조 모니터/음수 좌표 환경은 output_idx가 지정한 모니터와 일치해야 한다.
        # 현재는 같은 출력 안에 있을 때만 클라이언트 영역으로 잘라낸다.
        rect = game.client_rect
        if 0 <= rect.left < frame.shape[1] and 0 <= rect.top < frame.shape[0]:
            right, bottom = min(rect.right, frame.shape[1]), min(rect.bottom, frame.shape[0])
            if right - rect.left >= 640 and bottom - rect.top >= 360:
                return frame[rect.top:bottom, rect.left:right].copy()
        return frame

    @staticmethod
    def save_diagnostic(frame: np.ndarray, directory: str | Path, label: str = "capture") -> Path:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        filename = path / f"{label}_{datetime.now():%Y%m%d_%H%M%S_%f}.png"
        if not cv2.imwrite(str(filename), frame):
            raise CaptureError(f"진단 이미지를 저장하지 못했습니다: {filename}")
        return filename
