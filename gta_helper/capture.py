from __future__ import annotations

import json
import shutil
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from .config import AppConfig
from .windowing import GameWindow, Rect


class CaptureError(RuntimeError):
    pass


class DiagnosticFrameRecorder:
    """해킹 시도별 프레임과 판정 정보를 폴더 단위로 저장한다."""

    def __init__(
        self,
        directory: str | Path,
        duration_seconds: float = 7.0,
        fps: float = 8.0,
        max_total_bytes: int = 1024 * 1024 * 1024,
        jpeg_quality: int = 92,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.directory = Path(directory)
        self.duration_seconds = max(1.0, float(duration_seconds))
        self.fps = max(1.0, float(fps))
        self.max_total_bytes = max(1, int(max_total_bytes))
        self.jpeg_quality = min(100, max(70, int(jpeg_quality)))
        self._clock = clock
        self._session_dir: Path | None = None
        self._started_at = 0.0
        self._next_frame_at = 0.0
        self._frame_number = 0
        self._metadata: dict[str, object] = {}

    @property
    def active(self) -> bool:
        return self._session_dir is not None

    def start(self, frame: np.ndarray, label: str = "recognition", metadata: dict[str, object] | None = None) -> Path:
        if self._session_dir is not None:
            return self._session_dir
        if frame.ndim != 3 or frame.shape[0] < 1 or frame.shape[1] < 1:
            raise CaptureError("인식 개선 자료에 사용할 화면이 올바르지 않습니다.")

        self.directory.mkdir(parents=True, exist_ok=True)
        self.prune_old_sessions()
        safe_label = "".join(char if char.isalnum() or char in "-_" else "_" for char in label).strip("_") or "recognition"
        started = datetime.now()
        self._session_dir = self.directory / f"attempt_{started:%Y%m%d_%H%M%S_%f}_{safe_label}"
        self._session_dir.mkdir(parents=True, exist_ok=False)
        height, width = frame.shape[:2]
        self._started_at = self._clock()
        self._next_frame_at = self._started_at
        self._frame_number = 0
        self._metadata = {
            "label": label,
            "started_at": started.isoformat(timespec="milliseconds"),
            "duration_seconds": self.duration_seconds,
            "capture_fps": self.fps,
            "image_format": "jpeg",
            "jpeg_quality": self.jpeg_quality,
            "width": width,
            "height": height,
            **(metadata or {}),
        }
        self._write(frame, self._started_at, force=True)
        return self._session_dir

    def add(self, frame: np.ndarray) -> Path | None:
        if self._session_dir is None:
            return None
        now = self._clock()
        if now - self._started_at >= self.duration_seconds:
            self._write(frame, now, force=True)
            return self.finish()
        self._write(frame, now)
        return None

    def _write(self, frame: np.ndarray, now: float, force: bool = False) -> None:
        if self._session_dir is None or (not force and now < self._next_frame_at):
            return
        elapsed_ms = max(0, round((now - self._started_at) * 1000))
        filename = self._session_dir / f"frame_{self._frame_number:04d}_{elapsed_ms:05d}ms.jpg"
        try:
            ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        except cv2.error as exc:
            raise CaptureError(f"인식 개선용 진단 사진을 변환하지 못했습니다: {filename}") from exc
        if not ok:
            raise CaptureError(f"인식 개선용 진단 사진을 변환하지 못했습니다: {filename}")
        try:
            filename.write_bytes(encoded.tobytes())
        except OSError as exc:
            raise CaptureError(f"인식 개선용 진단 사진을 저장하지 못했습니다: {filename}") from exc
        self._frame_number += 1
        self._next_frame_at = now + (1.0 / self.fps)

    def annotate(self, **metadata: object) -> None:
        if self._session_dir is not None:
            self._metadata.update(metadata)

    @staticmethod
    def _json_default(value: object) -> object:
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, Path):
            return str(value)
        return str(value)

    def finish(self) -> Path | None:
        if self._session_dir is None:
            return None
        session_dir = self._session_dir
        self._metadata.update({
            "completed_at": datetime.now().isoformat(timespec="milliseconds"),
            "frame_count": self._frame_number,
        })
        try:
            (session_dir / "session.json").write_text(
                json.dumps(self._metadata, ensure_ascii=False, indent=2, default=self._json_default),
                encoding="utf-8",
            )
        except OSError as exc:
            raise CaptureError(f"인식 개선 자료 정보를 저장하지 못했습니다: {session_dir}") from exc
        finally:
            self._session_dir = None
            self._metadata = {}
            self._frame_number = 0
        self.prune_old_sessions(protected=session_dir)
        return session_dir

    def close(self) -> Path | None:
        return self.finish()

    def prune_old_sessions(self, protected: Path | None = None) -> list[Path]:
        if not self.directory.exists():
            return []
        sessions = [path for path in self.directory.iterdir() if path.is_dir() and path.name.startswith("attempt_")]
        sessions.sort(key=lambda path: (path.stat().st_mtime, path.name))
        sizes = {path: sum(item.stat().st_size for item in path.rglob("*") if item.is_file()) for path in sessions}
        total = sum(sizes.values())
        removed: list[Path] = []
        for session in sessions:
            if total <= self.max_total_bytes:
                break
            if protected is not None and session == protected:
                continue
            try:
                shutil.rmtree(session)
            except OSError as exc:
                raise CaptureError(f"오래된 인식 개선 자료를 정리하지 못했습니다: {session}") from exc
            total -= sizes[session]
            removed.append(session)
        return removed


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
