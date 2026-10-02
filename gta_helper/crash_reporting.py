"""화면, 예외 메시지, 개인 경로 없이 앱 오류 위치를 기록하고 전송한다."""
from __future__ import annotations

import io
import json
import sys
import threading
import time
import traceback
import urllib.request
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from .config import DIAGNOSTIC_UPLOAD_URL
from .version import APP_VERSION


class CrashReporter:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.directory = root / "diagnostics" / "app-errors"
        self.lock = threading.Lock()
        self.wake = threading.Event()
        self.last_error: tuple | None = None
        self.last_error_at = 0.0
        self.worker = threading.Thread(target=self._run, daemon=True, name="gta-helper-crash-reporter")

    def enabled(self) -> bool:
        try:
            raw = json.loads((self.root / "config.json").read_text(encoding="utf-8"))
            return isinstance(raw, dict) and raw.get("diagnostic_upload_enabled", True) is True
        except FileNotFoundError:
            return True
        except (OSError, ValueError):
            return False

    def record(self, exc_type: type, exc: BaseException, tb: object, source: str) -> Path | None:
        if issubclass(exc_type, (KeyboardInterrupt, SystemExit)):
            return None
        # 메시지와 소스 줄에는 사용자 데이터가 들어갈 수 있으므로 포함하지 않는다.
        metadata = {
            "report_type": "app_error", "source": source,
            "app_version": APP_VERSION,
            "occurred_at": datetime.now(timezone.utc).isoformat(),
            "exception_type": exc_type.__name__,
            "stack": [{"file": Path(frame.filename).name, "line": frame.lineno,
                       "function": frame.name} for frame in traceback.extract_tb(tb)[-40:]],
        }
        try:
            with self.lock:
                signature = (source, metadata["exception_type"], str(metadata["stack"]))
                now = time.monotonic()
                if signature == self.last_error and now - self.last_error_at < 60:
                    return None
                self.directory.mkdir(parents=True, exist_ok=True)
                for old in sorted(self.directory.glob("*.json"), key=lambda p: p.stat().st_mtime)[:-99]:
                    old.unlink()
                path = self.directory / f"{uuid.uuid4().hex}.json"
                path.write_text(json.dumps(metadata, ensure_ascii=False), encoding="utf-8")
                self.last_error, self.last_error_at = signature, now
        except OSError:
            return None
        self.wake.set()
        return path

    def send(self, path: Path) -> bool:
        if not self.enabled():
            return False
        try:
            with self.lock:
                metadata = path.read_bytes()
            buffer = io.BytesIO()
            with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                archive.writestr("session.json", metadata)
            request = urllib.request.Request(DIAGNOSTIC_UPLOAD_URL, data=buffer.getvalue(), method="POST",
                headers={"Content-Type": "application/zip", "X-Report-Id": path.stem,
                         "X-App-Version": APP_VERSION, "User-Agent": f"gta-hacking-helper/{APP_VERSION}"})
            with urllib.request.urlopen(request, timeout=3) as response:
                if response.status != 200:
                    return False
            with self.lock:
                path.unlink(missing_ok=True)
            return True
        except (OSError, ValueError):
            return False

    def _run(self) -> None:
        while True:
            self.wake.wait(60)
            self.wake.clear()
            try:
                pending = sorted(self.directory.glob("*.json"))[:100]
                for path in pending:
                    if not self.send(path):
                        break
            except OSError:
                pass

    def install(self) -> None:
        original_sys = sys.excepthook
        original_thread = threading.excepthook

        def main_hook(exc_type, exc, tb):
            path = self.record(exc_type, exc, tb, "main")
            if path is not None:
                self.send(path)  # 종료 전 시도하고 실패한 자료는 다음 실행에서 재전송한다.
            original_sys(exc_type, exc, tb)

        def thread_hook(args):
            self.record(args.exc_type, args.exc_value, args.exc_traceback, "thread")
            original_thread(args)

        sys.excepthook = main_hook
        threading.excepthook = thread_hook
        self.worker.start()
        self.wake.set()

    def tk_exception(self, exc_type, exc, tb) -> None:
        self.record(exc_type, exc, tb, "ui")
        traceback.print_exception(exc_type, exc, tb)
