from __future__ import annotations

import json
import queue
import tempfile
import threading
import urllib.error
import urllib.request
import uuid
import zipfile
from collections.abc import Callable
from pathlib import Path

from .version import APP_VERSION


MAX_REPORT_BYTES = 50 * 1024 * 1024


class ReportError(RuntimeError):
    pass


def session_outcome(session_dir: str | Path, confidence_threshold: float) -> str | None:
    metadata_path = Path(session_dir) / "session.json"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    recorded = metadata.get("answer_outcome")
    if recorded in {"success", "failure"}:
        return str(recorded)
    confidence = metadata.get("result_confidence")
    if confidence is None or not metadata.get("result_summary"):
        return "failure"
    try:
        return "success" if float(confidence) >= confidence_threshold else "failure"
    except (TypeError, ValueError):
        return "failure"


def is_unresolved_session(session_dir: str | Path, confidence_threshold: float) -> bool:
    return session_outcome(session_dir, confidence_threshold) == "failure"


def build_report_archive(session_dir: str | Path, target: str | Path) -> Path:
    session = Path(session_dir)
    metadata_path = session / "session.json"
    if not metadata_path.is_file():
        raise ReportError("진단 세션 정보가 없습니다.")
    files = [metadata_path, *sorted(session.glob("frame_*.jpg"))]
    target_path = Path(target)
    with zipfile.ZipFile(target_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, arcname=path.name)
    if target_path.stat().st_size > MAX_REPORT_BYTES:
        target_path.unlink(missing_ok=True)
        raise ReportError("진단 자료가 전송 허용 크기 50MB를 초과했습니다.")
    return target_path


def upload_report(session_dir: str | Path, endpoint: str) -> str:
    if not endpoint.lower().startswith("https://"):
        raise ReportError("진단 전송 주소는 HTTPS여야 합니다.")
    report_id = uuid.uuid4().hex
    with tempfile.TemporaryDirectory(prefix="gta-helper-report-") as temporary:
        archive_path = build_report_archive(session_dir, Path(temporary) / f"{report_id}.zip")
        request = urllib.request.Request(
            endpoint,
            data=archive_path.read_bytes(),
            method="POST",
            headers={
                "Content-Type": "application/zip",
                "Content-Length": str(archive_path.stat().st_size),
                "User-Agent": f"gta-hacking-helper/{APP_VERSION}",
                "X-Report-Id": report_id,
                "X-App-Version": APP_VERSION,
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                if not 200 <= response.status < 300:
                    raise ReportError(f"진단 서버가 HTTP {response.status}를 반환했습니다.")
        except (OSError, urllib.error.HTTPError) as exc:
            raise ReportError(f"진단 자료를 전송하지 못했습니다: {exc}") from exc
    return report_id


class DiagnosticReporter(threading.Thread):
    def __init__(
        self,
        endpoint: str,
        confidence_threshold: float,
        notify: Callable[[str], None] | None = None,
    ) -> None:
        super().__init__(daemon=True, name="gta-helper-reporter")
        self.endpoint = endpoint.strip()
        self.confidence_threshold = confidence_threshold
        self.notify = notify or (lambda message: None)
        self._queue: queue.Queue[tuple[Path, str] | None] = queue.Queue()

    @property
    def configured(self) -> bool:
        return self.endpoint.lower().startswith("https://")

    def submit(self, session_dir: str | Path) -> bool:
        if not self.configured:
            return False
        outcome = session_outcome(session_dir, self.confidence_threshold)
        if outcome is None:
            return False
        session = Path(session_dir)
        try:
            metadata = json.loads((session / "session.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            metadata = {}
        report_kind = "manual" if metadata.get("capture_trigger") == "manual" else outcome
        self._queue.put((session, report_kind))
        return True

    def stop(self) -> None:
        self._queue.put(None)

    def run(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                return
            session, report_kind = item
            try:
                report_id = upload_report(session, self.endpoint)
            except ReportError as exc:
                self.notify(str(exc))
            else:
                label = {"success": "성공", "failure": "실패", "manual": "수동"}[report_kind]
                self.notify(f"인식 결과 자료 전송 완료 ({label}): {report_id[:8]}")
