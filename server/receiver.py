from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import tempfile
import zipfile
from datetime import datetime, timedelta, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


SERVICE_VERSION = "1.0.0"
MAX_REQUEST_BYTES = 50 * 1024 * 1024
MAX_EXTRACTED_BYTES = 100 * 1024 * 1024
MAX_METADATA_BYTES = 1024 * 1024
MAX_FRAME_BYTES = 10 * 1024 * 1024
MAX_FRAMES = 80
MAX_STORAGE_BYTES = 5 * 1024 * 1024 * 1024
REPORT_ID_RE = re.compile(r"[0-9a-f]{32}")
APP_VERSION_RE = re.compile(r"[0-9A-Za-z._-]{1,40}")
FRAME_NAME_RE = re.compile(r"frame_\d{4}_\d+ms\.jpg")


class ValidationError(ValueError):
    pass


class StorageFullError(OSError):
    pass


def validate_report_archive(data: bytes) -> dict[str, Any]:
    if not data or len(data) > MAX_REQUEST_BYTES:
        raise ValidationError("요청 크기가 허용 범위를 벗어났습니다.")
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            items = archive.infolist()
            names = [item.filename for item in items]
            if len(names) != len(set(names)):
                raise ValidationError("ZIP에 중복 파일 이름이 있습니다.")
            if "session.json" not in names:
                raise ValidationError("session.json이 없습니다.")
            if any("/" in name or "\\" in name or name in {"", ".", ".."} for name in names):
                raise ValidationError("ZIP 안의 경로를 허용하지 않습니다.")
            frames = [name for name in names if FRAME_NAME_RE.fullmatch(name)]
            if not 1 <= len(frames) <= MAX_FRAMES:
                raise ValidationError("진단 JPEG 개수가 허용 범위를 벗어났습니다.")
            if set(names) != {"session.json", *frames}:
                raise ValidationError("허용되지 않은 파일이 포함되어 있습니다.")
            if sum(item.file_size for item in items) > MAX_EXTRACTED_BYTES:
                raise ValidationError("압축 해제 크기가 허용 범위를 벗어났습니다.")
            metadata_item = archive.getinfo("session.json")
            if metadata_item.file_size > MAX_METADATA_BYTES:
                raise ValidationError("session.json이 너무 큽니다.")
            try:
                metadata = json.loads(archive.read(metadata_item).decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValidationError("session.json 형식이 올바르지 않습니다.") from exc
            if not isinstance(metadata, dict):
                raise ValidationError("session.json은 JSON 객체여야 합니다.")
            for name in frames:
                item = archive.getinfo(name)
                if not 4 <= item.file_size <= MAX_FRAME_BYTES:
                    raise ValidationError("JPEG 크기가 허용 범위를 벗어났습니다.")
                image = archive.read(item)
                if not image.startswith(b"\xff\xd8") or not image.endswith(b"\xff\xd9"):
                    raise ValidationError("JPEG 파일 형식이 올바르지 않습니다.")
    except zipfile.BadZipFile as exc:
        raise ValidationError("올바른 ZIP 파일이 아닙니다.") from exc
    return metadata


def store_report(data: bytes, report_id: str, storage: str | Path, now: datetime | None = None) -> tuple[Path, bool]:
    current = now or datetime.now(timezone.utc)
    day = current.strftime("%Y-%m-%d")
    directory = Path(storage) / day
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{report_id}.zip"
    if target.exists():
        return target, False
    stored_bytes = sum(path.stat().st_size for path in Path(storage).glob("????-??-??/*.zip") if path.is_file())
    if stored_bytes + len(data) > MAX_STORAGE_BYTES:
        raise StorageFullError("진단 자료 저장 한도에 도달했습니다.")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{report_id}.", suffix=".tmp", dir=directory)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(data)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_name, target)
        os.chmod(target, 0o600)
    finally:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
    return target, True


def cleanup_reports(storage: str | Path, retention_days: int, now: datetime | None = None) -> int:
    root = Path(storage)
    if not root.exists():
        return 0
    cutoff = (now or datetime.now(timezone.utc)) - timedelta(days=max(1, retention_days))
    removed = 0
    for path in root.glob("????-??-??/*.zip"):
        try:
            modified = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
            if modified < cutoff:
                path.unlink()
                removed += 1
        except StorageFullError:
            self._json_response(HTTPStatus.INSUFFICIENT_STORAGE, {"error": "storage_full"})
            return
        except OSError:
            continue
    for directory in root.glob("????-??-??"):
        try:
            directory.rmdir()
        except OSError:
            pass
    return removed


class ReportServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int], storage: str | Path) -> None:
        self.storage = Path(storage)
        super().__init__(address, ReportHandler)


class ReportHandler(BaseHTTPRequestHandler):
    server: ReportServer

    def log_message(self, format: str, *args: object) -> None:
        return

    def _json_response(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path != "/health":
            self._json_response(HTTPStatus.NOT_FOUND, {"error": "not_found"})
            return
        self._json_response(HTTPStatus.OK, {"status": "ok", "version": SERVICE_VERSION})

    def do_POST(self) -> None:
        if self.path != "/v1/reports":
            self._json_response(HTTPStatus.NOT_FOUND, {"error": "not_found"})
            return
        if self.headers.get_content_type() != "application/zip":
            self._json_response(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, {"error": "content_type"})
            return
        report_id = self.headers.get("X-Report-Id", "")
        app_version = self.headers.get("X-App-Version", "")
        if REPORT_ID_RE.fullmatch(report_id) is None or APP_VERSION_RE.fullmatch(app_version) is None:
            self._json_response(HTTPStatus.BAD_REQUEST, {"error": "headers"})
            return
        try:
            length = int(self.headers.get("Content-Length", ""))
        except ValueError:
            length = -1
        if not 1 <= length <= MAX_REQUEST_BYTES:
            self._json_response(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"error": "size"})
            return
        data = self.rfile.read(length)
        if len(data) != length:
            self._json_response(HTTPStatus.BAD_REQUEST, {"error": "length"})
            return
        try:
            validate_report_archive(data)
            _, created = store_report(data, report_id, self.server.storage)
        except ValidationError as exc:
            self._json_response(HTTPStatus.BAD_REQUEST, {"error": "invalid_report", "message": str(exc)})
            return
        except OSError:
            self._json_response(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": "storage"})
            return
        digest = hashlib.sha256(data).hexdigest()
        print(f"report accepted id={report_id[:8]} created={created} bytes={len(data)} sha256={digest}", flush=True)
        self._json_response(HTTPStatus.OK, {"report_id": report_id, "created": created})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GTA 미판단 인식 자료 수신기")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--storage", type=Path, default=Path("/var/lib/gta-report-receiver"))
    parser.add_argument("--cleanup", action="store_true")
    parser.add_argument("--retention-days", type=int, default=30)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.cleanup:
        print(f"removed={cleanup_reports(args.storage, args.retention_days)}")
        return 0
    args.storage.mkdir(parents=True, exist_ok=True)
    server = ReportServer((args.host, args.port), args.storage)
    print(f"receiver listening on {args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
