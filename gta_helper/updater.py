from __future__ import annotations

import hashlib
import json
import re
import shutil
import tempfile
import urllib.parse
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any


RELEASES_URL = "https://api.github.com/repos/sssh7710/gta5-Hacking-Helper-/releases"
USER_AGENT = "gta-hacking-helper-updater"
MAX_ARCHIVE_BYTES = 100 * 1024 * 1024
MAX_EXTRACTED_BYTES = 300 * 1024 * 1024
PROTECTED_ROOT_NAMES = {".git", ".venv", "config.json", "diagnostics", "updates"}
UPDATE_CHANNELS = {"release", "beta"}


class UpdateError(RuntimeError):
    pass


@dataclass(frozen=True)
class UpdateInfo:
    tag: str
    version: str
    archive_url: str
    checksum_url: str
    release_url: str


def version_key(value: str) -> tuple[int, int, int, int, int]:
    match = re.fullmatch(
        r"v?(\d+)\.(\d+)\.(\d+)(?:[-.]?(dev|alpha|beta|rc)[-.]?(\d+))?",
        value.strip(),
        re.IGNORECASE,
    )
    if match is None:
        raise ValueError(f"지원하지 않는 버전 형식입니다: {value}")
    major, minor, patch = (int(match.group(index)) for index in range(1, 4))
    stage = (match.group(4) or "final").lower()
    stage_number = int(match.group(5) or 0)
    return major, minor, patch, {"dev": 0, "alpha": 1, "beta": 2, "rc": 3, "final": 4}[stage], stage_number


def select_update(releases: list[dict[str, Any]], current_version: str, channel: str = "beta") -> UpdateInfo | None:
    if channel not in UPDATE_CHANNELS:
        raise ValueError(f"지원하지 않는 업데이트 채널입니다: {channel}")
    current_key = version_key(current_version)
    candidates: list[tuple[tuple[int, int, int, int, int], UpdateInfo]] = []
    for release in releases:
        if release.get("draft"):
            continue
        tag = str(release.get("tag_name", ""))
        try:
            key = version_key(tag)
        except ValueError:
            continue
        if channel == "release" and (release.get("prerelease") or key[3] != 4):
            continue
        if key <= current_key:
            continue
        assets = release.get("assets") or []
        archive = next(
            (asset for asset in assets if str(asset.get("name", "")).endswith("-full-files.zip")),
            None,
        )
        checksum = next(
            (asset for asset in assets if str(asset.get("name", "")).endswith("-full-files.zip.sha256")),
            None,
        )
        if archive is None or checksum is None:
            continue
        candidates.append((key, UpdateInfo(
            tag=tag,
            version=tag.removeprefix("v"),
            archive_url=str(archive["browser_download_url"]),
            checksum_url=str(checksum["browser_download_url"]),
            release_url=str(release.get("html_url", "")),
        )))
    return max(candidates, key=lambda item: item[0])[1] if candidates else None


def _request_bytes(url: str, max_bytes: int) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/vnd.github+json"})
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            length = response.headers.get("Content-Length")
            if length is not None and int(length) > max_bytes:
                raise UpdateError("업데이트 파일이 허용 크기를 초과했습니다.")
            data = response.read(max_bytes + 1)
    except (OSError, ValueError) as exc:
        raise UpdateError(f"업데이트 서버에 연결하지 못했습니다: {exc}") from exc
    if len(data) > max_bytes:
        raise UpdateError("업데이트 파일이 허용 크기를 초과했습니다.")
    return data


def check_for_update(current_version: str, channel: str = "beta") -> UpdateInfo | None:
    try:
        releases = json.loads(_request_bytes(RELEASES_URL, 2 * 1024 * 1024).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise UpdateError("GitHub 릴리스 응답을 읽지 못했습니다.") from exc
    if not isinstance(releases, list):
        raise UpdateError("GitHub 릴리스 응답 형식이 올바르지 않습니다.")
    return select_update(releases, current_version, channel)


def download_update(info: UpdateInfo, directory: str | Path) -> Path:
    target_dir = Path(directory) / info.tag
    target_dir.mkdir(parents=True, exist_ok=True)
    archive_path = target_dir / Path(urllib.parse.urlparse(info.archive_url).path).name
    try:
        checksum_text = _request_bytes(info.checksum_url, 4096).decode("ascii", errors="strict")
    except UnicodeDecodeError as exc:
        raise UpdateError("업데이트 체크섬 파일이 올바르지 않습니다.") from exc
    expected_match = re.search(r"\b([0-9a-fA-F]{64})\b", checksum_text)
    if expected_match is None:
        raise UpdateError("업데이트 체크섬 파일이 올바르지 않습니다.")
    archive_data = _request_bytes(info.archive_url, MAX_ARCHIVE_BYTES)
    actual = hashlib.sha256(archive_data).hexdigest()
    if actual.lower() != expected_match.group(1).lower():
        raise UpdateError("업데이트 파일의 SHA-256 검증에 실패했습니다.")
    archive_path.write_bytes(archive_data)
    return archive_path


def safe_extract_archive(archive_path: str | Path, destination: str | Path) -> Path:
    destination_path = Path(destination).resolve()
    destination_path.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(archive_path) as archive:
            if sum(item.file_size for item in archive.infolist()) > MAX_EXTRACTED_BYTES:
                raise UpdateError("압축 해제된 업데이트 파일이 허용 크기를 초과합니다.")
            for item in archive.infolist():
                relative = PurePosixPath(item.filename)
                if relative.is_absolute() or ".." in relative.parts or not relative.parts or any(":" in part for part in relative.parts):
                    raise UpdateError(f"안전하지 않은 업데이트 경로입니다: {item.filename}")
                if (item.external_attr >> 16) & 0o170000 == 0o120000:
                    raise UpdateError(f"심볼릭 링크는 업데이트에 포함할 수 없습니다: {item.filename}")
                resolved = (destination_path / Path(*relative.parts)).resolve()
                if destination_path not in resolved.parents and resolved != destination_path:
                    raise UpdateError(f"업데이트 경로가 대상 폴더를 벗어납니다: {item.filename}")
            archive.extractall(destination_path)
    except (OSError, zipfile.BadZipFile) as exc:
        raise UpdateError(f"업데이트 ZIP을 풀지 못했습니다: {exc}") from exc
    return destination_path


def apply_update(archive_path: str | Path, project_root: str | Path) -> None:
    root = Path(project_root).resolve()
    with tempfile.TemporaryDirectory(prefix="gta-helper-update-") as temporary:
        extracted = safe_extract_archive(archive_path, temporary)
        children = [path for path in extracted.iterdir()]
        source = children[0] if len(children) == 1 and children[0].is_dir() else extracted
        if not (source / "app.py").is_file() or not (source / "run.bat").is_file():
            raise UpdateError("업데이트 ZIP에 필수 프로그램 파일이 없습니다.")
        for item in source.rglob("*"):
            relative = item.relative_to(source)
            if relative.parts[0] in PROTECTED_ROOT_NAMES:
                continue
            target = root / relative
            if item.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            elif item.is_file():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(item, target)
