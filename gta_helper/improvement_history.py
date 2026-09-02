"""두 PC의 진단 자료 개선 구간을 이어서 기록하는 도구 지원 함수."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class ImprovementHistoryError(ValueError):
    """개선 작업 이력의 순서 또는 형식이 잘못된 경우 발생한다."""


def parse_utc_timestamp(value: str) -> datetime:
    """`YYYY-MM-DDTHH:MM:SSZ` 형식의 UTC 시각을 읽는다."""
    if not value.endswith("Z"):
        raise ImprovementHistoryError("시각은 UTC Z 형식이어야 합니다. 예: 2026-09-02T01:00:00Z")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise ImprovementHistoryError("시각 형식이 올바르지 않습니다.") from exc
    if parsed.tzinfo != timezone.utc:
        raise ImprovementHistoryError("시각은 UTC여야 합니다.")
    return parsed.replace(microsecond=0)


def format_utc_timestamp(value: datetime) -> str:
    return value.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def read_history(path: str | Path) -> list[dict[str, Any]]:
    history_path = Path(path)
    if not history_path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(history_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ImprovementHistoryError(f"{history_path} {line_number}행이 JSON이 아닙니다.") from exc
        if not isinstance(record, dict):
            raise ImprovementHistoryError(f"{history_path} {line_number}행은 JSON 객체여야 합니다.")
        records.append(record)
    return records


def last_successful_to(records: list[dict[str, Any]]) -> str | None:
    successful = [record for record in records if record.get("result") == "success"]
    if not successful:
        return None
    value = successful[-1].get("server_log_to_utc")
    if not isinstance(value, str):
        raise ImprovementHistoryError("마지막 성공 기록에 server_log_to_utc가 없습니다.")
    parse_utc_timestamp(value)
    return value


def build_record(
    records: list[dict[str, Any]],
    *,
    computer: str,
    server_log_from_utc: str,
    server_log_to_utc: str,
    report_count: int,
    commit: str,
    result: str,
    summary: str,
    completed_at_utc: str | None = None,
) -> dict[str, Any]:
    if not computer.strip() or not summary.strip():
        raise ImprovementHistoryError("computer와 summary는 비워 둘 수 없습니다.")
    if result not in {"success", "failure"}:
        raise ImprovementHistoryError("result는 success 또는 failure여야 합니다.")
    if report_count < 0:
        raise ImprovementHistoryError("report_count는 0 이상이어야 합니다.")
    start = parse_utc_timestamp(server_log_from_utc)
    end = parse_utc_timestamp(server_log_to_utc)
    if end <= start:
        raise ImprovementHistoryError("server_log_to_utc는 server_log_from_utc보다 뒤여야 합니다.")
    previous = last_successful_to(records)
    if result == "success" and previous is not None and server_log_from_utc != previous:
        raise ImprovementHistoryError("성공 기록의 시작 시각은 마지막 성공 기록의 종료 시각과 같아야 합니다.")
    finished = parse_utc_timestamp(completed_at_utc) if completed_at_utc else datetime.now(timezone.utc).replace(microsecond=0)
    if finished < end:
        raise ImprovementHistoryError("completed_at_utc는 처리 구간 종료 시각보다 빠를 수 없습니다.")
    return {
        "completed_at_utc": format_utc_timestamp(finished),
        "server_log_from_utc": format_utc_timestamp(start),
        "server_log_to_utc": format_utc_timestamp(end),
        "computer": computer.strip(),
        "report_count": report_count,
        "commit": commit.strip(),
        "result": result,
        "summary": summary.strip(),
    }


def append_record(path: str | Path, record: dict[str, Any]) -> None:
    history_path = Path(path)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with history_path.open("a", encoding="utf-8", newline="\n") as output:
        output.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
