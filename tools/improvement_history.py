"""예약된 진단 개선 작업의 서버 조회 구간과 결과를 기록한다.

예시:
    .venv\\Scripts\\python.exe -B tools\\improvement_history.py next --initial-from-utc 2026-09-01T00:00:00Z
    .venv\\Scripts\\python.exe -B tools\\improvement_history.py record --computer PC-A --from-utc 2026-09-01T00:00:00Z --to-utc 2026-09-08T01:00:00Z --report-count 24 --commit abc1234 --summary "점멸 인식 회귀 보정"
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gta_helper.improvement_history import (
    ImprovementHistoryError,
    append_record,
    build_record,
    format_utc_timestamp,
    last_successful_to,
    parse_utc_timestamp,
    read_history,
)


DEFAULT_HISTORY = ROOT / "automation" / "improvement-history.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="두 PC의 진단 개선 작업 이력 관리")
    parser.add_argument("--history", type=Path, default=DEFAULT_HISTORY, help="공유할 JSONL 이력 파일")
    commands = parser.add_subparsers(dest="command", required=True)

    next_command = commands.add_parser("next", help="다음 서버 조회 구간을 출력")
    next_command.add_argument("--initial-from-utc", help="첫 작업일 때만 지정할 초기 조회 시작 UTC 시각")
    next_command.add_argument("--to-utc", help="조회 종료 UTC 시각(기본값: 현재 시각)")

    record_command = commands.add_parser("record", help="개선 작업 결과를 이력에 추가")
    record_command.add_argument("--computer", required=True, help="작업한 컴퓨터 이름")
    record_command.add_argument("--from-utc", required=True, dest="from_utc", help="처리 시작 UTC 시각")
    record_command.add_argument("--to-utc", required=True, dest="to_utc", help="처리 종료 UTC 시각")
    record_command.add_argument("--report-count", required=True, type=int, help="가져온 서버 진단 자료 수")
    record_command.add_argument("--commit", required=True, help="개선 작업 커밋 SHA")
    record_command.add_argument("--result", choices=("success", "failure"), default="success")
    record_command.add_argument("--summary", required=True, help="변경 또는 실패 요약")
    record_command.add_argument("--completed-at-utc", help="완료 UTC 시각(기본값: 현재 시각)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        records = read_history(args.history)
        if args.command == "next":
            previous = last_successful_to(records)
            start = previous or args.initial_from_utc
            if not start:
                raise ImprovementHistoryError("첫 작업은 --initial-from-utc를 지정해야 합니다.")
            end = args.to_utc or format_utc_timestamp(datetime.now(timezone.utc))
            if parse_utc_timestamp(end) <= parse_utc_timestamp(start):
                raise ImprovementHistoryError("조회 종료 시각은 시작 시각보다 뒤여야 합니다.")
            print(json.dumps({"server_log_from_utc": start, "server_log_to_utc": end}, ensure_ascii=False))
            return 0

        record = build_record(
            records,
            computer=args.computer,
            server_log_from_utc=args.from_utc,
            server_log_to_utc=args.to_utc,
            report_count=args.report_count,
            commit=args.commit,
            result=args.result,
            summary=args.summary,
            completed_at_utc=args.completed_at_utc,
        )
        append_record(args.history, record)
        print(f"기록 완료: {args.history}")
        return 0
    except ImprovementHistoryError as exc:
        print(f"오류: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
