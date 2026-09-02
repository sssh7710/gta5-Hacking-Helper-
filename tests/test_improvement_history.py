from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from gta_helper.improvement_history import ImprovementHistoryError, append_record, build_record, last_successful_to, read_history


class ImprovementHistoryTests(unittest.TestCase):
    def test_successful_records_form_contiguous_windows(self) -> None:
        first = build_record(
            [],
            computer="PC-A",
            server_log_from_utc="2026-09-01T00:00:00Z",
            server_log_to_utc="2026-09-03T00:00:00Z",
            report_count=3,
            commit="abc1234",
            result="success",
            summary="first",
            completed_at_utc="2026-09-03T01:00:00Z",
        )
        second = build_record(
            [first],
            computer="PC-B",
            server_log_from_utc="2026-09-03T00:00:00Z",
            server_log_to_utc="2026-09-05T00:00:00Z",
            report_count=5,
            commit="def5678",
            result="success",
            summary="second",
            completed_at_utc="2026-09-05T01:00:00Z",
        )
        self.assertEqual(last_successful_to([first, second]), "2026-09-05T00:00:00Z")

    def test_successful_record_cannot_skip_a_window(self) -> None:
        first = build_record(
            [], computer="PC-A", server_log_from_utc="2026-09-01T00:00:00Z", server_log_to_utc="2026-09-03T00:00:00Z",
            report_count=1, commit="abc1234", result="success", summary="first", completed_at_utc="2026-09-03T01:00:00Z",
        )
        with self.assertRaises(ImprovementHistoryError):
            build_record(
                [first], computer="PC-B", server_log_from_utc="2026-09-04T00:00:00Z", server_log_to_utc="2026-09-05T00:00:00Z",
                report_count=1, commit="def5678", result="success", summary="skipped", completed_at_utc="2026-09-05T01:00:00Z",
            )

    def test_failure_does_not_advance_successful_cursor(self) -> None:
        failure = build_record(
            [], computer="PC-A", server_log_from_utc="2026-09-01T00:00:00Z", server_log_to_utc="2026-09-03T00:00:00Z",
            report_count=2, commit="", result="failure", summary="test failure", completed_at_utc="2026-09-03T01:00:00Z",
        )
        self.assertIsNone(last_successful_to([failure]))

    def test_history_round_trip(self) -> None:
        record = build_record(
            [], computer="PC-A", server_log_from_utc="2026-09-01T00:00:00Z", server_log_to_utc="2026-09-03T00:00:00Z",
            report_count=0, commit="abc1234", result="success", summary="no reports", completed_at_utc="2026-09-03T01:00:00Z",
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "automation" / "improvement-history.jsonl"
            append_record(path, record)
            self.assertEqual(read_history(path), [record])


if __name__ == "__main__":
    unittest.main()
