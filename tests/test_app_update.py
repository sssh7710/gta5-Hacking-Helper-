from __future__ import annotations

import queue
import unittest
from unittest.mock import patch

from app import HelperApp
from gta_helper.updater import UpdateError, UpdateInfo
from gta_helper.version import APP_VERSION


class AppUpdateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.app = object.__new__(HelperApp)
        self.app.events = queue.Queue()

    def test_manual_check_reports_current_version(self) -> None:
        button = object()
        with patch("app.check_for_update", return_value=None) as check_for_update_mock:
            self.app._check_for_updates(manual=True, source_button=button, channel="release")  # type: ignore[arg-type]

        self.assertEqual(self.app.events.get_nowait(), ("update_current", button))
        self.assertEqual(self.app.events.get_nowait(), ("update_check_finished", button))
        check_for_update_mock.assert_called_once_with(APP_VERSION, "release")

    def test_manual_check_reports_available_update(self) -> None:
        button = object()
        info = UpdateInfo("v1.0.0-beta.14", "1.0.0-beta.14", "archive", "checksum", "release")
        with patch("app.check_for_update", return_value=info):
            self.app._check_for_updates(manual=True, source_button=button, channel="beta")  # type: ignore[arg-type]

        self.assertEqual(self.app.events.get_nowait(), ("update_available", (info, button)))
        self.assertEqual(self.app.events.get_nowait(), ("update_check_finished", button))

    def test_manual_check_reports_connection_error(self) -> None:
        button = object()
        with patch("app.check_for_update", side_effect=UpdateError("연결 실패")):
            self.app._check_for_updates(manual=True, source_button=button, channel="release")  # type: ignore[arg-type]

        self.assertEqual(self.app.events.get_nowait(), ("update_check_error", ("연결 실패", True, button)))
        self.assertEqual(self.app.events.get_nowait(), ("update_check_finished", button))


if __name__ == "__main__":
    unittest.main()
