from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from gta_helper.windowing import exclude_from_capture


class WindowingTests(unittest.TestCase):
    def test_capture_exclusion_uses_top_level_window(self) -> None:
        user32 = Mock()
        user32.GetAncestor.return_value = 456
        user32.SetWindowDisplayAffinity.return_value = 1

        with patch(
            "gta_helper.windowing.ctypes.windll",
            SimpleNamespace(user32=user32),
            create=True,
        ):
            applied = exclude_from_capture(123)

        self.assertTrue(applied)
        user32.GetAncestor.assert_called_once_with(123, 2)
        user32.SetWindowDisplayAffinity.assert_called_once_with(456, 0x11)


if __name__ == "__main__":
    unittest.main()
