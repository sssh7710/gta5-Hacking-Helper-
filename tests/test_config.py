from __future__ import annotations

import unittest
from pathlib import Path

from gta_helper.config import AppConfig


class ConfigTests(unittest.TestCase):
    def test_load_creates_default_and_preserves_custom_keys(self) -> None:
        path = Path(__file__).resolve().parents[1] / "diagnostics" / "_test_config.json"
        try:
            config = AppConfig.load(path)
            self.assertTrue(path.exists())
            config.custom_keys["select"] = "Space"
            config.save(path)
            loaded = AppConfig.load(path)
            self.assertEqual(loaded.custom_keys["select"], "Space")
            self.assertIn("up", loaded.custom_keys)
        finally:
            if path.exists():
                path.unlink()
