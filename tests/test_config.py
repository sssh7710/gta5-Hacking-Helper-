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
            self.assertFalse(config.controls_legend_enabled)
            self.assertTrue(config.diagnostic_capture_enabled)
            self.assertEqual(config.diagnostic_capture_max_mb, 1024)
            self.assertTrue(config.auto_update_enabled)
            self.assertTrue(config.diagnostic_upload_enabled)
            self.assertTrue(config.diagnostic_upload_url.startswith("https://"))
            config.custom_keys["select"] = "Space"
            config.controls_legend_enabled = True
            config.diagnostic_capture_enabled = False
            config.diagnostic_capture_max_mb = 500
            config.auto_update_enabled = False
            config.diagnostic_upload_enabled = False
            config.save(path)
            loaded = AppConfig.load(path)
            self.assertEqual(loaded.custom_keys["select"], "Space")
            self.assertTrue(loaded.controls_legend_enabled)
            self.assertFalse(loaded.diagnostic_capture_enabled)
            self.assertEqual(loaded.diagnostic_capture_max_mb, 500)
            self.assertFalse(loaded.auto_update_enabled)
            self.assertFalse(loaded.diagnostic_upload_enabled)
            self.assertIn("up", loaded.custom_keys)
        finally:
            if path.exists():
                path.unlink()
