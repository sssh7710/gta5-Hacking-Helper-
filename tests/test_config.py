from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from gta_helper.config import AppConfig, DIAGNOSTIC_UPLOAD_URL


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
            self.assertEqual(config.update_channel, "beta")
            self.assertTrue(config.diagnostic_upload_enabled)
            self.assertTrue(config.diagnostic_upload_url.startswith("https://"))
            self.assertEqual(config.guide_font_size, 11)
            config.custom_keys["select"] = "Space"
            config.controls_legend_enabled = True
            config.diagnostic_capture_enabled = False
            config.diagnostic_capture_max_mb = 500
            config.auto_update_enabled = False
            config.update_channel = "release"
            config.diagnostic_upload_enabled = False
            config.guide_font_size = 18
            config.save(path)
            loaded = AppConfig.load(path)
            self.assertEqual(loaded.custom_keys["select"], "Space")
            self.assertTrue(loaded.controls_legend_enabled)
            self.assertFalse(loaded.diagnostic_capture_enabled)
            self.assertEqual(loaded.diagnostic_capture_max_mb, 500)
            self.assertFalse(loaded.auto_update_enabled)
            self.assertEqual(loaded.update_channel, "release")
            self.assertFalse(loaded.diagnostic_upload_enabled)
            self.assertEqual(loaded.guide_font_size, 18)
            self.assertIn("up", loaded.custom_keys)
        finally:
            if path.exists():
                path.unlink()

    def test_font_size_is_backward_compatible_and_bounded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps({"capture_backend": "auto"}), encoding="utf-8")
            self.assertEqual(AppConfig.load(path).guide_font_size, 11)

            path.write_text(json.dumps({"guide_font_size": 99}), encoding="utf-8")
            self.assertEqual(AppConfig.load(path).guide_font_size, 24)

            path.write_text(json.dumps({"guide_font_size": "invalid"}), encoding="utf-8")
            self.assertEqual(AppConfig.load(path).guide_font_size, 11)

    def test_update_channel_is_backward_compatible_and_validated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps({"capture_backend": "auto"}), encoding="utf-8")
            self.assertEqual(AppConfig.load(path).update_channel, "beta")

            path.write_text(json.dumps({"update_channel": "release"}), encoding="utf-8")
            self.assertEqual(AppConfig.load(path).update_channel, "release")

            path.write_text(json.dumps({"update_channel": "invalid"}), encoding="utf-8")
            self.assertEqual(AppConfig.load(path).update_channel, "beta")

    def test_load_replaces_user_configured_upload_url(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(
                json.dumps({"diagnostic_upload_url": "https://example.invalid/v1/reports"}),
                encoding="utf-8",
            )

            config = AppConfig.load(path)

            self.assertEqual(config.diagnostic_upload_url, DIAGNOSTIC_UPLOAD_URL)
            persisted = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(persisted["diagnostic_upload_url"], DIAGNOSTIC_UPLOAD_URL)
