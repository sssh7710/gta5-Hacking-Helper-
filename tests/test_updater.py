from __future__ import annotations

import tempfile
import unittest
import zipfile
from pathlib import Path

from gta_helper.updater import UpdateError, apply_update, safe_extract_archive, select_update, version_key


class UpdaterTests(unittest.TestCase):
    def test_beta_versions_are_compared_numerically(self) -> None:
        self.assertGreater(version_key("v1.0.0-beta.10"), version_key("1.0.0-beta.9"))
        self.assertGreater(version_key("1.0.0"), version_key("1.0.0-rc.2"))

    def test_selects_newest_release_with_full_archive_and_checksum(self) -> None:
        releases = [
            {
                "tag_name": "v1.0.0-beta.10",
                "html_url": "https://example.test/release",
                "assets": [
                    {"name": "helper-full-files.zip", "browser_download_url": "https://example.test/update.zip"},
                    {"name": "helper-full-files.zip.sha256", "browser_download_url": "https://example.test/update.sha256"},
                ],
            },
            {
                "tag_name": "v1.0.0-beta.11",
                "assets": [{"name": "source.zip", "browser_download_url": "https://example.test/source.zip"}],
            },
        ]

        selected = select_update(releases, "1.0.0-beta.9")

        self.assertIsNotNone(selected)
        assert selected is not None
        self.assertEqual(selected.tag, "v1.0.0-beta.10")
        self.assertEqual(selected.archive_url, "https://example.test/update.zip")

    def test_rejects_archive_path_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "bad.zip"
            with zipfile.ZipFile(archive, "w") as output:
                output.writestr("../outside.txt", "bad")

            with self.assertRaises(UpdateError):
                safe_extract_archive(archive, root / "extract")

    def test_apply_update_preserves_local_data(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "app"
            root.mkdir()
            (root / "app.py").write_text("old", encoding="utf-8")
            (root / "config.json").write_text("local config", encoding="utf-8")
            (root / "diagnostics").mkdir()
            (root / "diagnostics" / "local.jpg").write_text("local image", encoding="utf-8")
            archive = Path(directory) / "update.zip"
            with zipfile.ZipFile(archive, "w") as output:
                output.writestr("package/app.py", "new")
                output.writestr("package/run.bat", "run")
                output.writestr("package/config.json", "release config")
                output.writestr("package/diagnostics/remote.jpg", "release image")
                output.writestr("package/gta_helper/new_module.py", "added")

            apply_update(archive, root)

            self.assertEqual((root / "app.py").read_text(encoding="utf-8"), "new")
            self.assertEqual((root / "config.json").read_text(encoding="utf-8"), "local config")
            self.assertTrue((root / "diagnostics" / "local.jpg").exists())
            self.assertFalse((root / "diagnostics" / "remote.jpg").exists())
            self.assertTrue((root / "gta_helper" / "new_module.py").exists())


if __name__ == "__main__":
    unittest.main()
