from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .models import DisplayMode


DEFAULT_KEYS = {
    "up": "↑ / W", "down": "↓ / S", "left": "← / A", "right": "→ / D",
    "select": "Enter / 마우스 1", "back": "Backspace / Esc",
}
UPDATE_CHANNELS = {"release", "beta"}


@dataclass
class AppConfig:
    capture_backend: str = "auto"
    capture_output: int = 0
    target_fps: int = 15
    confidence_threshold: float = 0.68
    display_mode: str = DisplayMode.CLICK_THROUGH.value
    guide_monitor: str = "auto"
    overlay_x: int = 20
    overlay_y: int = 80
    overlay_width: int = 390
    overlay_height: int = 245
    overlay_opacity: float = 0.90
    guide_font_size: int = 11
    voice_enabled: bool = False
    voice_rate: int = 165
    controls_legend_enabled: bool = False
    input_profile: str = "기본 키보드"
    custom_keys: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_KEYS))
    diagnostic_dir: str = "diagnostics"
    diagnostic_capture_enabled: bool = True
    diagnostic_capture_seconds: float = 7.0
    diagnostic_capture_fps: int = 8
    diagnostic_capture_max_mb: int = 1024
    auto_update_enabled: bool = True
    update_channel: str = "beta"
    diagnostic_upload_enabled: bool = True
    diagnostic_upload_url: str = "https://gta-reports.64-110-118-28.sslip.io/v1/reports"
    game_title_patterns: list[str] = field(default_factory=lambda: ["grand theft auto", "gta v"])

    @classmethod
    def load(cls, path: Path) -> "AppConfig":
        if not path.exists():
            config = cls()
            config.save(path)
            return config
        try:
            raw: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return cls()
        known = {key: raw[key] for key in cls.__dataclass_fields__ if key in raw}
        config = cls(**known)
        config.custom_keys = {**DEFAULT_KEYS, **config.custom_keys}
        try:
            config.guide_font_size = max(8, min(24, int(config.guide_font_size)))
        except (TypeError, ValueError):
            config.guide_font_size = 11
        if not isinstance(config.update_channel, str) or config.update_channel not in UPDATE_CHANNELS:
            config.update_channel = "beta"
        return config

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2), encoding="utf-8")
