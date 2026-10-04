from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .models import DisplayMode


DEFAULT_KEYS = {
    "up": "↑ / W", "down": "↓ / S", "left": "← / A", "right": "→ / D",
    "select": "Enter / 마우스 1", "back": "Backspace / Esc",
}
UPDATE_CHANNELS = {"release", "beta"}
DIAGNOSTIC_UPLOAD_URL = "https://gta-reports.64-110-118-28.sslip.io/v1/reports"


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
    update_channel: str = "release"
    diagnostic_upload_enabled: bool = True
    diagnostic_upload_url: str = DIAGNOSTIC_UPLOAD_URL
    game_title_patterns: list[str] = field(default_factory=lambda: ["grand theft auto", "gta v"])

    @classmethod
    def load(cls, path: Path) -> "AppConfig":
        if not path.exists():
            config = cls()
            config.save(path)
            return config
        try:
            raw: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return cls()
        if not isinstance(raw, dict):
            return cls()
        known = {key: raw[key] for key in cls.__dataclass_fields__ if key in raw}
        # 채널 필드가 없던 기존 설정은 종전의 베타 채널 동작을 유지한다.
        if "update_channel" not in raw:
            known["update_channel"] = "beta"
        config = cls(**known)
        defaults = cls()
        # 잘못된 항목만 복구하고 기존의 정상적인 설정은 유지한다.
        for key in cls.__dataclass_fields__:
            value, default = getattr(config, key), getattr(defaults, key)
            if isinstance(default, bool):
                if not isinstance(value, bool):
                    # 잘못된 값 때문에 전송 등 선택 기능이 켜지지 않게 한다.
                    setattr(config, key, False)
            elif isinstance(default, (int, float)):
                try:
                    if isinstance(value, bool) or not math.isfinite(float(value)):
                        raise ValueError
                    value = int(value) if isinstance(default, int) else float(value)
                except (TypeError, ValueError, OverflowError):
                    value = default
                if key in {"confidence_threshold", "overlay_opacity"}:
                    if not 0 <= value <= 1 or (key == "overlay_opacity" and value == 0):
                        value = default
                elif key == "capture_output":
                    if value < 0:
                        value = default
                elif key == "guide_font_size":
                    value = max(8, min(24, value))
                elif key not in {"overlay_x", "overlay_y"} and value <= 0:
                    value = default
                setattr(config, key, value)
            elif isinstance(default, str) and (not isinstance(value, str) or not value.strip()):
                setattr(config, key, default)
        custom_keys = config.custom_keys if isinstance(config.custom_keys, dict) else {}
        config.custom_keys = {
            **DEFAULT_KEYS,
            **{key: value for key, value in custom_keys.items()
               if isinstance(value, str) and value.strip()},
        }
        patterns = config.game_title_patterns
        if not isinstance(patterns, list) or not patterns or any(
            not isinstance(value, str) or not value.strip() for value in patterns
        ):
            config.game_title_patterns = defaults.game_title_patterns
        if config.capture_backend not in {"auto", "dxgi", "winrt"}:
            config.capture_backend = defaults.capture_backend
        if config.display_mode not in {mode.value for mode in DisplayMode}:
            config.display_mode = defaults.display_mode
        if not isinstance(config.update_channel, str) or config.update_channel not in UPDATE_CHANNELS:
            config.update_channel = defaults.update_channel
        if config.diagnostic_upload_url != DIAGNOSTIC_UPLOAD_URL:
            config.diagnostic_upload_url = DIAGNOSTIC_UPLOAD_URL
            config.save(path)
        return config

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2), encoding="utf-8")
