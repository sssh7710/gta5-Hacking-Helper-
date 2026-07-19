from __future__ import annotations

import ctypes
from ctypes import wintypes
from dataclasses import dataclass


@dataclass(frozen=True)
class Rect:
    left: int
    top: int
    right: int
    bottom: int

    @property
    def width(self) -> int: return max(0, self.right - self.left)
    @property
    def height(self) -> int: return max(0, self.bottom - self.top)


@dataclass(frozen=True)
class GameWindow:
    hwnd: int
    title: str
    client_rect: Rect


def set_dpi_aware() -> None:
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except (AttributeError, OSError):
        pass


def find_game_window(patterns: list[str]) -> GameWindow | None:
    user32 = ctypes.windll.user32
    found: list[GameWindow] = []
    EnumProc = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

    def callback(hwnd: int, _: int) -> bool:
        if not user32.IsWindowVisible(hwnd):
            return True
        length = user32.GetWindowTextLengthW(hwnd)
        if not length:
            return True
        title = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, title, len(title))
        value = title.value
        if not any(pattern.lower() in value.lower() for pattern in patterns):
            return True
        rect = wintypes.RECT()
        if not user32.GetClientRect(hwnd, ctypes.byref(rect)):
            return True
        points = (wintypes.POINT * 2)(wintypes.POINT(rect.left, rect.top), wintypes.POINT(rect.right, rect.bottom))
        user32.ClientToScreen(hwnd, ctypes.byref(points[0]))
        user32.ClientToScreen(hwnd, ctypes.byref(points[1]))
        client = Rect(points[0].x, points[0].y, points[1].x, points[1].y)
        if client.width >= 640 and client.height >= 360:
            found.append(GameWindow(hwnd, value, client))
        return True

    user32.EnumWindows(EnumProc(callback), 0)
    return max(found, key=lambda item: item.client_rect.width * item.client_rect.height, default=None)


def enable_click_through(hwnd: int, enabled: bool) -> None:
    GWL_EXSTYLE, WS_EX_TRANSPARENT, WS_EX_LAYERED = -20, 0x20, 0x80000
    user32 = ctypes.windll.user32
    style = user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
    if enabled:
        style |= WS_EX_TRANSPARENT | WS_EX_LAYERED
    else:
        style &= ~WS_EX_TRANSPARENT
    user32.SetWindowLongW(hwnd, GWL_EXSTYLE, style)


def exclude_from_capture(hwnd: int) -> bool:
    # Windows 10 2004+: WDA_EXCLUDEFROMCAPTURE. 실패해도 분석 전 잠시 숨기는 방식으로 보완한다.
    WDA_EXCLUDEFROMCAPTURE = 0x11
    try:
        return bool(ctypes.windll.user32.SetWindowDisplayAffinity(hwnd, WDA_EXCLUDEFROMCAPTURE))
    except (AttributeError, OSError):
        return False
