from __future__ import annotations

import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent
VENV_PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
VENV_PYTHONW = ROOT / ".venv" / "Scripts" / "pythonw.exe"


def _restart_in_project_venv() -> None:
    """Windows에서 app.py를 직접 실행해도 설치된 프로젝트 환경을 사용한다."""
    if __name__ != "__main__" or os.name != "nt" or not VENV_PYTHON.exists():
        return
    target = VENV_PYTHONW if VENV_PYTHONW.exists() else VENV_PYTHON
    if Path(sys.executable).resolve() in {VENV_PYTHON.resolve(), target.resolve()}:
        return
    subprocess.Popen(
        [str(target), "-B", str(Path(__file__).resolve()), *sys.argv[1:]],
        cwd=str(ROOT),
        close_fds=True,
    )
    raise SystemExit(0)


_restart_in_project_venv()

import tkinter as tk
from tkinter import messagebox, ttk

import numpy as np

from gta_helper.analyzer import PuzzleAnalyzer
from gta_helper.capture import CaptureError, DxCapture
from gta_helper.config import AppConfig
from gta_helper.models import AppState, DisplayMode, PuzzleType, SolveResult
from gta_helper.speech import SpeechService
from gta_helper.windowing import enable_click_through, exclude_from_capture, find_game_window, set_dpi_aware


INPUT_PROFILES = {
    "기본 키보드": {"up": "↑ / W", "down": "↓ / S", "left": "← / A", "right": "→ / D", "select": "Enter / 마우스 1", "back": "Backspace / Esc"},
    "사용자 지정 키보드": None,
    "Xbox": {"up": "D-pad ↑", "down": "D-pad ↓", "left": "D-pad ←", "right": "D-pad →", "select": "A", "back": "B"},
    "PlayStation": {"up": "D-pad ↑", "down": "D-pad ↓", "left": "D-pad ←", "right": "D-pad →", "select": "✕", "back": "○"},
}


class Scanner(threading.Thread):
    def __init__(self, config: AppConfig, events: queue.Queue[tuple[str, object]]) -> None:
        super().__init__(daemon=True, name="gta-helper-scanner")
        self.config, self.events = config, events
        self.stop_event = threading.Event()
        self.diagnostic_event = threading.Event()
        self.reset_event = threading.Event()
        self.latest_frame: np.ndarray | None = None
        self._capture: DxCapture | None = None

    def stop(self) -> None:
        self.stop_event.set()

    def save_diagnostic(self) -> None:
        self.diagnostic_event.set()

    def reset_analysis(self) -> None:
        self.reset_event.set()

    def run(self) -> None:
        analyzer = PuzzleAnalyzer()
        last_signature: tuple | None = None
        try:
            self._capture = DxCapture(self.config)
            self._capture.open()
            self.events.put(("status", f"캡처 준비 완료 ({self._capture.backend})"))
            last_observed_pattern = None
            while not self.stop_event.is_set():
                if self.reset_event.is_set():
                    analyzer.reset()
                    last_signature = None
                    last_observed_pattern = None
                    self.reset_event.clear()
                    self.events.put(("reset", None))
                game = find_game_window(self.config.game_title_patterns)
                if game is None:
                    self.events.put(("state", (AppState.WAITING, "GTA V 창을 기다리는 중")))
                    time.sleep(0.8)
                    continue
                try:
                    frame = self._capture.grab(game)
                except CaptureError as exc:
                    self.events.put(("state", (AppState.ERROR, str(exc))))
                    time.sleep(1.0)
                    continue
                self.latest_frame = frame
                if self.diagnostic_event.is_set():
                    path = self._capture.save_diagnostic(frame, self.config.diagnostic_dir, "gta")
                    self.events.put(("status", f"진단 이미지 저장: {path.name}"))
                    self.diagnostic_event.clear()
                self.events.put(("state", (AppState.ANALYZING, "해킹 화면 자동 감시 중")))
                result = analyzer.update(frame)
                observed_pattern = analyzer.dot.current_pattern
                if observed_pattern and observed_pattern != last_observed_pattern:
                    coordinates = "_".join(f"r{point.row}c{point.column}" for point in observed_pattern)
                    try:
                        self._capture.save_diagnostic(frame, self.config.diagnostic_dir, f"observed_{coordinates}")
                    except CaptureError as exc:
                        self.events.put(("status", str(exc)))
                    last_observed_pattern = observed_pattern
                if result is not None and result.signature != last_signature:
                    last_signature = result.signature
                    if result.puzzle == PuzzleType.DOT_MEMORY:
                        coordinates = "_".join(f"r{point.row}c{point.column}" for point in result.locations)
                        try:
                            path = self._capture.save_diagnostic(frame, self.config.diagnostic_dir, f"detected_{coordinates}")
                            result.debug["diagnostic_path"] = str(path)
                            self.events.put(("status", f"키패드 판정 화면 저장: {path.name}"))
                        except CaptureError as exc:
                            self.events.put(("status", str(exc)))
                    self.events.put(("result", result))
                time.sleep(max(0.02, 1 / max(1, self.config.target_fps)))
        except Exception as exc:
            self.events.put(("state", (AppState.ERROR, f"스캐너 오류: {exc}")))
        finally:
            if self._capture:
                self._capture.close()


class HelperApp:
    def __init__(self) -> None:
        set_dpi_aware()
        self.config_path = ROOT / "config.json"
        self.config = AppConfig.load(self.config_path)
        self.events: queue.Queue[tuple[str, object]] = queue.Queue()
        self.root = tk.Tk()
        self.root.title("GTA 해킹 안내 도우미")
        self.root.geometry(f"{self.config.overlay_width}x{self.config.overlay_height}+{self.config.overlay_x}+{self.config.overlay_y}")
        self.root.attributes("-topmost", True)
        self.root.attributes("-alpha", self.config.overlay_opacity)
        self.root.configure(bg="#111827")
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.state_var = tk.StringVar(value=AppState.WAITING.value)
        self.detail_var = tk.StringVar(value="시작 중")
        self.answer_var = tk.StringVar(value="GTA V 해킹 화면을 기다립니다.")
        self.confidence_var = tk.StringVar(value="")
        self.controls_var = tk.StringVar(value="")
        self.mode_var = tk.StringVar(value=self.config.display_mode)
        self._fit_scheduled = False
        self.voice = SpeechService(self.config.voice_enabled, self.config.voice_rate)
        self.scanner = Scanner(self.config, self.events)
        self._build_ui()
        self._apply_mode(initial=True)
        self.scanner.start()
        self.root.after(100, self._drain_events)

    def _build_ui(self) -> None:
        panel = tk.Frame(self.root, bg="#111827", padx=14, pady=12)
        self.panel = panel
        panel.pack(fill="both", expand=True)
        tk.Label(panel, text="GTA 해킹 안내 도우미", bg="#111827", fg="#f9fafb", font=("맑은 고딕", 14, "bold")).pack(anchor="w")
        tk.Label(panel, textvariable=self.state_var, bg="#111827", fg="#60a5fa", font=("맑은 고딕", 10, "bold")).pack(anchor="w", pady=(5, 0))
        tk.Label(panel, textvariable=self.detail_var, bg="#111827", fg="#d1d5db", wraplength=350, justify="left").pack(anchor="w")
        tk.Label(panel, textvariable=self.answer_var, bg="#111827", fg="#fef3c7", font=("맑은 고딕", 11, "bold"), wraplength=355, justify="left").pack(anchor="w", pady=(12, 2))
        tk.Label(panel, textvariable=self.confidence_var, bg="#111827", fg="#9ca3af").pack(anchor="w")
        tk.Label(panel, textvariable=self.controls_var, bg="#111827", fg="#9ca3af", wraplength=355, justify="left").pack(anchor="w")
        buttons = tk.Frame(panel, bg="#111827")
        buttons.pack(side="bottom", fill="x", pady=(10, 0))
        ttk.Button(buttons, text="설정", command=self.show_settings).pack(side="left")
        ttk.Button(buttons, text="진단 저장", command=self.scanner.save_diagnostic).pack(side="left", padx=5)
        ttk.Button(buttons, text="재인식", command=self.scanner.reset_analysis).pack(side="left")
        self.lock_button = ttk.Button(buttons, text="오버레이 잠금", command=self.toggle_lock)
        self.lock_button.pack(side="left")
        ttk.Button(buttons, text="종료", command=self.close).pack(side="right")
        self._refresh_controls()

    def _refresh_controls(self) -> None:
        if not self.config.controls_legend_enabled:
            self.controls_var.set("")
            self._schedule_fit()
            return
        mapping = INPUT_PROFILES.get(self.config.input_profile)
        if mapping is None:
            mapping = self.config.custom_keys
        self.controls_var.set(f"조작 범례 ({self.config.input_profile}): 이동 {mapping['up']} {mapping['down']} {mapping['left']} {mapping['right']} / 선택 {mapping['select']}")
        self._schedule_fit()

    def _schedule_fit(self) -> None:
        if self._fit_scheduled:
            return
        self._fit_scheduled = True
        self.root.after_idle(self._fit_window_to_content)

    def _fit_window_to_content(self) -> None:
        self._fit_scheduled = False
        if self.config.display_mode == DisplayMode.VOICE_ONLY.value:
            return
        self.root.update_idletasks()
        required_width = self.panel.winfo_reqwidth()
        required_height = self.panel.winfo_reqheight()
        current_width = self.root.winfo_width()
        current_height = self.root.winfo_height()
        new_width = max(current_width, required_width)
        new_height = max(current_height, required_height)
        if (new_width, new_height) != (current_width, current_height):
            self.root.geometry(f"{new_width}x{new_height}+{self.root.winfo_x()}+{self.root.winfo_y()}")

    def _apply_mode(self, initial: bool = False) -> None:
        mode = self.mode_var.get()
        self.config.display_mode = mode
        if mode == DisplayMode.VOICE_ONLY.value:
            self.root.iconify()
            return
        self.root.deiconify()
        click_through = mode == DisplayMode.CLICK_THROUGH.value and not initial
        enable_click_through(self.root.winfo_id(), click_through)
        self.lock_button.configure(text="오버레이 잠금 해제" if click_through else "오버레이 잠금")
        exclude_from_capture(self.root.winfo_id())

    def toggle_lock(self) -> None:
        currently_locked = self.config.display_mode == DisplayMode.CLICK_THROUGH.value and self.lock_button.cget("text") == "오버레이 잠금 해제"
        enable_click_through(self.root.winfo_id(), not currently_locked)
        self.lock_button.configure(text="오버레이 잠금" if currently_locked else "오버레이 잠금 해제")

    def show_settings(self) -> None:
        dialog = tk.Toplevel(self.root)
        dialog.title("설정")
        dialog.attributes("-topmost", True)
        dialog.resizable(False, False)
        body = ttk.Frame(dialog, padding=12)
        body.grid()
        ttk.Label(body, text="안내 방식").grid(row=0, column=0, sticky="w", pady=4)
        mode = ttk.Combobox(body, textvariable=self.mode_var, state="readonly", width=22, values=[item.value for item in DisplayMode])
        mode.grid(row=0, column=1, padx=8)
        voice_var = tk.BooleanVar(value=self.config.voice_enabled)
        ttk.Checkbutton(body, text="음성 안내 사용", variable=voice_var).grid(row=1, column=0, columnspan=2, sticky="w", pady=4)
        controls_legend_var = tk.BooleanVar(value=self.config.controls_legend_enabled)
        ttk.Checkbutton(body, text="조작 범례 표시", variable=controls_legend_var).grid(row=2, column=0, columnspan=2, sticky="w", pady=4)
        ttk.Label(body, text="캡처 백엔드").grid(row=3, column=0, sticky="w", pady=4)
        backend = ttk.Combobox(body, state="readonly", width=22, values=["auto", "dxgi", "winrt"])
        backend.set(self.config.capture_backend)
        backend.grid(row=3, column=1, padx=8)
        ttk.Label(body, text="입력 프로필").grid(row=4, column=0, sticky="w", pady=4)
        profile = ttk.Combobox(body, state="readonly", width=22, values=list(INPUT_PROFILES))
        profile.set(self.config.input_profile)
        profile.grid(row=4, column=1, padx=8)
        ttk.Label(body, text="사용자 키 (위/아래/왼쪽/오른쪽/선택/뒤로)").grid(row=5, column=0, columnspan=2, sticky="w", pady=(8, 2))
        key_vars: dict[str, tk.StringVar] = {}
        for row, key in enumerate(("up", "down", "left", "right", "select", "back"), start=6):
            ttk.Label(body, text=key).grid(row=row, column=0, sticky="w", pady=1)
            value = tk.StringVar(value=self.config.custom_keys[key])
            key_vars[key] = value
            ttk.Entry(body, textvariable=value, width=25).grid(row=row, column=1, padx=8, pady=1)

        def save() -> None:
            self.config.voice_enabled = voice_var.get()
            self.config.controls_legend_enabled = controls_legend_var.get()
            self.config.capture_backend = backend.get()
            self.config.input_profile = profile.get()
            self.config.custom_keys = {key: value.get().strip() or self.config.custom_keys[key] for key, value in key_vars.items()}
            self._apply_mode()
            self._refresh_controls()
            self.config.save(self.config_path)
            dialog.destroy()
            messagebox.showinfo("설정 저장", "설정을 저장했습니다. 캡처/음성 변경은 다음 실행부터 적용됩니다.")

        ttk.Button(body, text="저장", command=save).grid(row=12, column=1, sticky="e", pady=(10, 0))

    def _drain_events(self) -> None:
        content_changed = False
        while True:
            try:
                kind, payload = self.events.get_nowait()
            except queue.Empty:
                break
            if kind == "state":
                state, message = payload  # type: ignore[misc]
                content_changed = content_changed or self.state_var.get() != state.value or self.detail_var.get() != message
                self.state_var.set(state.value)
                self.detail_var.set(message)
            elif kind == "status":
                content_changed = content_changed or self.detail_var.get() != str(payload)
                self.detail_var.set(str(payload))
            elif kind == "reset":
                self.state_var.set(AppState.ANALYZING.value)
                self.detail_var.set("인식 기록을 초기화했습니다. 점멸 패턴을 기다리는 중")
                self.answer_var.set("GTA V 해킹 화면을 기다립니다.")
                self.confidence_var.set("")
                content_changed = True
            elif kind == "result":
                result: SolveResult = payload  # type: ignore[assignment]
                if result.confidence >= self.config.confidence_threshold:
                    self.state_var.set(AppState.SOLVED.value)
                    self.detail_var.set(result.puzzle.value)
                    self.answer_var.set(result.display_text())
                    self.confidence_var.set(f"인식 신뢰도: {result.confidence:.0%}")
                    self.voice.say(result.summary + ". " + ". ".join(result.details))
                    content_changed = True
                else:
                    self.detail_var.set("신뢰도가 낮아 답을 표시하지 않았습니다. 진단 저장을 사용하세요.")
                    content_changed = True
        if content_changed:
            self._schedule_fit()
        self.root.after(100, self._drain_events)

    def close(self) -> None:
        self.config.overlay_x, self.config.overlay_y = self.root.winfo_x(), self.root.winfo_y()
        self.config.overlay_width, self.config.overlay_height = self.root.winfo_width(), self.root.winfo_height()
        self.config.save(self.config_path)
        self.scanner.stop()
        self.voice.close()
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


if __name__ == "__main__":
    HelperApp().run()
