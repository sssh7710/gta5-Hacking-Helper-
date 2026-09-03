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
from gta_helper.capture import CaptureError, DiagnosticFrameRecorder, DxCapture
from gta_helper.config import AppConfig
from gta_helper.models import AppState, DisplayMode, PuzzleType, SolveResult
from gta_helper.reporting import DiagnosticReporter
from gta_helper.speech import SpeechService
from gta_helper.updater import UpdateError, UpdateInfo, check_for_update, download_update
from gta_helper.version import APP_VERSION
from gta_helper.windowing import enable_click_through, exclude_from_capture, find_game_window, set_dpi_aware


INPUT_PROFILES = {
    "기본 키보드": {"up": "↑ / W", "down": "↓ / S", "left": "← / A", "right": "→ / D", "select": "Enter / 마우스 1", "back": "Backspace / Esc"},
    "사용자 지정 키보드": None,
    "Xbox": {"up": "D-pad ↑", "down": "D-pad ↓", "left": "D-pad ←", "right": "D-pad →", "select": "A", "back": "B"},
    "PlayStation": {"up": "D-pad ↑", "down": "D-pad ↓", "left": "D-pad ←", "right": "D-pad →", "select": "✕", "back": "○"},
}
UPDATE_CHANNEL_LABELS = {"릴리즈": "release", "베타": "beta"}
UPDATE_CHANNEL_NAMES = {value: label for label, value in UPDATE_CHANNEL_LABELS.items()}


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
        recorder = DiagnosticFrameRecorder(
            self.config.diagnostic_dir,
            duration_seconds=self.config.diagnostic_capture_seconds,
            fps=self.config.diagnostic_capture_fps,
            max_total_bytes=self.config.diagnostic_capture_max_mb * 1024 * 1024,
            answer_confidence_threshold=self.config.confidence_threshold,
        )
        try:
            removed = recorder.prune_old_sessions()
        except CaptureError as exc:
            self.events.put(("status", str(exc)))
        else:
            if removed:
                self.events.put(("status", f"오래된 인식 개선 자료 {len(removed)}개를 정리했습니다."))
        last_signature: tuple | None = None
        try:
            self._capture = DxCapture(self.config)
            self._capture.open()
            self.events.put(("status", f"캡처 준비 완료 ({self._capture.backend})"))
            last_observed_pattern = None
            keypad_screen_seen = False
            keypad_result_shown = False
            keypad_missing_checks = 0
            casino_attempt_seen = False
            casino_rearm = False
            casino_missing_checks = 0
            while not self.stop_event.is_set():
                scan_started_at = time.monotonic()
                if self.reset_event.is_set():
                    analyzer.reset()
                    last_signature = None
                    last_observed_pattern = None
                    keypad_screen_seen = False
                    keypad_result_shown = False
                    keypad_missing_checks = 0
                    casino_attempt_seen = False
                    casino_rearm = False
                    casino_missing_checks = 0
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
                try:
                    completed_session = recorder.add(frame)
                    if completed_session is not None:
                        self.events.put(("status", f"인식 개선 사진 저장: {completed_session.name}"))
                        self.events.put(("diagnostic_completed", completed_session))
                except CaptureError as exc:
                    self.events.put(("status", str(exc)))
                if self.diagnostic_event.is_set():
                    self.events.put(("diagnostic_frame", frame.copy()))
                    self.events.put(("status", "수동 진단 서버 전송 대기"))
                    self.diagnostic_event.clear()
                self.events.put(("state", (AppState.ANALYZING, "해킹 화면 자동 감시 중")))
                result = analyzer.update(frame)
                if analyzer.dot.grid_visible:
                    keypad_missing_checks = 0
                    if not keypad_screen_seen:
                        keypad_screen_seen = True
                        keypad_result_shown = False
                        self.events.put(("keypad_start", analyzer.dot.current_grid_shape))
                else:
                    keypad_missing_checks += 1
                    if keypad_missing_checks >= 15:
                        keypad_screen_seen = False
                        keypad_result_shown = False
                        last_observed_pattern = None
                if analyzer.dot.input_visible and keypad_result_shown:
                    # 같은 배열이 연속 라운드에 나와도 다음 점멸이 새 분석으로
                    # 처리되도록 입력 단계에서 이전 관찰 배열을 해제한다.
                    last_observed_pattern = None
                if analyzer.casino_layout_checked:
                    if analyzer.casino_screen_visible:
                        casino_missing_checks = 0
                        if analyzer.casino_selection_visible:
                            casino_rearm = True
                        elif (
                            self.config.diagnostic_capture_enabled
                            and not recorder.active
                            and (not casino_attempt_seen or casino_rearm)
                        ):
                            try:
                                session_dir = recorder.start(
                                    frame,
                                    "fragment_fingerprint_pending",
                                    {
                                        "puzzle": PuzzleType.FRAGMENT_FINGERPRINT.name,
                                        "capture_backend": self._capture.backend,
                                        "detection_status": "pending",
                                    },
                                )
                                casino_attempt_seen = True
                                casino_rearm = False
                                self.events.put(("fingerprint_start", None))
                                self.events.put(("status", f"지문 인식 진단 수집 시작: {session_dir.name}"))
                            except CaptureError as exc:
                                self.events.put(("status", str(exc)))
                    else:
                        casino_missing_checks += 1
                        if casino_missing_checks >= 2:
                            casino_attempt_seen = False
                            casino_rearm = False
                observed_pattern = analyzer.dot.current_pattern
                if observed_pattern and observed_pattern != last_observed_pattern:
                    if keypad_result_shown:
                        keypad_result_shown = False
                        self.events.put(("keypad_start", analyzer.dot.current_grid_shape))
                    if self.config.diagnostic_capture_enabled and not recorder.active:
                        rows, columns = analyzer.dot.current_grid_shape
                        try:
                            session_dir = recorder.start(
                                frame,
                                f"keypad_{columns}x{rows}",
                                {
                                    "puzzle": PuzzleType.DOT_MEMORY.name,
                                    "grid_rows": rows,
                                    "grid_columns": columns,
                                    "trigger_pattern": [
                                        {"row": point.row, "column": point.column}
                                        for point in observed_pattern
                                    ],
                                    "capture_backend": self._capture.backend,
                                },
                            )
                            self.events.put(("status", f"인식 개선 사진 수집 시작: {session_dir.name}"))
                        except CaptureError as exc:
                            self.events.put(("status", str(exc)))
                    last_observed_pattern = observed_pattern
                # 점멸 해킹 결과는 솔버가 판마다 한 번만 반환한다. 다음 판의
                # 배열이 우연히 같아도 다시 안내해야 하므로 전역 서명 중복
                # 차단을 적용하지 않는다.
                if result is not None and (result.puzzle == PuzzleType.DOT_MEMORY or result.signature != last_signature):
                    last_signature = result.signature
                    if result.puzzle == PuzzleType.DOT_MEMORY:
                        keypad_result_shown = True
                    if self.config.diagnostic_capture_enabled and not recorder.active:
                        try:
                            session_dir = recorder.start(
                                frame,
                                result.puzzle.name.lower(),
                                {"puzzle": result.puzzle.name, "capture_backend": self._capture.backend},
                            )
                            self.events.put(("status", f"인식 개선 사진 수집 시작: {session_dir.name}"))
                        except CaptureError as exc:
                            self.events.put(("status", str(exc)))
                    recorder.annotate(
                        expected_puzzle=result.puzzle.name,
                        result_summary=result.summary,
                        result_confidence=result.confidence,
                        result_locations=[
                            {"row": point.row, "column": point.column}
                            for point in result.locations
                        ],
                        result_debug=result.debug,
                    )
                    self.events.put(("result", result))
                frame_interval = 1 / max(1, self.config.target_fps)
                remaining = frame_interval - (time.monotonic() - scan_started_at)
                if remaining > 0:
                    self.stop_event.wait(remaining)
        except Exception as exc:
            self.events.put(("state", (AppState.ERROR, f"스캐너 오류: {exc}")))
        finally:
            completed_session = recorder.close()
            if completed_session is not None:
                self.events.put(("diagnostic_completed", completed_session))
            if self._capture:
                self._capture.close()


class HelperApp:
    def __init__(self) -> None:
        set_dpi_aware()
        self.config_path = ROOT / "config.json"
        self.config = AppConfig.load(self.config_path)
        self.events: queue.Queue[tuple[str, object]] = queue.Queue()
        self.root = tk.Tk()
        self.root.title(f"GTA 해킹 안내 도우미 {APP_VERSION}")
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
        self._pending_update_archive: Path | None = None
        self._update_check_in_progress = False
        self.voice = SpeechService(self.config.voice_enabled, self.config.voice_rate)
        self.reporter = DiagnosticReporter(
            self.config.diagnostic_upload_url,
            self.config.confidence_threshold,
            notify=lambda message: self.events.put(("status", message)),
        )
        self.scanner = Scanner(self.config, self.events)
        self._build_ui()
        self._apply_mode(initial=True)
        self.reporter.start()
        self.scanner.start()
        if self.config.auto_update_enabled:
            self._start_update_check()
        self.root.after(100, self._drain_events)

    def _build_ui(self) -> None:
        panel = tk.Frame(self.root, bg="#111827", padx=14, pady=12)
        self.panel = panel
        panel.pack(fill="both", expand=True)
        self.title_label = tk.Label(panel, text="GTA 해킹 안내 도우미", bg="#111827", fg="#f9fafb")
        self.title_label.pack(anchor="w")
        self.state_label = tk.Label(panel, textvariable=self.state_var, bg="#111827", fg="#60a5fa")
        self.state_label.pack(anchor="w", pady=(5, 0))
        self.detail_label = tk.Label(panel, textvariable=self.detail_var, bg="#111827", fg="#d1d5db", wraplength=350, justify="left")
        self.detail_label.pack(anchor="w")
        self.answer_label = tk.Label(panel, textvariable=self.answer_var, bg="#111827", fg="#fef3c7", wraplength=355, justify="left")
        self.answer_label.pack(anchor="w", pady=(12, 2))
        self.confidence_label = tk.Label(panel, textvariable=self.confidence_var, bg="#111827", fg="#9ca3af")
        self.confidence_label.pack(anchor="w")
        self.controls_label = tk.Label(panel, textvariable=self.controls_var, bg="#111827", fg="#9ca3af", wraplength=355, justify="left")
        self.controls_label.pack(anchor="w")
        buttons = tk.Frame(panel, bg="#111827")
        buttons.pack(side="bottom", fill="x", pady=(10, 0))
        ttk.Button(buttons, text="설정", command=self.show_settings).pack(side="left")
        ttk.Button(buttons, text="진단 전송", command=self.scanner.save_diagnostic).pack(side="left", padx=5)
        ttk.Button(buttons, text="재인식", command=self.scanner.reset_analysis).pack(side="left")
        self.lock_button = ttk.Button(buttons, text="오버레이 잠금", command=self.toggle_lock)
        self.lock_button.pack(side="left")
        ttk.Button(buttons, text="종료", command=self.close).pack(side="right")
        self._apply_font_size()
        self._refresh_controls()

    def _apply_font_size(self) -> None:
        size = max(8, min(24, int(self.config.guide_font_size)))
        self.config.guide_font_size = size
        secondary_size = max(8, size - 2)
        self.title_label.configure(font=("맑은 고딕", size + 3, "bold"))
        self.state_label.configure(font=("맑은 고딕", max(8, size - 1), "bold"))
        self.detail_label.configure(font=("맑은 고딕", secondary_size))
        self.answer_label.configure(font=("맑은 고딕", size, "bold"))
        self.confidence_label.configure(font=("맑은 고딕", secondary_size))
        self.controls_label.configure(font=("맑은 고딕", secondary_size))
        self._schedule_fit()

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
        diagnostic_capture_var = tk.BooleanVar(value=self.config.diagnostic_capture_enabled)
        ttk.Checkbutton(body, text="인식 개선 사진 자동 저장 (약 7초)", variable=diagnostic_capture_var).grid(row=3, column=0, columnspan=2, sticky="w", pady=4)
        auto_update_var = tk.BooleanVar(value=self.config.auto_update_enabled)
        ttk.Checkbutton(body, text="새 버전 자동 확인", variable=auto_update_var).grid(row=4, column=0, sticky="w", pady=4)
        update_channel_var = tk.StringVar(value=UPDATE_CHANNEL_NAMES.get(self.config.update_channel, "베타"))
        check_update_button = ttk.Button(
            body,
            text="지금 업데이트 확인",
            command=lambda: self._start_update_check(
                manual=True,
                source_button=check_update_button,
                channel=UPDATE_CHANNEL_LABELS.get(update_channel_var.get(), "beta"),
            ),
        )
        check_update_button.grid(row=4, column=1, sticky="e", padx=8, pady=4)
        ttk.Label(body, text="업데이트 채널").grid(row=5, column=0, sticky="w", pady=4)
        update_channel = ttk.Combobox(
            body,
            textvariable=update_channel_var,
            state="readonly",
            width=22,
            values=list(UPDATE_CHANNEL_LABELS),
        )
        update_channel.grid(row=5, column=1, padx=8)
        diagnostic_upload_var = tk.BooleanVar(value=self.config.diagnostic_upload_enabled)
        upload_text = "진단 자료 자동 전송" if self.reporter.configured else "진단 자료 자동 전송 (서버 준비 전)"
        ttk.Checkbutton(body, text=upload_text, variable=diagnostic_upload_var).grid(row=6, column=0, columnspan=2, sticky="w", pady=4)
        ttk.Label(body, text="※ 전송 자료에는 GTA 게임 화면이 포함될 수 있습니다.", foreground="#9a6700").grid(row=7, column=0, columnspan=2, sticky="w", pady=(0, 4))
        ttk.Label(body, text="안내 글자 크기").grid(row=8, column=0, sticky="w", pady=4)
        font_size_var = tk.IntVar(value=self.config.guide_font_size)
        ttk.Spinbox(body, from_=8, to=24, increment=1, textvariable=font_size_var, width=20).grid(row=8, column=1, padx=8)
        ttk.Label(body, text="인식 자료 최대 용량 (MB)").grid(row=9, column=0, sticky="w", pady=4)
        diagnostic_max_var = tk.IntVar(value=self.config.diagnostic_capture_max_mb)
        ttk.Spinbox(body, from_=100, to=10240, increment=100, textvariable=diagnostic_max_var, width=20).grid(row=9, column=1, padx=8)
        ttk.Label(body, text="캡처 백엔드").grid(row=10, column=0, sticky="w", pady=4)
        backend = ttk.Combobox(body, state="readonly", width=22, values=["auto", "dxgi", "winrt"])
        backend.set(self.config.capture_backend)
        backend.grid(row=10, column=1, padx=8)
        ttk.Label(body, text="입력 프로필").grid(row=11, column=0, sticky="w", pady=4)
        profile = ttk.Combobox(body, state="readonly", width=22, values=list(INPUT_PROFILES))
        profile.set(self.config.input_profile)
        profile.grid(row=11, column=1, padx=8)
        ttk.Label(body, text="사용자 키 (위/아래/왼쪽/오른쪽/선택/뒤로)").grid(row=12, column=0, columnspan=2, sticky="w", pady=(8, 2))
        key_vars: dict[str, tk.StringVar] = {}
        for row, key in enumerate(("up", "down", "left", "right", "select", "back"), start=13):
            ttk.Label(body, text=key).grid(row=row, column=0, sticky="w", pady=1)
            value = tk.StringVar(value=self.config.custom_keys[key])
            key_vars[key] = value
            ttk.Entry(body, textvariable=value, width=25).grid(row=row, column=1, padx=8, pady=1)

        def save() -> None:
            self.config.voice_enabled = voice_var.get()
            self.config.controls_legend_enabled = controls_legend_var.get()
            self.config.diagnostic_capture_enabled = diagnostic_capture_var.get()
            self.config.auto_update_enabled = auto_update_var.get()
            self.config.update_channel = UPDATE_CHANNEL_LABELS.get(update_channel_var.get(), "beta")
            self.config.diagnostic_upload_enabled = diagnostic_upload_var.get()
            try:
                self.config.guide_font_size = max(8, min(24, int(font_size_var.get())))
            except (tk.TclError, ValueError):
                self.config.guide_font_size = 11
            try:
                self.config.diagnostic_capture_max_mb = max(100, min(10240, int(diagnostic_max_var.get())))
            except (tk.TclError, ValueError):
                self.config.diagnostic_capture_max_mb = 1024
            self.config.capture_backend = backend.get()
            self.config.input_profile = profile.get()
            self.config.custom_keys = {key: value.get().strip() or self.config.custom_keys[key] for key, value in key_vars.items()}
            self._apply_mode()
            self._apply_font_size()
            self._refresh_controls()
            self.config.save(self.config_path)
            dialog.destroy()
            messagebox.showinfo("설정 저장", "설정을 저장했습니다. 일부 변경은 다음 실행부터 적용됩니다.")

        ttk.Button(body, text="저장", command=save).grid(row=19, column=1, sticky="e", pady=(10, 0))

    def _start_update_check(
        self,
        manual: bool = False,
        source_button: ttk.Button | None = None,
        channel: str | None = None,
    ) -> None:
        if self._update_check_in_progress:
            if manual:
                messagebox.showinfo("업데이트 확인", "이미 새 버전을 확인하고 있습니다.", parent=self._update_dialog_parent(source_button))
            return
        self._update_check_in_progress = True
        if source_button is not None:
            source_button.state(["disabled"])
        if manual:
            self.detail_var.set("새 버전 확인 중")
        threading.Thread(
            target=self._check_for_updates,
            args=(manual, source_button, channel or self.config.update_channel),
            daemon=True,
            name="gta-helper-update-check",
        ).start()

    def _check_for_updates(
        self,
        manual: bool = False,
        source_button: ttk.Button | None = None,
        channel: str | None = None,
    ) -> None:
        try:
            info = check_for_update(APP_VERSION, channel or self.config.update_channel)
        except UpdateError as exc:
            self.events.put(("update_check_error", (str(exc), manual, source_button)))
        else:
            if info is not None:
                self.events.put(("update_available", (info, source_button)))
            elif manual:
                self.events.put(("update_current", source_button))
        finally:
            self.events.put(("update_check_finished", source_button))

    def _download_update(self, info: UpdateInfo) -> None:
        try:
            archive = download_update(info, ROOT / "updates")
        except (OSError, UpdateError) as exc:
            self.events.put(("update_error", str(exc)))
        else:
            self.events.put(("update_downloaded", (info, archive)))

    def _update_dialog_parent(self, source_button: ttk.Button | None) -> tk.Misc:
        if source_button is not None:
            try:
                if source_button.winfo_exists():
                    return source_button.winfo_toplevel()
            except tk.TclError:
                pass
        return self.root

    def _offer_update(self, info: UpdateInfo, source_button: ttk.Button | None = None) -> None:
        if not messagebox.askyesno(
            "새 버전 발견",
            f"{info.tag} 버전이 있습니다.\n지금 내려받아 업데이트할까요?\n\n설정과 진단 자료는 유지됩니다.",
            parent=self._update_dialog_parent(source_button),
        ):
            return
        self.detail_var.set(f"{info.tag} 업데이트 다운로드 중")
        threading.Thread(target=self._download_update, args=(info,), daemon=True, name="gta-helper-update-download").start()

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
            elif kind == "fingerprint_start":
                self.state_var.set(AppState.ANALYZING.value)
                self.detail_var.set("새 지문 화면 감지")
                self.answer_var.set("지문 정답을 분석 중입니다.")
                self.confidence_var.set("")
                content_changed = True
            elif kind == "keypad_start":
                rows, columns = payload  # type: ignore[misc]
                self.state_var.set(AppState.ANALYZING.value)
                self.detail_var.set(f"키패드 화면 감지 ({columns}열×{rows}행)")
                self.answer_var.set("키패드 해킹 인식 중입니다.")
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
                    self.detail_var.set("신뢰도가 낮아 답을 표시하지 않았습니다. 진단 전송을 사용하세요.")
                    content_changed = True
            elif kind == "diagnostic_completed":
                if self.config.diagnostic_upload_enabled:
                    self.reporter.submit(payload)  # type: ignore[arg-type]
            elif kind == "diagnostic_frame":
                self.reporter.submit_frame(payload)  # type: ignore[arg-type]
            elif kind == "update_available":
                info, source_button = payload  # type: ignore[misc]
                self._offer_update(info, source_button)
            elif kind == "update_current":
                messagebox.showinfo("업데이트 확인", f"현재 {APP_VERSION} 버전이 최신입니다.", parent=self._update_dialog_parent(payload))  # type: ignore[arg-type]
                self.detail_var.set(f"현재 {APP_VERSION} 버전이 최신입니다.")
                content_changed = True
            elif kind == "update_check_error":
                message, manual, source_button = payload  # type: ignore[misc]
                self.detail_var.set(f"업데이트 확인 건너뜀: {message}")
                if manual:
                    messagebox.showerror("업데이트 확인 실패", str(message), parent=self._update_dialog_parent(source_button))
                content_changed = True
            elif kind == "update_check_finished":
                self._update_check_in_progress = False
                source_button = payload
                if source_button is not None:
                    try:
                        if source_button.winfo_exists():
                            source_button.state(["!disabled"])
                    except tk.TclError:
                        pass
            elif kind == "update_downloaded":
                info, archive = payload  # type: ignore[misc]
                self._pending_update_archive = archive
                messagebox.showinfo("업데이트 준비 완료", f"{info.tag} 업데이트를 적용하기 위해 프로그램을 다시 시작합니다.", parent=self.root)
                self.close()
                return
            elif kind == "update_error":
                # 네트워크가 없는 환경에서도 프로그램 사용을 방해하지 않는다.
                self.detail_var.set(f"업데이트 확인 건너뜀: {payload}")
                content_changed = True
        if content_changed:
            self._schedule_fit()
        self.root.after(100, self._drain_events)

    def close(self) -> None:
        self.config.overlay_x, self.config.overlay_y = self.root.winfo_x(), self.root.winfo_y()
        self.config.overlay_width, self.config.overlay_height = self.root.winfo_width(), self.root.winfo_height()
        self.config.save(self.config_path)
        self.scanner.stop()
        self.reporter.stop()
        self.voice.close()
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()
        self.scanner.join(timeout=3.0)
        self.reporter.join(timeout=3.0)
        if self._pending_update_archive is not None:
            subprocess.Popen(
                [str(VENV_PYTHON), "-B", str(ROOT / "update_installer.py"), str(self._pending_update_archive), str(ROOT)],
                cwd=str(ROOT),
                close_fds=True,
            )


if __name__ == "__main__":
    HelperApp().run()
