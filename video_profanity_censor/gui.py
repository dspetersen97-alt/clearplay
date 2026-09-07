"""Tkinter GUI for the Video Profanity Censor.

Provides a desktop interface to:
- select a source video file,
- optionally select an external subtitle file (when none is chosen, the engine
  falls back to probing the video for embedded subtitle tracks),
- choose the censor mode (mute or tone) and output path,
- verify that all required dependencies are installed, and
- run the censoring pipeline with live progress feedback.

The GUI collects inputs and calls CensorEngine.process() on a background
thread, marshalling progress updates back to the Tkinter main loop through a
thread-safe queue.
"""

from __future__ import annotations

import importlib.util
import queue
import shutil
import threading
from pathlib import Path

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from video_profanity_censor.censor_engine import CensorEngine
from video_profanity_censor.cli import FILTERED_DIR_NAME, discover_videos
from video_profanity_censor.models import (
    CensorMode,
    ProcessingResult,
    ProcessingStage,
)

# File dialog filters
VIDEO_FILETYPES = [
    ("Video files", "*.mp4 *.mkv *.avi *.mov *.wmv"),
    ("All files", "*.*"),
]
SUBTITLE_FILETYPES = [
    ("Subtitle files", "*.srt *.ass *.ssa"),
    ("All files", "*.*"),
]

# Python packages the pipeline imports at runtime, mapped to install names.
REQUIRED_PACKAGES: dict[str, str] = {
    "faster_whisper": "faster-whisper",
    "ffmpeg": "ffmpeg-python",
    "pydub": "pydub",
    "nltk": "nltk",
}


def check_dependencies() -> tuple[bool, list[str]]:
    """Check that FFmpeg and all required Python packages are available.

    Returns:
        A tuple of (all_ok, lines) where lines is a human-readable status
        report describing each dependency.
    """
    lines: list[str] = []
    all_ok = True

    # FFmpeg binary on PATH (required for probing, extraction, and muxing).
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path:
        lines.append(f"[OK]      FFmpeg found: {ffmpeg_path}")
    else:
        all_ok = False
        lines.append("[MISSING] FFmpeg not found on PATH. Install it and restart.")

    # Required Python packages.
    for module_name, install_name in REQUIRED_PACKAGES.items():
        if importlib.util.find_spec(module_name) is not None:
            lines.append(f"[OK]      Python package '{install_name}' installed")
        else:
            all_ok = False
            lines.append(
                f"[MISSING] Python package '{install_name}' "
                f"(pip install {install_name})"
            )

    return all_ok, lines


class CensorGUI:
    """The main application window."""

    # Sentinel keys placed on the queue by the worker thread.
    _PROGRESS = "progress"
    _DONE = "done"
    _ERROR = "error"
    _LOG = "log"          # (kind, text) — append a line to the log
    _BATCH_DONE = "batch_done"  # (kind, succeeded, failed) — batch summary

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Video Profanity Censor")
        self.root.minsize(640, 560)

        # Thread-safe channel from the worker thread to the UI loop.
        self._queue: queue.Queue = queue.Queue()
        self._worker: threading.Thread | None = None

        # Tk variables bound to widgets.
        self.input_mode_var = tk.StringVar(value="file")  # "file" or "folder"
        self.video_var = tk.StringVar()
        self.folder_var = tk.StringVar()
        self.subtitle_var = tk.StringVar()
        self.output_var = tk.StringVar()
        self.mode_var = tk.StringVar(value="mute")

        self._build_widgets()
        self._update_input_mode()  # set initial enabled/disabled state

    # --- UI construction ---

    def _build_widgets(self) -> None:
        pad = {"padx": 8, "pady": 4}
        frm = ttk.Frame(self.root, padding=12)
        frm.pack(fill="both", expand=True)
        frm.columnconfigure(1, weight=1)

        row = 0

        # Input mode: single file or whole folder (batch)
        ttk.Label(frm, text="Input mode:").grid(row=row, column=0, sticky="w", **pad)
        mode_sel = ttk.Frame(frm)
        mode_sel.grid(row=row, column=1, columnspan=2, sticky="w", **pad)
        ttk.Radiobutton(
            mode_sel, text="Single file", value="file",
            variable=self.input_mode_var, command=self._update_input_mode,
        ).pack(side="left")
        ttk.Radiobutton(
            mode_sel, text="Folder (batch)", value="folder",
            variable=self.input_mode_var, command=self._update_input_mode,
        ).pack(side="left", padx=(12, 0))

        row += 1
        # Source video (single-file mode)
        self.video_label = ttk.Label(frm, text="Source video:")
        self.video_label.grid(row=row, column=0, sticky="w", **pad)
        self.video_entry = ttk.Entry(frm, textvariable=self.video_var)
        self.video_entry.grid(row=row, column=1, sticky="ew", **pad)
        self.video_btn = ttk.Button(frm, text="Browse...", command=self._browse_video)
        self.video_btn.grid(row=row, column=2, **pad)

        row += 1
        # Source folder (batch mode)
        self.folder_label = ttk.Label(frm, text="Source folder:")
        self.folder_label.grid(row=row, column=0, sticky="w", **pad)
        self.folder_entry = ttk.Entry(frm, textvariable=self.folder_var)
        self.folder_entry.grid(row=row, column=1, sticky="ew", **pad)
        self.folder_btn = ttk.Button(frm, text="Browse...", command=self._browse_folder)
        self.folder_btn.grid(row=row, column=2, **pad)

        row += 1
        self.folder_hint = ttk.Label(
            frm,
            text="Every supported video in the folder is processed into a "
            "'filtered' subfolder.",
            foreground="gray",
            wraplength=560,
            justify="left",
        )
        self.folder_hint.grid(row=row, column=0, columnspan=3, sticky="w", padx=8)

        row += 1
        # Subtitle file (optional, single-file mode only)
        self.subtitle_label = ttk.Label(frm, text="Subtitle file\n(optional):")
        self.subtitle_label.grid(row=row, column=0, sticky="w", **pad)
        self.subtitle_entry = ttk.Entry(frm, textvariable=self.subtitle_var)
        self.subtitle_entry.grid(row=row, column=1, sticky="ew", **pad)
        sub_btns = ttk.Frame(frm)
        sub_btns.grid(row=row, column=2, **pad)
        self.subtitle_btn = ttk.Button(
            sub_btns, text="Browse...", command=self._browse_subtitle
        )
        self.subtitle_btn.pack(side="left")
        self.subtitle_clear_btn = ttk.Button(
            sub_btns, text="Clear", command=lambda: self.subtitle_var.set("")
        )
        self.subtitle_clear_btn.pack(side="left", padx=(4, 0))

        row += 1
        self.subtitle_hint = ttk.Label(
            frm,
            text="If no subtitle file is selected, embedded subtitle tracks in the "
            "video are checked automatically.",
            foreground="gray",
            wraplength=560,
            justify="left",
        )
        self.subtitle_hint.grid(row=row, column=0, columnspan=3, sticky="w", padx=8)

        row += 1
        # Output file (optional, single-file mode only)
        self.output_label = ttk.Label(frm, text="Output file\n(optional):")
        self.output_label.grid(row=row, column=0, sticky="w", **pad)
        self.output_entry = ttk.Entry(frm, textvariable=self.output_var)
        self.output_entry.grid(row=row, column=1, sticky="ew", **pad)
        self.output_btn = ttk.Button(frm, text="Browse...", command=self._browse_output)
        self.output_btn.grid(row=row, column=2, **pad)

        row += 1
        self.output_hint = ttk.Label(
            frm,
            text="Leave blank to write <input>_censored.<ext> next to the source.",
            foreground="gray",
            wraplength=560,
            justify="left",
        )
        self.output_hint.grid(row=row, column=0, columnspan=3, sticky="w", padx=8)

        row += 1
        # Censor mode
        ttk.Label(frm, text="Censor mode:").grid(row=row, column=0, sticky="w", **pad)
        mode_frame = ttk.Frame(frm)
        mode_frame.grid(row=row, column=1, columnspan=2, sticky="w", **pad)
        ttk.Radiobutton(
            mode_frame, text="Mute (silence)", value="mute", variable=self.mode_var
        ).pack(side="left")
        ttk.Radiobutton(
            mode_frame, text="Tone (beep)", value="tone", variable=self.mode_var
        ).pack(side="left", padx=(12, 0))

        row += 1
        # Action buttons
        btn_frame = ttk.Frame(frm)
        btn_frame.grid(row=row, column=0, columnspan=3, sticky="w", **pad)
        self.check_btn = ttk.Button(
            btn_frame, text="Check Dependencies", command=self._on_check_dependencies
        )
        self.check_btn.pack(side="left")
        self.run_btn = ttk.Button(btn_frame, text="Run", command=self._on_run)
        self.run_btn.pack(side="left", padx=(8, 0))

        row += 1
        # Progress bar + status label
        self.progress = ttk.Progressbar(frm, mode="determinate", maximum=100.0)
        self.progress.grid(row=row, column=0, columnspan=3, sticky="ew", **pad)

        row += 1
        self.status_var = tk.StringVar(value="Ready.")
        ttk.Label(frm, textvariable=self.status_var).grid(
            row=row, column=0, columnspan=3, sticky="w", padx=8
        )

        row += 1
        # Log output
        frm.rowconfigure(row, weight=1)
        log_frame = ttk.Frame(frm)
        log_frame.grid(row=row, column=0, columnspan=3, sticky="nsew", **pad)
        log_frame.rowconfigure(0, weight=1)
        log_frame.columnconfigure(0, weight=1)
        self.log = tk.Text(log_frame, height=10, wrap="word", state="disabled")
        self.log.grid(row=0, column=0, sticky="nsew")
        log_scroll = ttk.Scrollbar(log_frame, command=self.log.yview)
        log_scroll.grid(row=0, column=1, sticky="ns")
        self.log.configure(yscrollcommand=log_scroll.set)

    # --- File pickers ---

    def _browse_video(self) -> None:
        path = filedialog.askopenfilename(
            title="Select source video", filetypes=VIDEO_FILETYPES
        )
        if path:
            self.video_var.set(path)

    def _browse_folder(self) -> None:
        path = filedialog.askdirectory(title="Select folder of videos")
        if path:
            self.folder_var.set(path)

    def _update_input_mode(self) -> None:
        """Enable the widgets for the selected input mode and disable the others.

        Single-file mode uses the video picker plus the optional subtitle/output
        pickers. Folder mode uses only the folder picker — subtitle and output are
        per-file concepts that batch mode derives automatically (into 'filtered/'),
        so they are disabled to match the CLI's behavior.
        """
        is_folder = self.input_mode_var.get() == "folder"
        file_state = "disabled" if is_folder else "normal"
        folder_state = "normal" if is_folder else "disabled"

        for w in (self.video_entry, self.video_btn):
            w.configure(state=file_state)
        for w in (
            self.subtitle_entry, self.subtitle_btn, self.subtitle_clear_btn,
            self.output_entry, self.output_btn,
        ):
            w.configure(state=file_state)
        for w in (self.folder_entry, self.folder_btn):
            w.configure(state=folder_state)

    def _browse_subtitle(self) -> None:
        path = filedialog.askopenfilename(
            title="Select subtitle file", filetypes=SUBTITLE_FILETYPES
        )
        if path:
            self.subtitle_var.set(path)

    def _browse_output(self) -> None:
        path = filedialog.asksaveasfilename(
            title="Select output video", filetypes=VIDEO_FILETYPES
        )
        if path:
            self.output_var.set(path)

    # --- Logging helpers ---

    def _log(self, text: str) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", text + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    # --- Dependency check ---

    def _on_check_dependencies(self) -> None:
        all_ok, lines = check_dependencies()
        self._log("--- Dependency check ---")
        for line in lines:
            self._log(line)
        if all_ok:
            self._log("All dependencies are installed.")
            self.status_var.set("Dependencies OK.")
        else:
            self._log("Some dependencies are missing. See above.")
            self.status_var.set("Missing dependencies.")

    # --- Run pipeline ---

    def _on_run(self) -> None:
        if self._worker is not None and self._worker.is_alive():
            messagebox.showinfo("Busy", "Processing is already running.")
            return

        is_folder = self.input_mode_var.get() == "folder"
        censor_mode = CensorMode.MUTE if self.mode_var.get() == "mute" else CensorMode.TONE

        # Validate the selected input for the current mode.
        if is_folder:
            folder = self.folder_var.get().strip()
            if not folder:
                messagebox.showwarning("No folder", "Please select a source folder.")
                return
            folder_path = Path(folder)
            if not folder_path.is_dir():
                messagebox.showerror("Not found", f"Folder not found:\n{folder}")
                return
            videos = discover_videos(folder_path)
            if not videos:
                messagebox.showwarning(
                    "No videos",
                    f"No supported video files found in:\n{folder}",
                )
                return
        else:
            video = self.video_var.get().strip()
            if not video:
                messagebox.showwarning("No video", "Please select a source video file.")
                return
            if not Path(video).is_file():
                messagebox.showerror("Not found", f"Video file not found:\n{video}")
                return

        # Verify dependencies before starting a long job.
        all_ok, lines = check_dependencies()
        if not all_ok:
            self._log("--- Dependency check ---")
            for line in lines:
                self._log(line)
            messagebox.showerror(
                "Missing dependencies",
                "Cannot run: required dependencies are missing. "
                "See the log for details.",
            )
            self.status_var.set("Missing dependencies.")
            return

        # Disable controls while running.
        self.run_btn.configure(state="disabled")
        self.check_btn.configure(state="disabled")
        self.progress.configure(value=0.0)
        self.status_var.set("Starting...")

        if is_folder:
            self._log(f"--- Starting batch processing of {len(videos)} file(s) ---")
            self._worker = threading.Thread(
                target=self._run_batch_worker,
                args=(folder_path, videos, censor_mode),
                daemon=True,
            )
        else:
            self._log("--- Starting processing ---")
            subtitle = self.subtitle_var.get().strip()
            output = self.output_var.get().strip()
            args = {
                "input_path": Path(video),
                "output_path": Path(output) if output else None,
                "censor_mode": censor_mode,
                "subtitle_path": Path(subtitle) if subtitle else None,
            }
            self._worker = threading.Thread(
                target=self._run_worker, args=(args,), daemon=True
            )

        self._worker.start()
        self.root.after(100, self._poll_queue)

    def _run_worker(self, args: dict) -> None:
        """Runs on a background thread; must not touch Tk widgets directly."""

        def progress_callback(
            stage: ProcessingStage, percent: float, message: str
        ) -> None:
            self._queue.put((self._PROGRESS, stage, percent, message))

        try:
            engine = CensorEngine()
            result = engine.process(progress_callback=progress_callback, **args)
            self._queue.put((self._DONE, result))
        except Exception as exc:  # noqa: BLE001 - surface any failure to the UI
            self._queue.put((self._ERROR, str(exc)))

    def _run_batch_worker(
        self, folder: Path, videos: list[Path], censor_mode: CensorMode
    ) -> None:
        """Batch worker: process each video into <folder>/filtered/.

        Runs on a background thread. Emits log lines and per-file progress through
        the queue, continues past per-file failures, and finishes with a batch
        summary. Never touches Tk widgets directly.
        """
        filtered_dir = folder / FILTERED_DIR_NAME
        try:
            filtered_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            self._queue.put((self._ERROR, f"Cannot create output folder: {exc}"))
            return

        self._queue.put((self._LOG, f"Output folder: {filtered_dir}"))
        engine = CensorEngine()
        total = len(videos)
        succeeded: list[str] = []
        failed: list[tuple[str, str]] = []

        for i, video in enumerate(videos, start=1):
            self._queue.put((self._LOG, f"[{i}/{total}] {video.name}"))

            def progress_callback(
                stage: ProcessingStage, percent: float, message: str,
                _i=i, _total=total, _name=video.name,
            ) -> None:
                # Overall progress spans all files; scale each file into its slice.
                overall = ((_i - 1) + percent / 100.0) / _total * 100.0
                self._queue.put(
                    (self._PROGRESS, stage, overall, f"{_name}: {message}")
                )

            output_path = filtered_dir / f"{video.stem}_censored{video.suffix}"
            report_path = filtered_dir / f"{video.stem}_censored_report.txt"

            try:
                result = engine.process(
                    input_path=video,
                    output_path=output_path,
                    report_path=report_path,
                    censor_mode=censor_mode,
                    progress_callback=progress_callback,
                )
            except Exception as exc:  # noqa: BLE001 - one file must not abort the batch
                self._queue.put((self._LOG, f"  ERROR: {exc}"))
                failed.append((video.name, str(exc)))
                continue

            if result.success:
                if result.output_path:
                    self._queue.put((
                        self._LOG,
                        f"  Done: {result.profane_instances_censored} censored -> "
                        f"{result.output_path.name}",
                    ))
                else:
                    self._queue.put((self._LOG, "  Done: no profanity found."))
                succeeded.append(video.name)
            else:
                msg = result.error_message or "unknown error"
                self._queue.put((self._LOG, f"  FAILED: {msg}"))
                failed.append((video.name, msg))

        self._queue.put((self._BATCH_DONE, succeeded, failed))

    def _poll_queue(self) -> None:
        """Drains the worker queue and updates the UI. Runs on the Tk thread."""
        try:
            while True:
                item = self._queue.get_nowait()
                kind = item[0]
                if kind == self._PROGRESS:
                    _, stage, percent, message = item
                    self.progress.configure(value=percent)
                    stage_name = stage.value.replace("_", " ").title()
                    status = f"[{percent:5.1f}%] {stage_name}"
                    if message:
                        status += f" - {message}"
                    self.status_var.set(status)
                    self._log(status)
                elif kind == self._LOG:
                    self._log(item[1])
                elif kind == self._DONE:
                    self._on_finished(item[1])
                    return
                elif kind == self._BATCH_DONE:
                    self._on_batch_finished(item[1], item[2])
                    return
                elif kind == self._ERROR:
                    self._log(f"Error: {item[1]}")
                    self.status_var.set("Failed.")
                    messagebox.showerror("Processing failed", item[1])
                    self._reset_controls()
                    return
        except queue.Empty:
            pass

        # Keep polling until a DONE/ERROR item is received (those branches return
        # early). Continue while the worker is alive or items may still be queued.
        if self._worker is not None and self._worker.is_alive():
            self.root.after(100, self._poll_queue)
        else:
            # Worker finished; drain any final items on one more pass.
            self.root.after(100, self._drain_final_queue)

    def _drain_final_queue(self) -> None:
        """Final drain after the worker thread has exited."""
        try:
            while True:
                item = self._queue.get_nowait()
                kind = item[0]
                if kind == self._DONE:
                    self._on_finished(item[1])
                elif kind == self._BATCH_DONE:
                    self._on_batch_finished(item[1], item[2])
                elif kind == self._LOG:
                    self._log(item[1])
                elif kind == self._ERROR:
                    self._log(f"Error: {item[1]}")
                    self.status_var.set("Failed.")
                    messagebox.showerror("Processing failed", item[1])
                    self._reset_controls()
                elif kind == self._PROGRESS:
                    _, stage, percent, message = item
                    self.progress.configure(value=percent)
        except queue.Empty:
            pass

    def _on_finished(self, result: ProcessingResult) -> None:
        if result.success:
            self._log("--- Processing complete ---")
            self._log(f"Elapsed: {result.total_elapsed_seconds:.1f}s")
            self._log(f"Profane instances found: {result.profane_instances_detected}")
            self._log(f"Profane instances censored: {result.profane_instances_censored}")
            if result.model_size_used:
                self._log(f"Model size used: {result.model_size_used}")
            if result.output_path:
                self._log(f"Output: {result.output_path}")
            else:
                self._log("No profanity found - no output file created.")
            if result.report_path:
                self._log(f"Report: {result.report_path}")
            self.progress.configure(value=100.0)
            self.status_var.set("Done.")
            messagebox.showinfo("Done", "Processing complete. See the log for details.")
        else:
            stage = result.error_stage.value if result.error_stage else "unknown"
            msg = result.error_message or "Unknown error."
            self._log(f"--- Processing failed at stage: {stage} ---")
            self._log(f"Error: {msg}")
            self.status_var.set("Failed.")
            messagebox.showerror("Processing failed", f"Failed at {stage}:\n{msg}")
        self._reset_controls()

    def _on_batch_finished(
        self, succeeded: list[str], failed: list[tuple[str, str]]
    ) -> None:
        total = len(succeeded) + len(failed)
        self._log("--- Batch complete ---")
        self._log(f"Total files: {total}")
        self._log(f"Succeeded:   {len(succeeded)}")
        self._log(f"Failed:      {len(failed)}")
        if failed:
            self._log("Failed files:")
            for name, reason in failed:
                self._log(f"  - {name}: {reason}")
        self.progress.configure(value=100.0)

        if failed:
            self.status_var.set(f"Batch done: {len(succeeded)} ok, {len(failed)} failed.")
            messagebox.showwarning(
                "Batch complete",
                f"{len(succeeded)} succeeded, {len(failed)} failed. "
                "See the log for details.",
            )
        else:
            self.status_var.set(f"Batch done: {len(succeeded)} file(s) processed.")
            messagebox.showinfo(
                "Batch complete",
                f"All {len(succeeded)} file(s) processed. See the log for details.",
            )
        self._reset_controls()

    def _reset_controls(self) -> None:
        self.run_btn.configure(state="normal")
        self.check_btn.configure(state="normal")


def main() -> int:
    """Launch the GUI. Entry point for the video-profanity-censor-gui script."""
    root = tk.Tk()
    CensorGUI(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
