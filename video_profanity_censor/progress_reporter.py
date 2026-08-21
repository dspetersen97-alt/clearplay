"""Progress reporting for the Video Profanity Censor pipeline.

Provides real-time progress feedback during video processing including
percentage completion, current stage, time estimation, and final summary.
"""

import sys
import time

from video_profanity_censor.models import (
    AccelerationBackend,
    ProcessingStage,
)


class ProgressReporter:
    """Reports processing progress to the user.

    Displays progress percentage, current processing stage, estimated time,
    and a final summary upon completion. Updates are throttled to occur at
    most every 2 seconds or every 1% increment, whichever occurs first.
    """

    # Processing speed multiplier: estimated processing time relative to file duration.
    # Based on typical Whisper transcription speeds (the bottleneck stage).
    _PROCESSING_SPEED_FACTOR: float = 0.5  # ~0.5x real-time on modern hardware

    def __init__(self, output=None):
        """Initialize the progress reporter.

        Args:
            output: File-like object for writing progress output.
                    Defaults to sys.stdout.
        """
        self._output = output or sys.stdout
        self._last_update_time: float = 0.0
        self._last_percent: float = -1.0
        self._start_time: float | None = None
        self._current_stage: ProcessingStage | None = None

    def update(self, stage: ProcessingStage, percent: float, message: str = "") -> None:
        """Report progress to the user.

        Updates are displayed when at least 2 seconds have elapsed since the
        last update OR when progress has increased by at least 1%, whichever
        occurs first (Req 6.1).

        Args:
            stage: The current processing stage (Req 6.2).
            percent: Completion percentage (0.0 to 100.0).
            message: Optional descriptive message about current activity.
        """
        if self._start_time is None:
            self._start_time = time.time()

        self._current_stage = stage

        # Clamp percent to valid range
        percent = max(0.0, min(100.0, percent))

        now = time.time()
        time_elapsed = now - self._last_update_time
        percent_delta = percent - self._last_percent

        # Display update if 2 seconds have elapsed OR 1% increment reached (Req 6.1)
        should_update = (
            self._last_percent < 0  # First update always displays
            or time_elapsed >= 2.0
            or percent_delta >= 1.0
        )

        if should_update:
            self._last_update_time = now
            self._last_percent = percent
            self._display_progress(stage, percent, message)

    def estimate_total_time(self, file_duration_seconds: float) -> float:
        """Estimate total processing time based on file duration.

        The estimate aims to be accurate within 25% of actual processing
        time (Req 6.3). The estimate is based on the processing speed factor
        which accounts for transcription (the main bottleneck) plus overhead
        for other pipeline stages.

        Args:
            file_duration_seconds: Duration of the input video file in seconds.

        Returns:
            Estimated total processing time in seconds.
        """
        if file_duration_seconds <= 0:
            return 0.0

        # Base estimate: file duration * speed factor for transcription
        transcription_estimate = file_duration_seconds * self._PROCESSING_SPEED_FACTOR

        # Add overhead for non-transcription stages (~20% of transcription time)
        # Covers: validation, audio extraction, backend detection, subtitle scanning,
        # profanity detection, audio censoring, output writing
        overhead = transcription_estimate * 0.2

        return transcription_estimate + overhead

    def display_estimate(self, file_duration_seconds: float) -> None:
        """Display the estimated total processing time at the start (Req 6.3).

        Args:
            file_duration_seconds: Duration of the input video file in seconds.
        """
        estimate = self.estimate_total_time(file_duration_seconds)
        if estimate > 0:
            self._write(f"Estimated processing time: {self._format_duration(estimate)}\n")

    def display_summary(
        self,
        elapsed_seconds: float,
        instances_detected: int,
        instances_censored: int,
        active_backend: AccelerationBackend | None = None,
    ) -> None:
        """Display the final processing summary (Req 6.4, 9.11).

        Args:
            elapsed_seconds: Total elapsed processing time in seconds.
            instances_detected: Number of profane instances detected.
            instances_censored: Number of profane instances censored.
            active_backend: The acceleration backend used for processing.
        """
        self._write("\n--- Processing Summary ---\n")
        self._write(f"Elapsed time: {self._format_duration(elapsed_seconds)}\n")
        self._write(f"Profane instances found: {instances_detected}\n")
        self._write(f"Profane instances censored: {instances_censored}\n")
        if active_backend is not None:
            self._write(f"Active backend: {active_backend.value}\n")
        self._write("--------------------------\n")

    def get_callback(self):
        """Return a ProgressCallback-compatible function.

        Returns:
            A callable matching the ProgressCallback type signature:
            Callable[[ProcessingStage, float, str], None]
        """
        return self.update

    def _display_progress(self, stage: ProcessingStage, percent: float, message: str) -> None:
        """Write a progress line to output.

        Args:
            stage: Current processing stage.
            percent: Completion percentage.
            message: Optional descriptive message.
        """
        stage_display = stage.value.replace("_", " ").title()
        line = f"[{percent:5.1f}%] {stage_display}"
        if message:
            line += f" - {message}"
        self._write(line + "\n")

    def _format_duration(self, seconds: float) -> str:
        """Format a duration in seconds to a human-readable string.

        Args:
            seconds: Duration in seconds.

        Returns:
            Formatted string like "1m 30s" or "5s".
        """
        if seconds < 60:
            return f"{seconds:.1f}s"
        minutes = int(seconds // 60)
        remaining = seconds % 60
        if minutes < 60:
            return f"{minutes}m {remaining:.0f}s"
        hours = minutes // 60
        remaining_minutes = minutes % 60
        return f"{hours}h {remaining_minutes}m"

    def _write(self, text: str) -> None:
        """Write text to the output stream.

        Args:
            text: Text to write.
        """
        self._output.write(text)
        if hasattr(self._output, "flush"):
            self._output.flush()
