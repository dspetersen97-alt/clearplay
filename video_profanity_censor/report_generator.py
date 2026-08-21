"""Report generator for the Video Profanity Censor pipeline.

Generates a plain-text profanity detection report listing each detected
profane word with its timestamp range and censoring action applied.
"""

from pathlib import Path

from video_profanity_censor.models import Detection


class ReportGenerationError(Exception):
    """Raised when the report cannot be written to the specified path."""

    def __init__(self, path: Path, reason: str):
        self.path = path
        self.reason = reason
        super().__init__(
            f"Failed to write report to '{path}': {reason}"
        )


class ReportGenerator:
    """Generates a plain-text profanity detection report."""

    def generate(
        self,
        detections: list[Detection],
        output_path: Path,
        report_path: Path | None = None,
    ) -> Path:
        """Generates a plain-text profanity detection report.

        Args:
            detections: List of Detection objects from profanity detection.
            output_path: Path to the output video file (used to derive default report path).
            report_path: Explicit report output path. If None, derives from output_path.

        Returns:
            The path where the report was written.

        Raises:
            ReportGenerationError: If the report file cannot be written.
        """
        target_path = self._resolve_report_path(output_path, report_path)
        content = self._format_report(detections)
        self._write_report(target_path, content)
        return target_path

    def _resolve_report_path(
        self, output_path: Path, report_path: Path | None
    ) -> Path:
        """Resolves the report file path.

        If report_path is provided, uses it directly.
        Otherwise, derives from output_path by appending '_report.txt' to the stem.
        """
        if report_path is not None:
            return report_path

        # Derive default path: same directory, filename with "_report.txt" suffix
        return output_path.parent / f"{output_path.stem}_report.txt"

    def _format_report(self, detections: list[Detection]) -> str:
        """Formats the detection report as plain text.

        Entries are sorted chronologically by start timestamp.
        """
        lines: list[str] = []
        lines.append("Profanity Detection Report")
        lines.append("=" * 40)
        lines.append("")

        if not detections:
            lines.append("No profanity was found in the source file.")
            lines.append("")
            return "\n".join(lines)

        # Sort detections chronologically by start timestamp
        sorted_detections = sorted(
            detections, key=lambda d: d.timestamp_range.start
        )

        lines.append(f"Total detections: {len(sorted_detections)}")
        lines.append("")
        lines.append(f"{'Word':<20} {'Start':<14} {'End':<14} {'Action'}")
        lines.append("-" * 65)

        for detection in sorted_detections:
            start_fmt = self._format_timestamp(detection.timestamp_range.start)
            end_fmt = self._format_timestamp(detection.timestamp_range.end)
            word = detection.word
            action = detection.censor_action.value

            lines.append(f"{word:<20} {start_fmt:<14} {end_fmt:<14} {action}")

        lines.append("")
        return "\n".join(lines)

    def _format_timestamp(self, seconds: float) -> str:
        """Formats a timestamp in seconds to HH:MM:SS.mmm format."""
        total_ms = round(seconds * 1000)
        hours = total_ms // 3_600_000
        remainder = total_ms % 3_600_000
        minutes = remainder // 60_000
        remainder = remainder % 60_000
        secs = remainder // 1000
        ms = remainder % 1000

        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{ms:03d}"

    def _write_report(self, path: Path, content: str) -> None:
        """Writes the report content to the specified path.

        Raises:
            ReportGenerationError: If the path is not writable.
        """
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
        except (OSError, ValueError) as e:
            raise ReportGenerationError(path, str(e)) from e
