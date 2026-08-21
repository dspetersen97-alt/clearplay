"""Tests for the ProgressReporter component."""

import io
import time
from unittest.mock import patch

import pytest

from video_profanity_censor.models import AccelerationBackend, ProcessingStage
from video_profanity_censor.progress_reporter import ProgressReporter


class TestProgressReporterUpdate:
    """Tests for the update() method."""

    def test_first_update_always_displays(self):
        """First update should always be displayed regardless of thresholds."""
        output = io.StringIO()
        reporter = ProgressReporter(output=output)

        reporter.update(ProcessingStage.VALIDATION, 0.0, "Starting")

        result = output.getvalue()
        assert "0.0%" in result
        assert "Validation" in result
        assert "Starting" in result

    def test_displays_processing_stage(self):
        """Update should display the current processing stage name (Req 6.2)."""
        output = io.StringIO()
        reporter = ProgressReporter(output=output)

        reporter.update(ProcessingStage.TRANSCRIPTION, 50.0)

        assert "Transcription" in output.getvalue()

    def test_displays_backend_detection_stage(self):
        """Update should display BACKEND_DETECTION stage (Req 6.2)."""
        output = io.StringIO()
        reporter = ProgressReporter(output=output)

        reporter.update(ProcessingStage.BACKEND_DETECTION, 10.0, "Probing GPU")

        result = output.getvalue()
        assert "Backend Detection" in result
        assert "Probing GPU" in result

    def test_displays_all_processing_stages(self):
        """All processing stages should be displayable."""
        output = io.StringIO()
        reporter = ProgressReporter(output=output)

        for stage in ProcessingStage:
            reporter.update(stage, 0.0)
            # Reset to allow next update
            reporter._last_percent = -1.0

        result = output.getvalue()
        assert "Validation" in result
        assert "Backend Detection" in result
        assert "Subtitle Scanning" in result
        assert "Transcription" in result
        assert "Profanity Detection" in result
        assert "Audio Censoring" in result
        assert "Output Writing" in result

    def test_throttles_updates_within_2_seconds_and_less_than_1_percent(self):
        """Updates within 2 seconds AND less than 1% increment should be suppressed (Req 6.1)."""
        output = io.StringIO()
        reporter = ProgressReporter(output=output)

        reporter.update(ProcessingStage.TRANSCRIPTION, 10.0, "First")
        reporter.update(ProcessingStage.TRANSCRIPTION, 10.5, "Should be suppressed")

        result = output.getvalue()
        assert "First" in result
        assert "Should be suppressed" not in result

    def test_updates_on_1_percent_increment(self):
        """Update should display when 1% increment is reached (Req 6.1)."""
        output = io.StringIO()
        reporter = ProgressReporter(output=output)

        reporter.update(ProcessingStage.TRANSCRIPTION, 10.0, "First")
        reporter.update(ProcessingStage.TRANSCRIPTION, 11.0, "One percent later")

        result = output.getvalue()
        assert "First" in result
        assert "One percent later" in result

    def test_updates_after_2_seconds_elapsed(self):
        """Update should display when 2 seconds have elapsed (Req 6.1)."""
        output = io.StringIO()
        reporter = ProgressReporter(output=output)

        reporter.update(ProcessingStage.TRANSCRIPTION, 10.0, "First")

        # Simulate 2 seconds passing by manipulating the last update time
        reporter._last_update_time -= 2.0

        reporter.update(ProcessingStage.TRANSCRIPTION, 10.3, "After delay")

        result = output.getvalue()
        assert "First" in result
        assert "After delay" in result

    def test_percent_clamped_to_valid_range(self):
        """Percent values should be clamped between 0 and 100."""
        output = io.StringIO()
        reporter = ProgressReporter(output=output)

        reporter.update(ProcessingStage.VALIDATION, -5.0)
        result = output.getvalue()
        assert "0.0%" in result

        # Reset for next test
        reporter._last_percent = -1.0
        output.truncate(0)
        output.seek(0)

        reporter.update(ProcessingStage.VALIDATION, 150.0)
        result = output.getvalue()
        assert "100.0%" in result

    def test_message_is_optional(self):
        """Update should work without a message."""
        output = io.StringIO()
        reporter = ProgressReporter(output=output)

        reporter.update(ProcessingStage.VALIDATION, 5.0)

        result = output.getvalue()
        assert "5.0%" in result
        assert "Validation" in result


class TestProgressReporterEstimate:
    """Tests for the estimate_total_time() method."""

    def test_estimate_positive_for_positive_duration(self):
        """Estimation should return a positive value for positive file duration (Req 6.3)."""
        reporter = ProgressReporter(output=io.StringIO())

        estimate = reporter.estimate_total_time(3600.0)  # 1 hour file

        assert estimate > 0

    def test_estimate_zero_for_zero_duration(self):
        """Estimation should return 0 for a zero-duration file."""
        reporter = ProgressReporter(output=io.StringIO())

        estimate = reporter.estimate_total_time(0.0)

        assert estimate == 0.0

    def test_estimate_zero_for_negative_duration(self):
        """Estimation should return 0 for a negative duration."""
        reporter = ProgressReporter(output=io.StringIO())

        estimate = reporter.estimate_total_time(-10.0)

        assert estimate == 0.0

    def test_estimate_scales_with_duration(self):
        """Longer files should have proportionally longer estimates."""
        reporter = ProgressReporter(output=io.StringIO())

        short_estimate = reporter.estimate_total_time(600.0)  # 10 min
        long_estimate = reporter.estimate_total_time(3600.0)  # 1 hour

        assert long_estimate > short_estimate
        # Should scale linearly
        ratio = long_estimate / short_estimate
        assert abs(ratio - 6.0) < 0.01

    def test_estimate_within_25_percent_accuracy_range(self):
        """The estimate formula should produce values in a reasonable range (Req 6.3).

        While we can't test actual processing time here, we verify the estimate
        is within a plausible range based on the speed factor.
        """
        reporter = ProgressReporter(output=io.StringIO())

        file_duration = 3600.0  # 1 hour
        estimate = reporter.estimate_total_time(file_duration)

        # The estimate should be based on the speed factor (0.5) plus overhead
        # Expected: 3600 * 0.5 * 1.2 = 2160 seconds
        expected = file_duration * 0.5 * 1.2
        assert abs(estimate - expected) < 0.01


class TestProgressReporterSummary:
    """Tests for the display_summary() method."""

    def test_summary_includes_elapsed_time(self):
        """Summary should display elapsed time (Req 6.4)."""
        output = io.StringIO()
        reporter = ProgressReporter(output=output)

        reporter.display_summary(
            elapsed_seconds=125.5,
            instances_detected=10,
            instances_censored=10,
        )

        result = output.getvalue()
        assert "Elapsed time:" in result
        assert "2m 6s" in result  # 125.5 seconds

    def test_summary_includes_instances_found(self):
        """Summary should display number of instances found (Req 6.4)."""
        output = io.StringIO()
        reporter = ProgressReporter(output=output)

        reporter.display_summary(
            elapsed_seconds=60.0,
            instances_detected=15,
            instances_censored=15,
        )

        result = output.getvalue()
        assert "Profane instances found: 15" in result

    def test_summary_includes_instances_censored(self):
        """Summary should display number of instances censored (Req 6.4)."""
        output = io.StringIO()
        reporter = ProgressReporter(output=output)

        reporter.display_summary(
            elapsed_seconds=60.0,
            instances_detected=15,
            instances_censored=12,
        )

        result = output.getvalue()
        assert "Profane instances censored: 12" in result

    def test_summary_includes_active_backend(self):
        """Summary should display the active backend (Req 9.11)."""
        output = io.StringIO()
        reporter = ProgressReporter(output=output)

        reporter.display_summary(
            elapsed_seconds=60.0,
            instances_detected=5,
            instances_censored=5,
            active_backend=AccelerationBackend.DIRECTML,
        )

        result = output.getvalue()
        assert "Active backend: directml" in result

    def test_summary_with_cpu_backend(self):
        """Summary should display CPU backend name (Req 9.11)."""
        output = io.StringIO()
        reporter = ProgressReporter(output=output)

        reporter.display_summary(
            elapsed_seconds=300.0,
            instances_detected=20,
            instances_censored=20,
            active_backend=AccelerationBackend.CPU,
        )

        result = output.getvalue()
        assert "Active backend: cpu" in result

    def test_summary_without_backend(self):
        """Summary should work without active backend (e.g., when pipeline fails early)."""
        output = io.StringIO()
        reporter = ProgressReporter(output=output)

        reporter.display_summary(
            elapsed_seconds=5.0,
            instances_detected=0,
            instances_censored=0,
            active_backend=None,
        )

        result = output.getvalue()
        assert "Active backend" not in result
        assert "Elapsed time:" in result

    def test_summary_format_seconds_under_minute(self):
        """Short durations should be formatted as seconds."""
        output = io.StringIO()
        reporter = ProgressReporter(output=output)

        reporter.display_summary(
            elapsed_seconds=45.3,
            instances_detected=0,
            instances_censored=0,
        )

        result = output.getvalue()
        assert "45.3s" in result

    def test_summary_format_hours(self):
        """Long durations should be formatted with hours."""
        output = io.StringIO()
        reporter = ProgressReporter(output=output)

        reporter.display_summary(
            elapsed_seconds=3725.0,  # 1h 2m 5s
            instances_detected=0,
            instances_censored=0,
        )

        result = output.getvalue()
        assert "1h 2m" in result


class TestProgressReporterDisplayEstimate:
    """Tests for the display_estimate() method."""

    def test_displays_estimated_time(self):
        """Should display the estimated processing time (Req 6.3)."""
        output = io.StringIO()
        reporter = ProgressReporter(output=output)

        reporter.display_estimate(3600.0)

        result = output.getvalue()
        assert "Estimated processing time:" in result

    def test_no_display_for_zero_duration(self):
        """Should not display estimate for zero-duration files."""
        output = io.StringIO()
        reporter = ProgressReporter(output=output)

        reporter.display_estimate(0.0)

        assert output.getvalue() == ""


class TestProgressReporterCallback:
    """Tests for the get_callback() method."""

    def test_callback_returns_callable(self):
        """get_callback should return a callable matching ProgressCallback type."""
        reporter = ProgressReporter(output=io.StringIO())

        callback = reporter.get_callback()

        assert callable(callback)

    def test_callback_invokes_update(self):
        """Calling the callback should invoke update()."""
        output = io.StringIO()
        reporter = ProgressReporter(output=output)
        callback = reporter.get_callback()

        callback(ProcessingStage.TRANSCRIPTION, 50.0, "Halfway")

        result = output.getvalue()
        assert "50.0%" in result
        assert "Transcription" in result
        assert "Halfway" in result
