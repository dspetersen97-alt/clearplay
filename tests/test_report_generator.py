"""Unit tests for the ReportGenerator component."""

from pathlib import Path

import pytest

from video_profanity_censor.models import CensorMode, Detection, TimestampRange
from video_profanity_censor.report_generator import ReportGenerationError, ReportGenerator


@pytest.fixture
def generator():
    return ReportGenerator()


@pytest.fixture
def sample_detections():
    """A list of detections in non-chronological order."""
    return [
        Detection(
            word="damn",
            timestamp_range=TimestampRange(start=45.123, end=45.678),
            censor_action=CensorMode.TONE,
        ),
        Detection(
            word="hell",
            timestamp_range=TimestampRange(start=12.500, end=12.900),
            censor_action=CensorMode.MUTE,
        ),
        Detection(
            word="crap",
            timestamp_range=TimestampRange(start=120.000, end=120.450),
            censor_action=CensorMode.TONE,
        ),
    ]


class TestTimestampFormatting:
    """Tests for HH:MM:SS.mmm timestamp formatting (Req 7.1)."""

    def test_zero_timestamp(self, generator):
        assert generator._format_timestamp(0.0) == "00:00:00.000"

    def test_milliseconds_only(self, generator):
        assert generator._format_timestamp(0.123) == "00:00:00.123"

    def test_seconds_and_milliseconds(self, generator):
        assert generator._format_timestamp(5.456) == "00:00:05.456"

    def test_minutes_seconds_milliseconds(self, generator):
        assert generator._format_timestamp(125.789) == "00:02:05.789"

    def test_hours_minutes_seconds(self, generator):
        # 1 hour, 30 minutes, 45.123 seconds = 5445.123
        assert generator._format_timestamp(5445.123) == "01:30:45.123"

    def test_large_timestamp(self, generator):
        # 2 hours, 15 minutes, 30.500 seconds = 8130.5
        assert generator._format_timestamp(8130.5) == "02:15:30.500"

    def test_exact_minute(self, generator):
        assert generator._format_timestamp(60.0) == "00:01:00.000"

    def test_exact_hour(self, generator):
        assert generator._format_timestamp(3600.0) == "01:00:00.000"


class TestReportSorting:
    """Tests for chronological sorting by start timestamp (Req 7.2)."""

    def test_entries_sorted_chronologically(self, generator, sample_detections, tmp_path):
        output_path = tmp_path / "output.mp4"
        report_path = tmp_path / "report.txt"

        generator.generate(sample_detections, output_path, report_path)

        content = report_path.read_text(encoding="utf-8")
        lines = content.strip().split("\n")

        # Find the data lines (after the header)
        data_lines = [
            line for line in lines
            if line.strip() and not line.startswith("Profanity")
            and not line.startswith("=")
            and not line.startswith("Total")
            and not line.startswith("-")
            and not line.startswith("Word")
        ]

        # Verify ordering: hell (12.5) < damn (45.123) < crap (120.0)
        assert "hell" in data_lines[0]
        assert "damn" in data_lines[1]
        assert "crap" in data_lines[2]

    def test_already_sorted_input(self, generator, tmp_path):
        detections = [
            Detection(
                word="first",
                timestamp_range=TimestampRange(start=1.0, end=1.5),
                censor_action=CensorMode.TONE,
            ),
            Detection(
                word="second",
                timestamp_range=TimestampRange(start=2.0, end=2.5),
                censor_action=CensorMode.MUTE,
            ),
        ]
        output_path = tmp_path / "output.mp4"
        report_path = tmp_path / "report.txt"

        generator.generate(detections, output_path, report_path)

        content = report_path.read_text(encoding="utf-8")
        first_idx = content.index("first")
        second_idx = content.index("second")
        assert first_idx < second_idx


class TestReportContent:
    """Tests for report content including word, timestamp, and action (Req 7.1)."""

    def test_contains_word(self, generator, tmp_path):
        detections = [
            Detection(
                word="damn",
                timestamp_range=TimestampRange(start=10.0, end=10.5),
                censor_action=CensorMode.TONE,
            ),
        ]
        output_path = tmp_path / "output.mp4"
        report_path = tmp_path / "report.txt"

        generator.generate(detections, output_path, report_path)

        content = report_path.read_text(encoding="utf-8")
        assert "damn" in content

    def test_contains_timestamp_range(self, generator, tmp_path):
        detections = [
            Detection(
                word="hell",
                timestamp_range=TimestampRange(start=65.123, end=65.678),
                censor_action=CensorMode.MUTE,
            ),
        ]
        output_path = tmp_path / "output.mp4"
        report_path = tmp_path / "report.txt"

        generator.generate(detections, output_path, report_path)

        content = report_path.read_text(encoding="utf-8")
        assert "00:01:05.123" in content
        assert "00:01:05.678" in content

    def test_contains_censor_action_tone(self, generator, tmp_path):
        detections = [
            Detection(
                word="word",
                timestamp_range=TimestampRange(start=1.0, end=1.5),
                censor_action=CensorMode.TONE,
            ),
        ]
        output_path = tmp_path / "output.mp4"
        report_path = tmp_path / "report.txt"

        generator.generate(detections, output_path, report_path)

        content = report_path.read_text(encoding="utf-8")
        assert "tone_replaced" in content

    def test_contains_censor_action_mute(self, generator, tmp_path):
        detections = [
            Detection(
                word="word",
                timestamp_range=TimestampRange(start=1.0, end=1.5),
                censor_action=CensorMode.MUTE,
            ),
        ]
        output_path = tmp_path / "output.mp4"
        report_path = tmp_path / "report.txt"

        generator.generate(detections, output_path, report_path)

        content = report_path.read_text(encoding="utf-8")
        assert "muted" in content


class TestNoProfanityCase:
    """Tests for 'no profanity found' handling (Req 7.6)."""

    def test_empty_detections_produces_no_profanity_message(self, generator, tmp_path):
        output_path = tmp_path / "output.mp4"
        report_path = tmp_path / "report.txt"

        generator.generate([], output_path, report_path)

        content = report_path.read_text(encoding="utf-8")
        assert "No profanity was found" in content

    def test_empty_detections_still_writes_report(self, generator, tmp_path):
        output_path = tmp_path / "output.mp4"
        report_path = tmp_path / "report.txt"

        generator.generate([], output_path, report_path)

        assert report_path.exists()


class TestReportPath:
    """Tests for report path handling (Req 7.3, 7.4)."""

    def test_writes_to_configured_report_path(self, generator, tmp_path):
        """Req 7.3: Write to configured report path."""
        output_path = tmp_path / "output.mp4"
        report_path = tmp_path / "custom_report.txt"

        result = generator.generate([], output_path, report_path)

        assert result == report_path
        assert report_path.exists()

    def test_derives_default_path_from_output_file(self, generator, tmp_path):
        """Req 7.4: Default path is output file name + '_report.txt'."""
        output_path = tmp_path / "my_movie_censored.mp4"

        result = generator.generate([], output_path, report_path=None)

        expected_path = tmp_path / "my_movie_censored_report.txt"
        assert result == expected_path
        assert expected_path.exists()

    def test_default_path_same_directory_as_output(self, generator, tmp_path):
        """Req 7.4: Report in same directory as output file."""
        subdir = tmp_path / "output_dir"
        subdir.mkdir()
        output_path = subdir / "video.mp4"

        result = generator.generate([], output_path, report_path=None)

        assert result.parent == subdir

    def test_creates_parent_directories_if_needed(self, generator, tmp_path):
        """Report path's parent directories are created if they don't exist."""
        output_path = tmp_path / "output.mp4"
        report_path = tmp_path / "nested" / "dir" / "report.txt"

        generator.generate([], output_path, report_path)

        assert report_path.exists()


class TestUnwritablePath:
    """Tests for unwritable report path error handling (Req 7.5)."""

    def test_unwritable_path_raises_error(self, generator, tmp_path):
        """Req 7.5: Error message indicates report could not be written and includes path."""
        output_path = tmp_path / "output.mp4"
        # Use a path with characters invalid on Windows (CON is reserved)
        invalid_path = tmp_path / "report\x00invalid.txt"

        with pytest.raises(ReportGenerationError) as exc_info:
            generator.generate([], output_path, invalid_path)

        assert str(invalid_path) in str(exc_info.value)

    def test_unwritable_path_error_contains_path(self, generator, tmp_path):
        """Req 7.5: Error includes the path that failed."""
        output_path = tmp_path / "output.mp4"
        # Create a file where the parent directory would need to be
        blocker = tmp_path / "blocker"
        blocker.write_text("I am a file")
        # Try to write a report inside what is actually a file (not a directory)
        invalid_path = blocker / "subdir" / "report.txt"

        with pytest.raises(ReportGenerationError) as exc_info:
            generator.generate([], output_path, invalid_path)

        error = exc_info.value
        assert error.path == invalid_path


class TestReturnValue:
    """Tests that generate returns the path where report was written."""

    def test_returns_configured_path(self, generator, tmp_path):
        output_path = tmp_path / "output.mp4"
        report_path = tmp_path / "report.txt"

        result = generator.generate([], output_path, report_path)
        assert result == report_path

    def test_returns_derived_path(self, generator, tmp_path):
        output_path = tmp_path / "video.mp4"

        result = generator.generate([], output_path, report_path=None)
        assert result == tmp_path / "video_report.txt"
