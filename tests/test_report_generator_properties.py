"""Property-based tests for ReportGenerator.

**Validates: Requirements 7.1, 7.2**
"""

import re
import tempfile
from pathlib import Path

from hypothesis import given, settings, strategies as st

from video_profanity_censor.models import (
    CensorMode,
    Detection,
    TimestampRange,
)
from video_profanity_censor.report_generator import ReportGenerator


# Strategy: generate a word (alphabetic, reasonable length)
_words = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyz",
    min_size=2,
    max_size=15,
).filter(lambda w: w.isalpha())

# Strategy: generate timestamp starts (non-negative, up to 24 hours)
_timestamp_starts = st.floats(
    min_value=0.0, max_value=86400.0, allow_nan=False, allow_infinity=False
)

# Strategy: generate positive durations for timestamp ranges
_durations = st.floats(
    min_value=0.01, max_value=10.0, allow_nan=False, allow_infinity=False
)

# Strategy: generate a censor mode
_censor_modes = st.sampled_from([CensorMode.TONE, CensorMode.MUTE])


@st.composite
def detection_strategy(draw):
    """Generate a single Detection with valid timestamps (start < end)."""
    word = draw(_words)
    start = draw(_timestamp_starts)
    duration = draw(_durations)
    mode = draw(_censor_modes)

    return Detection(
        word=word,
        timestamp_range=TimestampRange(start=start, end=start + duration),
        censor_action=mode,
    )


# Strategy: generate a list of Detections (possibly empty for edge case, but at least 1 for main tests)
_detection_lists = st.lists(detection_strategy(), min_size=1, max_size=20)


# Regex for HH:MM:SS.mmm format
_TIMESTAMP_PATTERN = re.compile(r"\d{2}:\d{2}:\d{2}\.\d{3}")


class TestReportGenerationCorrectness:
    """Property 10: Report Generation Correctness.

    For any list of detections (in any order), the generated report SHALL contain
    one entry per detection with the word, timestamp range formatted as HH:MM:SS.mmm,
    and censoring action — and all entries SHALL be sorted chronologically by start
    timestamp.

    **Validates: Requirements 7.1, 7.2**
    """

    @given(detections=_detection_lists)
    @settings(max_examples=200)
    def test_one_entry_per_detection(self, detections: list[Detection]):
        """The report must contain exactly one entry per detection."""
        generator = ReportGenerator()

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "output.mp4"
            report_path = Path(tmp_dir) / "report.txt"

            generator.generate(detections, output_path, report_path=report_path)
            content = report_path.read_text(encoding="utf-8")

        # Each detection word should appear in the report content.
        # Count data lines (lines after the header separator that contain timestamps)
        lines = content.strip().split("\n")
        data_lines = [
            line for line in lines
            if _TIMESTAMP_PATTERN.search(line)
        ]

        assert len(data_lines) == len(detections), (
            f"Expected {len(detections)} data entries in report, "
            f"found {len(data_lines)}. Report content:\n{content}"
        )

    @given(detections=_detection_lists)
    @settings(max_examples=200)
    def test_entries_sorted_chronologically(self, detections: list[Detection]):
        """Report entries must be sorted chronologically by start timestamp."""
        generator = ReportGenerator()

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "output.mp4"
            report_path = Path(tmp_dir) / "report.txt"

            generator.generate(detections, output_path, report_path=report_path)
            content = report_path.read_text(encoding="utf-8")

        # Extract all timestamp pairs from data lines
        lines = content.strip().split("\n")
        data_lines = [
            line for line in lines
            if _TIMESTAMP_PATTERN.search(line)
        ]

        # Parse the start timestamps from data lines
        start_timestamps: list[str] = []
        for line in data_lines:
            matches = _TIMESTAMP_PATTERN.findall(line)
            assert len(matches) >= 1, (
                f"Expected at least one timestamp in data line: '{line}'"
            )
            start_timestamps.append(matches[0])

        # Verify chronological order (string comparison works for HH:MM:SS.mmm)
        for i in range(len(start_timestamps) - 1):
            assert start_timestamps[i] <= start_timestamps[i + 1], (
                f"Entries not sorted chronologically: "
                f"'{start_timestamps[i]}' should come before or equal "
                f"'{start_timestamps[i + 1]}'"
            )

    @given(detections=_detection_lists)
    @settings(max_examples=200)
    def test_entries_contain_word_timestamps_and_action(
        self, detections: list[Detection]
    ):
        """Each entry must contain the word, timestamp range in HH:MM:SS.mmm, and action."""
        generator = ReportGenerator()

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "output.mp4"
            report_path = Path(tmp_dir) / "report.txt"

            generator.generate(detections, output_path, report_path=report_path)
            content = report_path.read_text(encoding="utf-8")

        # Sort detections same way the report does for matching
        sorted_detections = sorted(
            detections, key=lambda d: d.timestamp_range.start
        )

        # Extract data lines (lines containing timestamps)
        lines = content.strip().split("\n")
        data_lines = [
            line for line in lines
            if _TIMESTAMP_PATTERN.search(line)
        ]

        assert len(data_lines) == len(sorted_detections)

        for detection, line in zip(sorted_detections, data_lines):
            # Verify word is present in the line
            assert detection.word in line, (
                f"Expected word '{detection.word}' in report line: '{line}'"
            )

            # Verify timestamps are in HH:MM:SS.mmm format
            timestamps = _TIMESTAMP_PATTERN.findall(line)
            assert len(timestamps) == 2, (
                f"Expected 2 timestamps (start, end) in line: '{line}', "
                f"found {len(timestamps)}"
            )

            # Verify the censoring action is present
            assert detection.censor_action.value in line, (
                f"Expected action '{detection.censor_action.value}' "
                f"in report line: '{line}'"
            )

    @given(detections=_detection_lists)
    @settings(max_examples=200)
    def test_timestamp_format_is_correct(self, detections: list[Detection]):
        """All timestamps in the report must use HH:MM:SS.mmm format."""
        generator = ReportGenerator()

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "output.mp4"
            report_path = Path(tmp_dir) / "report.txt"

            generator.generate(detections, output_path, report_path=report_path)
            content = report_path.read_text(encoding="utf-8")

        # Extract data lines
        lines = content.strip().split("\n")
        data_lines = [
            line for line in lines
            if _TIMESTAMP_PATTERN.search(line)
        ]

        # Each data line must have exactly 2 valid HH:MM:SS.mmm timestamps
        strict_pattern = re.compile(r"\d{2}:\d{2}:\d{2}\.\d{3}")
        for line in data_lines:
            timestamps = strict_pattern.findall(line)
            assert len(timestamps) == 2, (
                f"Expected exactly 2 HH:MM:SS.mmm timestamps in line: '{line}'"
            )

            # Validate the time components are within valid ranges
            for ts in timestamps:
                parts = ts.split(":")
                hours = int(parts[0])
                minutes = int(parts[1])
                sec_parts = parts[2].split(".")
                seconds = int(sec_parts[0])
                milliseconds = int(sec_parts[1])

                assert 0 <= hours <= 99, f"Invalid hours: {hours}"
                assert 0 <= minutes <= 59, f"Invalid minutes: {minutes}"
                assert 0 <= seconds <= 59, f"Invalid seconds: {seconds}"
                assert 0 <= milliseconds <= 999, f"Invalid milliseconds: {milliseconds}"


# ============================================================================
# Property 11: Default Report Path Derivation
# ============================================================================

# Valid filename characters (avoiding reserved chars on Windows)
_filename_chars = st.sampled_from(
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "0123456789"
    "_-"
)

# Strategy: generate a valid filename stem (non-empty, no dots or path separators)
_filename_stems = st.text(
    alphabet=_filename_chars,
    min_size=1,
    max_size=30,
)

# Strategy: generate common file extensions (with leading dot)
_extensions = st.sampled_from([
    ".mp4", ".mkv", ".avi", ".mov", ".wmv",
    ".txt", ".wav", ".flac", ".mp3",
    ".video", ".output", ".censored",
])

# Strategy: generate directory path components
_dir_components = st.text(
    alphabet=_filename_chars,
    min_size=1,
    max_size=15,
)

# Strategy: generate a full directory path with 1-4 components
_directory_paths = st.lists(
    _dir_components,
    min_size=1,
    max_size=4,
).map(lambda parts: Path("C:/") / Path(*parts))


@st.composite
def output_file_paths(draw):
    """Generate random output file paths with varying directories, filenames, and extensions."""
    directory = draw(_directory_paths)
    stem = draw(_filename_stems)
    extension = draw(_extensions)
    return directory / f"{stem}{extension}"


class TestDefaultReportPathDerivation:
    """Property 11: Default Report Path Derivation.

    For any output file path, when no report path is configured, the report
    SHALL be written to the same directory as the output file using the output
    filename stem appended with "_report.txt".

    **Validates: Requirements 7.4**
    """

    @given(output_path=output_file_paths())
    @settings(max_examples=300)
    def test_report_path_in_same_directory(self, output_path: Path):
        """The derived report path must be in the same directory as the output file."""
        generator = ReportGenerator()
        report_path = generator._resolve_report_path(output_path, report_path=None)

        assert report_path.parent == output_path.parent, (
            f"Report path parent '{report_path.parent}' does not match "
            f"output path parent '{output_path.parent}'"
        )

    @given(output_path=output_file_paths())
    @settings(max_examples=300)
    def test_report_filename_is_stem_plus_report_txt(self, output_path: Path):
        """The derived report filename must be the output stem + '_report.txt'."""
        generator = ReportGenerator()
        report_path = generator._resolve_report_path(output_path, report_path=None)

        expected_filename = f"{output_path.stem}_report.txt"
        assert report_path.name == expected_filename, (
            f"Report filename '{report_path.name}' does not match "
            f"expected '{expected_filename}' (output stem: '{output_path.stem}')"
        )

    @given(output_path=output_file_paths())
    @settings(max_examples=300)
    def test_report_path_ends_with_report_txt_suffix(self, output_path: Path):
        """The derived report path must always end with '_report.txt'."""
        generator = ReportGenerator()
        report_path = generator._resolve_report_path(output_path, report_path=None)

        assert report_path.name.endswith("_report.txt"), (
            f"Report path name '{report_path.name}' does not end with '_report.txt'"
        )

    @given(output_path=output_file_paths())
    @settings(max_examples=300)
    def test_report_path_preserves_output_stem(self, output_path: Path):
        """The derived report path must contain the original output file stem."""
        generator = ReportGenerator()
        report_path = generator._resolve_report_path(output_path, report_path=None)

        # The report filename should start with the output stem
        assert report_path.name.startswith(output_path.stem), (
            f"Report filename '{report_path.name}' does not start with "
            f"output stem '{output_path.stem}'"
        )

    @given(output_path=output_file_paths())
    @settings(max_examples=300)
    def test_explicit_report_path_overrides_default(self, output_path: Path):
        """When an explicit report path is provided, it is used as-is."""
        generator = ReportGenerator()
        explicit_path = Path("C:/custom/location/my_report.txt")

        report_path = generator._resolve_report_path(output_path, report_path=explicit_path)

        assert report_path == explicit_path, (
            f"Explicit report path '{explicit_path}' was not honored; "
            f"got '{report_path}' instead"
        )
