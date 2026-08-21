"""Property-based tests for InputValidator file format and size validation.

**Validates: Requirements 1.1, 1.3, 1.7**
"""

from pathlib import Path
from unittest.mock import patch, MagicMock
import string

import pytest
from hypothesis import given, strategies as st, settings

from video_profanity_censor.input_validator import InputValidator


# Constants mirroring the validator's configuration
SUPPORTED_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".wmv"}
MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024 * 1024  # 50 GB

# Strategy: generate random file extensions (both valid and invalid)
valid_extensions = st.sampled_from(sorted(SUPPORTED_EXTENSIONS))
invalid_extensions = st.text(
    alphabet=string.ascii_lowercase + string.digits,
    min_size=1,
    max_size=6,
).map(lambda s: f".{s}").filter(lambda ext: ext.lower() not in SUPPORTED_EXTENSIONS)

all_extensions = st.one_of(valid_extensions, invalid_extensions)

# Strategy: generate file sizes covering both within-limit and exceeding-limit
file_sizes = st.one_of(
    st.integers(min_value=1, max_value=MAX_FILE_SIZE_BYTES),  # within limit
    st.integers(min_value=MAX_FILE_SIZE_BYTES + 1, max_value=MAX_FILE_SIZE_BYTES * 3),  # exceeding
    st.just(MAX_FILE_SIZE_BYTES),  # exact boundary
    st.just(MAX_FILE_SIZE_BYTES + 1),  # just over
)


def _make_mock_stat(file_size: int):
    """Create a mock stat_result with the given file size."""
    mock_stat = MagicMock()
    mock_stat.st_size = file_size
    return mock_stat


def _make_ffprobe_result():
    """Create a minimal valid ffprobe result with video and audio streams."""
    return {
        "format": {"format_name": "mov,mp4,m4a,3gp,3g2,mj2"},
        "streams": [
            {"codec_type": "video", "codec_name": "h264"},
            {
                "codec_type": "audio",
                "codec_name": "aac",
                "sample_rate": "48000",
                "channels": 2,
                "channel_layout": "stereo",
                "bits_per_sample": "0",
                "bits_per_raw_sample": "32",
                "bit_rate": "128000",
                "duration": "120.5",
            },
        ],
    }


def _run_validation(extension: str, file_size: int):
    """Run the InputValidator with mocked filesystem for the given extension and size."""
    file_path = Path(f"C:/fake/test_video{extension}")
    validator = InputValidator()
    mock_stat = _make_mock_stat(file_size)

    with (
        patch.object(Path, "exists", return_value=True),
        patch.object(Path, "is_file", return_value=True),
        patch("os.access", return_value=True),
        patch.object(Path, "stat", return_value=mock_stat),
        patch("ffmpeg.probe", return_value=_make_ffprobe_result()),
    ):
        return validator.validate(file_path)


class TestFileFormatValidationProperty:
    """Property 1: File Format Validation.

    For any file path with a file extension, the InputValidator SHALL accept the
    file if and only if the extension (case-insensitive) is one of
    {.mp4, .mkv, .avi, .mov, .wmv} and the file size is at most 50 GB.
    Files with unsupported extensions or exceeding the size limit SHALL be
    rejected with an appropriate error message.

    **Validates: Requirements 1.1, 1.3, 1.7**
    """

    @given(extension=all_extensions, file_size=file_sizes)
    @settings(max_examples=200)
    def test_format_and_size_acceptance_property(
        self, extension: str, file_size: int
    ):
        """Acceptance iff extension is supported AND size <= 50 GB."""
        # Determine expected outcome based on the property
        extension_valid = extension.lower() in SUPPORTED_EXTENSIONS
        size_valid = file_size <= MAX_FILE_SIZE_BYTES
        should_accept = extension_valid and size_valid

        result = _run_validation(extension, file_size)

        if should_accept:
            assert result.is_valid, (
                f"Expected valid for ext={extension}, size={file_size}, "
                f"but got error: {result.error_message}"
            )
        else:
            assert not result.is_valid, (
                f"Expected invalid for ext={extension}, size={file_size}, "
                f"but was accepted"
            )
            # Verify appropriate error message content
            if not extension_valid:
                assert "unsupported" in result.error_message.lower() or \
                       "format" in result.error_message.lower(), (
                    f"Error for unsupported format should mention format: "
                    f"{result.error_message}"
                )
            elif not size_valid:
                assert "size" in result.error_message.lower() or \
                       "exceed" in result.error_message.lower() or \
                       "gb" in result.error_message.lower(), (
                    f"Error for oversized file should mention size: "
                    f"{result.error_message}"
                )

    @given(
        extension=valid_extensions,
        file_size=st.integers(min_value=1, max_value=MAX_FILE_SIZE_BYTES),
    )
    @settings(max_examples=100)
    def test_valid_format_and_size_always_accepted(
        self, extension: str, file_size: int
    ):
        """Any supported extension with size <= 50 GB is always accepted."""
        result = _run_validation(extension, file_size)

        assert result.is_valid, (
            f"Valid extension {extension} with size {file_size} should be accepted, "
            f"but got: {result.error_message}"
        )

    @given(
        extension=invalid_extensions,
        file_size=file_sizes,
    )
    @settings(max_examples=100)
    def test_invalid_format_always_rejected(
        self, extension: str, file_size: int
    ):
        """Any unsupported extension is always rejected, regardless of size."""
        result = _run_validation(extension, file_size)

        assert not result.is_valid, (
            f"Invalid extension {extension} should be rejected"
        )
        assert "unsupported" in result.error_message.lower() or \
               "format" in result.error_message.lower(), (
            f"Error message should mention unsupported format: {result.error_message}"
        )

    @given(
        extension=valid_extensions,
        file_size=st.integers(
            min_value=MAX_FILE_SIZE_BYTES + 1,
            max_value=MAX_FILE_SIZE_BYTES * 3,
        ),
    )
    @settings(max_examples=100)
    def test_oversized_file_always_rejected(
        self, extension: str, file_size: int
    ):
        """Any file exceeding 50 GB is always rejected, even with valid format."""
        result = _run_validation(extension, file_size)

        assert not result.is_valid, (
            f"File of size {file_size} bytes should be rejected"
        )
        assert "size" in result.error_message.lower() or \
               "exceed" in result.error_message.lower() or \
               "gb" in result.error_message.lower(), (
            f"Error message should mention size: {result.error_message}"
        )

    @given(
        extension=st.sampled_from(sorted(SUPPORTED_EXTENSIONS)),
        upper=st.booleans(),
    )
    @settings(max_examples=50)
    def test_case_insensitive_extension_matching(
        self, extension: str, upper: bool
    ):
        """Extension matching is case-insensitive (e.g., .MP4, .Mp4 accepted)."""
        # Apply case transformation
        if upper:
            ext = extension.upper()
        else:
            ext = extension.swapcase()

        result = _run_validation(ext, 1024 * 1024)  # 1 MB, well within limits

        assert result.is_valid, (
            f"Extension {ext} (case variant of {extension}) should be accepted, "
            f"but got: {result.error_message}"
        )
