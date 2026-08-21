"""Unit tests for InputValidator component.

Tests cover:
- File not found error (Req 1.6)
- Unsupported format error listing supported formats (Req 1.3)
- File exceeds 50 GB error (Req 1.7)
- No audio track error (Req 1.4)
- Corrupt/unparseable container error (Req 1.5)
"""

from pathlib import Path
from unittest.mock import patch

import pytest

from video_profanity_censor.input_validator import InputValidator


@pytest.fixture
def validator() -> InputValidator:
    """Provides an InputValidator instance."""
    return InputValidator()


class TestFileNotFound:
    """Tests for file not found error (Req 1.6)."""

    def test_nonexistent_file_returns_invalid(self, validator: InputValidator, tmp_path: Path):
        """A file path that does not exist should return is_valid=False."""
        missing_file = tmp_path / "nonexistent_video.mp4"
        result = validator.validate(missing_file)

        assert result.is_valid is False
        assert "not found" in result.error_message.lower()
        assert str(missing_file) in result.error_message

    def test_nonexistent_nested_path_returns_invalid(self, validator: InputValidator, tmp_path: Path):
        """A path in a nonexistent directory should return an appropriate error."""
        missing_file = tmp_path / "no_such_dir" / "video.mp4"
        result = validator.validate(missing_file)

        assert result.is_valid is False
        assert "not found" in result.error_message.lower()


class TestUnsupportedFormat:
    """Tests for unsupported format error message (Req 1.3)."""

    def test_unsupported_extension_returns_invalid(self, validator: InputValidator, tmp_path: Path):
        """A file with an unsupported extension should be rejected."""
        bad_file = tmp_path / "video.flv"
        bad_file.write_bytes(b"\x00" * 100)
        result = validator.validate(bad_file)

        assert result.is_valid is False
        assert "unsupported" in result.error_message.lower()

    def test_error_message_lists_supported_formats(self, validator: InputValidator, tmp_path: Path):
        """The error message for unsupported format should list all supported formats."""
        bad_file = tmp_path / "video.webm"
        bad_file.write_bytes(b"\x00" * 100)
        result = validator.validate(bad_file)

        assert result.is_valid is False
        for fmt in InputValidator.SUPPORTED_FORMATS:
            assert fmt in result.error_message

    def test_error_message_mentions_provided_extension(self, validator: InputValidator, tmp_path: Path):
        """The error should mention the invalid extension that was provided."""
        bad_file = tmp_path / "video.xyz"
        bad_file.write_bytes(b"\x00" * 100)
        result = validator.validate(bad_file)

        assert result.is_valid is False
        assert ".xyz" in result.error_message


class TestFileSizeExceedsLimit:
    """Tests for file exceeds 50 GB error (Req 1.7)."""

    def test_file_over_50gb_returns_invalid(self, validator: InputValidator, tmp_path: Path):
        """A file larger than 50 GB should be rejected."""
        large_file = tmp_path / "huge_video.mp4"
        large_file.write_bytes(b"\x00" * 10)  # Small actual file

        # Mock stat to report a size over 50 GB
        oversized = InputValidator.MAX_FILE_SIZE_BYTES + 1
        with patch.object(Path, "stat") as mock_stat:
            mock_stat.return_value.st_size = oversized
            result = validator.validate(large_file)

        assert result.is_valid is False
        assert "50" in result.error_message or "size" in result.error_message.lower()

    def test_file_exactly_50gb_is_accepted_at_format_stage(self, validator: InputValidator, tmp_path: Path):
        """A file exactly at the 50 GB limit should pass the size check (proceeds to FFmpeg probe)."""
        file_at_limit = tmp_path / "big_video.mp4"
        file_at_limit.write_bytes(b"\x00" * 10)

        exactly_50gb = InputValidator.MAX_FILE_SIZE_BYTES
        probe_result = {
            "format": {"format_name": "mov,mp4,m4a,3gp,3g2,mj2"},
            "streams": [
                {"codec_type": "video", "codec_name": "h264"},
                {"codec_type": "audio", "codec_name": "aac", "sample_rate": "44100",
                 "channels": 2, "channel_layout": "stereo", "bit_rate": "128000",
                 "bits_per_sample": "16", "duration": "120.0"},
            ],
        }
        with patch.object(Path, "stat") as mock_stat, \
             patch("video_profanity_censor.input_validator.ffmpeg.probe", return_value=probe_result):
            mock_stat.return_value.st_size = exactly_50gb
            result = validator.validate(file_at_limit)

        # Should pass size check (may pass or fail on probe, but not on size)
        assert "size" not in (result.error_message or "").lower()


class TestNoAudioTrack:
    """Tests for no audio track error (Req 1.4)."""

    def test_video_without_audio_returns_invalid(self, validator: InputValidator, tmp_path: Path):
        """A valid video file with no audio streams should be rejected."""
        video_file = tmp_path / "silent_video.mp4"
        video_file.write_bytes(b"\x00" * 100)

        probe_result = {
            "format": {"format_name": "mov,mp4,m4a,3gp,3g2,mj2"},
            "streams": [
                {"codec_type": "video", "codec_name": "h264"},
            ],
        }
        with patch.object(Path, "stat") as mock_stat, \
             patch("video_profanity_censor.input_validator.ffmpeg.probe", return_value=probe_result):
            mock_stat.return_value.st_size = 1024 * 1024  # 1 MB
            result = validator.validate(video_file)

        assert result.is_valid is False
        assert "no audio track" in result.error_message.lower()

    def test_no_audio_error_still_reports_video_codec(self, validator: InputValidator, tmp_path: Path):
        """When no audio track is found, the result should still contain video codec info."""
        video_file = tmp_path / "no_audio.mkv"
        video_file.write_bytes(b"\x00" * 100)

        probe_result = {
            "format": {"format_name": "matroska,webm"},
            "streams": [
                {"codec_type": "video", "codec_name": "hevc"},
            ],
        }
        with patch.object(Path, "stat") as mock_stat, \
             patch("video_profanity_censor.input_validator.ffmpeg.probe", return_value=probe_result):
            mock_stat.return_value.st_size = 5 * 1024 * 1024
            result = validator.validate(video_file)

        assert result.is_valid is False
        assert result.video_codec == "hevc"


class TestCorruptContainer:
    """Tests for corrupt/unparseable container error (Req 1.5)."""

    def test_corrupt_file_returns_invalid(self, validator: InputValidator, tmp_path: Path):
        """A file that FFmpeg cannot parse should be rejected with an appropriate error."""
        import ffmpeg as ffmpeg_lib

        corrupt_file = tmp_path / "corrupt.mp4"
        corrupt_file.write_bytes(b"\x00" * 100)

        with patch.object(Path, "stat") as mock_stat, \
             patch("video_profanity_censor.input_validator.ffmpeg.probe",
                   side_effect=ffmpeg_lib.Error("ffprobe", b"", b"Invalid data")):
            mock_stat.return_value.st_size = 1024 * 1024
            result = validator.validate(corrupt_file)

        assert result.is_valid is False
        assert "cannot parse" in result.error_message.lower() or "valid video container" in result.error_message.lower()

    def test_corrupt_error_includes_file_path(self, validator: InputValidator, tmp_path: Path):
        """The corrupt container error should reference the file path."""
        import ffmpeg as ffmpeg_lib

        corrupt_file = tmp_path / "bad_container.avi"
        corrupt_file.write_bytes(b"\x00" * 100)

        with patch.object(Path, "stat") as mock_stat, \
             patch("video_profanity_censor.input_validator.ffmpeg.probe",
                   side_effect=ffmpeg_lib.Error("ffprobe", b"", b"Error")):
            mock_stat.return_value.st_size = 2 * 1024 * 1024
            result = validator.validate(corrupt_file)

        assert result.is_valid is False
        assert str(corrupt_file) in result.error_message
