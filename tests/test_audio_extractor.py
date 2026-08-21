"""Unit tests for AudioExtractor component."""

import subprocess
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from video_profanity_censor.audio_extractor import AudioExtractor
from video_profanity_censor.models import AudioExtractionResult, AudioMetadata


@pytest.fixture
def extractor() -> AudioExtractor:
    """Provides an AudioExtractor instance."""
    return AudioExtractor()


@pytest.fixture
def sample_video(tmp_path: Path) -> Path:
    """Creates a small sample video file with audio using FFmpeg."""
    video_path = tmp_path / "sample.mp4"
    try:
        # Generate a 2-second video with a sine wave audio track
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-f", "lavfi", "-i", "testsrc=duration=2:size=320x240:rate=24",
                "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
                "-c:v", "libx264", "-preset", "ultrafast",
                "-c:a", "aac", "-b:a", "128k", "-ar", "44100", "-ac", "2",
                str(video_path),
            ],
            check=True,
            capture_output=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        pytest.skip("FFmpeg not available for test")
    return video_path


@pytest.fixture
def multi_audio_video(tmp_path: Path) -> Path:
    """Creates a video file with two audio tracks."""
    video_path = tmp_path / "multi_audio.mp4"
    try:
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-f", "lavfi", "-i", "testsrc=duration=2:size=320x240:rate=24",
                "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
                "-f", "lavfi", "-i", "sine=frequency=880:duration=2",
                "-c:v", "libx264", "-preset", "ultrafast",
                "-c:a", "aac", "-b:a", "128k",
                "-map", "0:v", "-map", "1:a", "-map", "2:a",
                str(video_path),
            ],
            check=True,
            capture_output=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        pytest.skip("FFmpeg not available for test")
    return video_path


class TestAudioExtractor:
    """Tests for AudioExtractor.extract()."""

    def test_extract_default_track(self, extractor: AudioExtractor, sample_video: Path, tmp_path: Path):
        """Test extracting the default (first) audio track."""
        output_path = tmp_path / "extracted.wav"
        result = extractor.extract(sample_video, output_path=output_path)

        assert isinstance(result, AudioExtractionResult)
        assert result.output_path == output_path
        assert result.output_path.exists()
        assert result.audio_metadata.track_index == 0
        assert result.audio_metadata.channels == 2
        assert result.audio_metadata.sample_rate == 44100
        assert result.audio_metadata.codec == "aac"

    def test_extract_to_temp_file(self, extractor: AudioExtractor, sample_video: Path):
        """Test extraction to auto-generated temporary file when output_path is None."""
        result = extractor.extract(sample_video)

        assert result.output_path.exists()
        assert result.output_path.suffix == ".wav"
        # Cleanup
        result.output_path.unlink(missing_ok=True)

    def test_extract_specific_track_index(
        self, extractor: AudioExtractor, multi_audio_video: Path, tmp_path: Path
    ):
        """Test extracting a specific audio track by index (Req 2.7)."""
        output_path = tmp_path / "track1.wav"
        result = extractor.extract(multi_audio_video, track_index=1, output_path=output_path)

        assert result.output_path.exists()
        assert result.audio_metadata.track_index == 1

    def test_extract_invalid_track_index(self, extractor: AudioExtractor, sample_video: Path):
        """Test that an invalid track index raises ValueError."""
        with pytest.raises(ValueError, match="does not exist"):
            extractor.extract(sample_video, track_index=99)

    def test_extract_file_not_found(self, extractor: AudioExtractor):
        """Test that a non-existent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="Source file not found"):
            extractor.extract(Path("/nonexistent/video.mp4"))

    def test_audio_metadata_captured(self, extractor: AudioExtractor, sample_video: Path, tmp_path: Path):
        """Test that audio metadata is correctly captured."""
        output_path = tmp_path / "meta_test.wav"
        result = extractor.extract(sample_video, output_path=output_path)

        meta = result.audio_metadata
        assert isinstance(meta, AudioMetadata)
        assert meta.codec == "aac"
        assert meta.sample_rate == 44100
        assert meta.channels == 2
        assert meta.channel_layout in ("stereo", "2 channels")
        assert meta.duration_seconds > 0
        assert meta.bitrate > 0


class TestDefaultChannelLayout:
    """Tests for _default_channel_layout static method."""

    def test_mono(self):
        assert AudioExtractor._default_channel_layout(1) == "mono"

    def test_stereo(self):
        assert AudioExtractor._default_channel_layout(2) == "stereo"

    def test_surround_5_1(self):
        assert AudioExtractor._default_channel_layout(6) == "5.1"

    def test_surround_7_1(self):
        assert AudioExtractor._default_channel_layout(8) == "7.1"

    def test_unknown_channels(self):
        assert AudioExtractor._default_channel_layout(3) == "3 channels"
