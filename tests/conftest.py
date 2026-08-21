"""Shared test fixtures and configuration for the video profanity censor test suite."""

import pytest
from pathlib import Path


@pytest.fixture
def tmp_video_path(tmp_path: Path) -> Path:
    """Provides a temporary path for test video files."""
    return tmp_path / "test_video.mp4"


@pytest.fixture
def tmp_audio_path(tmp_path: Path) -> Path:
    """Provides a temporary path for test audio files."""
    return tmp_path / "test_audio.wav"


@pytest.fixture
def tmp_output_path(tmp_path: Path) -> Path:
    """Provides a temporary path for output files."""
    return tmp_path / "output.mp4"


@pytest.fixture
def sample_profanity_list() -> set[str]:
    """Provides a small sample profanity list for testing."""
    return {"damn", "hell", "shit", "fuck", "ass", "bitch", "crap"}
