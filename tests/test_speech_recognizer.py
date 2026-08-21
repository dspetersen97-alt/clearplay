"""Unit tests for SpeechRecognizer component.

Requirements: 2.1, 2.2, 2.6, 2.8, 9.8, 9.9, 9.10
"""

from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from video_profanity_censor.models import (
    AccelerationBackend,
    ProfanityRegion,
    SubtitleCue,
    TimestampRange,
    TranscribedWord,
    TranscriptionResult,
)
from video_profanity_censor.speech_recognizer import SpeechRecognizer


# --- Helpers for creating mock Whisper model outputs ---


def make_mock_segment(words):
    """Create a mock faster-whisper segment with word-level info."""
    segment = MagicMock()
    mock_words = []
    for w in words:
        word_info = MagicMock()
        word_info.word = w["word"]
        word_info.start = w["start"]
        word_info.end = w["end"]
        word_info.probability = w.get("probability", 0.95)
        mock_words.append(word_info)
    segment.words = mock_words
    return segment


def make_mock_segment_no_words():
    """Create a mock segment with no words (no speech detected)."""
    segment = MagicMock()
    segment.words = None
    return segment


# --- Test Classes ---


class TestNoSpeechResultHandling:
    """Test no-speech result handling (Req 2.6)."""

    def test_no_speech_detected_returns_empty_result(self, tmp_path):
        """When Whisper detects no speech, result should indicate no speech."""
        audio_file = tmp_path / "silence.wav"
        audio_file.write_bytes(b"\x00" * 100)

        recognizer = SpeechRecognizer(model_size="medium", backend=AccelerationBackend.CPU)

        # Mock the model loading and transcription
        mock_model = MagicMock()
        mock_info = MagicMock()
        # Return segments with no words (no speech)
        segments = [make_mock_segment_no_words()]
        mock_model.transcribe.return_value = (iter(segments), mock_info)

        # Pre-load the model with our mock
        recognizer._model = {
            "type": "cpu",
            "model_size": "medium",
            "whisper_model": mock_model,
        }

        result = recognizer.transcribe(audio_file)

        assert result.has_speech is False
        assert len(result.words) == 0
        assert any("no speech" in w.lower() for w in result.warnings)

    def test_no_speech_with_empty_segments(self, tmp_path):
        """When Whisper returns no segments at all, should report no speech."""
        audio_file = tmp_path / "empty.wav"
        audio_file.write_bytes(b"\x00" * 100)

        recognizer = SpeechRecognizer(model_size="medium", backend=AccelerationBackend.CPU)

        mock_model = MagicMock()
        mock_info = MagicMock()
        # Empty iterator - no segments at all
        mock_model.transcribe.return_value = (iter([]), mock_info)

        recognizer._model = {
            "type": "cpu",
            "model_size": "medium",
            "whisper_model": mock_model,
        }

        result = recognizer.transcribe(audio_file)

        assert result.has_speech is False
        assert len(result.words) == 0


class TestSegmentSkipBehavior:
    """Test segment skip behavior with warning (Req 2.8)."""

    def test_corrupted_region_skipped_with_warning(self, tmp_path):
        """When a region fails to transcribe, it should be skipped with a warning."""
        audio_file = tmp_path / "test.wav"
        audio_file.write_bytes(b"\x00" * 100)

        recognizer = SpeechRecognizer(model_size="medium", backend=AccelerationBackend.CPU)

        # Pre-load model
        mock_model = MagicMock()
        recognizer._model = {
            "type": "cpu",
            "model_size": "medium",
            "whisper_model": mock_model,
        }

        regions = [
            ProfanityRegion(
                start=10.0, end=15.0,
                source_cues=[SubtitleCue(index=1, start=12.0, end=13.0, text="bad")]
            ),
        ]

        # Mock subprocess to fail (simulating FFmpeg extraction failure for region)
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = RuntimeError("FFmpeg failed to extract region: decode error")
            result = recognizer._transcribe_regions(audio_file, regions)

        # Should have a warning about the skipped region
        assert len(result.warnings) > 0
        assert any("10.000" in w or "15.000" in w for w in result.warnings)
        # Should have the skipped range recorded
        assert len(result.skipped_ranges) == 1
        assert result.skipped_ranges[0].start == 10.0
        assert result.skipped_ranges[0].end == 15.0

    def test_multiple_regions_one_fails_others_continue(self, tmp_path):
        """When one region fails, other regions should still be processed."""
        audio_file = tmp_path / "test.wav"
        audio_file.write_bytes(b"\x00" * 100)

        recognizer = SpeechRecognizer(model_size="medium", backend=AccelerationBackend.CPU)

        mock_model = MagicMock()
        recognizer._model = {
            "type": "cpu",
            "model_size": "medium",
            "whisper_model": mock_model,
        }

        regions = [
            ProfanityRegion(
                start=5.0, end=10.0,
                source_cues=[SubtitleCue(index=1, start=6.0, end=8.0, text="word1")]
            ),
            ProfanityRegion(
                start=20.0, end=25.0,
                source_cues=[SubtitleCue(index=2, start=21.0, end=23.0, text="word2")]
            ),
        ]

        # Mock _transcribe_region_segment: first call succeeds, second raises
        good_result = TranscriptionResult(
            words=[TranscribedWord(word="hello", start=5.5, end=6.0, confidence=0.9)],
            has_speech=True,
        )

        with patch.object(recognizer, "_transcribe_region_segment") as mock_transcribe:
            mock_transcribe.side_effect = [
                good_result,
                RuntimeError("Corrupt audio segment"),
            ]
            result = recognizer._transcribe_regions(audio_file, regions)

        # First region's words should be present
        assert len(result.words) == 1
        assert result.words[0].word == "hello"
        # Should have a warning for the second region
        assert any("20.000" in w or "25.000" in w for w in result.warnings)
        # Should have a skipped range for the second region
        assert len(result.skipped_ranges) == 1
        assert result.skipped_ranges[0].start == 20.0


class TestRegionTargetedVsFullTranscription:
    """Test region-targeted transcription vs full transcription routing (Req 2.1, 2.2)."""

    def test_full_transcription_when_no_regions(self, tmp_path):
        """When no regions provided, full transcription should be used (Req 2.1)."""
        audio_file = tmp_path / "test.wav"
        audio_file.write_bytes(b"\x00" * 100)

        recognizer = SpeechRecognizer(model_size="medium", backend=AccelerationBackend.CPU)

        mock_model = MagicMock()
        mock_info = MagicMock()
        segments = [make_mock_segment([
            {"word": "hello", "start": 0.0, "end": 0.5},
            {"word": "world", "start": 0.6, "end": 1.0},
        ])]
        mock_model.transcribe.return_value = (iter(segments), mock_info)

        recognizer._model = {
            "type": "cpu",
            "model_size": "medium",
            "whisper_model": mock_model,
        }

        result = recognizer.transcribe(audio_file, regions=None)

        assert result.has_speech is True
        assert len(result.words) == 2
        assert result.words[0].word == "hello"
        assert result.words[1].word == "world"
        # Verify model.transcribe was called (full mode)
        mock_model.transcribe.assert_called_once()

    def test_targeted_transcription_when_regions_provided(self, tmp_path):
        """When regions provided, only those segments should be transcribed (Req 2.2)."""
        audio_file = tmp_path / "test.wav"
        audio_file.write_bytes(b"\x00" * 100)

        recognizer = SpeechRecognizer(model_size="medium", backend=AccelerationBackend.CPU)

        mock_model = MagicMock()
        recognizer._model = {
            "type": "cpu",
            "model_size": "medium",
            "whisper_model": mock_model,
        }

        regions = [
            ProfanityRegion(
                start=10.0, end=15.0,
                source_cues=[SubtitleCue(index=1, start=11.0, end=13.0, text="profanity")]
            ),
        ]

        # Mock _transcribe_region_segment to return specific result
        region_result = TranscriptionResult(
            words=[
                TranscribedWord(word="bad", start=11.0, end=11.5, confidence=0.9),
                TranscribedWord(word="word", start=11.6, end=12.0, confidence=0.88),
            ],
            has_speech=True,
        )

        with patch.object(recognizer, "_transcribe_region_segment", return_value=region_result):
            result = recognizer.transcribe(audio_file, regions=regions)

        assert result.has_speech is True
        assert len(result.words) == 2
        assert result.words[0].word == "bad"
        # Full transcription should NOT have been called
        mock_model.transcribe.assert_not_called()

    def test_regions_empty_list_uses_full_transcription(self, tmp_path):
        """An empty regions list should be treated the same as None (full mode)."""
        audio_file = tmp_path / "test.wav"
        audio_file.write_bytes(b"\x00" * 100)

        recognizer = SpeechRecognizer(model_size="medium", backend=AccelerationBackend.CPU)

        mock_model = MagicMock()
        mock_info = MagicMock()
        segments = [make_mock_segment([
            {"word": "test", "start": 0.0, "end": 0.4},
        ])]
        mock_model.transcribe.return_value = (iter(segments), mock_info)

        recognizer._model = {
            "type": "cpu",
            "model_size": "medium",
            "whisper_model": mock_model,
        }

        # Empty list is falsy in Python, should route to full transcription
        result = recognizer.transcribe(audio_file, regions=[])

        mock_model.transcribe.assert_called_once()
        assert result.has_speech is True


class TestModelLoadingCPU:
    """Test model loading for CPU backend (mock faster-whisper) (Req 9.8)."""

    def test_cpu_model_loads_successfully(self):
        """CPU model should load via faster-whisper WhisperModel."""
        recognizer = SpeechRecognizer(
            model_size="medium", backend=AccelerationBackend.CPU
        )

        mock_whisper_model = MagicMock()

        with patch(
            "video_profanity_censor.speech_recognizer.WhisperModel",
            return_value=mock_whisper_model,
            create=True,
        ) as mock_cls:
            # Patch the import mechanism
            mock_module = MagicMock()
            mock_module.WhisperModel = mock_cls
            with patch.dict("sys.modules", {"faster_whisper": mock_module}):
                with patch(
                    "builtins.__import__",
                    side_effect=lambda name, *args, **kwargs: (
                        mock_module if name == "faster_whisper"
                        else __builtins__.__import__(name, *args, **kwargs)
                    ),
                ):
                    recognizer._load_model_cpu("medium")

        assert recognizer._model is not None
        assert recognizer._model["type"] == "cpu"
        assert recognizer._model["model_size"] == "medium"
        assert recognizer._model["whisper_model"] == mock_whisper_model

    def test_cpu_raises_when_faster_whisper_not_installed(self):
        """Should raise RuntimeError when faster-whisper is not installed."""
        recognizer = SpeechRecognizer(
            model_size="medium", backend=AccelerationBackend.CPU
        )

        with patch(
            "builtins.__import__",
            side_effect=ImportError("No module named 'faster_whisper'"),
        ):
            with pytest.raises(RuntimeError, match="faster-whisper is not installed"):
                recognizer._load_model_cpu("medium")

    def test_cpu_model_uses_int8_compute_type(self):
        """CPU model should use int8 compute type for optimized inference."""
        recognizer = SpeechRecognizer(
            model_size="medium", backend=AccelerationBackend.CPU
        )

        mock_whisper_model = MagicMock()
        mock_whisper_cls = MagicMock(return_value=mock_whisper_model)

        mock_module = MagicMock()
        mock_module.WhisperModel = mock_whisper_cls

        with patch.dict("sys.modules", {"faster_whisper": mock_module}):
            with patch(
                "builtins.__import__",
                side_effect=lambda name, *args, **kwargs: (
                    mock_module if name == "faster_whisper"
                    else __builtins__.__import__(name, *args, **kwargs)
                ),
            ):
                recognizer._load_model_cpu("medium")

        # Verify WhisperModel was called with correct params
        mock_whisper_cls.assert_called_once_with(
            "medium",
            device="cpu",
            compute_type="int8",
        )


class TestMockWhisperPredictableOutput:
    """Mock Whisper model for predictable test output."""

    def test_transcription_returns_correct_word_timestamps(self, tmp_path):
        """Whisper output should be correctly mapped to TranscribedWord objects."""
        audio_file = tmp_path / "test.wav"
        audio_file.write_bytes(b"\x00" * 100)

        recognizer = SpeechRecognizer(model_size="medium", backend=AccelerationBackend.CPU)

        mock_model = MagicMock()
        mock_info = MagicMock()

        # Create predictable segment output
        words_data = [
            {"word": "this", "start": 0.0, "end": 0.3, "probability": 0.98},
            {"word": "is", "start": 0.35, "end": 0.5, "probability": 0.97},
            {"word": "a", "start": 0.55, "end": 0.6, "probability": 0.96},
            {"word": "test", "start": 0.65, "end": 1.0, "probability": 0.99},
        ]
        segments = [make_mock_segment(words_data)]
        mock_model.transcribe.return_value = (iter(segments), mock_info)

        recognizer._model = {
            "type": "cpu",
            "model_size": "medium",
            "whisper_model": mock_model,
        }

        result = recognizer.transcribe(audio_file)

        assert result.has_speech is True
        assert len(result.words) == 4
        assert result.words[0].word == "this"
        assert result.words[0].start == 0.0
        assert result.words[0].end == 0.3
        assert result.words[0].confidence == 0.98
        assert result.words[3].word == "test"
        assert result.words[3].start == 0.65
        assert result.words[3].end == 1.0

    def test_multi_segment_transcription(self, tmp_path):
        """Multiple segments should be combined into a single word list."""
        audio_file = tmp_path / "test.wav"
        audio_file.write_bytes(b"\x00" * 100)

        recognizer = SpeechRecognizer(model_size="medium", backend=AccelerationBackend.CPU)

        mock_model = MagicMock()
        mock_info = MagicMock()

        seg1 = make_mock_segment([
            {"word": "first", "start": 0.0, "end": 0.5},
            {"word": "segment", "start": 0.6, "end": 1.0},
        ])
        seg2 = make_mock_segment([
            {"word": "second", "start": 5.0, "end": 5.5},
            {"word": "segment", "start": 5.6, "end": 6.0},
        ])

        mock_model.transcribe.return_value = (iter([seg1, seg2]), mock_info)

        recognizer._model = {
            "type": "cpu",
            "model_size": "medium",
            "whisper_model": mock_model,
        }

        result = recognizer.transcribe(audio_file)

        assert result.has_speech is True
        assert len(result.words) == 4
        assert result.words[0].word == "first"
        assert result.words[2].word == "second"

    def test_empty_words_are_skipped(self, tmp_path):
        """Words with empty text after stripping should be skipped."""
        audio_file = tmp_path / "test.wav"
        audio_file.write_bytes(b"\x00" * 100)

        recognizer = SpeechRecognizer(model_size="medium", backend=AccelerationBackend.CPU)

        mock_model = MagicMock()
        mock_info = MagicMock()

        segments = [make_mock_segment([
            {"word": "hello", "start": 0.0, "end": 0.5},
            {"word": "  ", "start": 0.6, "end": 0.7},  # whitespace-only word
            {"word": "world", "start": 0.8, "end": 1.2},
        ])]
        mock_model.transcribe.return_value = (iter(segments), mock_info)

        recognizer._model = {
            "type": "cpu",
            "model_size": "medium",
            "whisper_model": mock_model,
        }

        result = recognizer.transcribe(audio_file)

        assert len(result.words) == 2
        assert result.words[0].word == "hello"
        assert result.words[1].word == "world"

    def test_audio_file_not_found_returns_no_speech(self, tmp_path):
        """If audio file doesn't exist, should return no-speech result with warning."""
        audio_file = tmp_path / "nonexistent.wav"

        recognizer = SpeechRecognizer(model_size="medium", backend=AccelerationBackend.CPU)

        result = recognizer.transcribe(audio_file)

        assert result.has_speech is False
        assert any("not found" in w.lower() for w in result.warnings)
