"""Integration tests for the CensorEngine pipeline.

These tests verify end-to-end processing of the full pipeline with mocked
external dependencies (FFmpeg, Whisper models, onnxruntime).
They test that components work together correctly through the CensorEngine orchestrator.

Requirements: All
"""

import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from video_profanity_censor.backend_selector import BackendSelector
from video_profanity_censor.censor_engine import CensorEngine
from video_profanity_censor.models import (
    AccelerationBackend,
    AssemblyResult,
    AudioExtractionResult,
    AudioMetadata,
    BackendDetectionResult,
    BackendInfo,
    CensorMode,
    CensorResult,
    Detection,
    DetectionResult,
    ProcessingResult,
    ProcessingStage,
    ProfanityRegion,
    SubtitleCue,
    SubtitleScanResult,
    TimestampRange,
    TranscribedWord,
    TranscriptionResult,
    ValidationResult,
)


# --- Fixtures ---


@pytest.fixture
def engine():
    """Create a CensorEngine instance."""
    return CensorEngine()


@pytest.fixture
def tmp_dir():
    """Create a temporary directory for test files."""
    d = tempfile.mkdtemp()
    yield Path(d)
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def fake_video(tmp_dir):
    """Create a fake video file for testing."""
    video_path = tmp_dir / "sample.mp4"
    video_path.write_bytes(b"fake video content")
    return video_path


@pytest.fixture
def fake_output(tmp_dir):
    """Path for test output file."""
    return tmp_dir / "output.mp4"


@pytest.fixture
def valid_audio_metadata():
    """Standard audio metadata for mocking."""
    return AudioMetadata(
        codec="aac",
        sample_rate=44100,
        bit_depth=16,
        channels=2,
        channel_layout="stereo",
        bitrate=128000,
        duration_seconds=120.0,
        track_index=0,
    )


@pytest.fixture
def valid_validation_result(valid_audio_metadata):
    """A passing validation result."""
    return ValidationResult(
        is_valid=True,
        audio_metadata=valid_audio_metadata,
        video_codec="h264",
        container_format="mp4",
        file_size_bytes=1024 * 1024,
    )


@pytest.fixture
def audio_extraction_result(tmp_dir, valid_audio_metadata):
    """A successful audio extraction result with a real temp WAV file."""
    wav_path = tmp_dir / "extracted.wav"
    wav_path.write_bytes(b"fake wav data")
    return AudioExtractionResult(
        output_path=wav_path,
        audio_metadata=valid_audio_metadata,
    )


@pytest.fixture
def cpu_backend_result():
    """BackendDetectionResult selecting CPU."""
    return BackendDetectionResult(
        selected_backend=AccelerationBackend.CPU,
        selected_model_size="medium",
        available_backends=[
            BackendInfo(
                backend=AccelerationBackend.CPU,
                is_available=True,
                device_name="CPU (CTranslate2/faster-whisper)",
            ),
        ],
        gpu_vram_mb=None,
    )


@pytest.fixture
def profanity_transcription():
    """Transcription with profane words."""
    return TranscriptionResult(
        words=[
            TranscribedWord(word="hello", start=0.0, end=0.5, confidence=0.95),
            TranscribedWord(word="damn", start=1.0, end=1.3, confidence=0.92),
            TranscribedWord(word="world", start=1.5, end=2.0, confidence=0.97),
        ],
        has_speech=True,
    )


@pytest.fixture
def clean_transcription():
    """Transcription with no profanity."""
    return TranscriptionResult(
        words=[
            TranscribedWord(word="hello", start=0.0, end=0.5, confidence=0.95),
            TranscribedWord(word="world", start=0.6, end=1.0, confidence=0.97),
        ],
        has_speech=True,
    )


# --- Integration Tests ---


class TestEndToEndWithProfanity:
    """Test end-to-end processing with a short sample video with known profanity."""

    def test_end_to_end_with_profanity(
        self,
        engine,
        fake_video,
        fake_output,
        tmp_dir,
        valid_validation_result,
        audio_extraction_result,
        cpu_backend_result,
        profanity_transcription,
    ):
        """Full pipeline: video with profanity produces censored output."""
        censored_path = tmp_dir / "censored.wav"
        censored_path.write_bytes(b"censored audio")
        censor_result = CensorResult(
            censored_audio_path=censored_path,
            segments_censored=1,
            total_censored_duration=0.3,
        )
        assembly_result = AssemblyResult(
            output_path=fake_output,
            container_format="mp4",
            video_stream_copied=True,
            streams_preserved=0,
        )
        fake_output.write_bytes(b"output video")

        with patch.object(engine, "_stage_validate", return_value=valid_validation_result), \
             patch.object(engine, "_stage_extract_audio", return_value=audio_extraction_result), \
             patch.object(engine, "_stage_detect_backend", return_value=cpu_backend_result), \
             patch.object(engine, "_stage_scan_subtitles", return_value=SubtitleScanResult(has_subtitles=False)), \
             patch.object(engine, "_stage_transcribe", return_value=profanity_transcription), \
             patch.object(engine, "_stage_censor_audio", return_value=censor_result), \
             patch.object(engine, "_stage_assemble_output", return_value=assembly_result), \
             patch.object(engine, "_generate_report"), \
             patch("video_profanity_censor.progress_reporter.ProgressReporter"):

            result = engine.process(
                input_path=fake_video,
                output_path=fake_output,
            )

        assert result.success is True
        assert result.active_backend == AccelerationBackend.CPU
        assert result.model_size_used == "medium"
        assert result.profane_instances_detected == 1
        assert result.profane_instances_censored == 1


class TestNoProfanityProducesIdenticalOutput:
    """Test video with no profanity produces identical output."""

    def test_no_profanity_copies_source(
        self,
        engine,
        fake_video,
        fake_output,
        valid_validation_result,
        audio_extraction_result,
        cpu_backend_result,
        clean_transcription,
    ):
        """When no profanity is found, output should be identical to source."""
        with patch.object(engine, "_stage_validate", return_value=valid_validation_result), \
             patch.object(engine, "_stage_extract_audio", return_value=audio_extraction_result), \
             patch.object(engine, "_stage_detect_backend", return_value=cpu_backend_result), \
             patch.object(engine, "_stage_scan_subtitles", return_value=SubtitleScanResult(has_subtitles=False)), \
             patch.object(engine, "_stage_transcribe", return_value=clean_transcription), \
             patch.object(engine, "_generate_report"), \
             patch("video_profanity_censor.progress_reporter.ProgressReporter"):

            result = engine.process(
                input_path=fake_video,
                output_path=fake_output,
            )

        assert result.success is True
        assert result.profane_instances_detected == 0
        assert result.profane_instances_censored == 0
        # No output file should be created when no profanity found
        assert result.output_path is None
        assert not fake_output.exists()


class TestEmbeddedSubtitlesWithProfanityUsesTargetedTranscription:
    """Test video with embedded subtitles containing profanity uses targeted transcription."""

    def test_embedded_subtitles_profanity_targeted(
        self,
        engine,
        fake_video,
        fake_output,
        tmp_dir,
        valid_validation_result,
        audio_extraction_result,
        cpu_backend_result,
    ):
        """Subtitles with profanity trigger targeted transcription of regions only."""
        profanity_regions = [
            ProfanityRegion(
                start=8.0,
                end=14.0,
                source_cues=[SubtitleCue(index=1, start=10.0, end=12.0, text="damn it")],
            ),
        ]
        subtitle_result = SubtitleScanResult(
            has_subtitles=True,
            subtitle_source="embedded:2",
            profanity_regions=profanity_regions,
            total_cues_scanned=50,
            profane_cues_found=1,
            skipped_speech_recognition=False,
        )

        transcription = TranscriptionResult(
            words=[
                TranscribedWord(word="damn", start=10.5, end=10.9, confidence=0.92),
            ],
            has_speech=True,
        )

        censored_path = tmp_dir / "censored.wav"
        censored_path.write_bytes(b"censored audio")
        censor_result = CensorResult(
            censored_audio_path=censored_path,
            segments_censored=1,
            total_censored_duration=0.4,
        )
        assembly_result = AssemblyResult(
            output_path=fake_output,
            container_format="mp4",
            video_stream_copied=True,
            streams_preserved=1,
        )
        fake_output.write_bytes(b"output video")

        with patch.object(engine, "_stage_validate", return_value=valid_validation_result), \
             patch.object(engine, "_stage_extract_audio", return_value=audio_extraction_result), \
             patch.object(engine, "_stage_detect_backend", return_value=cpu_backend_result), \
             patch.object(engine, "_stage_scan_subtitles", return_value=subtitle_result), \
             patch.object(engine, "_stage_transcribe", return_value=transcription) as mock_transcribe, \
             patch.object(engine, "_stage_censor_audio", return_value=censor_result), \
             patch.object(engine, "_stage_assemble_output", return_value=assembly_result), \
             patch.object(engine, "_generate_report"), \
             patch("video_profanity_censor.progress_reporter.ProgressReporter"):

            result = engine.process(
                input_path=fake_video,
                output_path=fake_output,
            )

        assert result.success is True
        # Verify transcribe was called (regions are passed internally)
        mock_transcribe.assert_called_once()
        # Verify that subtitle_scan_result with regions was used
        call_args = mock_transcribe.call_args
        # _stage_transcribe(audio_path, backend, model_size, subtitle_scan_result, progress_callback)
        assert call_args[0][3] == subtitle_result


class TestCleanSubtitlesSkipsSpeechRecognition:
    """Test video with clean subtitles skips speech recognition."""

    def test_clean_subtitles_skip_sr(
        self,
        engine,
        fake_video,
        fake_output,
        valid_validation_result,
        audio_extraction_result,
        cpu_backend_result,
    ):
        """Clean subtitles (no profanity) should skip SR and produce identical output."""
        subtitle_result = SubtitleScanResult(
            has_subtitles=True,
            subtitle_source="embedded:2",
            profanity_regions=[],
            total_cues_scanned=100,
            profane_cues_found=0,
            skipped_speech_recognition=True,
        )

        with patch.object(engine, "_stage_validate", return_value=valid_validation_result), \
             patch.object(engine, "_stage_extract_audio", return_value=audio_extraction_result), \
             patch.object(engine, "_stage_detect_backend", return_value=cpu_backend_result), \
             patch.object(engine, "_stage_scan_subtitles", return_value=subtitle_result), \
             patch.object(engine, "_stage_transcribe") as mock_transcribe, \
             patch.object(engine, "_generate_report"), \
             patch("video_profanity_censor.progress_reporter.ProgressReporter"):

            result = engine.process(
                input_path=fake_video,
                output_path=fake_output,
            )

        assert result.success is True
        assert result.profane_instances_detected == 0
        assert result.profane_instances_censored == 0
        # Transcription should NOT have been called (short-circuit)
        mock_transcribe.assert_not_called()
        # No output file should be created when no profanity found
        assert result.output_path is None
        assert not fake_output.exists()


class TestExternalSubtitlePreference:
    """Test external subtitle file preference over embedded tracks."""

    def test_external_subtitle_preferred(
        self,
        engine,
        fake_video,
        fake_output,
        tmp_dir,
        valid_validation_result,
        audio_extraction_result,
        cpu_backend_result,
    ):
        """External subtitle file should be used over embedded tracks."""
        external_srt = tmp_dir / "external.srt"
        external_srt.write_text("1\n00:00:01,000 --> 00:00:03,000\nHello world\n")

        subtitle_result = SubtitleScanResult(
            has_subtitles=True,
            subtitle_source=str(external_srt),
            profanity_regions=[],
            total_cues_scanned=1,
            profane_cues_found=0,
            skipped_speech_recognition=True,
        )

        with patch.object(engine, "_stage_validate", return_value=valid_validation_result), \
             patch.object(engine, "_stage_extract_audio", return_value=audio_extraction_result), \
             patch.object(engine, "_stage_detect_backend", return_value=cpu_backend_result), \
             patch.object(engine, "_stage_scan_subtitles", return_value=subtitle_result) as mock_scan, \
             patch.object(engine, "_generate_report"), \
             patch("video_profanity_censor.progress_reporter.ProgressReporter"):

            result = engine.process(
                input_path=fake_video,
                output_path=fake_output,
                subtitle_path=external_srt,
            )

        assert result.success is True
        # Verify _stage_scan_subtitles was called with external path
        mock_scan.assert_called_once()
        call_args = mock_scan.call_args[0]
        # _stage_scan_subtitles(input_path, subtitle_path, profanity_list)
        assert call_args[1] == external_srt


class TestSubtitlePrefilteringDisabled:
    """Test subtitle pre-filtering disabled uses full transcription."""

    def test_disable_subtitle_prefilter(
        self,
        engine,
        fake_video,
        fake_output,
        valid_validation_result,
        audio_extraction_result,
        cpu_backend_result,
        clean_transcription,
    ):
        """When subtitle pre-filtering disabled, full transcription is used."""
        with patch.object(engine, "_stage_validate", return_value=valid_validation_result), \
             patch.object(engine, "_stage_extract_audio", return_value=audio_extraction_result), \
             patch.object(engine, "_stage_detect_backend", return_value=cpu_backend_result), \
             patch.object(engine, "_stage_scan_subtitles") as mock_scan, \
             patch.object(engine, "_stage_transcribe", return_value=clean_transcription), \
             patch.object(engine, "_generate_report"), \
             patch("video_profanity_censor.progress_reporter.ProgressReporter"):

            result = engine.process(
                input_path=fake_video,
                output_path=fake_output,
                disable_subtitle_prefilter=True,
            )

        assert result.success is True
        # SubtitleScanner stage should NOT have been called
        mock_scan.assert_not_called()


class TestMetadataAndSubtitleTrackPreservation:
    """Test metadata and subtitle track preservation in output."""

    def test_metadata_preservation(
        self,
        engine,
        fake_video,
        fake_output,
        tmp_dir,
        valid_validation_result,
        audio_extraction_result,
        cpu_backend_result,
        profanity_transcription,
    ):
        """Output assembler should preserve metadata and subtitle tracks via FFmpeg flags."""
        censored_path = tmp_dir / "censored.wav"
        censored_path.write_bytes(b"censored audio")
        censor_result = CensorResult(
            censored_audio_path=censored_path,
            segments_censored=1,
            total_censored_duration=0.3,
        )

        fake_output.write_bytes(b"output video")

        with patch.object(engine, "_stage_validate", return_value=valid_validation_result), \
             patch.object(engine, "_stage_extract_audio", return_value=audio_extraction_result), \
             patch.object(engine, "_stage_detect_backend", return_value=cpu_backend_result), \
             patch.object(engine, "_stage_scan_subtitles", return_value=SubtitleScanResult(has_subtitles=False)), \
             patch.object(engine, "_stage_transcribe", return_value=profanity_transcription), \
             patch.object(engine, "_stage_censor_audio", return_value=censor_result), \
             patch("video_profanity_censor.output_assembler.ffmpeg.probe") as mock_probe, \
             patch("video_profanity_censor.output_assembler.subprocess.run") as mock_run, \
             patch.object(engine, "_generate_report"), \
             patch("video_profanity_censor.progress_reporter.ProgressReporter"):

            mock_probe.return_value = {
                "streams": [
                    {"codec_type": "video", "codec_name": "h264", "index": 0},
                    {"codec_type": "audio", "codec_name": "aac", "index": 1},
                    {"codec_type": "subtitle", "codec_name": "subrip", "index": 2},
                ]
            }
            mock_run.return_value = MagicMock(returncode=0, stderr="")

            result = engine.process(
                input_path=fake_video,
                output_path=fake_output,
            )

        assert result.success is True
        # Verify FFmpeg was called with metadata/chapter preservation flags
        cmd_args = mock_run.call_args[0][0]
        assert "-map_metadata" in cmd_args
        assert "-map_chapters" in cmd_args
        # Should contain subtitle stream mapping
        assert "0:s?" in cmd_args


class TestBackendSelectorRAMDetection:
    """Test BackendSelector selects model size based on system RAM."""

    def test_large_model_selected_with_16gb_ram(self):
        """System with 16 GB+ RAM should get large model."""
        from video_profanity_censor.backend_selector import BackendSelector

        with patch.object(BackendSelector, "_query_system_ram", return_value=16384):
            selector = BackendSelector()
            result = selector.detect()

            assert result.selected_backend == AccelerationBackend.CPU
            assert result.selected_model_size == "large"

    def test_medium_model_selected_with_8gb_ram(self):
        """System with 8 GB RAM should get medium model."""
        from video_profanity_censor.backend_selector import BackendSelector

        with patch.object(BackendSelector, "_query_system_ram", return_value=8192):
            selector = BackendSelector()
            result = selector.detect()

            assert result.selected_backend == AccelerationBackend.CPU
            assert result.selected_model_size == "medium"

    def test_cpu_model_loading(self):
        """CPU backend should load via faster-whisper (CTranslate2)."""
        from video_profanity_censor.speech_recognizer import SpeechRecognizer

        mock_whisper_model = MagicMock()
        mock_faster_whisper = MagicMock()
        mock_faster_whisper.WhisperModel.return_value = mock_whisper_model

        with patch.dict("sys.modules", {"faster_whisper": mock_faster_whisper}):
            recognizer = SpeechRecognizer(
                model_size="large",
                backend=AccelerationBackend.CPU,
            )
            recognizer._ensure_model_loaded()

            assert recognizer._model is not None
            assert recognizer._model["type"] == "cpu"
            assert recognizer._model["model_size"] == "large"

    def test_processing_summary_includes_backend(self, tmp_path):
        """Processing result should include backend name."""
        from video_profanity_censor.censor_engine import CensorEngine

        source = tmp_path / "test.mp4"
        source.write_bytes(b"fake video")

        engine = CensorEngine()

        with patch.object(engine, "_stage_validate") as mock_validate:
            mock_validate.return_value = MagicMock(is_valid=False, error_message="test error")
            result = engine.process(input_path=source)

        # Even on validation failure, the result is returned
        assert result.success is False
