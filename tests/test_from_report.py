"""Tests for the detections-from-report bypass in CensorEngine and the CLI flag."""

from pathlib import Path
from unittest.mock import patch

import pytest

from video_profanity_censor.censor_engine import CensorEngine
from video_profanity_censor.models import (
    CensorMode,
    ProcessingResult,
    ValidationResult,
    AudioExtractionResult,
    AudioMetadata,
)


def _write_report(tmp_path) -> Path:
    p = tmp_path / "edited_report.txt"
    p.write_text(
        "Word                 Start          End            Action\n"
        "-----------------------------------------------------------\n"
        "manual               00:00:03.000   00:00:03.600   muted\n"
        "again                00:00:07.000   00:00:07.800   tone\n"
    )
    return p


class TestEngineBypass:
    def test_bypass_skips_transcription_and_uses_report_timings(self, tmp_path):
        report = _write_report(tmp_path)
        video = tmp_path / "clip.mkv"
        video.write_bytes(b"\x00")
        out = tmp_path / "out.mkv"

        meta = AudioMetadata(
            codec="eac3", sample_rate=48000, bit_depth=0, channels=6,
            channel_layout="5.1", bitrate=640000, duration_seconds=10.0, track_index=0,
        )

        engine = CensorEngine()
        captured = {}

        with patch.object(CensorEngine, "_stage_validate",
                          return_value=ValidationResult(is_valid=True)), \
             patch.object(CensorEngine, "_stage_extract_audio",
                          return_value=AudioExtractionResult(
                              output_path=tmp_path / "audio.wav", audio_metadata=meta)), \
             patch.object(CensorEngine, "_stage_transcribe",
                          side_effect=AssertionError("transcription must be skipped")), \
             patch.object(CensorEngine, "_stage_detect_backend",
                          side_effect=AssertionError("backend detection must be skipped")), \
             patch.object(CensorEngine, "_stage_censor_audio") as m_censor, \
             patch.object(CensorEngine, "_stage_assemble_output") as m_assemble, \
             patch.object(CensorEngine, "_generate_report"), \
             patch.object(CensorEngine, "_cleanup_temp_files"):

            from video_profanity_censor.models import CensorResult, AssemblyResult

            def fake_censor(audio_path, detection_result, censor_mode, progress_callback, **kw):
                captured["detections"] = detection_result.detections
                return CensorResult(
                    censored_audio_path=tmp_path / "censored.wav",
                    segments_censored=len(detection_result.detections),
                    total_censored_duration=1.0,
                )

            m_censor.side_effect = fake_censor
            m_assemble.return_value = AssemblyResult(
                output_path=out, container_format="matroska"
            )

            result = engine.process(
                input_path=video,
                output_path=out,
                censor_mode=CensorMode.MUTE,
                detections_path=report,
            )

        assert result.success
        # Two detections came from the report, with the report's timings.
        dets = captured["detections"]
        assert len(dets) == 2
        assert dets[0].timestamp_range.start == pytest.approx(3.0)
        assert dets[0].timestamp_range.end == pytest.approx(3.6)
        assert dets[0].censor_action == CensorMode.MUTE
        assert dets[1].censor_action == CensorMode.TONE
        # No transcription => no backend/model recorded.
        assert result.active_backend is None
        assert result.model_size_used is None

    def test_bad_report_fails_gracefully(self, tmp_path):
        bad = tmp_path / "bad.txt"
        bad.write_text("no detections here\n")
        video = tmp_path / "clip.mkv"
        video.write_bytes(b"\x00")

        engine = CensorEngine()
        with patch.object(CensorEngine, "_stage_validate",
                          return_value=ValidationResult(is_valid=True)), \
             patch.object(CensorEngine, "_stage_extract_audio",
                          return_value=AudioExtractionResult(
                              output_path=tmp_path / "a.wav",
                              audio_metadata=AudioMetadata(
                                  codec="aac", sample_rate=48000, bit_depth=16,
                                  channels=2, channel_layout="stereo", bitrate=128000,
                                  duration_seconds=5.0, track_index=0))), \
             patch.object(CensorEngine, "_cleanup_temp_files"):
            result = engine.process(
                input_path=video, detections_path=bad, censor_mode=CensorMode.MUTE
            )

        assert not result.success
        assert "detections" in (result.error_message or "").lower()


class TestMaxDurationCapBypass:
    """The per-word duration cap must be disabled for from-report timings (so a
    deliberately long edited window is applied verbatim) but kept for the normal
    transcription path (to defend against bad Whisper timestamps)."""

    def _run_and_capture_cap(self, tmp_path, detections_path):
        """Run process() far enough to capture the AudioProcessor's cap, then stop."""
        video = tmp_path / "clip.mkv"
        video.write_bytes(b"\x00")

        meta = AudioMetadata(
            codec="aac", sample_rate=48000, bit_depth=16, channels=2,
            channel_layout="stereo", bitrate=128000, duration_seconds=10.0,
            track_index=0,
        )
        captured = {}

        # Stop the run right after AudioProcessor is constructed by having the
        # patched class raise once we've recorded the cap it was built with.
        class _StopRun(Exception):
            pass

        def fake_ap(mode, max_word_duration_ms=750, **kw):
            captured["cap"] = max_word_duration_ms
            raise _StopRun

        from video_profanity_censor.models import DetectionResult, Detection, TimestampRange

        with patch.object(CensorEngine, "_stage_validate",
                          return_value=ValidationResult(is_valid=True)), \
             patch.object(CensorEngine, "_stage_extract_audio",
                          return_value=AudioExtractionResult(
                              output_path=tmp_path / "a.wav", audio_metadata=meta)), \
             patch.object(CensorEngine, "_stage_detect_backend") as m_backend, \
             patch.object(CensorEngine, "_stage_load_detections",
                          return_value=DetectionResult(detections=[
                              Detection("x", TimestampRange(2.0, 7.0), CensorMode.MUTE)])), \
             patch.object(CensorEngine, "_stage_transcribe"), \
             patch.object(CensorEngine, "_stage_detect_profanity",
                          return_value=DetectionResult(detections=[
                              Detection("x", TimestampRange(2.0, 7.0), CensorMode.MUTE)])), \
             patch.object(CensorEngine, "_stage_scan_subtitles"), \
             patch("video_profanity_censor.audio_processor.AudioProcessor", fake_ap), \
             patch.object(CensorEngine, "_cleanup_temp_files"):

            # Backend detect returns a minimal object for the transcription path.
            from video_profanity_censor.models import (
                BackendDetectionResult, AccelerationBackend,
            )
            m_backend.return_value = BackendDetectionResult(
                selected_backend=AccelerationBackend.CPU,
                selected_model_size="medium",
            )
            CensorEngine().process(
                input_path=video,
                censor_mode=CensorMode.MUTE,
                detections_path=detections_path,
                disable_subtitle_prefilter=True,
            )
        return captured.get("cap")

    def test_from_report_disables_cap(self, tmp_path):
        report = tmp_path / "r.txt"
        report.write_text("x 00:00:02.000 00:00:07.000 muted\n")
        cap = self._run_and_capture_cap(tmp_path, detections_path=report)
        assert cap == 0  # cap disabled for edited report timings

    def test_transcription_path_keeps_cap(self, tmp_path):
        cap = self._run_and_capture_cap(tmp_path, detections_path=None)
        assert cap == 750  # default cap preserved for Whisper timings


class TestCliFromReport:
    def test_from_report_forwarded(self, tmp_path):
        from video_profanity_censor.cli import main

        video = tmp_path / "m.mkv"
        video.write_bytes(b"\x00")
        report = tmp_path / "r.txt"
        report.write_text("w 00:00:01.000 00:00:01.500 muted\n")

        seen = {}

        def fake(self, **kwargs):
            seen["detections_path"] = kwargs.get("detections_path")
            return ProcessingResult(
                success=True, output_path=tmp_path / "o.mkv",
                profane_instances_censored=1,
            )

        with patch("video_profanity_censor.cli.CensorEngine.process", fake):
            rc = main([str(video), "--from-report", str(report)])

        assert rc == 0
        assert seen["detections_path"] == report

    def test_from_report_missing_file_errors(self, tmp_path):
        from video_profanity_censor.cli import main

        video = tmp_path / "m.mkv"
        video.write_bytes(b"\x00")
        rc = main([str(video), "--from-report", str(tmp_path / "nope.txt")])
        assert rc == 1

    def test_from_report_with_folder_errors(self, tmp_path):
        from video_profanity_censor.cli import main

        (tmp_path / "a.mkv").write_bytes(b"\x00")
        rc = main([str(tmp_path), "--from-report", "r.txt"])
        assert rc == 1
