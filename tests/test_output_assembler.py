"""Unit tests for the OutputAssembler component."""

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from video_profanity_censor.models import AssemblyResult
from video_profanity_censor.output_assembler import (
    CONTAINER_FORMAT_MAP,
    SUPPORTED_AUDIO_CODECS,
    OutputAssembler,
)


@pytest.fixture
def assembler():
    """Create an OutputAssembler instance."""
    return OutputAssembler()


class TestAssembleFileValidation:
    """Tests for file existence validation."""

    def test_source_file_not_found(self, assembler, tmp_path):
        """Should raise FileNotFoundError when source doesn't exist."""
        fake_source = tmp_path / "nonexistent.mp4"
        censored = tmp_path / "censored.wav"
        censored.write_bytes(b"fake audio data")
        output = tmp_path / "output.mp4"

        with pytest.raises(FileNotFoundError, match="Source file not found"):
            assembler.assemble(fake_source, censored, output)

    def test_censored_audio_not_found(self, assembler, tmp_path):
        """Should raise FileNotFoundError when censored audio doesn't exist."""
        source = tmp_path / "source.mp4"
        source.write_bytes(b"fake video data")
        fake_censored = tmp_path / "nonexistent.wav"
        output = tmp_path / "output.mp4"

        with pytest.raises(FileNotFoundError, match="Censored audio file not found"):
            assembler.assemble(source, fake_censored, output)


class TestContainerFormat:
    """Tests for container format detection."""

    def test_mp4_format(self, assembler):
        """Should detect mp4 container format."""
        assert assembler._get_container_format(Path("video.mp4")) == "mp4"

    def test_mkv_format(self, assembler):
        """Should detect matroska container format for .mkv."""
        assert assembler._get_container_format(Path("video.mkv")) == "matroska"

    def test_avi_format(self, assembler):
        """Should detect avi container format."""
        assert assembler._get_container_format(Path("video.avi")) == "avi"

    def test_mov_format(self, assembler):
        """Should detect mov container format."""
        assert assembler._get_container_format(Path("video.mov")) == "mov"

    def test_wmv_format(self, assembler):
        """Should detect asf container format for .wmv."""
        assert assembler._get_container_format(Path("video.wmv")) == "asf"

    def test_case_insensitive(self, assembler):
        """Should handle uppercase extensions."""
        assert assembler._get_container_format(Path("video.MP4")) == "mp4"
        assert assembler._get_container_format(Path("video.MKV")) == "matroska"

    def test_unknown_format_fallback(self, assembler):
        """Should fall back to extension without dot for unknown formats."""
        assert assembler._get_container_format(Path("video.webm")) == "webm"


class TestAudioCodecValidation:
    """Tests for audio codec validation."""

    def test_supported_codec_passes(self, assembler):
        """Should return None for supported codecs (no fallback needed)."""
        for codec in SUPPORTED_AUDIO_CODECS:
            assert assembler._validate_audio_codec(codec) is None

    def test_unsupported_codec_returns_fallback(self, assembler):
        """Should return fallback codec for unsupported codec."""
        result = assembler._validate_audio_codec("dts_hd_ma")
        assert result == "ac3"

    def test_unsupported_codec_high_channel_returns_eac3(self, assembler):
        """Should return eac3 fallback when source has more than 6 channels."""
        result = assembler._validate_audio_codec("dts_hd_ma", channels=8)
        assert result == "eac3"

    def test_empty_codec_passes(self, assembler):
        """Should return None for empty codec string (unknown/missing)."""
        assert assembler._validate_audio_codec("") is None


class TestStreamCounting:
    """Tests for preserved stream counting."""

    def test_single_video_single_audio(self, assembler):
        """With only video and one audio track, no additional streams preserved."""
        streams = [
            {"codec_type": "video"},
            {"codec_type": "audio"},
        ]
        assert assembler._count_preserved_streams(streams, 0) == 0

    def test_subtitle_stream_counted(self, assembler):
        """Should count subtitle streams as preserved."""
        streams = [
            {"codec_type": "video"},
            {"codec_type": "audio"},
            {"codec_type": "subtitle"},
        ]
        assert assembler._count_preserved_streams(streams, 0) == 1

    def test_multiple_audio_tracks(self, assembler):
        """Should count non-replaced audio tracks as preserved."""
        streams = [
            {"codec_type": "video"},
            {"codec_type": "audio"},  # Track 0 - being replaced
            {"codec_type": "audio"},  # Track 1 - preserved
            {"codec_type": "audio"},  # Track 2 - preserved
        ]
        assert assembler._count_preserved_streams(streams, 0) == 2

    def test_mixed_streams(self, assembler):
        """Should count all non-video non-target-audio streams."""
        streams = [
            {"codec_type": "video"},
            {"codec_type": "audio"},  # Track 0 - being replaced
            {"codec_type": "audio"},  # Track 1 - preserved
            {"codec_type": "subtitle"},
            {"codec_type": "subtitle"},
            {"codec_type": "data"},
        ]
        assert assembler._count_preserved_streams(streams, 0) == 4

    def test_replacing_non_first_track(self, assembler):
        """Should correctly count when replacing a non-first audio track."""
        streams = [
            {"codec_type": "video"},
            {"codec_type": "audio"},  # Track 0 - preserved
            {"codec_type": "audio"},  # Track 1 - being replaced
        ]
        assert assembler._count_preserved_streams(streams, 1) == 1


class TestAssemblyWithMocks:
    """Tests for the full assembly process with mocked FFmpeg calls."""

    @patch("video_profanity_censor.output_assembler.subprocess.run")
    @patch("video_profanity_censor.output_assembler.ffmpeg.probe")
    def test_successful_assembly(self, mock_probe, mock_run, assembler, tmp_path):
        """Should return AssemblyResult on successful assembly."""
        source = tmp_path / "source.mp4"
        source.write_bytes(b"fake video")
        censored = tmp_path / "censored.wav"
        censored.write_bytes(b"fake audio")
        output = tmp_path / "output.mp4"

        mock_probe.return_value = {
            "streams": [
                {"codec_type": "video", "codec_name": "h264", "index": 0},
                {"codec_type": "audio", "codec_name": "aac", "index": 1},
            ],
            "format": {"format_name": "mov,mp4,m4a,3gp,3g2,mj2"},
        }
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="", stderr=""
        )

        result = assembler.assemble(source, censored, output)

        assert isinstance(result, AssemblyResult)
        assert result.output_path == output
        assert result.container_format == "mp4"
        assert result.video_stream_copied is True
        assert result.streams_preserved == 0

    @patch("video_profanity_censor.output_assembler.subprocess.run")
    @patch("video_profanity_censor.output_assembler.ffmpeg.probe")
    def test_preserves_subtitle_streams(self, mock_probe, mock_run, assembler, tmp_path):
        """Should preserve subtitle streams and count them."""
        source = tmp_path / "source.mkv"
        source.write_bytes(b"fake video")
        censored = tmp_path / "censored.wav"
        censored.write_bytes(b"fake audio")
        output = tmp_path / "output.mkv"

        mock_probe.return_value = {
            "streams": [
                {"codec_type": "video", "codec_name": "h264", "index": 0},
                {"codec_type": "audio", "codec_name": "aac", "index": 1},
                {"codec_type": "subtitle", "codec_name": "subrip", "index": 2},
                {"codec_type": "subtitle", "codec_name": "ass", "index": 3},
            ],
            "format": {"format_name": "matroska,webm"},
        }
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="", stderr=""
        )

        result = assembler.assemble(source, censored, output)

        assert result.container_format == "matroska"
        assert result.streams_preserved == 2

    @patch("video_profanity_censor.output_assembler.subprocess.run")
    @patch("video_profanity_censor.output_assembler.ffmpeg.probe")
    def test_video_copy_flag_in_command(self, mock_probe, mock_run, assembler, tmp_path):
        """Should use -c:v copy to avoid video re-encoding."""
        source = tmp_path / "source.mp4"
        source.write_bytes(b"fake video")
        censored = tmp_path / "censored.wav"
        censored.write_bytes(b"fake audio")
        output = tmp_path / "output.mp4"

        mock_probe.return_value = {
            "streams": [
                {"codec_type": "video", "codec_name": "h264", "index": 0},
                {"codec_type": "audio", "codec_name": "aac", "index": 1},
            ],
            "format": {},
        }
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="", stderr=""
        )

        assembler.assemble(source, censored, output)

        # Verify the command includes -c:v copy
        call_args = mock_run.call_args[0][0]
        assert "-c:v" in call_args
        cv_idx = call_args.index("-c:v")
        assert call_args[cv_idx + 1] == "copy"

    @patch("video_profanity_censor.output_assembler.subprocess.run")
    @patch("video_profanity_censor.output_assembler.ffmpeg.probe")
    def test_metadata_preservation_flags(self, mock_probe, mock_run, assembler, tmp_path):
        """Should include -map_metadata and -map_chapters flags."""
        source = tmp_path / "source.mp4"
        source.write_bytes(b"fake video")
        censored = tmp_path / "censored.wav"
        censored.write_bytes(b"fake audio")
        output = tmp_path / "output.mp4"

        mock_probe.return_value = {
            "streams": [
                {"codec_type": "video", "codec_name": "h264", "index": 0},
                {"codec_type": "audio", "codec_name": "aac", "index": 1},
            ],
            "format": {},
        }
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="", stderr=""
        )

        assembler.assemble(source, censored, output)

        call_args = mock_run.call_args[0][0]
        assert "-map_metadata" in call_args
        assert "-map_chapters" in call_args
        # Verify they reference input 0 (the source)
        meta_idx = call_args.index("-map_metadata")
        assert call_args[meta_idx + 1] == "0"
        chap_idx = call_args.index("-map_chapters")
        assert call_args[chap_idx + 1] == "0"

    @patch("video_profanity_censor.output_assembler.subprocess.run")
    @patch("video_profanity_censor.output_assembler.ffmpeg.probe")
    def test_replaces_correct_audio_track(self, mock_probe, mock_run, assembler, tmp_path):
        """Should map censored audio for the target track and keep others."""
        source = tmp_path / "source.mkv"
        source.write_bytes(b"fake video")
        censored = tmp_path / "censored.wav"
        censored.write_bytes(b"fake audio")
        output = tmp_path / "output.mkv"

        mock_probe.return_value = {
            "streams": [
                {"codec_type": "video", "codec_name": "h264", "index": 0},
                {"codec_type": "audio", "codec_name": "aac", "index": 1},
                {"codec_type": "audio", "codec_name": "ac3", "index": 2},
            ],
            "format": {},
        }
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="", stderr=""
        )

        # Replace track 0 (first audio)
        assembler.assemble(source, censored, output, audio_track_index=0)

        call_args = mock_run.call_args[0][0]
        # Should map 1:a:0 for the replaced track, and 0:a:1 for the other
        assert "1:a:0" in call_args
        assert "0:a:1" in call_args

    @patch("video_profanity_censor.output_assembler.subprocess.run")
    @patch("video_profanity_censor.output_assembler.ffmpeg.probe")
    def test_replaces_second_audio_track(self, mock_probe, mock_run, assembler, tmp_path):
        """Should correctly replace the second audio track when index=1."""
        source = tmp_path / "source.mkv"
        source.write_bytes(b"fake video")
        censored = tmp_path / "censored.wav"
        censored.write_bytes(b"fake audio")
        output = tmp_path / "output.mkv"

        mock_probe.return_value = {
            "streams": [
                {"codec_type": "video", "codec_name": "h264", "index": 0},
                {"codec_type": "audio", "codec_name": "aac", "index": 1},
                {"codec_type": "audio", "codec_name": "ac3", "index": 2},
            ],
            "format": {},
        }
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="", stderr=""
        )

        # Replace track 1 (second audio)
        assembler.assemble(source, censored, output, audio_track_index=1)

        call_args = mock_run.call_args[0][0]
        # Should map 0:a:0 for first (kept), 1:a:0 for second (replaced)
        assert "0:a:0" in call_args
        assert "1:a:0" in call_args

    @patch("video_profanity_censor.output_assembler.subprocess.run")
    @patch("video_profanity_censor.output_assembler.ffmpeg.probe")
    def test_unsupported_codec_uses_fallback(self, mock_probe, mock_run, assembler, tmp_path):
        """Should use fallback codec (ac3) for unsupported audio codec instead of raising."""
        source = tmp_path / "source.mp4"
        source.write_bytes(b"fake video")
        censored = tmp_path / "censored.wav"
        censored.write_bytes(b"fake audio")
        output = tmp_path / "output.mp4"

        mock_probe.return_value = {
            "streams": [
                {"codec_type": "video", "codec_name": "h264", "index": 0},
                {"codec_type": "audio", "codec_name": "dts_hd_ma", "channels": 6, "index": 1},
            ],
            "format": {},
        }

        mock_run.return_value = MagicMock(returncode=0, stderr="")

        result = assembler.assemble(source, censored, output)
        assert result.codec_fallback == "ac3"

        # Verify the FFmpeg command includes the fallback codec
        call_args = mock_run.call_args[0][0]
        # Find the -c:a:0 argument and verify it's "ac3"
        for i, arg in enumerate(call_args):
            if arg == "-c:a:0":
                assert call_args[i + 1] == "ac3"
                break

    @patch("video_profanity_censor.output_assembler.subprocess.run")
    @patch("video_profanity_censor.output_assembler.ffmpeg.probe")
    def test_ffmpeg_failure_raises_runtime_error(self, mock_probe, mock_run, assembler, tmp_path):
        """Should raise RuntimeError when FFmpeg command fails."""
        source = tmp_path / "source.mp4"
        source.write_bytes(b"fake video")
        censored = tmp_path / "censored.wav"
        censored.write_bytes(b"fake audio")
        output = tmp_path / "output.mp4"

        mock_probe.return_value = {
            "streams": [
                {"codec_type": "video", "codec_name": "h264", "index": 0},
                {"codec_type": "audio", "codec_name": "aac", "index": 1},
            ],
            "format": {},
        }
        mock_run.side_effect = subprocess.CalledProcessError(
            returncode=1, cmd="ffmpeg", stderr="Encoding failed"
        )

        with pytest.raises(RuntimeError, match="FFmpeg output assembly failed"):
            assembler.assemble(source, censored, output)

    @patch("video_profanity_censor.output_assembler.subprocess.run")
    @patch("video_profanity_censor.output_assembler.ffmpeg.probe")
    def test_output_format_matches_source(self, mock_probe, mock_run, assembler, tmp_path):
        """Should use the same container format as source (Req 5.4)."""
        for ext, expected_fmt in CONTAINER_FORMAT_MAP.items():
            source = tmp_path / f"source{ext}"
            source.write_bytes(b"fake video")
            censored = tmp_path / "censored.wav"
            censored.write_bytes(b"fake audio")
            output = tmp_path / f"output{ext}"

            mock_probe.return_value = {
                "streams": [
                    {"codec_type": "video", "codec_name": "h264", "index": 0},
                    {"codec_type": "audio", "codec_name": "aac", "index": 1},
                ],
                "format": {},
            }
            mock_run.return_value = subprocess.CompletedProcess(
                args=[], returncode=0, stdout="", stderr=""
            )

            result = assembler.assemble(source, censored, output)
            assert result.container_format == expected_fmt

            # Verify the -f flag uses the correct format
            call_args = mock_run.call_args[0][0]
            f_idx = call_args.index("-f")
            assert call_args[f_idx + 1] == expected_fmt
