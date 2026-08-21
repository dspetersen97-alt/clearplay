"""Tests for the CLI entry point."""

import pytest

from video_profanity_censor.cli import (
    _resolve_backend,
    _resolve_censor_mode,
    main,
    parse_args,
)
from video_profanity_censor.models import AccelerationBackend, CensorMode


class TestParseArgs:
    """Tests for argument parsing."""

    def test_required_input_file(self):
        """Input file is required positional argument."""
        args = parse_args(["movie.mp4"])
        assert args.input == "movie.mp4"

    def test_default_values(self):
        """All optional arguments have sensible defaults."""
        args = parse_args(["movie.mp4"])
        assert args.output is None
        assert args.audio_track == 0
        assert args.mode == "mute"
        assert args.backend == "auto"
        assert args.profanity_list is None
        assert args.report_path is None
        assert args.subtitle_path is None
        assert args.disable_subtitle_prefilter is False
        assert args.model_size is None

    def test_output_path(self):
        """Output path can be specified with --output or -o."""
        args = parse_args(["movie.mp4", "--output", "out.mp4"])
        assert args.output == "out.mp4"

        args = parse_args(["movie.mp4", "-o", "out.mp4"])
        assert args.output == "out.mp4"

    def test_audio_track_index(self):
        """Audio track index can be specified."""
        args = parse_args(["movie.mp4", "--audio-track", "2"])
        assert args.audio_track == 2

    def test_censor_mode_tone(self):
        """Mode accepts 'tone'."""
        args = parse_args(["movie.mp4", "--mode", "tone"])
        assert args.mode == "tone"

    def test_censor_mode_mute(self):
        """Mode accepts 'mute'."""
        args = parse_args(["movie.mp4", "--mode", "mute"])
        assert args.mode == "mute"

    def test_invalid_mode_rejected(self):
        """Invalid mode values are rejected."""
        with pytest.raises(SystemExit):
            parse_args(["movie.mp4", "--mode", "beep"])

    def test_profanity_list_path(self):
        """Custom profanity list path can be specified."""
        args = parse_args(["movie.mp4", "--profanity-list", "custom.txt"])
        assert args.profanity_list == "custom.txt"

    def test_report_path(self):
        """Report path can be specified."""
        args = parse_args(["movie.mp4", "--report-path", "report.txt"])
        assert args.report_path == "report.txt"

    def test_subtitle_path(self):
        """External subtitle path can be specified."""
        args = parse_args(["movie.mp4", "--subtitle-path", "subs.srt"])
        assert args.subtitle_path == "subs.srt"

    def test_disable_subtitle_prefilter_flag(self):
        """Subtitle pre-filtering can be disabled."""
        args = parse_args(["movie.mp4", "--disable-subtitle-prefilter"])
        assert args.disable_subtitle_prefilter is True

    def test_model_size_choices(self):
        """Model size accepts valid Whisper model sizes."""
        for size in ["tiny", "base", "small", "medium", "large"]:
            args = parse_args(["movie.mp4", "--model-size", size])
            assert args.model_size == size

    def test_invalid_model_size_rejected(self):
        """Invalid model size values are rejected."""
        with pytest.raises(SystemExit):
            parse_args(["movie.mp4", "--model-size", "huge"])

    def test_missing_input_fails(self):
        """Missing input file causes argparse to exit."""
        with pytest.raises(SystemExit):
            parse_args([])

    def test_backend_choices(self):
        """Backend accepts valid choices."""
        for backend in ["auto", "directml", "cpu"]:
            args = parse_args(["movie.mp4", "--backend", backend])
            assert args.backend == backend

    def test_invalid_backend_rejected(self):
        """Invalid backend values are rejected."""
        with pytest.raises(SystemExit):
            parse_args(["movie.mp4", "--backend", "cuda"])

    def test_all_arguments_together(self):
        """All arguments can be specified together."""
        args = parse_args([
            "movie.mp4",
            "--output", "out.mp4",
            "--audio-track", "1",
            "--mode", "mute",
            "--backend", "directml",
            "--profanity-list", "words.txt",
            "--report-path", "report.txt",
            "--subtitle-path", "subs.srt",
            "--disable-subtitle-prefilter",
            "--model-size", "large",
        ])
        assert args.input == "movie.mp4"
        assert args.output == "out.mp4"
        assert args.audio_track == 1
        assert args.mode == "mute"
        assert args.backend == "directml"
        assert args.profanity_list == "words.txt"
        assert args.report_path == "report.txt"
        assert args.subtitle_path == "subs.srt"
        assert args.disable_subtitle_prefilter is True
        assert args.model_size == "large"


class TestResolveCensorMode:
    """Tests for _resolve_censor_mode helper."""

    def test_tone_mode(self):
        assert _resolve_censor_mode("tone") == CensorMode.TONE

    def test_mute_mode(self):
        assert _resolve_censor_mode("mute") == CensorMode.MUTE


class TestResolveBackend:
    """Tests for _resolve_backend helper."""

    def test_auto_returns_none(self):
        assert _resolve_backend("auto") is None

    def test_directml_returns_directml(self):
        assert _resolve_backend("directml") == AccelerationBackend.DIRECTML

    def test_cpu_returns_cpu(self):
        assert _resolve_backend("cpu") == AccelerationBackend.CPU


class TestMainFunction:
    """Tests for the main() function."""

    def test_nonexistent_file_returns_failure(self):
        """Processing a nonexistent file should return exit code 1."""
        exit_code = main(["nonexistent_file_xyz.mp4"])
        assert exit_code == 1
