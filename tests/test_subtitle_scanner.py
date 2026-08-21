"""Unit tests for the SubtitleScanner component.

Tests cover:
- Embedded subtitle track discovery via FFprobe (Req 8.1)
- External SRT file parsing with valid content (Req 8.2)
- External ASS/SSA file parsing with valid content (Req 8.2)
- External file preference over embedded tracks (Req 8.8)
- No subtitles available returns has_subtitles=False (Req 8.6)
- Clean subtitles (no profanity) sets skipped_speech_recognition (Req 8.7)
- Malformed subtitle file handling
"""

import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
from nltk.stem import SnowballStemmer

from video_profanity_censor.models import ProfanityList, SubtitleScanResult
from video_profanity_censor.subtitle_scanner import SubtitleScanner


_stemmer = SnowballStemmer("english")


def _make_profanity_list(words: set[str]) -> ProfanityList:
    """Create a ProfanityList with pre-computed stems from a set of words."""
    lower_words = {w.lower() for w in words}
    stems = {_stemmer.stem(w) for w in lower_words}
    return ProfanityList(words=lower_words, stems=stems, source_path=None)


class TestEmbeddedSubtitleTrackDiscovery:
    """Tests for embedded subtitle track discovery via FFprobe (Req 8.1)."""

    def test_discovers_embedded_subtitle_tracks(self, tmp_path: Path):
        """FFprobe output with subtitle streams is correctly parsed into SubtitleTrackInfo."""
        profanity_list = _make_profanity_list({"damn"})
        scanner = SubtitleScanner(profanity_list=profanity_list)

        ffprobe_output = json.dumps({
            "streams": [
                {
                    "index": 2,
                    "codec_name": "subrip",
                    "tags": {"language": "eng"},
                },
                {
                    "index": 3,
                    "codec_name": "ass",
                    "tags": {"language": "fra"},
                },
            ]
        })

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = ffprobe_output

        video_path = tmp_path / "video.mkv"
        video_path.write_bytes(b"fake video content")

        with patch("subprocess.run", return_value=mock_result) as mock_run:
            tracks = scanner._find_subtitle_tracks(video_path)

        assert len(tracks) == 2
        assert tracks[0].track_index == 2
        assert tracks[0].language == "eng"
        assert tracks[0].codec == "subrip"
        assert tracks[0].is_embedded is True
        assert tracks[1].track_index == 3
        assert tracks[1].language == "fra"
        assert tracks[1].codec == "ass"

    def test_no_embedded_tracks_returns_empty(self, tmp_path: Path):
        """FFprobe output with no subtitle streams returns an empty list."""
        profanity_list = _make_profanity_list({"damn"})
        scanner = SubtitleScanner(profanity_list=profanity_list)

        ffprobe_output = json.dumps({"streams": []})
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = ffprobe_output

        video_path = tmp_path / "video.mp4"
        video_path.write_bytes(b"fake video content")

        with patch("subprocess.run", return_value=mock_result):
            tracks = scanner._find_subtitle_tracks(video_path)

        assert tracks == []

    def test_ffprobe_failure_returns_empty(self, tmp_path: Path):
        """When FFprobe returns a non-zero exit code, returns empty list."""
        profanity_list = _make_profanity_list({"damn"})
        scanner = SubtitleScanner(profanity_list=profanity_list)

        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = ""

        video_path = tmp_path / "video.mp4"
        video_path.write_bytes(b"fake video content")

        with patch("subprocess.run", return_value=mock_result):
            tracks = scanner._find_subtitle_tracks(video_path)

        assert tracks == []

    def test_scan_with_embedded_track_containing_profanity(self, tmp_path: Path):
        """Full scan flow with embedded subtitles containing profanity."""
        profanity_list = _make_profanity_list({"damn"})
        scanner = SubtitleScanner(profanity_list=profanity_list)

        # Mock FFprobe to return one subtitle track
        ffprobe_output = json.dumps({
            "streams": [
                {"index": 2, "codec_name": "subrip", "tags": {"language": "eng"}}
            ]
        })

        # Mock FFmpeg extraction to return SRT content with profanity
        srt_content = (
            "1\n"
            "00:01:00,000 --> 00:01:02,000\n"
            "What the damn is going on?\n"
        )

        video_path = tmp_path / "video.mkv"
        video_path.write_bytes(b"fake video")

        def mock_subprocess_run(cmd, **kwargs):
            result = MagicMock()
            if "ffprobe" in cmd[0]:
                result.returncode = 0
                result.stdout = ffprobe_output
            elif "ffmpeg" in cmd[0]:
                result.returncode = 0
                result.stdout = srt_content
            else:
                result.returncode = 1
                result.stdout = ""
            return result

        with patch("subprocess.run", side_effect=mock_subprocess_run):
            scan_result = scanner.scan(video_path)

        assert scan_result.has_subtitles is True
        assert scan_result.subtitle_source == "embedded:2"
        assert scan_result.profane_cues_found == 1
        assert len(scan_result.profanity_regions) == 1


class TestExternalSRTParsing:
    """Tests for external SRT file parsing with valid content (Req 8.2)."""

    def test_parse_valid_srt_file(self, tmp_path: Path):
        """A well-formed SRT file is parsed into subtitle cues."""
        profanity_list = _make_profanity_list({"damn"})
        scanner = SubtitleScanner(profanity_list=profanity_list)

        srt_content = (
            "1\n"
            "00:00:10,500 --> 00:00:13,000\n"
            "Hello everyone\n"
            "\n"
            "2\n"
            "00:00:15,000 --> 00:00:18,500\n"
            "What the damn is this?\n"
            "\n"
            "3\n"
            "00:00:20,000 --> 00:00:22,000\n"
            "Let's find out\n"
        )

        srt_file = tmp_path / "subtitles.srt"
        srt_file.write_text(srt_content, encoding="utf-8")

        result = scanner.scan(tmp_path / "video.mp4", external_subtitle_path=srt_file)

        assert result.has_subtitles is True
        assert result.subtitle_source == str(srt_file)
        assert result.total_cues_scanned == 3
        assert result.profane_cues_found == 1

    def test_srt_timestamps_parsed_correctly(self, tmp_path: Path):
        """SRT timestamps are correctly converted to seconds."""
        profanity_list = _make_profanity_list({"bad"})
        scanner = SubtitleScanner(profanity_list=profanity_list)

        srt_content = (
            "1\n"
            "01:02:03,456 --> 01:02:05,789\n"
            "This is bad\n"
        )

        srt_file = tmp_path / "subtitles.srt"
        srt_file.write_text(srt_content, encoding="utf-8")

        result = scanner.scan(tmp_path / "video.mp4", external_subtitle_path=srt_file)

        assert result.profane_cues_found == 1
        region = result.profanity_regions[0]
        # Start should be 1*3600 + 2*60 + 3.456 - 2.0 buffer = 3721.456
        expected_start = 1 * 3600 + 2 * 60 + 3.456 - 2.0
        assert abs(region.start - expected_start) < 0.01

    def test_multiline_srt_text(self, tmp_path: Path):
        """SRT cues with multi-line text are correctly parsed."""
        profanity_list = _make_profanity_list({"damn"})
        scanner = SubtitleScanner(profanity_list=profanity_list)

        srt_content = (
            "1\n"
            "00:00:05,000 --> 00:00:08,000\n"
            "First line\n"
            "damn second line\n"
        )

        srt_file = tmp_path / "subtitles.srt"
        srt_file.write_text(srt_content, encoding="utf-8")

        result = scanner.scan(tmp_path / "video.mp4", external_subtitle_path=srt_file)

        assert result.profane_cues_found == 1


class TestExternalASSParsing:
    """Tests for external ASS/SSA file parsing with valid content (Req 8.2)."""

    def test_parse_valid_ass_file(self, tmp_path: Path):
        """A well-formed ASS file with Dialogue lines is parsed into cues."""
        profanity_list = _make_profanity_list({"damn"})
        scanner = SubtitleScanner(profanity_list=profanity_list)

        ass_content = (
            "[Script Info]\n"
            "Title: Test\n"
            "\n"
            "[Events]\n"
            "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
            "Dialogue: 0,0:00:10.50,0:00:13.00,Default,,0,0,0,,Hello everyone\n"
            "Dialogue: 0,0:00:15.00,0:00:18.50,Default,,0,0,0,,What the damn is this?\n"
            "Dialogue: 0,0:00:20.00,0:00:22.00,Default,,0,0,0,,Let's find out\n"
        )

        ass_file = tmp_path / "subtitles.ass"
        ass_file.write_text(ass_content, encoding="utf-8")

        result = scanner.scan(tmp_path / "video.mp4", external_subtitle_path=ass_file)

        assert result.has_subtitles is True
        assert result.subtitle_source == str(ass_file)
        assert result.total_cues_scanned == 3
        assert result.profane_cues_found == 1

    def test_ass_style_tags_stripped(self, tmp_path: Path):
        """ASS style override tags like {\\b1} are stripped before profanity matching."""
        profanity_list = _make_profanity_list({"damn"})
        scanner = SubtitleScanner(profanity_list=profanity_list)

        ass_content = (
            "[Events]\n"
            "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
            "Dialogue: 0,0:00:10.00,0:00:12.00,Default,,0,0,0,,{\\b1}damn{\\b0} it\n"
        )

        ass_file = tmp_path / "subtitles.ass"
        ass_file.write_text(ass_content, encoding="utf-8")

        result = scanner.scan(tmp_path / "video.mp4", external_subtitle_path=ass_file)

        assert result.profane_cues_found == 1

    def test_ssa_file_extension_accepted(self, tmp_path: Path):
        """SSA file extension is handled the same as ASS."""
        profanity_list = _make_profanity_list({"shit"})
        scanner = SubtitleScanner(profanity_list=profanity_list)

        ssa_content = (
            "[Events]\n"
            "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
            "Dialogue: 0,0:00:05.00,0:00:07.00,Default,,0,0,0,,Oh shit\n"
        )

        ssa_file = tmp_path / "subtitles.ssa"
        ssa_file.write_text(ssa_content, encoding="utf-8")

        result = scanner.scan(tmp_path / "video.mp4", external_subtitle_path=ssa_file)

        assert result.has_subtitles is True
        assert result.profane_cues_found == 1

    def test_ass_timestamps_parsed_correctly(self, tmp_path: Path):
        """ASS timestamps (centiseconds) are correctly converted to seconds."""
        profanity_list = _make_profanity_list({"bad"})
        scanner = SubtitleScanner(profanity_list=profanity_list)

        ass_content = (
            "[Events]\n"
            "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
            "Dialogue: 0,1:02:03.45,1:02:05.78,Default,,0,0,0,,This is bad\n"
        )

        ass_file = tmp_path / "subtitles.ass"
        ass_file.write_text(ass_content, encoding="utf-8")

        result = scanner.scan(tmp_path / "video.mp4", external_subtitle_path=ass_file)

        assert result.profane_cues_found == 1
        region = result.profanity_regions[0]
        # Start should be 1*3600 + 2*60 + 3.45 - 2.0 buffer = 3721.45
        expected_start = 1 * 3600 + 2 * 60 + 3.45 - 2.0
        assert abs(region.start - expected_start) < 0.01


class TestExternalFilePreference:
    """Tests for external subtitle file preference over embedded tracks (Req 8.8)."""

    def test_external_file_used_over_embedded(self, tmp_path: Path):
        """When external subtitle file is provided, embedded tracks are not probed."""
        profanity_list = _make_profanity_list({"damn"})
        scanner = SubtitleScanner(profanity_list=profanity_list)

        srt_content = (
            "1\n"
            "00:00:05,000 --> 00:00:08,000\n"
            "Oh damn it\n"
        )

        srt_file = tmp_path / "external.srt"
        srt_file.write_text(srt_content, encoding="utf-8")

        video_path = tmp_path / "video.mkv"
        video_path.write_bytes(b"fake video")

        with patch("subprocess.run") as mock_run:
            result = scanner.scan(video_path, external_subtitle_path=srt_file)

        # subprocess.run should NOT be called since external file is used
        mock_run.assert_not_called()
        assert result.subtitle_source == str(srt_file)
        assert result.profane_cues_found == 1

    def test_external_file_source_recorded(self, tmp_path: Path):
        """The subtitle_source field records the external file path."""
        profanity_list = _make_profanity_list({"bad"})
        scanner = SubtitleScanner(profanity_list=profanity_list)

        srt_content = (
            "1\n"
            "00:00:01,000 --> 00:00:03,000\n"
            "This is bad\n"
        )

        srt_file = tmp_path / "my_subs.srt"
        srt_file.write_text(srt_content, encoding="utf-8")

        result = scanner.scan(tmp_path / "video.mp4", external_subtitle_path=srt_file)

        assert result.subtitle_source == str(srt_file)


class TestNoSubtitlesAvailable:
    """Tests for no subtitles available scenario (Req 8.6)."""

    def test_no_subtitles_returns_has_subtitles_false(self, tmp_path: Path):
        """When no embedded tracks and no external file, has_subtitles is False."""
        profanity_list = _make_profanity_list({"damn"})
        scanner = SubtitleScanner(profanity_list=profanity_list)

        ffprobe_output = json.dumps({"streams": []})
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = ffprobe_output

        video_path = tmp_path / "video.mp4"
        video_path.write_bytes(b"fake video")

        with patch("subprocess.run", return_value=mock_result):
            result = scanner.scan(video_path)

        assert result.has_subtitles is False
        assert result.subtitle_source is None
        assert result.profanity_regions == []
        assert result.total_cues_scanned == 0

    def test_no_subtitles_skipped_speech_recognition_false(self, tmp_path: Path):
        """When no subtitles available, skipped_speech_recognition is False (full transcription needed)."""
        profanity_list = _make_profanity_list({"damn"})
        scanner = SubtitleScanner(profanity_list=profanity_list)

        ffprobe_output = json.dumps({"streams": []})
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = ffprobe_output

        video_path = tmp_path / "video.mp4"
        video_path.write_bytes(b"fake video")

        with patch("subprocess.run", return_value=mock_result):
            result = scanner.scan(video_path)

        assert result.skipped_speech_recognition is False


class TestCleanSubtitlesSkipSpeechRecognition:
    """Tests for clean subtitles (no profanity) setting skipped_speech_recognition (Req 8.7)."""

    def test_clean_srt_skips_speech_recognition(self, tmp_path: Path):
        """When subtitles exist but contain no profanity, skipped_speech_recognition is True."""
        profanity_list = _make_profanity_list({"damn", "shit", "fuck"})
        scanner = SubtitleScanner(profanity_list=profanity_list)

        srt_content = (
            "1\n"
            "00:00:05,000 --> 00:00:08,000\n"
            "Hello everyone, welcome to the show\n"
            "\n"
            "2\n"
            "00:00:10,000 --> 00:00:13,000\n"
            "Today we have a wonderful episode\n"
            "\n"
            "3\n"
            "00:00:15,000 --> 00:00:18,000\n"
            "Let's get started with the first segment\n"
        )

        srt_file = tmp_path / "clean.srt"
        srt_file.write_text(srt_content, encoding="utf-8")

        result = scanner.scan(tmp_path / "video.mp4", external_subtitle_path=srt_file)

        assert result.has_subtitles is True
        assert result.skipped_speech_recognition is True
        assert result.profane_cues_found == 0
        assert result.profanity_regions == []
        assert result.total_cues_scanned == 3

    def test_profanity_found_does_not_skip_speech_recognition(self, tmp_path: Path):
        """When subtitles contain profanity, skipped_speech_recognition is False."""
        profanity_list = _make_profanity_list({"damn"})
        scanner = SubtitleScanner(profanity_list=profanity_list)

        srt_content = (
            "1\n"
            "00:00:05,000 --> 00:00:08,000\n"
            "Oh damn it\n"
        )

        srt_file = tmp_path / "profane.srt"
        srt_file.write_text(srt_content, encoding="utf-8")

        result = scanner.scan(tmp_path / "video.mp4", external_subtitle_path=srt_file)

        assert result.skipped_speech_recognition is False
        assert result.profane_cues_found == 1


class TestMalformedSubtitleHandling:
    """Tests for malformed subtitle file handling."""

    def test_empty_srt_file(self, tmp_path: Path):
        """An empty SRT file results in no cues parsed."""
        profanity_list = _make_profanity_list({"damn"})
        scanner = SubtitleScanner(profanity_list=profanity_list)

        srt_file = tmp_path / "empty.srt"
        srt_file.write_text("", encoding="utf-8")

        result = scanner.scan(tmp_path / "video.mp4", external_subtitle_path=srt_file)

        # Empty file has no cues, so has_subtitles depends on subtitle_source being set
        # but no cues means the scanner treats it as no valid subtitle content
        assert result.total_cues_scanned == 0

    def test_srt_missing_timestamps(self, tmp_path: Path):
        """An SRT file with blocks missing timestamps is handled gracefully."""
        profanity_list = _make_profanity_list({"damn"})
        scanner = SubtitleScanner(profanity_list=profanity_list)

        srt_content = (
            "1\n"
            "This has no timestamp line\n"
            "\n"
            "2\n"
            "00:00:05,000 --> 00:00:08,000\n"
            "Valid cue with damn word\n"
        )

        srt_file = tmp_path / "malformed.srt"
        srt_file.write_text(srt_content, encoding="utf-8")

        result = scanner.scan(tmp_path / "video.mp4", external_subtitle_path=srt_file)

        # Only the valid cue should be parsed
        assert result.total_cues_scanned == 1
        assert result.profane_cues_found == 1

    def test_nonexistent_external_file(self, tmp_path: Path):
        """A nonexistent external subtitle file path is handled gracefully."""
        profanity_list = _make_profanity_list({"damn"})
        scanner = SubtitleScanner(profanity_list=profanity_list)

        missing_file = tmp_path / "does_not_exist.srt"

        result = scanner.scan(tmp_path / "video.mp4", external_subtitle_path=missing_file)

        # File doesn't exist so no cues returned, but subtitle_source is still set
        assert result.subtitle_source == str(missing_file)
        assert result.total_cues_scanned == 0

    def test_ass_file_with_no_dialogue_lines(self, tmp_path: Path):
        """An ASS file with no Dialogue lines results in no cues."""
        profanity_list = _make_profanity_list({"damn"})
        scanner = SubtitleScanner(profanity_list=profanity_list)

        ass_content = (
            "[Script Info]\n"
            "Title: Empty subtitles\n"
            "\n"
            "[V4+ Styles]\n"
            "Format: Name, Fontname, Fontsize\n"
            "Style: Default,Arial,20\n"
        )

        ass_file = tmp_path / "no_dialogue.ass"
        ass_file.write_text(ass_content, encoding="utf-8")

        result = scanner.scan(tmp_path / "video.mp4", external_subtitle_path=ass_file)

        assert result.total_cues_scanned == 0

    def test_binary_content_file(self, tmp_path: Path):
        """A file with binary content does not crash the scanner."""
        profanity_list = _make_profanity_list({"damn"})
        scanner = SubtitleScanner(profanity_list=profanity_list)

        binary_file = tmp_path / "garbage.srt"
        binary_file.write_bytes(b"\x00\x01\x02\xff\xfe\xfd" * 100)

        # Should not raise; falls back to latin-1 decoding
        result = scanner.scan(tmp_path / "video.mp4", external_subtitle_path=binary_file)

        assert result.total_cues_scanned == 0
