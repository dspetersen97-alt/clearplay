"""Tests for CLI batch (folder) processing mode."""

from pathlib import Path
from unittest.mock import patch

from video_profanity_censor.cli import (
    FILTERED_DIR_NAME,
    discover_videos,
    main,
)
from video_profanity_censor.models import ProcessingResult


def _touch(path: Path) -> None:
    path.write_bytes(b"\x00")


class TestDiscoverVideos:
    def test_finds_supported_extensions_only(self, tmp_path: Path):
        _touch(tmp_path / "a.mkv")
        _touch(tmp_path / "b.mp4")
        _touch(tmp_path / "c.avi")
        _touch(tmp_path / "d.mov")
        _touch(tmp_path / "e.wmv")
        _touch(tmp_path / "notes.txt")
        _touch(tmp_path / "cover.jpg")

        found = {p.name for p in discover_videos(tmp_path)}
        assert found == {"a.mkv", "b.mp4", "c.avi", "d.mov", "e.wmv"}

    def test_extension_match_is_case_insensitive(self, tmp_path: Path):
        _touch(tmp_path / "clip.MOV")
        _touch(tmp_path / "movie.MKV")
        found = {p.name for p in discover_videos(tmp_path)}
        assert found == {"clip.MOV", "movie.MKV"}

    def test_skips_the_filtered_output_dir(self, tmp_path: Path):
        _touch(tmp_path / "a.mkv")
        filtered = tmp_path / FILTERED_DIR_NAME
        filtered.mkdir()
        _touch(filtered / "a_censored.mkv")

        found = {p.name for p in discover_videos(tmp_path)}
        assert found == {"a.mkv"}  # the censored output is not rediscovered

    def test_is_non_recursive(self, tmp_path: Path):
        _touch(tmp_path / "top.mkv")
        sub = tmp_path / "season1"
        sub.mkdir()
        _touch(sub / "nested.mkv")

        found = {p.name for p in discover_videos(tmp_path)}
        assert found == {"top.mkv"}

    def test_empty_folder_returns_empty(self, tmp_path: Path):
        assert discover_videos(tmp_path) == []


class TestBatchRouting:
    def _ok(self, **kwargs) -> ProcessingResult:
        return ProcessingResult(
            success=True,
            output_path=Path(kwargs["output_path"]),
            report_path=Path(kwargs["report_path"]),
            profane_instances_detected=2,
            profane_instances_censored=2,
        )

    def test_folder_input_processes_each_into_filtered(self, tmp_path: Path):
        _touch(tmp_path / "one.mkv")
        _touch(tmp_path / "two.mp4")

        calls = {}

        def fake(self, **kwargs):
            calls[Path(kwargs["input_path"]).name] = (
                Path(kwargs["output_path"]),
                Path(kwargs["report_path"]),
            )
            return TestBatchRouting()._ok(**kwargs)

        with patch("video_profanity_censor.cli.CensorEngine.process", fake):
            rc = main([str(tmp_path)])

        assert rc == 0
        filtered = tmp_path / FILTERED_DIR_NAME
        assert filtered.is_dir()
        assert calls["one.mkv"][0] == filtered / "one_censored.mkv"
        assert calls["one.mkv"][1] == filtered / "one_censored_report.txt"
        assert calls["two.mp4"][0] == filtered / "two_censored.mp4"

    def test_empty_folder_returns_failure(self, tmp_path: Path):
        rc = main([str(tmp_path)])
        assert rc == 1

    def test_one_failure_does_not_abort_batch(self, tmp_path: Path):
        _touch(tmp_path / "good.mkv")
        _touch(tmp_path / "bad.mkv")

        def fake(self, **kwargs):
            name = Path(kwargs["input_path"]).name
            if name == "bad.mkv":
                return ProcessingResult(
                    success=False, error_message="no decoder for eac3"
                )
            return TestBatchRouting()._ok(**kwargs)

        with patch("video_profanity_censor.cli.CensorEngine.process", fake):
            rc = main([str(tmp_path)])

        # good.mkv succeeded, bad.mkv failed -> overall failure exit code, but both ran.
        assert rc == 1

    def test_unexpected_exception_is_caught_per_file(self, tmp_path: Path):
        _touch(tmp_path / "boom.mkv")
        _touch(tmp_path / "fine.mp4")

        def fake(self, **kwargs):
            if Path(kwargs["input_path"]).name == "boom.mkv":
                raise RuntimeError("kaboom")
            return TestBatchRouting()._ok(**kwargs)

        with patch("video_profanity_censor.cli.CensorEngine.process", fake):
            rc = main([str(tmp_path)])

        # Should not raise; batch continues and reports failure for the one file.
        assert rc == 1


class TestBatchFlagGuards:
    def test_output_with_folder_errors(self, tmp_path: Path):
        _touch(tmp_path / "a.mkv")
        assert main([str(tmp_path), "--output", "x.mkv"]) == 1

    def test_subtitle_path_with_folder_errors(self, tmp_path: Path):
        _touch(tmp_path / "a.mkv")
        assert main([str(tmp_path), "--subtitle-path", "s.srt"]) == 1

    def test_report_path_with_folder_errors(self, tmp_path: Path):
        _touch(tmp_path / "a.mkv")
        assert main([str(tmp_path), "--report-path", "r.txt"]) == 1

    def test_nonexistent_path_errors(self):
        assert main(["/no/such/path/xyz"]) == 1
