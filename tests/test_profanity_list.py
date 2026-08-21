"""Unit tests for the profanity list loader."""

import tempfile
from pathlib import Path

import pytest

from video_profanity_censor.profanity_list import (
    ProfanityListError,
    load_profanity_list,
)


class TestLoadProfanityListDefault:
    """Tests for loading the bundled default profanity list."""

    def test_loads_default_list_when_no_path_given(self):
        result = load_profanity_list()
        assert result.is_valid
        assert len(result.words) > 0

    def test_default_list_has_stems(self):
        result = load_profanity_list()
        assert len(result.stems) > 0

    def test_default_list_source_path_points_to_data_dir(self):
        result = load_profanity_list()
        assert result.source_path is not None
        assert result.source_path.name == "default_profanity_list.txt"

    def test_default_list_words_are_lowercase(self):
        result = load_profanity_list()
        for word in result.words:
            assert word == word.lower()


class TestLoadProfanityListFromFile:
    """Tests for loading a custom profanity list from a file path."""

    def test_loads_custom_file(self, tmp_path: Path):
        custom_file = tmp_path / "custom.txt"
        custom_file.write_text("badword\nanotherbad\n", encoding="utf-8")

        result = load_profanity_list(custom_file)
        assert result.words == {"badword", "anotherbad"}
        assert result.source_path == custom_file

    def test_normalizes_words_to_lowercase(self, tmp_path: Path):
        custom_file = tmp_path / "custom.txt"
        custom_file.write_text("BadWord\nUPPER\nmixed\n", encoding="utf-8")

        result = load_profanity_list(custom_file)
        assert result.words == {"badword", "upper", "mixed"}

    def test_strips_whitespace(self, tmp_path: Path):
        custom_file = tmp_path / "custom.txt"
        custom_file.write_text("  padded  \n\ttabs\t\n", encoding="utf-8")

        result = load_profanity_list(custom_file)
        assert result.words == {"padded", "tabs"}

    def test_skips_blank_lines(self, tmp_path: Path):
        custom_file = tmp_path / "custom.txt"
        custom_file.write_text("word1\n\n\nword2\n\n", encoding="utf-8")

        result = load_profanity_list(custom_file)
        assert result.words == {"word1", "word2"}

    def test_skips_comment_lines(self, tmp_path: Path):
        custom_file = tmp_path / "custom.txt"
        custom_file.write_text("# This is a comment\nword1\n# Another comment\nword2\n", encoding="utf-8")

        result = load_profanity_list(custom_file)
        assert result.words == {"word1", "word2"}

    def test_precomputes_stems(self, tmp_path: Path):
        custom_file = tmp_path / "custom.txt"
        custom_file.write_text("running\njumping\n", encoding="utf-8")

        result = load_profanity_list(custom_file)
        # SnowballStemmer should stem these
        assert "run" in result.stems
        assert "jump" in result.stems


class TestLoadProfanityListErrors:
    """Tests for error conditions (Req 3.6)."""

    def test_raises_error_for_nonexistent_file(self):
        with pytest.raises(ProfanityListError, match="not found"):
            load_profanity_list(Path("nonexistent_file.txt"))

    def test_raises_error_for_empty_file(self, tmp_path: Path):
        empty_file = tmp_path / "empty.txt"
        empty_file.write_text("", encoding="utf-8")

        with pytest.raises(ProfanityListError, match="empty"):
            load_profanity_list(empty_file)

    def test_raises_error_for_only_blank_lines(self, tmp_path: Path):
        blank_file = tmp_path / "blank.txt"
        blank_file.write_text("\n\n  \n\n", encoding="utf-8")

        with pytest.raises(ProfanityListError, match="empty"):
            load_profanity_list(blank_file)

    def test_raises_error_for_only_comments(self, tmp_path: Path):
        comments_file = tmp_path / "comments.txt"
        comments_file.write_text("# comment 1\n# comment 2\n", encoding="utf-8")

        with pytest.raises(ProfanityListError, match="empty"):
            load_profanity_list(comments_file)

    def test_error_message_mentions_detection_requirement(self, tmp_path: Path):
        empty_file = tmp_path / "empty.txt"
        empty_file.write_text("", encoding="utf-8")

        with pytest.raises(ProfanityListError, match="at least one entry"):
            load_profanity_list(empty_file)
