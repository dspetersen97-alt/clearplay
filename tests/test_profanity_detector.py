"""Unit tests for the ProfanityDetector component.

Tests cover:
- Custom list loading and validation (Req 3.5)
- Empty list behavior (Req 3.6)
- Case-insensitive matching (Req 3.1)
- Morphological variants via stemming (Req 3.1, 3.3)
- Scunthorpe problem — no substring false positives (Req 3.4)
"""

from pathlib import Path

import pytest
from nltk.stem import SnowballStemmer

from video_profanity_censor.models import (
    CensorMode,
    DetectionResult,
    ProfanityList,
    TranscribedWord,
    TranscriptionResult,
)
from video_profanity_censor.profanity_detector import ProfanityDetector


# Helper to build a ProfanityList from a set of words
_stemmer = SnowballStemmer("english")


def _make_profanity_list(words: set[str]) -> ProfanityList:
    """Create a ProfanityList with pre-computed stems from a set of words."""
    lower_words = {w.lower() for w in words}
    stems = {_stemmer.stem(w) for w in lower_words}
    return ProfanityList(words=lower_words, stems=stems, source_path=None)


def _make_transcription(words: list[str]) -> TranscriptionResult:
    """Create a TranscriptionResult with dummy timestamps for the given words."""
    transcribed = []
    t = 0.0
    for w in words:
        transcribed.append(TranscribedWord(word=w, start=t, end=t + 0.5, confidence=0.95))
        t += 0.6
    return TranscriptionResult(words=transcribed)


class TestCustomListLoading:
    """Tests for custom profanity list loading and usage (Req 3.5)."""

    def test_custom_list_passed_to_detector(self):
        """A manually-created ProfanityList is used for detection."""
        custom_list = _make_profanity_list({"badword", "nastything"})
        detector = ProfanityDetector(profanity_list=custom_list)

        transcription = _make_transcription(["badword", "goodword", "nastything"])
        result = detector.detect(transcription)

        detected_words = [d.word for d in result.detections]
        assert "badword" in detected_words
        assert "nastything" in detected_words
        assert "goodword" not in detected_words

    def test_custom_list_single_word(self):
        """A custom list with a single word still works correctly."""
        custom_list = _make_profanity_list({"onlythis"})
        detector = ProfanityDetector(profanity_list=custom_list)

        transcription = _make_transcription(["onlythis", "other", "words"])
        result = detector.detect(transcription)

        assert len(result.detections) == 1
        assert result.detections[0].word == "onlythis"

    def test_total_words_scanned_count(self):
        """DetectionResult reports the total number of words scanned."""
        custom_list = _make_profanity_list({"bad"})
        detector = ProfanityDetector(profanity_list=custom_list)

        transcription = _make_transcription(["hello", "world", "bad", "day"])
        result = detector.detect(transcription)

        assert result.total_words_scanned == 4


class TestEmptyList:
    """Tests for empty profanity list behavior (Req 3.6)."""

    def test_empty_list_detects_nothing(self):
        """An empty profanity list results in no detections."""
        empty_list = ProfanityList(words=set(), stems=set(), source_path=None)
        detector = ProfanityDetector(profanity_list=empty_list)

        transcription = _make_transcription(["damn", "hell", "anything"])
        result = detector.detect(transcription)

        assert len(result.detections) == 0

    def test_empty_list_is_not_valid(self):
        """An empty ProfanityList reports is_valid as False."""
        empty_list = ProfanityList(words=set(), stems=set(), source_path=None)
        assert not empty_list.is_valid


class TestCaseInsensitiveMatching:
    """Tests for case-insensitive whole-word matching (Req 3.1)."""

    def test_uppercase_match(self):
        """'DAMN' matches when 'damn' is in the list."""
        custom_list = _make_profanity_list({"damn"})
        detector = ProfanityDetector(profanity_list=custom_list)

        transcription = _make_transcription(["DAMN"])
        result = detector.detect(transcription)

        assert len(result.detections) == 1

    def test_mixed_case_match(self):
        """'Damn' matches when 'damn' is in the list."""
        custom_list = _make_profanity_list({"damn"})
        detector = ProfanityDetector(profanity_list=custom_list)

        transcription = _make_transcription(["Damn"])
        result = detector.detect(transcription)

        assert len(result.detections) == 1

    def test_lowercase_match(self):
        """'damn' matches when 'damn' is in the list."""
        custom_list = _make_profanity_list({"damn"})
        detector = ProfanityDetector(profanity_list=custom_list)

        transcription = _make_transcription(["damn"])
        result = detector.detect(transcription)

        assert len(result.detections) == 1

    def test_all_caps_word_in_list(self):
        """Words provided in uppercase in list are normalized to lowercase."""
        # ProfanityList words should be stored lowercase
        custom_list = _make_profanity_list({"HELL"})
        detector = ProfanityDetector(profanity_list=custom_list)

        transcription = _make_transcription(["hell", "Hell", "HELL"])
        result = detector.detect(transcription)

        assert len(result.detections) == 3


class TestMorphologicalVariants:
    """Tests for morphological variant matching via stemming (Req 3.1, 3.3)."""

    def test_plural_form(self):
        """'fucks' (plural) is detected when 'fuck' is in the list."""
        custom_list = _make_profanity_list({"fuck"})
        detector = ProfanityDetector(profanity_list=custom_list)

        transcription = _make_transcription(["fucks"])
        result = detector.detect(transcription)

        assert len(result.detections) == 1

    def test_gerund_form(self):
        """'shitting' (gerund) is detected when 'shit' is in the list."""
        custom_list = _make_profanity_list({"shit"})
        detector = ProfanityDetector(profanity_list=custom_list)

        transcription = _make_transcription(["shitting"])
        result = detector.detect(transcription)

        assert len(result.detections) == 1

    def test_verb_form(self):
        """'damning' (verb form) is detected when 'damn' is in the list."""
        custom_list = _make_profanity_list({"damn"})
        detector = ProfanityDetector(profanity_list=custom_list)

        transcription = _make_transcription(["damning"])
        result = detector.detect(transcription)

        assert len(result.detections) == 1

    def test_past_tense(self):
        """'damned' (past tense) is detected when 'damn' is in the list."""
        custom_list = _make_profanity_list({"damn"})
        detector = ProfanityDetector(profanity_list=custom_list)

        transcription = _make_transcription(["damned"])
        result = detector.detect(transcription)

        assert len(result.detections) == 1

    def test_non_variant_not_matched(self):
        """A completely unrelated word is not matched by stemming."""
        custom_list = _make_profanity_list({"damn"})
        detector = ProfanityDetector(profanity_list=custom_list)

        transcription = _make_transcription(["dancing", "morning", "running"])
        result = detector.detect(transcription)

        assert len(result.detections) == 0


class TestScunthorpeProblem:
    """Tests that substring-only matches are NOT flagged (Req 3.4).

    The Scunthorpe problem: words that contain a profane term as a substring
    should not be falsely flagged. The detector uses whole-word matching only.
    """

    def test_class_does_not_match_ass(self):
        """'class' does NOT match 'ass' despite containing it as a substring."""
        custom_list = _make_profanity_list({"ass"})
        detector = ProfanityDetector(profanity_list=custom_list)

        transcription = _make_transcription(["class"])
        result = detector.detect(transcription)

        assert len(result.detections) == 0

    def test_basement_does_not_match_ass(self):
        """'basement' does NOT match 'ass'."""
        custom_list = _make_profanity_list({"ass"})
        detector = ProfanityDetector(profanity_list=custom_list)

        transcription = _make_transcription(["basement"])
        result = detector.detect(transcription)

        assert len(result.detections) == 0

    def test_classic_does_not_match_ass(self):
        """'classic' does NOT match 'ass'."""
        custom_list = _make_profanity_list({"ass"})
        detector = ProfanityDetector(profanity_list=custom_list)

        transcription = _make_transcription(["classic"])
        result = detector.detect(transcription)

        assert len(result.detections) == 0

    def test_scunthorpe_does_not_match(self):
        """'scunthorpe' does NOT match any profane term containing 'cunt'."""
        custom_list = _make_profanity_list({"cunt"})
        detector = ProfanityDetector(profanity_list=custom_list)

        transcription = _make_transcription(["scunthorpe"])
        result = detector.detect(transcription)

        assert len(result.detections) == 0

    def test_shell_does_not_match_hell(self):
        """'shell' does NOT match 'hell'."""
        custom_list = _make_profanity_list({"hell"})
        detector = ProfanityDetector(profanity_list=custom_list)

        transcription = _make_transcription(["shell"])
        result = detector.detect(transcription)

        assert len(result.detections) == 0

    def test_assess_does_not_match_ass(self):
        """'assess' does NOT match 'ass'."""
        custom_list = _make_profanity_list({"ass"})
        detector = ProfanityDetector(profanity_list=custom_list)

        transcription = _make_transcription(["assess"])
        result = detector.detect(transcription)

        assert len(result.detections) == 0

    def test_actual_profane_word_still_detected(self):
        """The actual profane word 'ass' IS detected even when Scunthorpe words aren't."""
        custom_list = _make_profanity_list({"ass"})
        detector = ProfanityDetector(profanity_list=custom_list)

        transcription = _make_transcription(["class", "ass", "classic"])
        result = detector.detect(transcription)

        assert len(result.detections) == 1
        assert result.detections[0].word == "ass"


class TestCensorModeRecording:
    """Tests that the correct censor mode is recorded in detections."""

    def test_default_tone_mode(self):
        """Default censor mode is TONE."""
        custom_list = _make_profanity_list({"bad"})
        detector = ProfanityDetector(profanity_list=custom_list)

        transcription = _make_transcription(["bad"])
        result = detector.detect(transcription)

        assert result.detections[0].censor_action == CensorMode.TONE

    def test_mute_mode(self):
        """Mute censor mode is recorded when specified."""
        custom_list = _make_profanity_list({"bad"})
        detector = ProfanityDetector(profanity_list=custom_list, censor_mode=CensorMode.MUTE)

        transcription = _make_transcription(["bad"])
        result = detector.detect(transcription)

        assert result.detections[0].censor_action == CensorMode.MUTE
