"""Property-based tests for ProfanityDetector.

**Validates: Requirements 3.1, 3.2, 3.3, 3.4**
"""

import string

from hypothesis import given, assume, strategies as st, settings
from nltk.stem import SnowballStemmer

from video_profanity_censor.models import (
    ProfanityList,
    TranscribedWord,
    TranscriptionResult,
    TimestampRange,
)
from video_profanity_censor.profanity_detector import ProfanityDetector
from video_profanity_censor.profanity_list import load_profanity_list


# Load the default profanity list for use in tests
_profanity_list = load_profanity_list()

# Known profane words from the default list to use as explicit profane entries
KNOWN_PROFANE_WORDS = sorted(_profanity_list.words)

# Clean words that should never match (not substrings of profane words either)
CLEAN_WORDS = ["hello", "world", "computer", "python", "garden", "table", "window"]

# Strategy: generate a non-negative float for a start timestamp
# Using finite floats avoids NaN/inf issues
timestamp_starts = st.floats(min_value=0.0, max_value=3600.0, allow_nan=False, allow_infinity=False)

# Strategy: generate a positive duration to ensure start < end
durations = st.floats(min_value=0.01, max_value=5.0, allow_nan=False, allow_infinity=False)

# Strategy: generate confidence values
confidences = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)


def _make_transcribed_word(word: str, start: float, duration: float, confidence: float) -> TranscribedWord:
    """Create a TranscribedWord with guaranteed start < end."""
    return TranscribedWord(
        word=word,
        start=start,
        end=start + duration,
        confidence=confidence,
    )


# Strategy: generate a list of TranscribedWords that includes at least one profane word
@st.composite
def transcription_with_profanity(draw):
    """Generate a TranscriptionResult containing a mix of profane and clean words.

    Ensures at least one profane word is present and all words have valid timestamps
    where start < end. Timestamps are sequential (each word starts after the previous ends).
    """
    # Decide how many clean words before, between, and after profane words
    num_prefix_clean = draw(st.integers(min_value=0, max_value=3))
    num_profane = draw(st.integers(min_value=1, max_value=5))
    num_suffix_clean = draw(st.integers(min_value=0, max_value=3))

    words: list[TranscribedWord] = []
    current_time = draw(st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False))

    # Add clean prefix words
    for _ in range(num_prefix_clean):
        clean_word = draw(st.sampled_from(CLEAN_WORDS))
        duration = draw(durations)
        confidence = draw(confidences)
        words.append(_make_transcribed_word(clean_word, current_time, duration, confidence))
        current_time += duration + draw(st.floats(min_value=0.01, max_value=1.0, allow_nan=False, allow_infinity=False))

    # Add profane words
    for _ in range(num_profane):
        profane_word = draw(st.sampled_from(KNOWN_PROFANE_WORDS))
        duration = draw(durations)
        confidence = draw(confidences)
        words.append(_make_transcribed_word(profane_word, current_time, duration, confidence))
        current_time += duration + draw(st.floats(min_value=0.01, max_value=1.0, allow_nan=False, allow_infinity=False))

    # Add clean suffix words
    for _ in range(num_suffix_clean):
        clean_word = draw(st.sampled_from(CLEAN_WORDS))
        duration = draw(durations)
        confidence = draw(confidences)
        words.append(_make_transcribed_word(clean_word, current_time, duration, confidence))
        current_time += duration + draw(st.floats(min_value=0.01, max_value=1.0, allow_nan=False, allow_infinity=False))

    return TranscriptionResult(words=words, has_speech=True)


class TestDetectionTimestampPreservation:
    """Property 3: Detection Timestamp Preservation.

    For any transcription containing profane words, the Timestamp_Range recorded
    for each detection SHALL exactly equal the word-level timestamps (start, end)
    from the transcription input — no modification, rounding, or loss of precision.

    **Validates: Requirements 3.2**
    """

    @given(transcription=transcription_with_profanity())
    @settings(max_examples=200)
    def test_timestamp_range_exactly_equals_word_timestamps(
        self, transcription: TranscriptionResult
    ):
        """Detected profanity timestamps must exactly equal the source word timestamps."""
        detector = ProfanityDetector(profanity_list=_profanity_list)
        result = detector.detect(transcription)

        # Build a lookup from profane words in the input to their original timestamps
        # For each detection, verify timestamp_range matches the original word exactly
        for detection in result.detections:
            # Find the corresponding TranscribedWord in the input
            # Match by word content AND timestamp (there could be duplicate words)
            matching_words = [
                w for w in transcription.words
                if w.start == detection.timestamp_range.start
                and w.end == detection.timestamp_range.end
            ]

            assert len(matching_words) >= 1, (
                f"Detection for '{detection.word}' at "
                f"[{detection.timestamp_range.start}, {detection.timestamp_range.end}] "
                f"has no matching input word with those exact timestamps"
            )

    @given(transcription=transcription_with_profanity())
    @settings(max_examples=200)
    def test_no_timestamp_modification_or_rounding(
        self, transcription: TranscriptionResult
    ):
        """Timestamps must be preserved with no modification, rounding, or precision loss."""
        detector = ProfanityDetector(profanity_list=_profanity_list)
        result = detector.detect(transcription)

        # Build index of input words by position for exact comparison
        input_word_timestamps: dict[int, tuple[float, float]] = {}
        for idx, word in enumerate(transcription.words):
            input_word_timestamps[idx] = (word.start, word.end)

        for detection in result.detections:
            # Find the exact input word that generated this detection
            found = False
            for idx, (start, end) in input_word_timestamps.items():
                word_obj = transcription.words[idx]
                if (
                    word_obj.word.lower().strip() == detection.word.lower().strip()
                    or word_obj.word == detection.word
                ) and start == detection.timestamp_range.start and end == detection.timestamp_range.end:
                    # Verify exact bit-level equality (no floating point rounding)
                    assert detection.timestamp_range.start == start, (
                        f"Start timestamp modified: expected {start}, "
                        f"got {detection.timestamp_range.start}"
                    )
                    assert detection.timestamp_range.end == end, (
                        f"End timestamp modified: expected {end}, "
                        f"got {detection.timestamp_range.end}"
                    )
                    found = True
                    break

            assert found, (
                f"Detection '{detection.word}' at "
                f"[{detection.timestamp_range.start}, {detection.timestamp_range.end}] "
                f"does not exactly match any input word timestamps"
            )

    @given(transcription=transcription_with_profanity())
    @settings(max_examples=200)
    def test_all_profane_words_have_timestamps_from_input(
        self, transcription: TranscriptionResult
    ):
        """Every detection's timestamp must originate from an input word's timestamps."""
        detector = ProfanityDetector(profanity_list=_profanity_list)
        result = detector.detect(transcription)

        # Collect all (start, end) pairs from the input transcription
        input_timestamp_pairs = {
            (w.start, w.end) for w in transcription.words
        }

        for detection in result.detections:
            detection_pair = (
                detection.timestamp_range.start,
                detection.timestamp_range.end,
            )
            assert detection_pair in input_timestamp_pairs, (
                f"Detection timestamp [{detection.timestamp_range.start}, "
                f"{detection.timestamp_range.end}] does not match any input "
                f"word's timestamps. Input pairs: {input_timestamp_pairs}"
            )



# ============================================================================
# Property 2: Profanity Detection Correctness
# ============================================================================

# Shared stemmer matching the production code
_stemmer = SnowballStemmer("english")

# Common morphological suffixes for English words
_MORPHOLOGICAL_SUFFIXES = ["s", "ed", "ing", "er", "est", "ly", "ness", "ment"]

# Strategy: generate base profanity words (simple lowercase alpha words)
_profanity_word_strategy = st.text(
    alphabet=string.ascii_lowercase,
    min_size=4,
    max_size=10,
).filter(lambda w: w.isalpha())

# Strategy: generate profanity lists (non-empty sets of words)
_profanity_list_strategy = st.frozensets(
    _profanity_word_strategy,
    min_size=1,
    max_size=10,
)

# Strategy: generate a morphological suffix
_morphological_suffix_strategy = st.sampled_from(_MORPHOLOGICAL_SUFFIXES)

# Strategy: generate safe padding for substring embedding
_safe_padding = st.text(
    alphabet=string.ascii_lowercase,
    min_size=3,
    max_size=8,
).filter(lambda w: w.isalpha())


def _make_profanity_list_from_words(words: frozenset[str]) -> ProfanityList:
    """Create a ProfanityList from a set of words with pre-computed stems."""
    word_set = set(words)
    stems = {_stemmer.stem(w) for w in word_set}
    return ProfanityList(words=word_set, stems=stems, source_path=None)


class TestProfanityDetectionCorrectnessProperty:
    """Property 2: Profanity Detection Correctness.

    For any word and profanity list, the ProfanityDetector SHALL flag the word
    as profane if and only if (a) the word matches an entry in the profanity
    list via case-insensitive whole-word comparison, or (b) the word's stem
    matches the stem of an entry in the profanity list (morphological variant).
    Words that merely contain a profane term as a substring but are not
    themselves profane or a morphological variant SHALL NOT be flagged.

    **Validates: Requirements 3.1, 3.3, 3.4**
    """

    @given(words=_profanity_list_strategy)
    @settings(max_examples=200)
    def test_exact_match_always_detected(self, words: frozenset[str]):
        """A word from the profanity list is always flagged as profane (exact match)."""
        profanity_list = _make_profanity_list_from_words(words)
        detector = ProfanityDetector(profanity_list=profanity_list)

        for word in words:
            assert detector._is_profane(word), (
                f"Word '{word}' is in the profanity list but was NOT flagged"
            )

    @given(words=_profanity_list_strategy)
    @settings(max_examples=200)
    def test_case_insensitive_match_always_detected(self, words: frozenset[str]):
        """Case variants of profanity list words are always flagged."""
        profanity_list = _make_profanity_list_from_words(words)
        detector = ProfanityDetector(profanity_list=profanity_list)

        for word in words:
            # Test uppercase variant
            assert detector._is_profane(word.upper()), (
                f"Word '{word.upper()}' (uppercase of '{word}') should be flagged"
            )
            # Test title case variant
            assert detector._is_profane(word.title()), (
                f"Word '{word.title()}' (title case of '{word}') should be flagged"
            )

    @given(words=_profanity_list_strategy, suffix=_morphological_suffix_strategy)
    @settings(max_examples=200)
    def test_morphological_variants_detected_via_stemming(
        self, words: frozenset[str], suffix: str
    ):
        """Morphological variants (word + suffix) are flagged when their stem
        matches a stem in the profanity list."""
        profanity_list = _make_profanity_list_from_words(words)
        detector = ProfanityDetector(profanity_list=profanity_list)

        for word in words:
            variant = word + suffix
            variant_stem = _stemmer.stem(variant)

            # The variant should be flagged if its stem is in the profanity list stems
            expected_profane = variant_stem in profanity_list.stems
            actual_profane = detector._is_profane(variant)

            if expected_profane:
                assert actual_profane, (
                    f"Variant '{variant}' (stem='{variant_stem}') should be flagged "
                    f"because its stem matches profanity list stems"
                )

    @given(
        words=_profanity_list_strategy,
        prefix=_safe_padding,
    )
    @settings(max_examples=200)
    def test_substring_prefix_not_flagged(
        self, words: frozenset[str], prefix: str
    ):
        """A word that merely contains a profane word as a suffix (prefix + profane)
        is NOT flagged — avoids the Scunthorpe problem."""
        profanity_list = _make_profanity_list_from_words(words)
        detector = ProfanityDetector(profanity_list=profanity_list)

        for word in words:
            compound = prefix + word
            compound_lower = compound.lower()
            compound_stem = _stemmer.stem(compound_lower)

            # Only test if the compound word itself is NOT in the list
            # and its stem doesn't happen to match a profanity stem
            if compound_lower not in profanity_list.words and compound_stem not in profanity_list.stems:
                assert not detector._is_profane(compound), (
                    f"'{compound}' merely contains '{word}' as substring "
                    f"and should NOT be flagged (Scunthorpe problem)"
                )

    @given(
        words=_profanity_list_strategy,
        suffix=_safe_padding,
    )
    @settings(max_examples=200)
    def test_substring_suffix_not_flagged(
        self, words: frozenset[str], suffix: str
    ):
        """A word that merely contains a profane word as a prefix (profane + suffix)
        is NOT flagged — avoids the Scunthorpe problem."""
        profanity_list = _make_profanity_list_from_words(words)
        detector = ProfanityDetector(profanity_list=profanity_list)

        for word in words:
            compound = word + suffix
            compound_lower = compound.lower()
            compound_stem = _stemmer.stem(compound_lower)

            # Only test if the compound word itself is NOT in the list
            # and its stem doesn't happen to match a profanity stem
            if compound_lower not in profanity_list.words and compound_stem not in profanity_list.stems:
                assert not detector._is_profane(compound), (
                    f"'{compound}' merely contains '{word}' as substring "
                    f"and should NOT be flagged (Scunthorpe problem)"
                )

    @given(
        words=_profanity_list_strategy,
        non_profane_word=st.text(
            alphabet=string.ascii_lowercase,
            min_size=4,
            max_size=12,
        ).filter(lambda w: w.isalpha()),
    )
    @settings(max_examples=200)
    def test_non_matching_word_never_flagged(
        self, words: frozenset[str], non_profane_word: str
    ):
        """A word not in the profanity list and whose stem doesn't match any
        profanity stem is never flagged."""
        profanity_list = _make_profanity_list_from_words(words)

        # Ensure the word is genuinely not profane (not in list, stem not in stems)
        assume(non_profane_word not in profanity_list.words)
        assume(_stemmer.stem(non_profane_word) not in profanity_list.stems)

        detector = ProfanityDetector(profanity_list=profanity_list)

        assert not detector._is_profane(non_profane_word), (
            f"Word '{non_profane_word}' is not in the profanity list and its "
            f"stem '{_stemmer.stem(non_profane_word)}' doesn't match any "
            f"profanity stem, but it was flagged as profane"
        )

    @given(words=_profanity_list_strategy)
    @settings(max_examples=100)
    def test_scunthorpe_embedded_word_not_flagged(self, words: frozenset[str]):
        """The classic Scunthorpe problem: embedding profane words within
        innocent longer words should never trigger a false positive."""
        profanity_list = _make_profanity_list_from_words(words)
        detector = ProfanityDetector(profanity_list=profanity_list)

        for word in words:
            # Create a word that embeds the profane word in the middle
            embedded = "abc" + word + "xyz"
            embedded_stem = _stemmer.stem(embedded.lower())

            # Only assert non-detection when the compound isn't accidentally profane
            if embedded.lower() not in profanity_list.words and embedded_stem not in profanity_list.stems:
                assert not detector._is_profane(embedded), (
                    f"'{embedded}' embeds '{word}' but should NOT be flagged"
                )
