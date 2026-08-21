"""Property-based tests for SubtitleScanner.

Tests Property 12: Subtitle Profanity Matching Parity and
Property 14: Profanity Region Merging from the design document.
"""

import string

import hypothesis.strategies as st
from hypothesis import assume, given, settings
from nltk.stem import SnowballStemmer

from video_profanity_censor.models import ProfanityList, SubtitleCue
from video_profanity_censor.profanity_detector import ProfanityDetector
from video_profanity_censor.subtitle_scanner import SubtitleScanner


# Shared stemmer matching the production code
_stemmer = SnowballStemmer("english")

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

# Common morphological suffixes for English words
_MORPHOLOGICAL_SUFFIXES = ["s", "ed", "ing", "er", "est", "ly", "ness", "ment"]

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


# ============================================================================
# Property 12: Subtitle Profanity Matching Parity
# ============================================================================


class TestSubtitleProfanityMatchingParity:
    """Property 12: Subtitle Profanity Matching Parity.

    **Validates: Requirements 8.2**

    For any subtitle text and profanity list, the SubtitleScanner SHALL identify
    the same words as profane that the ProfanityDetector would identify for the
    same text — that is, both components apply identical case-insensitive
    whole-word matching with morphological variant support.
    """

    @given(words=_profanity_list_strategy)
    @settings(max_examples=200)
    def test_exact_match_parity(self, words: frozenset[str]):
        """Both SubtitleScanner and ProfanityDetector flag exact-match profane words."""
        profanity_list = _make_profanity_list_from_words(words)
        scanner = SubtitleScanner(profanity_list=profanity_list)
        detector = ProfanityDetector(profanity_list=profanity_list)

        for word in words:
            scanner_result = scanner._is_profane(word)
            detector_result = detector._is_profane(word)
            assert scanner_result == detector_result, (
                f"Parity violation for exact match '{word}': "
                f"scanner={scanner_result}, detector={detector_result}"
            )

    @given(words=_profanity_list_strategy)
    @settings(max_examples=200)
    def test_case_insensitive_parity(self, words: frozenset[str]):
        """Both components produce the same result for case variants of profane words."""
        profanity_list = _make_profanity_list_from_words(words)
        scanner = SubtitleScanner(profanity_list=profanity_list)
        detector = ProfanityDetector(profanity_list=profanity_list)

        for word in words:
            for variant in [word.upper(), word.title(), word.swapcase()]:
                scanner_result = scanner._is_profane(variant)
                detector_result = detector._is_profane(variant)
                assert scanner_result == detector_result, (
                    f"Parity violation for case variant '{variant}' of '{word}': "
                    f"scanner={scanner_result}, detector={detector_result}"
                )

    @given(words=_profanity_list_strategy, suffix=_morphological_suffix_strategy)
    @settings(max_examples=200)
    def test_morphological_variant_parity(self, words: frozenset[str], suffix: str):
        """Both components produce the same result for morphological variants."""
        profanity_list = _make_profanity_list_from_words(words)
        scanner = SubtitleScanner(profanity_list=profanity_list)
        detector = ProfanityDetector(profanity_list=profanity_list)

        for word in words:
            variant = word + suffix
            scanner_result = scanner._is_profane(variant)
            detector_result = detector._is_profane(variant)
            assert scanner_result == detector_result, (
                f"Parity violation for morphological variant '{variant}' "
                f"(base='{word}', suffix='{suffix}'): "
                f"scanner={scanner_result}, detector={detector_result}"
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
    def test_non_profane_word_parity(self, words: frozenset[str], non_profane_word: str):
        """Both components agree that non-profane words are not flagged."""
        profanity_list = _make_profanity_list_from_words(words)

        # Ensure the word is genuinely not profane
        assume(non_profane_word not in profanity_list.words)
        assume(_stemmer.stem(non_profane_word) not in profanity_list.stems)

        scanner = SubtitleScanner(profanity_list=profanity_list)
        detector = ProfanityDetector(profanity_list=profanity_list)

        scanner_result = scanner._is_profane(non_profane_word)
        detector_result = detector._is_profane(non_profane_word)
        assert scanner_result == detector_result, (
            f"Parity violation for non-profane word '{non_profane_word}': "
            f"scanner={scanner_result}, detector={detector_result}"
        )

    @given(
        words=_profanity_list_strategy,
        prefix=_safe_padding,
    )
    @settings(max_examples=200)
    def test_substring_false_positive_parity(self, words: frozenset[str], prefix: str):
        """Both components agree on substring words (Scunthorpe problem avoidance)."""
        profanity_list = _make_profanity_list_from_words(words)
        scanner = SubtitleScanner(profanity_list=profanity_list)
        detector = ProfanityDetector(profanity_list=profanity_list)

        for word in words:
            compound = prefix + word
            scanner_result = scanner._is_profane(compound)
            detector_result = detector._is_profane(compound)
            assert scanner_result == detector_result, (
                f"Parity violation for substring compound '{compound}' "
                f"(prefix='{prefix}', profane='{word}'): "
                f"scanner={scanner_result}, detector={detector_result}"
            )

    @given(
        words=_profanity_list_strategy,
        sentence_words=st.lists(
            st.text(alphabet=string.ascii_lowercase, min_size=3, max_size=10).filter(lambda w: w.isalpha()),
            min_size=1,
            max_size=8,
        ),
    )
    @settings(max_examples=200)
    def test_cue_level_profanity_detection_parity(
        self, words: frozenset[str], sentence_words: list[str]
    ):
        """SubtitleScanner._cue_contains_profanity agrees with ProfanityDetector._is_profane
        applied word-by-word to the same text.

        This tests the full cue scanning path, verifying that the scanner's
        word splitting + matching produces the same outcome as checking each
        word individually with the detector.
        """
        profanity_list = _make_profanity_list_from_words(words)
        scanner = SubtitleScanner(profanity_list=profanity_list)
        detector = ProfanityDetector(profanity_list=profanity_list)

        # Build subtitle text from the generated words
        text = " ".join(sentence_words)
        cue = SubtitleCue(index=1, start=0.0, end=1.0, text=text)

        # Scanner says cue contains profanity
        scanner_cue_profane = scanner._cue_contains_profanity(cue)

        # Detector says at least one word is profane (using same word-splitting)
        detector_any_profane = any(
            detector._is_profane(w) for w in text.split()
        )

        assert scanner_cue_profane == detector_any_profane, (
            f"Cue-level parity violation for text '{text}': "
            f"scanner._cue_contains_profanity={scanner_cue_profane}, "
            f"detector word-by-word={detector_any_profane}"
        )

    @given(words=_profanity_list_strategy)
    @settings(max_examples=200)
    def test_punctuation_handling_parity(self, words: frozenset[str]):
        """Both components handle punctuation-wrapped words identically."""
        profanity_list = _make_profanity_list_from_words(words)
        scanner = SubtitleScanner(profanity_list=profanity_list)
        detector = ProfanityDetector(profanity_list=profanity_list)

        punctuation_wrappers = [
            ('', '.'),
            ('', ','),
            ('', '!'),
            ('', '?'),
            ('"', '"'),
            ("'", "'"),
            ('(', ')'),
        ]

        for word in words:
            for prefix_punct, suffix_punct in punctuation_wrappers:
                wrapped = f"{prefix_punct}{word}{suffix_punct}"
                scanner_result = scanner._is_profane(wrapped)
                detector_result = detector._is_profane(wrapped)
                assert scanner_result == detector_result, (
                    f"Parity violation for punctuation-wrapped word '{wrapped}': "
                    f"scanner={scanner_result}, detector={detector_result}"
                )


# ============================================================================
# Property 13: Profanity Region Buffer Expansion
# ============================================================================


class TestProfanityRegionBufferExpansion:
    """Property 13: Profanity Region Buffer Expansion.

    **Validates: Requirements 8.3**

    For any SubtitleCue timing and buffer value, the expanded region SHALL have:
    - start = max(0, cue.start - buffer)
    - end = cue.end + buffer
    - Region start SHALL never be negative.
    """

    @given(
        cue_start=st.floats(min_value=0.0, max_value=3600.0, allow_nan=False, allow_infinity=False),
        cue_duration=st.floats(min_value=0.01, max_value=60.0, allow_nan=False, allow_infinity=False),
        buffer_seconds=st.floats(min_value=0.0, max_value=30.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=200)
    def test_buffer_expansion_correctness(self, cue_start: float, cue_duration: float, buffer_seconds: float):
        """Expanded region start = max(0, cue.start - buffer) and end = cue.end + buffer.

        **Validates: Requirements 8.3**
        """
        cue_end = cue_start + cue_duration
        cue = SubtitleCue(index=1, start=cue_start, end=cue_end, text="profane")

        profanity_list = ProfanityList(words={"profane"}, stems={"profan"})
        scanner = SubtitleScanner(profanity_list=profanity_list, buffer_seconds=buffer_seconds)

        # Use a single cue to isolate buffer expansion from merging
        regions = scanner._expand_and_merge_regions([cue], buffer_seconds)

        assert len(regions) == 1, f"Expected 1 region for a single cue, got {len(regions)}"

        region = regions[0]
        expected_start = max(0.0, cue_start - buffer_seconds)
        expected_end = cue_end + buffer_seconds

        assert abs(region.start - expected_start) < 1e-9, (
            f"Region start mismatch: got {region.start}, expected {expected_start} "
            f"(cue.start={cue_start}, buffer={buffer_seconds})"
        )
        assert abs(region.end - expected_end) < 1e-9, (
            f"Region end mismatch: got {region.end}, expected {expected_end} "
            f"(cue.end={cue_end}, buffer={buffer_seconds})"
        )

    @given(
        cue_start=st.floats(min_value=0.0, max_value=3600.0, allow_nan=False, allow_infinity=False),
        cue_duration=st.floats(min_value=0.01, max_value=60.0, allow_nan=False, allow_infinity=False),
        buffer_seconds=st.floats(min_value=0.0, max_value=30.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=200)
    def test_region_start_never_negative(self, cue_start: float, cue_duration: float, buffer_seconds: float):
        """Region start SHALL never be negative, even when buffer exceeds cue start time.

        **Validates: Requirements 8.3**
        """
        cue_end = cue_start + cue_duration
        cue = SubtitleCue(index=1, start=cue_start, end=cue_end, text="profane")

        profanity_list = ProfanityList(words={"profane"}, stems={"profan"})
        scanner = SubtitleScanner(profanity_list=profanity_list, buffer_seconds=buffer_seconds)

        regions = scanner._expand_and_merge_regions([cue], buffer_seconds)

        assert len(regions) == 1
        assert regions[0].start >= 0.0, (
            f"Region start is negative: {regions[0].start} "
            f"(cue.start={cue_start}, buffer={buffer_seconds})"
        )

    @given(
        cue_start=st.floats(min_value=0.0, max_value=5.0, allow_nan=False, allow_infinity=False),
        cue_duration=st.floats(min_value=0.01, max_value=10.0, allow_nan=False, allow_infinity=False),
        buffer_seconds=st.floats(min_value=5.0, max_value=30.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=200)
    def test_buffer_larger_than_start_clamps_to_zero(self, cue_start: float, cue_duration: float, buffer_seconds: float):
        """When buffer exceeds cue start time, region start is clamped to 0.

        **Validates: Requirements 8.3**
        """
        assume(buffer_seconds > cue_start)

        cue_end = cue_start + cue_duration
        cue = SubtitleCue(index=1, start=cue_start, end=cue_end, text="profane")

        profanity_list = ProfanityList(words={"profane"}, stems={"profan"})
        scanner = SubtitleScanner(profanity_list=profanity_list, buffer_seconds=buffer_seconds)

        regions = scanner._expand_and_merge_regions([cue], buffer_seconds)

        assert len(regions) == 1
        assert regions[0].start == 0.0, (
            f"Region start should be clamped to 0.0 when buffer ({buffer_seconds}) > "
            f"cue.start ({cue_start}), but got {regions[0].start}"
        )


# ============================================================================
# Property 14: Profanity Region Merging
# ============================================================================


# Strategy to generate SubtitleCue objects with valid timings
def subtitle_cue_strategy(min_start: float = 0.0, max_end: float = 3600.0):
    """Generate a SubtitleCue with start < end and reasonable timing values."""
    return st.builds(
        SubtitleCue,
        index=st.integers(min_value=1, max_value=1000),
        start=st.floats(min_value=min_start, max_value=max_end - 0.1, allow_nan=False, allow_infinity=False),
        end=st.floats(min_value=min_start + 0.1, max_value=max_end, allow_nan=False, allow_infinity=False),
        text=st.just("profane"),  # Text doesn't matter for merging logic
    ).filter(lambda cue: cue.start < cue.end)


# Strategy to generate lists of overlapping/adjacent SubtitleCues
def overlapping_cues_strategy():
    """Generate a list of SubtitleCues that may overlap or be adjacent after buffer expansion."""
    return st.lists(
        subtitle_cue_strategy(min_start=0.0, max_end=600.0),
        min_size=1,
        max_size=20,
    )


class TestProfanityRegionMerging:
    """Property 14: Profanity Region Merging.

    **Validates: Requirements 8.5**

    For any list of ProfanityRegions, after merging:
    (a) no two regions in the output SHALL overlap or be adjacent (gap = 0), and
    (b) every time point covered by any input region SHALL also be covered by
        exactly one merged output region.
    The merge operation SHALL not lose or gain any covered time.
    """

    @given(
        cues=overlapping_cues_strategy(),
        buffer_seconds=st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=200)
    def test_no_output_regions_overlap_or_are_adjacent(self, cues: list[SubtitleCue], buffer_seconds: float):
        """After merging, no two output regions should overlap or be adjacent (gap = 0).

        **Validates: Requirements 8.5**
        """
        # Create a scanner with a dummy profanity list (not used for merging)
        profanity_list = ProfanityList(words={"profane"}, stems={"profan"})
        scanner = SubtitleScanner(profanity_list=profanity_list, buffer_seconds=buffer_seconds)

        # Call the merge method directly
        merged = scanner._expand_and_merge_regions(cues, buffer_seconds)

        # Property (a): No two output regions overlap or are adjacent
        for i in range(len(merged) - 1):
            current_region = merged[i]
            next_region = merged[i + 1]
            # The next region's start must be strictly greater than the current region's end
            assert next_region.start > current_region.end, (
                f"Regions {i} and {i+1} overlap or are adjacent: "
                f"region[{i}].end={current_region.end}, region[{i+1}].start={next_region.start}"
            )

    @given(
        cues=overlapping_cues_strategy(),
        buffer_seconds=st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=200)
    def test_every_input_time_point_is_covered_by_output(self, cues: list[SubtitleCue], buffer_seconds: float):
        """Every time point covered by any expanded input region must be covered by
        exactly one merged output region.

        **Validates: Requirements 8.5**
        """
        profanity_list = ProfanityList(words={"profane"}, stems={"profan"})
        scanner = SubtitleScanner(profanity_list=profanity_list, buffer_seconds=buffer_seconds)

        merged = scanner._expand_and_merge_regions(cues, buffer_seconds)

        # Compute the expanded input regions (same logic as _expand_and_merge_regions)
        input_regions = []
        for cue in cues:
            start = max(0.0, cue.start - buffer_seconds)
            end = cue.end + buffer_seconds
            input_regions.append((start, end))

        # For each input expanded region, verify it's fully covered by some merged region
        for inp_start, inp_end in input_regions:
            covered = False
            for region in merged:
                # The input region is covered if merged region fully contains it
                if region.start <= inp_start and region.end >= inp_end:
                    covered = True
                    break
            assert covered, (
                f"Input expanded region [{inp_start}, {inp_end}] is not fully covered "
                f"by any merged region. Merged regions: "
                f"{[(r.start, r.end) for r in merged]}"
            )

    @given(
        cues=overlapping_cues_strategy(),
        buffer_seconds=st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=200)
    def test_merged_regions_do_not_gain_extra_coverage(self, cues: list[SubtitleCue], buffer_seconds: float):
        """The merge operation shall not gain any covered time — every point in a merged
        region must be covered by at least one input expanded region.

        **Validates: Requirements 8.5**
        """
        profanity_list = ProfanityList(words={"profane"}, stems={"profan"})
        scanner = SubtitleScanner(profanity_list=profanity_list, buffer_seconds=buffer_seconds)

        merged = scanner._expand_and_merge_regions(cues, buffer_seconds)

        # Compute the expanded input regions
        input_regions = []
        for cue in cues:
            start = max(0.0, cue.start - buffer_seconds)
            end = cue.end + buffer_seconds
            input_regions.append((start, end))

        # For each merged region, verify its start and end are reachable from input regions
        # The merged region's bounds must be the min start and max end of overlapping inputs
        for region in merged:
            # Region start must equal some input region's start
            starts_match = any(
                abs(region.start - inp_start) < 1e-9
                for inp_start, _ in input_regions
            )
            assert starts_match, (
                f"Merged region start {region.start} doesn't match any input region start. "
                f"Input starts: {[s for s, _ in input_regions]}"
            )

            # Region end must equal some input region's end
            ends_match = any(
                abs(region.end - inp_end) < 1e-9
                for _, inp_end in input_regions
            )
            assert ends_match, (
                f"Merged region end {region.end} doesn't match any input region end. "
                f"Input ends: {[e for _, e in input_regions]}"
            )

    @given(
        cues=overlapping_cues_strategy(),
        buffer_seconds=st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=200)
    def test_merged_output_is_sorted_by_start_time(self, cues: list[SubtitleCue], buffer_seconds: float):
        """Merged output regions must be sorted by start time.

        **Validates: Requirements 8.5**
        """
        profanity_list = ProfanityList(words={"profane"}, stems={"profan"})
        scanner = SubtitleScanner(profanity_list=profanity_list, buffer_seconds=buffer_seconds)

        merged = scanner._expand_and_merge_regions(cues, buffer_seconds)

        for i in range(len(merged) - 1):
            assert merged[i].start <= merged[i + 1].start, (
                f"Merged regions not sorted: region[{i}].start={merged[i].start} > "
                f"region[{i+1}].start={merged[i+1].start}"
            )
