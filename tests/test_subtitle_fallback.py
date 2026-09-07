"""Tests for the subtitle-based censoring fallback.

When a profane word appears in the subtitles but the audio transcription missed
it (e.g. muttered under the breath), the engine still censors it using the
subtitle cue timing. These tests exercise CensorEngine._subtitle_fallback_detections
directly — no audio/Whisper needed.
"""

from nltk.stem import SnowballStemmer

from video_profanity_censor.censor_engine import CensorEngine
from video_profanity_censor.models import (
    CensorMode,
    Detection,
    ProfanityList,
    ProfanityRegion,
    SubtitleCue,
    SubtitleScanResult,
    TimestampRange,
)

_stemmer = SnowballStemmer("english")


def _make_list(*words: str) -> ProfanityList:
    lower = {w.lower() for w in words}
    stems = {_stemmer.stem(w) for w in lower}
    return ProfanityList(words=lower, stems=stems, source_path=None)


def _scan_result_with_cue(text: str, start: float, end: float) -> SubtitleScanResult:
    cue = SubtitleCue(index=1, start=start, end=end, text=text)
    region = ProfanityRegion(start=max(0.0, start - 2.0), end=end + 2.0, source_cues=[cue])
    return SubtitleScanResult(
        has_subtitles=True,
        subtitle_source="embedded:2",
        profanity_regions=[region],
        total_cues_scanned=1,
        profane_cues_found=1,
        skipped_speech_recognition=False,
    )


class TestSubtitleFallback:
    def setup_method(self):
        self.engine = CensorEngine()
        self.plist = _make_list("damn", "hell")

    def test_missed_word_is_censored_from_cue_timing(self):
        """A profane cue that the audio transcription missed produces a fallback
        detection using the cue's own start/end."""
        scan = _scan_result_with_cue("oh damn that hurt", 19.711, 20.231)
        existing: list[Detection] = []  # Whisper found nothing here

        fallback = self.engine._subtitle_fallback_detections(
            scan, existing, self.plist, CensorMode.MUTE
        )

        assert len(fallback) == 1
        d = fallback[0]
        assert d.word == "damn"
        assert d.timestamp_range.start == 19.711
        assert d.timestamp_range.end == 20.231
        assert d.censor_action == CensorMode.MUTE

    def test_already_detected_word_is_not_duplicated(self):
        """If the audio transcription already produced an overlapping detection,
        the fallback does not add a duplicate."""
        scan = _scan_result_with_cue("oh damn that hurt", 19.711, 20.231)
        existing = [
            Detection(
                word="damn",
                timestamp_range=TimestampRange(start=19.8, end=20.1),
                censor_action=CensorMode.MUTE,
            )
        ]

        fallback = self.engine._subtitle_fallback_detections(
            scan, existing, self.plist, CensorMode.MUTE
        )

        assert fallback == []

    def test_non_overlapping_existing_detection_does_not_suppress(self):
        """An existing detection elsewhere in the file does not suppress a fallback
        for a different, uncovered cue."""
        scan = _scan_result_with_cue("go to hell", 100.0, 101.0)
        existing = [
            Detection(
                word="damn",
                timestamp_range=TimestampRange(start=10.0, end=10.5),
                censor_action=CensorMode.MUTE,
            )
        ]

        fallback = self.engine._subtitle_fallback_detections(
            scan, existing, self.plist, CensorMode.MUTE
        )

        assert len(fallback) == 1
        assert fallback[0].word == "hell"

    def test_clean_cue_produces_no_fallback(self):
        """A cue with no profanity yields no fallback detection (defensive; the
        scanner would not normally include it)."""
        scan = _scan_result_with_cue("a perfectly clean line", 5.0, 6.0)

        fallback = self.engine._subtitle_fallback_detections(
            scan, [], self.plist, CensorMode.MUTE
        )

        assert fallback == []

    def test_no_subtitle_scan_result_yields_nothing(self):
        """When subtitles were not scanned (None), there is no fallback."""
        assert self.engine._subtitle_fallback_detections(
            None, [], self.plist, CensorMode.MUTE
        ) == []

    def test_scan_result_without_regions_yields_nothing(self):
        """A scan result with no profanity regions yields no fallback."""
        empty = SubtitleScanResult(has_subtitles=True, profanity_regions=[])
        assert self.engine._subtitle_fallback_detections(
            empty, [], self.plist, CensorMode.MUTE
        ) == []

    def test_stemmed_variant_in_subtitle_is_caught(self):
        """Morphological variants match via the shared stemmer (e.g. 'damned')."""
        scan = _scan_result_with_cue("you are damned", 30.0, 31.0)

        fallback = self.engine._subtitle_fallback_detections(
            scan, [], self.plist, CensorMode.MUTE
        )

        assert len(fallback) == 1
        assert fallback[0].word == "damned"

    def test_duplicate_cue_across_regions_counted_once(self):
        """If the same cue appears in multiple regions, it yields a single detection."""
        cue = SubtitleCue(index=1, start=40.0, end=41.0, text="damn it")
        region_a = ProfanityRegion(start=38.0, end=43.0, source_cues=[cue])
        region_b = ProfanityRegion(start=38.0, end=43.0, source_cues=[cue])
        scan = SubtitleScanResult(
            has_subtitles=True,
            profanity_regions=[region_a, region_b],
            total_cues_scanned=1,
            profane_cues_found=1,
        )

        fallback = self.engine._subtitle_fallback_detections(
            scan, [], self.plist, CensorMode.MUTE
        )

        assert len(fallback) == 1
