"""Profanity detection component for scanning transcriptions against a word list.

This module implements the ProfanityDetector which scans transcribed words against
a profanity list using case-insensitive whole-word matching with morphological
variant support via NLTK stemming. It avoids substring false positives
(the Scunthorpe problem) by comparing only whole words.
"""

import logging
import re

from nltk.stem import SnowballStemmer

logger = logging.getLogger(__name__)

from video_profanity_censor.models import (
    CensorMode,
    Detection,
    DetectionResult,
    ProfanityList,
    TimestampRange,
    TranscriptionResult,
)
from video_profanity_censor.profanity_list import load_profanity_list

# Shared stemmer instance (same as used in profanity_list.py)
_stemmer = SnowballStemmer("english")

# Pattern to strip leading/trailing punctuation from words
_PUNCTUATION_PATTERN = re.compile(r"^[^\w]+|[^\w]+$")


def _format_timestamp(seconds: float) -> str:
    """Format seconds as MM:SS.mmm."""
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes:02d}:{secs:06.3f}"


class ProfanityDetector:
    """Detects profanity in transcribed text using whole-word matching with stemming.

    The detector checks each transcribed word against the profanity list using:
    1. Case-insensitive exact whole-word matching
    2. Morphological variant matching via NLTK Snowball stemming

    It avoids substring false positives by comparing only the entire word
    (stripped of punctuation) against list entries — "class" will NOT match "ass".
    """

    def __init__(
        self,
        profanity_list: ProfanityList | None = None,
        censor_mode: CensorMode = CensorMode.TONE,
    ):
        """Initialize with a custom or default profanity list.

        Args:
            profanity_list: A pre-loaded ProfanityList. If None, loads the
                           default bundled profanity list.
            censor_mode: The censoring mode to record for detections.
        """
        if profanity_list is not None:
            self._profanity_list = profanity_list
        else:
            self._profanity_list = load_profanity_list()
        self._censor_mode = censor_mode

    def detect(self, transcription: TranscriptionResult) -> DetectionResult:
        """Scan transcription for profanity and return timestamp ranges.

        Iterates through each word in the transcription, checks if it is
        profane, and records the exact TimestampRange from the word-level
        timestamps without modification.

        Args:
            transcription: The transcription result containing word-level
                          timestamps from the speech recognizer.

        Returns:
            A DetectionResult with all detected profane words and their
            timestamp ranges.
        """
        detections: list[Detection] = []

        for transcribed_word in transcription.words:
            if self._is_profane(transcribed_word.word):
                detection = Detection(
                    word=transcribed_word.word,
                    timestamp_range=TimestampRange(
                        start=transcribed_word.start,
                        end=transcribed_word.end,
                    ),
                    censor_action=self._censor_mode,
                )
                detections.append(detection)
                logger.info(
                    "[PROFANITY] Detected \"%s\" at %s - %s",
                    transcribed_word.word,
                    _format_timestamp(transcribed_word.start),
                    _format_timestamp(transcribed_word.end),
                )

        return DetectionResult(
            detections=detections,
            total_words_scanned=len(transcription.words),
        )

    def _is_profane(self, word: str) -> bool:
        """Check if a word matches the profanity list.

        Uses case-insensitive whole-word comparison and morphological variant
        matching via stemming. The key to avoiding the Scunthorpe problem is
        that we compare the ENTIRE word against list entries — we never check
        if a profane term appears as a substring within the word.

        Args:
            word: The word to check (as received from the transcriber).

        Returns:
            True if the word is profane (exact match or stem match), False otherwise.
        """
        # Strip leading/trailing punctuation and normalize to lowercase
        cleaned = _PUNCTUATION_PATTERN.sub("", word).lower()

        # Empty string after cleaning is not profane
        if not cleaned:
            return False

        # Check 1: Exact whole-word match (case-insensitive)
        # This catches direct matches like "damn", "Damn", "DAMN"
        if cleaned in self._profanity_list.words:
            return True

        # Check 2: Morphological variant via stemming
        # This catches variants like "damning", "damned", "fucks", "shitting"
        # The stem of the word is compared against pre-computed stems of the
        # profanity list entries.
        word_stem = _stemmer.stem(cleaned)
        if word_stem in self._profanity_list.stems:
            return True

        return False
