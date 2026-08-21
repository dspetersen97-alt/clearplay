"""Profanity list loader with NLTK stemming for morphological matching.

This module handles loading profanity word lists from file paths or the bundled
default list, and pre-computes stems for morphological variant detection.
"""

from pathlib import Path

from nltk.stem import SnowballStemmer

from video_profanity_censor.models import ProfanityList

# Path to the bundled default profanity list
_DEFAULT_LIST_PATH = Path(__file__).parent / "data" / "default_profanity_list.txt"

# Shared stemmer instance
_stemmer = SnowballStemmer("english")


class ProfanityListError(Exception):
    """Raised when a profanity list cannot be loaded or is invalid."""


def _compute_stems(words: set[str]) -> set[str]:
    """Pre-compute stems for all words in the set using NLTK's Snowball stemmer.

    Args:
        words: Set of lowercase profanity words.

    Returns:
        Set of stems derived from the input words.
    """
    stems: set[str] = set()
    for word in words:
        stem = _stemmer.stem(word)
        stems.add(stem)
    return stems


def load_profanity_list(file_path: Path | None = None) -> ProfanityList:
    """Load a profanity list from a file or the bundled default list.

    Reads a text file with one word per line, normalizes to lowercase,
    and pre-computes stems for morphological matching.

    Args:
        file_path: Path to a custom profanity list file. If None, uses the
                   bundled default list.

    Returns:
        A ProfanityList with words, pre-computed stems, and source path.

    Raises:
        ProfanityListError: If the file doesn't exist, can't be read, or
                           contains no entries (Req 3.6).
    """
    source_path = file_path if file_path is not None else _DEFAULT_LIST_PATH

    if not source_path.exists():
        raise ProfanityListError(
            f"Profanity list file not found: {source_path}"
        )

    try:
        text = source_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as e:
        raise ProfanityListError(
            f"Cannot read profanity list file: {source_path} ({e})"
        ) from e

    # Parse one word per line, strip whitespace, normalize to lowercase
    words: set[str] = set()
    for line in text.splitlines():
        word = line.strip().lower()
        if word and not word.startswith("#"):  # Skip empty lines and comments
            words.add(word)

    # Validate that the list contains at least one entry (Req 3.6)
    if not words:
        raise ProfanityListError(
            "Profanity list is empty. Detection cannot proceed without "
            "at least one entry in the profanity list."
        )

    # Pre-compute stems for morphological matching
    stems = _compute_stems(words)

    return ProfanityList(
        words=words,
        stems=stems,
        source_path=source_path,
    )
