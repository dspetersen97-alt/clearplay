"""Property-based tests for SpeechRecognizer.

**Validates: Requirements 9.10**
"""

from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import hypothesis.strategies as st
from hypothesis import given, settings, assume

from video_profanity_censor.models import (
    AccelerationBackend,
    TranscribedWord,
    TranscriptionResult,
)
from video_profanity_censor.speech_recognizer import SpeechRecognizer


# --- Strategies ---

# Strategy for Whisper model sizes
model_sizes = st.sampled_from(["tiny", "base", "small", "medium", "large"])

# Strategy for generating transcribed words (simulating CPU fallback results)
@st.composite
def transcribed_words(draw, min_words=1, max_words=20):
    """Generate a list of transcribed words with valid timestamps."""
    num_words = draw(st.integers(min_value=min_words, max_value=max_words))
    words = []
    current_time = 0.0

    for _ in range(num_words):
        word_text = draw(
            st.text(
                alphabet="abcdefghijklmnopqrstuvwxyz",
                min_size=2,
                max_size=10,
            )
        )
        duration = draw(st.floats(min_value=0.1, max_value=2.0, allow_nan=False, allow_infinity=False))
        confidence = draw(st.floats(min_value=0.5, max_value=1.0, allow_nan=False, allow_infinity=False))

        start = current_time
        end = current_time + duration
        words.append(
            TranscribedWord(
                word=word_text,
                start=round(start, 3),
                end=round(end, 3),
                confidence=round(confidence, 3),
            )
        )
        # Small gap between words
        gap = draw(st.floats(min_value=0.0, max_value=0.5, allow_nan=False, allow_infinity=False))
        current_time = end + gap

    return words


# Strategy for OOM error types
oom_error_types = st.sampled_from([
    MemoryError("GPU out of memory"),
    RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB"),
    RuntimeError("DirectML: out of memory during inference"),
    RuntimeError("OOM when trying to allocate tensor"),
    MemoryError("Out of memory during transcription: GPU VRAM exhausted"),
])
