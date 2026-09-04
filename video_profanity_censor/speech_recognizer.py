"""Speech recognition component using Whisper via faster-whisper (CTranslate2)."""

import logging
import os
import tempfile
from pathlib import Path

from video_profanity_censor.models import (
    AccelerationBackend,
    ProfanityRegion,
    ProcessingStage,
    ProgressCallback,
    TimestampRange,
    TranscribedWord,
    TranscriptionResult,
)

logger = logging.getLogger(__name__)


class SpeechRecognizer:
    """Transcribes audio to text with word-level timestamps using Whisper.

    Uses faster-whisper (CTranslate2) for optimized CPU inference with INT8
    quantization, providing ~4x speedup over vanilla Whisper.
    """

    def __init__(
        self,
        model_size: str = "medium",
        backend: AccelerationBackend = AccelerationBackend.CPU,
    ):
        """Initialize with Whisper model size.

        Args:
            model_size: Whisper model size (tiny/base/small/medium/large).
            backend: Accepted for API compatibility. Only CPU is used.
        """
        self._model_size = model_size
        self._backend = AccelerationBackend.CPU  # Always CPU
        self._model = None
        self._fell_back_to_cpu = False

    def __enter__(self) -> "SpeechRecognizer":
        """Enter the context manager, returning this recognizer."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Release the underlying model when leaving the context."""
        self.release_model()

    def release_model(self) -> None:
        """Releases the loaded Whisper model and frees its memory.

        The faster-whisper/CTranslate2 model can consume several GB of RAM
        (the MEDIUM model in particular). It is only needed during
        transcription — profanity detection operates on the TranscriptionResult,
        not the model — so releasing it before the audio censoring stage frees
        that memory and prevents it from stacking with the censoring buffer.

        Idempotent: safe to call multiple times. Does nothing if no model is loaded.
        """
        import gc

        if self._model is None:
            return

        # Drop the reference to the underlying CTranslate2 model, then the dict.
        self._model.pop("whisper_model", None)
        self._model = None

        # Force a collection so the (potentially multi-GB) native model is freed.
        gc.collect()

        logger.info("Whisper model released and memory reclaimed.")

    @property
    def model_size(self) -> str:
        """The Whisper model size being used."""
        return self._model_size

    @property
    def backend(self) -> AccelerationBackend:
        """The acceleration backend being used."""
        return self._backend

    def transcribe(
        self,
        audio_path: Path,
        regions: list[ProfanityRegion] | None = None,
        progress_callback: ProgressCallback = None,
    ) -> TranscriptionResult:
        """Transcribes audio to text with word-level timestamps.

        If regions are provided, transcribes only those audio segments (targeted mode).
        Otherwise, transcribes the entire audio track (full mode).

        Args:
            audio_path: Path to the WAV audio file to transcribe.
            regions: Optional list of ProfanityRegions to transcribe (targeted mode).
            progress_callback: Optional callback for progress updates.

        Returns:
            TranscriptionResult with word-level timestamps and metadata.
        """
        if not audio_path.exists():
            return TranscriptionResult(
                has_speech=False,
                warnings=[f"Audio file not found: {audio_path}"],
            )

        # Load the model
        self._ensure_model_loaded()

        if regions:
            return self._transcribe_regions(audio_path, regions, progress_callback)
        else:
            return self._transcribe_full(audio_path, progress_callback)

    def _ensure_model_loaded(self) -> None:
        """Loads the model if not already loaded."""
        if self._model is not None:
            return
        self._load_model_cpu(self._model_size)

    def _load_model_cpu(self, model_size: str) -> None:
        """Loads Whisper model via CTranslate2/faster-whisper for optimized CPU inference.

        Uses faster-whisper which leverages CTranslate2 for INT8 quantization
        and optimized CPU kernels (~4x faster than vanilla Whisper).
        """
        try:
            from faster_whisper import WhisperModel
        except ImportError as e:
            raise RuntimeError(
                "faster-whisper is not installed. "
                "Install it with: pip install faster-whisper"
            ) from e

        logger.info(f"Loading Whisper '{model_size}' model with CPU backend (faster-whisper)...")

        model = WhisperModel(
            model_size,
            device="cpu",
            compute_type="int8",
        )

        self._model = {
            "type": "cpu",
            "model_size": model_size,
            "whisper_model": model,
        }
        logger.info(f"Whisper '{model_size}' model loaded on CPU backend.")

    def _transcribe_full(
        self,
        audio_path: Path,
        progress_callback: ProgressCallback = None,
    ) -> TranscriptionResult:
        """Transcribes the entire audio file (full mode).

        Args:
            audio_path: Path to the WAV audio file.
            progress_callback: Optional progress callback.

        Returns:
            TranscriptionResult with word-level timestamps.
        """
        if progress_callback:
            progress_callback(ProcessingStage.TRANSCRIPTION, 0.0, "Starting full transcription...")

        try:
            result = self._run_transcription(audio_path, progress_callback)
        except MemoryError:
            result = self._handle_oom(audio_path, progress_callback)
        except RuntimeError as e:
            if "out of memory" in str(e).lower() or "oom" in str(e).lower():
                result = self._handle_oom(audio_path, progress_callback)
            else:
                raise

        if progress_callback:
            progress_callback(ProcessingStage.TRANSCRIPTION, 100.0, "Transcription complete.")

        return result

    def _transcribe_regions(
        self,
        audio_path: Path,
        regions: list[ProfanityRegion],
        progress_callback: ProgressCallback = None,
    ) -> TranscriptionResult:
        """Transcribes only specified audio regions (targeted mode).

        Extracts each region from the audio file and transcribes them individually,
        then combines the results with correct time offsets.

        Args:
            audio_path: Path to the WAV audio file.
            regions: List of ProfanityRegions to transcribe.
            progress_callback: Optional progress callback.

        Returns:
            TranscriptionResult with combined word-level timestamps.
        """
        if progress_callback:
            progress_callback(
                ProcessingStage.TRANSCRIPTION,
                0.0,
                f"Starting targeted transcription of {len(regions)} region(s)...",
            )

        all_words: list[TranscribedWord] = []
        all_warnings: list[str] = []
        all_skipped: list[TimestampRange] = []
        has_any_speech = False

        for i, region in enumerate(regions):
            if progress_callback:
                percent = (i / len(regions)) * 100.0
                progress_callback(
                    ProcessingStage.TRANSCRIPTION,
                    percent,
                    f"Transcribing region {i + 1}/{len(regions)} "
                    f"({region.start:.1f}s - {region.end:.1f}s)...",
                )

            try:
                region_result = self._transcribe_region_segment(
                    audio_path, region
                )

                if region_result.has_speech:
                    has_any_speech = True

                all_words.extend(region_result.words)
                all_warnings.extend(region_result.warnings)
                all_skipped.extend(region_result.skipped_ranges)

            except Exception as e:
                # Skip corrupted segments and continue with warning (Req 2.8)
                warning_msg = (
                    f"Failed to transcribe region {region.start:.3f}s - "
                    f"{region.end:.3f}s: {e}"
                )
                logger.warning(warning_msg)
                all_warnings.append(warning_msg)
                all_skipped.append(
                    TimestampRange(start=region.start, end=region.end)
                )

        if progress_callback:
            progress_callback(ProcessingStage.TRANSCRIPTION, 100.0, "Transcription complete.")

        # Add fallback warning if we fell back to CPU during processing
        if self._fell_back_to_cpu:
            all_warnings.append(
                "GPU out-of-memory encountered. Fell back to CPU backend for processing."
            )

        return TranscriptionResult(
            words=all_words,
            has_speech=has_any_speech,
            warnings=all_warnings,
            skipped_ranges=all_skipped,
        )

    def _transcribe_region_segment(
        self,
        audio_path: Path,
        region: ProfanityRegion,
    ) -> TranscriptionResult:
        """Transcribes a single audio region by extracting it and running Whisper.

        The region audio is extracted to a temporary file, transcribed,
        and the resulting timestamps are offset back to the original audio timeline.

        Args:
            audio_path: Path to the full audio file.
            region: The ProfanityRegion to transcribe.

        Returns:
            TranscriptionResult with timestamps adjusted to original timeline.
        """
        import subprocess

        # Extract the region to a temporary file using ffmpeg
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)

        try:
            duration = region.end - region.start
            cmd = [
                "ffmpeg", "-y",
                "-i", str(audio_path),
                "-ss", str(region.start),
                "-t", str(duration),
                "-acodec", "pcm_s16le",
                "-ar", "16000",
                "-ac", "1",
                str(tmp_path),
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60,
            )

            if result.returncode != 0:
                raise RuntimeError(
                    f"FFmpeg failed to extract region: {result.stderr.strip()}"
                )

            # Transcribe the extracted segment
            try:
                segment_result = self._run_transcription(tmp_path, progress_callback=None)
            except MemoryError:
                segment_result = self._handle_oom(tmp_path, progress_callback=None)
            except RuntimeError as e:
                if "out of memory" in str(e).lower() or "oom" in str(e).lower():
                    segment_result = self._handle_oom(tmp_path, progress_callback=None)
                else:
                    raise

            # Offset timestamps back to original timeline
            offset_words = []
            
            # Get the earliest source cue start time — words detected before this
            # are likely Whisper timestamp errors (anchoring to segment start)
            earliest_cue_start = min(
                (cue.start for cue in region.source_cues),
                default=region.start,
            )
            
            for word in segment_result.words:
                word_start = word.start + region.start
                word_end = word.end + region.start
                
                # If word starts before the actual subtitle cue, Whisper's timestamp
                # is anchored to segment start rather than actual speech timing.
                # Clamp the word to start at the subtitle cue start instead.
                if word_start < earliest_cue_start:
                    word_start = earliest_cue_start
                    # If end is also before or at cue start, shift end to maintain
                    # a reasonable word duration (use Whisper's duration estimate)
                    word_duration = word.end - word.start
                    if word_end <= earliest_cue_start:
                        word_end = earliest_cue_start + word_duration
                
                offset_words.append(
                    TranscribedWord(
                        word=word.word,
                        start=word_start,
                        end=word_end,
                        confidence=word.confidence,
                    )
                )

            # Offset skipped ranges
            offset_skipped = []
            for skipped in segment_result.skipped_ranges:
                offset_skipped.append(
                    TimestampRange(
                        start=skipped.start + region.start,
                        end=skipped.end + region.start,
                    )
                )

            return TranscriptionResult(
                words=offset_words,
                has_speech=segment_result.has_speech,
                warnings=segment_result.warnings,
                skipped_ranges=offset_skipped,
            )
        finally:
            # Clean up temporary file
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    def _run_transcription(
        self,
        audio_path: Path,
        progress_callback: ProgressCallback = None,
    ) -> TranscriptionResult:
        """Runs the actual Whisper transcription on an audio file.

        Args:
            audio_path: Path to the audio file to transcribe.
            progress_callback: Optional progress callback.

        Returns:
            TranscriptionResult with word-level timestamps.
        """
        if self._model is None:
            raise RuntimeError("Model not loaded. Call _ensure_model_loaded() first.")

        return self._transcribe_with_faster_whisper(audio_path, progress_callback)

    def _transcribe_with_faster_whisper(
        self,
        audio_path: Path,
        progress_callback: ProgressCallback = None,
    ) -> TranscriptionResult:
        """Transcribes audio using faster-whisper (CTranslate2 CPU backend).

        Args:
            audio_path: Path to the audio file.
            progress_callback: Optional progress callback.

        Returns:
            TranscriptionResult with word-level timestamps.
        """
        whisper_model = self._model["whisper_model"]

        try:
            segments, info = whisper_model.transcribe(
                str(audio_path),
                word_timestamps=True,
                language="en",
            )
        except Exception as e:
            error_msg = str(e).lower()
            if "out of memory" in error_msg or "oom" in error_msg:
                raise MemoryError(f"Out of memory during transcription: {e}") from e
            raise

        words: list[TranscribedWord] = []
        warnings: list[str] = []
        skipped_ranges: list[TimestampRange] = []
        has_speech = False

        try:
            for segment in segments:
                if segment.words is None:
                    continue

                has_speech = True

                for word_info in segment.words:
                    # Skip words with very low confidence or empty text
                    word_text = word_info.word.strip()
                    if not word_text:
                        continue

                    words.append(
                        TranscribedWord(
                            word=word_text,
                            start=word_info.start,
                            end=word_info.end,
                            confidence=word_info.probability,
                        )
                    )
        except Exception as e:
            # Handle corrupted segment errors (Req 2.8)
            error_msg = str(e).lower()
            if "out of memory" in error_msg or "oom" in error_msg:
                raise MemoryError(f"Out of memory during transcription: {e}") from e
            warning_msg = f"Error processing transcription segment: {e}"
            logger.warning(warning_msg)
            warnings.append(warning_msg)

        # Handle no speech detected (Req 2.6)
        if not has_speech:
            return TranscriptionResult(
                words=[],
                has_speech=False,
                warnings=["No speech detected in audio."],
            )

        return TranscriptionResult(
            words=words,
            has_speech=True,
            warnings=warnings,
            skipped_ranges=skipped_ranges,
        )

    def _handle_oom(
        self,
        audio_path: Path,
        progress_callback: ProgressCallback = None,
    ) -> TranscriptionResult:
        """Handles GPU out-of-memory by falling back to CPU backend.

        Logs a warning, switches to CPU, reloads the model, and retries transcription.

        Args:
            audio_path: Path to the audio file to retry.
            progress_callback: Optional progress callback.

        Returns:
            TranscriptionResult from CPU backend transcription.
        """
        logger.warning(
            "GPU out-of-memory encountered during inference. "
            "Falling back to CPU backend (CTranslate2/faster-whisper)."
        )

        if progress_callback:
            progress_callback(
                ProcessingStage.TRANSCRIPTION,
                -1.0,  # Indeterminate progress during fallback
                "GPU out-of-memory. Falling back to CPU backend...",
            )

        # Switch to CPU backend
        self._backend = AccelerationBackend.CPU
        self._model = None
        self._fell_back_to_cpu = True

        # Reload model on CPU
        self._load_model_cpu(self._model_size)

        # Retry transcription on CPU
        result = self._run_transcription(audio_path, progress_callback)

        # Add OOM fallback warning
        result.warnings.append(
            "GPU out-of-memory encountered. Fell back to CPU backend for processing."
        )

        return result
