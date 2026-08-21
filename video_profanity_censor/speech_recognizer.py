"""Speech recognition using Whisper via faster-whisper (CPU) or ONNX Runtime DirectML (AMD GPU)."""

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

WHISPER_ONNX_MODEL_MAP = {
    "tiny": "openai/whisper-tiny",
    "base": "openai/whisper-base",
    "small": "openai/whisper-small",
    "medium": "openai/whisper-medium",
    "large": "openai/whisper-large-v3",
}


class SpeechRecognizer:
    """Transcribes audio to text with word-level timestamps using Whisper.

    Supports two backends:
    - CPU: faster-whisper (CTranslate2) with INT8 quantization
    - DirectML: ONNX Runtime with DmlExecutionProvider for AMD GPUs on Windows
    """

    def __init__(
        self,
        model_size: str = "medium",
        backend: AccelerationBackend = AccelerationBackend.CPU,
    ):
        self._model_size = model_size
        self._backend = backend
        self._model = None
        self._fell_back_to_cpu = False

    @property
    def model_size(self) -> str:
        return self._model_size

    @property
    def backend(self) -> AccelerationBackend:
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

        self._ensure_model_loaded()

        if regions:
            return self._transcribe_regions(audio_path, regions, progress_callback)
        else:
            return self._transcribe_full(audio_path, progress_callback)

    def _ensure_model_loaded(self) -> None:
        if self._model is not None:
            return
        if self._backend == AccelerationBackend.DIRECTML:
            self._load_model_directml(self._model_size)
        else:
            self._load_model_cpu(self._model_size)

    def _load_model_cpu(self, model_size: str) -> None:
        """Loads Whisper model via CTranslate2/faster-whisper for optimized CPU inference."""
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

    def _load_model_directml(self, model_size: str) -> None:
        """Loads Whisper ONNX model with DirectML acceleration for AMD GPUs."""
        try:
            from optimum.onnxruntime import ORTModelForSpeechSeq2Seq
            from transformers import AutoProcessor
        except ImportError as e:
            raise RuntimeError(
                "optimum and transformers are required for DirectML backend. "
                "Install with: pip install optimum[onnxruntime] transformers"
            ) from e

        model_id = WHISPER_ONNX_MODEL_MAP.get(model_size)
        if model_id is None:
            raise ValueError(
                f"Unsupported model size '{model_size}' for DirectML. "
                f"Supported: {', '.join(WHISPER_ONNX_MODEL_MAP.keys())}"
            )

        logger.info(
            f"Loading Whisper '{model_size}' ONNX model with DirectML backend "
            f"(model: {model_id})..."
        )

        try:
            processor = AutoProcessor.from_pretrained(model_id)
            model = ORTModelForSpeechSeq2Seq.from_pretrained(
                model_id,
                export=True,
                provider="DmlExecutionProvider",
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to load DirectML model '{model_id}': {e}. "
                "Ensure onnxruntime-directml is installed and a DirectX 12 GPU is available."
            ) from e

        self._model = {
            "type": "directml",
            "model_size": model_size,
            "model": model,
            "processor": processor,
            "model_id": model_id,
        }
        logger.info(f"Whisper '{model_size}' model loaded on DirectML backend.")

    def _transcribe_full(
        self,
        audio_path: Path,
        progress_callback: ProgressCallback = None,
    ) -> TranscriptionResult:
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
        import subprocess

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

            try:
                segment_result = self._run_transcription(tmp_path, progress_callback=None)
            except MemoryError:
                segment_result = self._handle_oom(tmp_path, progress_callback=None)
            except RuntimeError as e:
                if "out of memory" in str(e).lower() or "oom" in str(e).lower():
                    segment_result = self._handle_oom(tmp_path, progress_callback=None)
                else:
                    raise

            offset_words = []

            earliest_cue_start = min(
                (cue.start for cue in region.source_cues),
                default=region.start,
            )

            for word in segment_result.words:
                word_start = word.start + region.start
                word_end = word.end + region.start

                if word_start < earliest_cue_start:
                    word_start = earliest_cue_start
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
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    def _run_transcription(
        self,
        audio_path: Path,
        progress_callback: ProgressCallback = None,
    ) -> TranscriptionResult:
        if self._model is None:
            raise RuntimeError("Model not loaded. Call _ensure_model_loaded() first.")

        if self._model["type"] == "directml":
            return self._transcribe_with_directml(audio_path, progress_callback)
        return self._transcribe_with_faster_whisper(audio_path, progress_callback)

    def _transcribe_with_faster_whisper(
        self,
        audio_path: Path,
        progress_callback: ProgressCallback = None,
    ) -> TranscriptionResult:
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
            error_msg = str(e).lower()
            if "out of memory" in error_msg or "oom" in error_msg:
                raise MemoryError(f"Out of memory during transcription: {e}") from e
            warning_msg = f"Error processing transcription segment: {e}"
            logger.warning(warning_msg)
            warnings.append(warning_msg)

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

    def _transcribe_with_directml(
        self,
        audio_path: Path,
        progress_callback: ProgressCallback = None,
    ) -> TranscriptionResult:
        """Transcribes audio using ONNX Runtime with DirectML (AMD GPU)."""
        import torch
        from transformers import AutomaticSpeechRecognitionPipeline, pipeline

        model = self._model["model"]
        processor = self._model["processor"]
        model_id = self._model["model_id"]

        try:
            pipe = pipeline(
                "automatic-speech-recognition",
                model=model,
                tokenizer=processor.tokenizer,
                feature_extractor=processor.feature_extractor,
            )

            result = pipe(
                str(audio_path),
                return_timestamps="word",
                generate_kwargs={"language": "en"},
            )
        except Exception as e:
            error_msg = str(e).lower()
            if "out of memory" in error_msg or "oom" in error_msg:
                raise MemoryError(f"GPU out of memory during transcription: {e}") from e
            raise

        words: list[TranscribedWord] = []
        has_speech = False

        chunks = result.get("chunks", [])
        for chunk in chunks:
            text = chunk.get("text", "").strip()
            if not text:
                continue

            timestamp = chunk.get("timestamp", (None, None))
            if timestamp is None or len(timestamp) < 2:
                continue

            start, end = timestamp
            if start is None:
                continue
            if end is None:
                end = start + 0.5

            has_speech = True
            words.append(
                TranscribedWord(
                    word=text,
                    start=float(start),
                    end=float(end),
                    confidence=1.0,
                )
            )

        if not has_speech:
            return TranscriptionResult(
                words=[],
                has_speech=False,
                warnings=["No speech detected in audio."],
            )

        return TranscriptionResult(
            words=words,
            has_speech=True,
        )

    def _handle_oom(
        self,
        audio_path: Path,
        progress_callback: ProgressCallback = None,
    ) -> TranscriptionResult:
        logger.warning(
            "GPU out-of-memory encountered during inference. "
            "Falling back to CPU backend (CTranslate2/faster-whisper)."
        )

        if progress_callback:
            progress_callback(
                ProcessingStage.TRANSCRIPTION,
                -1.0,
                "GPU out-of-memory. Falling back to CPU backend...",
            )

        self._backend = AccelerationBackend.CPU
        self._model = None
        self._fell_back_to_cpu = True

        self._load_model_cpu(self._model_size)

        result = self._run_transcription(audio_path, progress_callback)

        result.warnings.append(
            "GPU out-of-memory encountered. Fell back to CPU backend for processing."
        )

        return result
