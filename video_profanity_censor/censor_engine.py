"""CensorEngine orchestrator for the video profanity censoring pipeline.

This module implements the main orchestrator that ties together all pipeline
components in sequence: validate → extract audio → detect backend → scan
subtitles → transcribe → detect profanity → censor audio → assemble output.

It manages temporary files, wires progress reporting through all stages,
and handles errors with stage identification.
"""

import logging
import shutil
import tempfile
import time
from pathlib import Path

from video_profanity_censor.audio_extractor import AudioExtractor
from video_profanity_censor.backend_selector import BackendSelector
from video_profanity_censor.input_validator import InputValidator
from video_profanity_censor.models import (
    AccelerationBackend,
    CensorMode,
    DetectionResult,
    ProcessingResult,
    ProcessingStage,
    ProgressCallback,
    SubtitleScanResult,
    TranscriptionResult,
)
from video_profanity_censor.profanity_detector import ProfanityDetector
from video_profanity_censor.profanity_list import load_profanity_list
from video_profanity_censor.subtitle_scanner import SubtitleScanner

logger = logging.getLogger(__name__)


class CensorEngine:
    """Orchestrates the full video profanity censoring pipeline.

    Coordinates all pipeline stages in sequence, manages temporary files,
    handles errors with stage identification, and wires progress callbacks
    through each component.
    """

    def process(
        self,
        input_path: Path,
        output_path: Path | None = None,
        audio_track: int = 0,
        censor_mode: CensorMode = CensorMode.TONE,
        profanity_list_path: Path | None = None,
        report_path: Path | None = None,
        subtitle_path: Path | None = None,
        disable_subtitle_prefilter: bool = False,
        backend: AccelerationBackend | None = None,
        model_size: str | None = None,
    ) -> ProcessingResult:
        """Orchestrates the full censoring pipeline.

        When subtitle_path is provided, uses that file for pre-filtering.
        When disable_subtitle_prefilter is True, skips subtitle scanning entirely.
        When backend is None, auto-detects the optimal acceleration backend.
        When model_size is None, selects based on available VRAM.

        Args:
            input_path: Path to the input video file.
            output_path: Path for the output video. If None, derives from input.
            audio_track: Index of the audio track to process (default: 0).
            censor_mode: Censoring mode — TONE or MUTE.
            profanity_list_path: Path to custom profanity list. None uses default.
            report_path: Path for the detection report. None uses default derivation.
            subtitle_path: Path to external subtitle file for pre-filtering.
            disable_subtitle_prefilter: If True, bypasses subtitle scanning (Req 8.9).
            backend: Specific backend to use. None triggers auto-detection (Req 9.4).
            model_size: Whisper model size override. None uses VRAM-based selection (Req 9.6, 9.7).

        Returns:
            ProcessingResult with success status, paths, and processing summary.
        """
        start_time = time.time()
        temp_files: list[Path] = []
        input_path = Path(input_path)

        # Derive default output path if not specified
        if output_path is None:
            output_path = self._derive_output_path(input_path)
        else:
            output_path = Path(output_path)

        # Derive default report path if not specified
        if report_path is None:
            report_path = self._derive_report_path(output_path)
        else:
            report_path = Path(report_path)

        # Create progress reporter
        from video_profanity_censor.progress_reporter import ProgressReporter

        progress_reporter = ProgressReporter()
        progress_callback: ProgressCallback = progress_reporter.update

        # Track the active backend and model size for the result
        active_backend: AccelerationBackend | None = None
        model_size_used: str | None = None

        try:
            # --- Stage 1: Validation ---
            progress_callback(ProcessingStage.VALIDATION, 0.0, "Validating input file...")
            validation_result = self._stage_validate(input_path)
            if not validation_result.is_valid:
                return ProcessingResult(
                    success=False,
                    error_message=validation_result.error_message,
                    error_stage=ProcessingStage.VALIDATION,
                    total_elapsed_seconds=time.time() - start_time,
                )
            progress_callback(ProcessingStage.VALIDATION, 100.0, "Validation complete")

            # --- Stage 2: Audio Extraction ---
            progress_callback(ProcessingStage.VALIDATION, 100.0, "Extracting audio track...")
            try:
                extraction_result = self._stage_extract_audio(input_path, audio_track)
                temp_files.append(extraction_result.output_path)
            except Exception as e:
                return ProcessingResult(
                    success=False,
                    error_message=f"Audio extraction failed: {e}",
                    error_stage=ProcessingStage.VALIDATION,
                    total_elapsed_seconds=time.time() - start_time,
                )

            # --- Stage 3: Backend Detection ---
            progress_callback(ProcessingStage.BACKEND_DETECTION, 0.0, "Detecting hardware acceleration...")
            try:
                backend_result = self._stage_detect_backend(backend)
                if backend_result.error_message:
                    return ProcessingResult(
                        success=False,
                        error_message=backend_result.error_message,
                        error_stage=ProcessingStage.BACKEND_DETECTION,
                        total_elapsed_seconds=time.time() - start_time,
                    )
                # Determine active backend and model size
                active_backend = backend_result.selected_backend
                if model_size is not None:
                    model_size_used = model_size
                else:
                    model_size_used = backend_result.selected_model_size
            except Exception as e:
                return ProcessingResult(
                    success=False,
                    error_message=f"Backend detection failed: {e}",
                    error_stage=ProcessingStage.BACKEND_DETECTION,
                    total_elapsed_seconds=time.time() - start_time,
                )
            progress_callback(ProcessingStage.BACKEND_DETECTION, 100.0, f"Using {active_backend.value} backend with {model_size_used} model")

            # --- Stage 4: Load Profanity List ---
            try:
                profanity_list = load_profanity_list(profanity_list_path)
            except Exception as e:
                return ProcessingResult(
                    success=False,
                    error_message=f"Failed to load profanity list: {e}",
                    error_stage=ProcessingStage.PROFANITY_DETECTION,
                    total_elapsed_seconds=time.time() - start_time,
                    active_backend=active_backend,
                    model_size_used=model_size_used,
                )

            # --- Stage 5: Subtitle Scanning (optional) ---
            subtitle_scan_result: SubtitleScanResult | None = None
            if not disable_subtitle_prefilter:
                progress_callback(ProcessingStage.SUBTITLE_SCANNING, 0.0, "Scanning subtitles for profanity...")
                try:
                    subtitle_scan_result = self._stage_scan_subtitles(
                        input_path, subtitle_path, profanity_list
                    )
                except Exception as e:
                    return ProcessingResult(
                        success=False,
                        error_message=f"Subtitle scanning failed: {e}",
                        error_stage=ProcessingStage.SUBTITLE_SCANNING,
                        total_elapsed_seconds=time.time() - start_time,
                        active_backend=active_backend,
                        model_size_used=model_size_used,
                    )
                progress_callback(ProcessingStage.SUBTITLE_SCANNING, 100.0, "Subtitle scan complete")

                # Short-circuit: subtitles available but no profanity found (Req 8.7)
                if subtitle_scan_result.has_subtitles and subtitle_scan_result.skipped_speech_recognition:
                    # No profanity found — skip output file creation
                    self._generate_report(DetectionResult(), report_path)
                    self._cleanup_temp_files(temp_files)
                    return ProcessingResult(
                        success=True,
                        output_path=None,
                        report_path=report_path,
                        total_elapsed_seconds=time.time() - start_time,
                        profane_instances_detected=0,
                        profane_instances_censored=0,
                        active_backend=active_backend,
                        model_size_used=model_size_used,
                    )

            # --- Stage 6: Transcription ---
            progress_callback(ProcessingStage.TRANSCRIPTION, 0.0, "Transcribing audio...")
            try:
                transcription_result = self._stage_transcribe(
                    extraction_result.output_path,
                    active_backend,
                    model_size_used,
                    subtitle_scan_result,
                    progress_callback,
                )
            except Exception as e:
                self._cleanup_temp_files(temp_files)
                return ProcessingResult(
                    success=False,
                    error_message=f"Transcription failed: {e}",
                    error_stage=ProcessingStage.TRANSCRIPTION,
                    total_elapsed_seconds=time.time() - start_time,
                    active_backend=active_backend,
                    model_size_used=model_size_used,
                )
            progress_callback(ProcessingStage.TRANSCRIPTION, 100.0, "Transcription complete")

            # --- Stage 7: Profanity Detection ---
            progress_callback(ProcessingStage.PROFANITY_DETECTION, 0.0, "Detecting profanity...")
            try:
                detection_result = self._stage_detect_profanity(
                    transcription_result, profanity_list, censor_mode
                )
            except Exception as e:
                self._cleanup_temp_files(temp_files)
                return ProcessingResult(
                    success=False,
                    error_message=f"Profanity detection failed: {e}",
                    error_stage=ProcessingStage.PROFANITY_DETECTION,
                    total_elapsed_seconds=time.time() - start_time,
                    active_backend=active_backend,
                    model_size_used=model_size_used,
                )
            progress_callback(ProcessingStage.PROFANITY_DETECTION, 100.0, "Profanity detection complete")

            # If no profanity detected, skip output file creation
            if not detection_result.detections:
                self._generate_report(detection_result, report_path)
                self._cleanup_temp_files(temp_files)
                return ProcessingResult(
                    success=True,
                    output_path=None,
                    report_path=report_path,
                    total_elapsed_seconds=time.time() - start_time,
                    profane_instances_detected=0,
                    profane_instances_censored=0,
                    active_backend=active_backend,
                    model_size_used=model_size_used,
                )

            # --- Stage 8: Audio Censoring ---
            progress_callback(ProcessingStage.AUDIO_CENSORING, 0.0, "Censoring audio...")
            try:
                censor_result = self._stage_censor_audio(
                    extraction_result.output_path,
                    detection_result,
                    censor_mode,
                    progress_callback,
                )
                temp_files.append(censor_result.censored_audio_path)
            except Exception as e:
                self._cleanup_temp_files(temp_files)
                return ProcessingResult(
                    success=False,
                    error_message=f"Audio censoring failed: {e}",
                    error_stage=ProcessingStage.AUDIO_CENSORING,
                    total_elapsed_seconds=time.time() - start_time,
                    active_backend=active_backend,
                    model_size_used=model_size_used,
                )
            progress_callback(ProcessingStage.AUDIO_CENSORING, 100.0, "Audio censoring complete")

            # --- Stage 9: Output Assembly ---
            progress_callback(ProcessingStage.OUTPUT_WRITING, 0.0, "Assembling output...")
            try:
                assembly_result = self._stage_assemble_output(
                    input_path,
                    censor_result.censored_audio_path,
                    output_path,
                    audio_track,
                )
                if assembly_result.error_message:
                    self._cleanup_temp_files(temp_files)
                    return ProcessingResult(
                        success=False,
                        error_message=assembly_result.error_message,
                        error_stage=ProcessingStage.OUTPUT_WRITING,
                        total_elapsed_seconds=time.time() - start_time,
                        active_backend=active_backend,
                        model_size_used=model_size_used,
                    )
                if assembly_result.codec_fallback:
                    progress_callback(
                        ProcessingStage.OUTPUT_WRITING,
                        50.0,
                        f"Warning: Source audio codec unsupported for re-encoding. "
                        f"Falling back to {assembly_result.codec_fallback.upper()} codec.",
                    )
            except Exception as e:
                self._cleanup_temp_files(temp_files)
                return ProcessingResult(
                    success=False,
                    error_message=f"Output assembly failed: {e}",
                    error_stage=ProcessingStage.OUTPUT_WRITING,
                    total_elapsed_seconds=time.time() - start_time,
                    active_backend=active_backend,
                    model_size_used=model_size_used,
                )
            progress_callback(ProcessingStage.OUTPUT_WRITING, 100.0, "Output assembly complete")

            # --- Stage 10: Report Generation ---
            self._generate_report(detection_result, report_path)

            # Cleanup and return success
            self._cleanup_temp_files(temp_files)

            elapsed = time.time() - start_time
            return ProcessingResult(
                success=True,
                output_path=output_path,
                report_path=report_path,
                total_elapsed_seconds=elapsed,
                profane_instances_detected=len(detection_result.detections),
                profane_instances_censored=censor_result.segments_censored,
                active_backend=active_backend,
                model_size_used=model_size_used,
            )

        except Exception as e:
            # Catch-all for unexpected errors
            self._cleanup_temp_files(temp_files)
            return ProcessingResult(
                success=False,
                error_message=f"Unexpected error: {e}",
                error_stage=None,
                total_elapsed_seconds=time.time() - start_time,
                active_backend=active_backend,
                model_size_used=model_size_used,
            )

    # --- Pipeline stage methods ---

    def _stage_validate(self, input_path: Path):
        """Run input validation stage."""
        validator = InputValidator()
        return validator.validate(input_path)

    def _stage_extract_audio(self, input_path: Path, audio_track: int):
        """Run audio extraction stage."""
        extractor = AudioExtractor()
        return extractor.extract(input_path, track_index=audio_track)

    def _stage_detect_backend(self, backend: AccelerationBackend | None):
        """Run backend detection stage.

        When backend is None, auto-detect (Req 9.2).
        When specified, pass as preferred_backend to BackendSelector (Req 9.4).
        """
        selector = BackendSelector()
        return selector.detect(preferred_backend=backend)

    def _stage_scan_subtitles(
        self, input_path: Path, subtitle_path: Path | None, profanity_list
    ) -> SubtitleScanResult:
        """Run subtitle scanning stage.

        Uses external subtitle file if provided, otherwise probes for embedded tracks.
        """
        scanner = SubtitleScanner(profanity_list=profanity_list)
        external_path = Path(subtitle_path) if subtitle_path is not None else None
        return scanner.scan(input_path, external_subtitle_path=external_path)

    def _stage_transcribe(
        self,
        audio_path: Path,
        active_backend: AccelerationBackend,
        model_size: str,
        subtitle_scan_result: SubtitleScanResult | None,
        progress_callback: ProgressCallback,
    ) -> TranscriptionResult:
        """Run transcription stage.

        Routes to targeted transcription (regions only) or full transcription
        based on subtitle scan results (Req 8.4).
        """
        from video_profanity_censor.speech_recognizer import SpeechRecognizer

        recognizer = SpeechRecognizer(
            model_size=model_size,
            backend=active_backend,
        )

        # Determine regions for targeted transcription
        regions = None
        if subtitle_scan_result is not None and subtitle_scan_result.profanity_regions:
            regions = subtitle_scan_result.profanity_regions

        try:
            return recognizer.transcribe(
                audio_path=audio_path,
                regions=regions,
                progress_callback=progress_callback,
            )
        finally:
            # The Whisper model (several GB via CTranslate2) is no longer needed
            # once transcription completes — profanity detection works on the
            # TranscriptionResult, not the model. Release it here so the memory
            # is reclaimed BEFORE the audio censoring stage runs, and even if a
            # later stage fails. This prevents the un-freed model from stacking
            # with the censoring buffer and driving the process into swap/OOM.
            recognizer.release_model()

    def _stage_detect_profanity(
        self,
        transcription_result: TranscriptionResult,
        profanity_list,
        censor_mode: CensorMode,
    ) -> DetectionResult:
        """Run profanity detection stage."""
        detector = ProfanityDetector(
            profanity_list=profanity_list,
            censor_mode=censor_mode,
        )
        return detector.detect(transcription_result)

    def _stage_censor_audio(
        self,
        audio_path: Path,
        detection_result: DetectionResult,
        censor_mode: CensorMode,
        progress_callback: ProgressCallback,
    ):
        """Run audio censoring stage."""
        from video_profanity_censor.audio_processor import AudioProcessor

        processor = AudioProcessor(mode=censor_mode)

        # Generate temp output path for censored audio
        censored_path = Path(
            tempfile.mktemp(suffix=".wav", prefix="censored_audio_")
        )

        return processor.censor(
            audio_path=audio_path,
            detections=detection_result.detections,
            output_path=censored_path,
            progress_callback=progress_callback,
        )

    def _stage_assemble_output(
        self,
        source_path: Path,
        censored_audio_path: Path,
        output_path: Path,
        audio_track_index: int,
    ):
        """Run output assembly stage."""
        from video_profanity_censor.output_assembler import OutputAssembler

        assembler = OutputAssembler()
        return assembler.assemble(
            source_path=source_path,
            censored_audio_path=censored_audio_path,
            output_path=output_path,
            audio_track_index=audio_track_index,
        )

    # --- Helper methods ---

    def _generate_report(self, detection_result: DetectionResult, report_path: Path) -> None:
        """Generate the profanity detection report."""
        from video_profanity_censor.report_generator import ReportGenerator

        try:
            generator = ReportGenerator()
            generator.generate(
                detections=detection_result.detections,
                output_path=report_path,
            )
        except Exception as e:
            logger.warning(f"Failed to generate report: {e}")

    def _derive_output_path(self, input_path: Path) -> Path:
        """Derive default output path from input path.

        Adds '_censored' suffix before the extension.
        """
        stem = input_path.stem
        suffix = input_path.suffix
        return input_path.parent / f"{stem}_censored{suffix}"

    def _derive_report_path(self, output_path: Path) -> Path:
        """Derive default report path from output path.

        Uses the output filename appended with '_report.txt' (Req 7.4).
        """
        return output_path.parent / f"{output_path.stem}_report.txt"

    def _cleanup_temp_files(self, temp_files: list[Path]) -> None:
        """Clean up temporary files created during processing."""
        for temp_file in temp_files:
            try:
                if temp_file.exists():
                    temp_file.unlink()
            except OSError as e:
                logger.warning(f"Failed to clean up temp file {temp_file}: {e}")
