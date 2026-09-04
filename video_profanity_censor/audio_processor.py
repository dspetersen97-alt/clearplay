"""Audio processing for censoring profane segments using FFmpeg.

Censoring is driven directly by FFmpeg filtergraphs and streamed from input to
output, so peak memory does NOT scale with track length: the full decoded PCM is
never loaded into a Python buffer. This is the streaming fix for the OOM crash on
long, multichannel feature films (see spec: audio-censor-oom-crash).
"""

import logging
import subprocess
from pathlib import Path

import ffmpeg
from pydub import AudioSegment
from pydub.generators import Sine

from video_profanity_censor.models import (
    AudioMetadata,
    CensorMode,
    CensorResult,
    Detection,
    ProcessingStage,
    ProgressCallback,
)

logger = logging.getLogger(__name__)


# Mapping of common audio codecs to FFmpeg encoder names
CODEC_TO_FFMPEG_ENCODER: dict[str, str] = {
    "aac": "aac",
    "mp3": "libmp3lame",
    "opus": "libopus",
    "vorbis": "libvorbis",
    "flac": "flac",
    "ac3": "ac3",
    "eac3": "eac3",
    "pcm_s16le": "pcm_s16le",
    "pcm_s24le": "pcm_s24le",
    "pcm_s32le": "pcm_s32le",
    "pcm_f32le": "pcm_f32le",
}

# Mapping of bit depth to FFmpeg sample format strings
BIT_DEPTH_TO_SAMPLE_FMT: dict[int, str] = {
    8: "u8",
    16: "s16",
    24: "s24",
    32: "s32",
}


class AudioProcessor:
    """Replaces profane audio segments with censor tone or silence."""

    # A detection whose raw (pre-buffer) duration exceeds this many times the
    # max_word_duration_ms cap is treated as a likely corrupt timestamp. Such a
    # detection is still censored, but bounded to a short window anchored at its
    # start (see the end cap in censor()); this constant only sets the threshold at
    # which that suspicious duration is logged.
    _SUSPICIOUS_DURATION_FACTOR: int = 10

    def __init__(
        self, mode: CensorMode = CensorMode.MUTE, tone_freq: int = 1000,
        censor_buffer_ms: int = 100, max_word_duration_ms: int = 750,
    ):
        """Initialize with censoring mode (TONE or MUTE) and tone frequency.

        Args:
            mode: Censoring mode - TONE inserts a beep, MUTE inserts silence.
            tone_freq: Frequency of the censor tone in Hz (default: 1000 Hz).
            censor_buffer_ms: Extra buffer in milliseconds added before and after
                             each censored word to ensure the full word is covered
                             (default: 100ms).
            max_word_duration_ms: Maximum duration in milliseconds for censoring a
                                 single word. If Whisper assigns a longer duration,
                                 it will be capped to this value (default: 750ms).
                                 Set to 0 to disable the cap.
        """
        self.mode = mode
        self.tone_freq = tone_freq
        self.censor_buffer_ms = censor_buffer_ms
        self.max_word_duration_ms = max_word_duration_ms

    def censor(
        self,
        audio_path: Path,
        detections: list[Detection],
        output_path: Path,
        audio_metadata: AudioMetadata | None = None,
        progress_callback: ProgressCallback = None,
    ) -> CensorResult:
        """Applies censoring to detected profanity segments.

        Args:
            audio_path: Path to the input audio file (WAV format).
            detections: List of profanity detections with timestamp ranges.
            output_path: Path to write the censored audio file.
            audio_metadata: Original audio metadata for re-encoding parameters.
            progress_callback: Optional callback for progress reporting.

        Returns:
            CensorResult with path to censored audio, segments censored, and total duration.

        Raises:
            FileNotFoundError: If the input audio file does not exist.
            RuntimeError: If audio processing fails.
        """
        audio_path = Path(audio_path)
        output_path = Path(output_path)

        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        if not detections:
            # No profanity detected; copy input to output unchanged
            import shutil
            shutil.copy2(str(audio_path), str(output_path))
            return CensorResult(
                censored_audio_path=output_path,
                segments_censored=0,
                total_censored_duration=0.0,
            )

        # Probe the working WAV for duration/rate/channels WITHOUT decoding the PCM
        # into RAM. audio_duration_ms is computed to match pydub's len() exactly
        # (round(1000 * frame_count / frame_rate)) so the window clamping below is
        # byte-identical to the previous pydub-based implementation.
        probe = self._probe_audio(audio_path)
        sample_rate = probe["sample_rate"]
        channels = probe["channels"]
        audio_duration_ms = probe["duration_ms"]

        # Sort detections by start time
        sorted_detections = sorted(
            detections, key=lambda d: d.timestamp_range.start
        )

        total_censored_duration = 0.0
        crossfade_ms = 10  # 10ms crossfade at boundaries

        # --- Phase A: compute resolved censor windows ---
        # Iterate sorted detections and produce (start_ms, end_ms) windows using the
        # IDENTICAL window math as before (buffer, cap trimmed from the start, boundary
        # clamping, and skipping non-positive-duration windows). This preserves the
        # per-detection accounting (total_censored_duration) and progress callback
        # semantics exactly. It never touches decoded PCM.
        resolved_windows: list[tuple[int, int]] = []
        for detection in sorted_detections:
            raw_start_s = detection.timestamp_range.start
            raw_end_s = detection.timestamp_range.end

            # A detection whose raw duration is implausibly long is almost certainly a
            # bad Whisper timestamp (e.g. the word inherited the segment/track end
            # time). We must NOT mute the whole span — that is the "muted from the
            # first swear to the end of the episode" bug — but we also want the word to
            # still be censored. The word's START marks where the profanity begins and
            # is the reliable anchor, so we keep the start and censor a bounded window
            # from it. The end cap below enforces this; the log here just flags that a
            # suspicious duration was bounded. Only active when a cap is configured.
            if self.max_word_duration_ms > 0:
                raw_duration_ms = int((raw_end_s - raw_start_s) * 1000)
                if raw_duration_ms > self.max_word_duration_ms * self._SUSPICIOUS_DURATION_FACTOR:
                    logger.warning(
                        "Detection %r has implausible duration %dms "
                        "(> %dx the %dms cap); likely a bad transcription timestamp. "
                        "Censoring a bounded %dms window from its start instead.",
                        detection.word, raw_duration_ms,
                        self._SUSPICIOUS_DURATION_FACTOR, self.max_word_duration_ms,
                        self.max_word_duration_ms,
                    )

            start_ms = int(raw_start_s * 1000) - self.censor_buffer_ms
            end_ms = int(raw_end_s * 1000) + self.censor_buffer_ms

            # Cap word duration to avoid overly long mutes from imprecise Whisper
            # timestamps. The detected word's START is the reliable anchor (it marks
            # where the profanity begins), so we keep the start and cap the END. This
            # bounds every censor window to at most max_word_duration_ms and guarantees
            # a single detection can never mute past start + cap, regardless of a
            # bogus end value — while still covering the word.
            if self.max_word_duration_ms > 0:
                max_end_ms = start_ms + self.max_word_duration_ms
                if end_ms > max_end_ms:
                    end_ms = max_end_ms

            # Clamp to audio boundaries
            start_ms = max(0, start_ms)
            end_ms = min(end_ms, audio_duration_ms)

            duration_ms = end_ms - start_ms

            if duration_ms <= 0:
                continue

            resolved_windows.append((start_ms, end_ms))

            total_censored_duration += duration_ms / 1000.0

            if progress_callback:
                progress = (sorted_detections.index(detection) + 1) / len(sorted_detections)
                progress_callback(
                    ProcessingStage.AUDIO_CENSORING,
                    progress * 100,
                    f"Censored segment {sorted_detections.index(detection) + 1}/{len(sorted_detections)}",
                )

        # If every window collapsed to a non-positive duration (all skipped), there is
        # nothing to censor — copy the input through unchanged. segments_censored still
        # reflects the detection count, matching the previous return value.
        if not resolved_windows:
            import shutil
            shutil.copy2(str(audio_path), str(output_path))
            return CensorResult(
                censored_audio_path=output_path,
                segments_censored=len(sorted_detections),
                total_censored_duration=total_censored_duration,
            )

        # --- Overlap resolution ---
        # The previous implementation resolved overlapping/adjacent windows with
        # last-writer-wins. For MUTE the silenced region is the UNION of windows
        # (overlap direction does not change silence). For TONE every window is replaced
        # by the same tone, so the toned content is identical regardless of which window
        # "wins" — again the union of windows. We therefore merge overlapping windows into
        # disjoint intervals, which reproduces the exact censored region boundaries.
        merged_windows = self._merge_windows(resolved_windows)

        # --- Phase B: stream the censored track through a single FFmpeg filtergraph ---
        # Peak memory is bounded and independent of track length: FFmpeg decodes, filters,
        # and encodes in a streaming fashion — the whole PCM buffer is never materialised.
        # Use the sample rate PROBED from the actual working WAV for the export,
        # not the sample rate carried in audio_metadata. The metadata rate can drift
        # from the file being encoded; encoding at a rate that differs from the real
        # PCM rate shifts pitch/speed. The probed rate is authoritative.
        export_params = self._build_export_params(
            audio_metadata, actual_sample_rate=sample_rate
        )
        self._run_censor_ffmpeg(
            audio_path=audio_path,
            output_path=output_path,
            windows=merged_windows,
            sample_rate=sample_rate,
            channels=channels,
            crossfade_ms=crossfade_ms,
            export_params=export_params,
        )

        return CensorResult(
            censored_audio_path=output_path,
            segments_censored=len(sorted_detections),
            total_censored_duration=total_censored_duration,
        )

    # --- FFmpeg streaming helpers ---

    def _probe_audio(self, audio_path: Path) -> dict:
        """Probes the working WAV for sample rate, channels, and duration.

        Duration is returned in milliseconds computed to match pydub's ``len()``
        semantics exactly — ``round(1000 * frame_count / frame_rate)`` — so window
        clamping stays byte-identical to the previous implementation. No PCM is decoded.

        Args:
            audio_path: Path to the WAV file to probe.

        Returns:
            Dict with keys ``sample_rate`` (int), ``channels`` (int), ``duration_ms`` (int).

        Raises:
            RuntimeError: If probing fails or no audio stream is present.
        """
        try:
            probe = ffmpeg.probe(str(audio_path))
        except ffmpeg.Error as e:
            stderr = e.stderr.decode() if e.stderr else str(e)
            raise RuntimeError(f"Failed to probe audio for censoring: {stderr}") from e

        audio_streams = [
            s for s in probe.get("streams", []) if s.get("codec_type") == "audio"
        ]
        if not audio_streams:
            raise RuntimeError(f"No audio stream found in: {audio_path}")

        stream = audio_streams[0]
        sample_rate = int(stream.get("sample_rate", 0))
        channels = int(stream.get("channels", 0))

        duration_seconds = float(stream.get("duration", 0.0))
        if duration_seconds == 0.0:
            duration_seconds = float(probe.get("format", {}).get("duration", 0.0))

        # Match pydub's len(): frame_count = round(duration * sample_rate), then
        # round(1000 * frame_count / sample_rate).
        if sample_rate > 0:
            frame_count = round(duration_seconds * sample_rate)
            duration_ms = round(1000 * frame_count / sample_rate)
        else:
            duration_ms = round(duration_seconds * 1000)

        return {
            "sample_rate": sample_rate,
            "channels": channels,
            "duration_ms": duration_ms,
        }

    @staticmethod
    def _merge_windows(
        windows: list[tuple[int, int]]
    ) -> list[tuple[int, int]]:
        """Merges overlapping/adjacent-overlapping windows into disjoint intervals.

        Reproduces the previous last-writer-wins censored regions as the union of
        windows (equivalent for both MUTE silence and a uniform TONE). The input from
        Phase A is sorted by detection start.

        Args:
            windows: List of (start_ms, end_ms) windows.

        Returns:
            List of merged, non-overlapping (start_ms, end_ms) windows.
        """
        if not windows:
            return []

        ordered = sorted(windows)
        merged: list[tuple[int, int]] = [ordered[0]]
        for start_ms, end_ms in ordered[1:]:
            last_start, last_end = merged[-1]
            if start_ms <= last_end:
                # Overlapping or touching — extend the current interval.
                merged[-1] = (last_start, max(last_end, end_ms))
            else:
                merged.append((start_ms, end_ms))
        return merged

    def _build_enable_expr(self, windows: list[tuple[int, int]]) -> str:
        """Builds an FFmpeg ``enable`` expression true within any censored window.

        Args:
            windows: Disjoint (start_ms, end_ms) windows.

        Returns:
            An FFmpeg timeline expression string (sum of ``between(t,start,end)`` terms).
        """
        terms = []
        for start_ms, end_ms in windows:
            start_s = start_ms / 1000.0
            end_s = end_ms / 1000.0
            terms.append(f"between(t,{start_s:.6f},{end_s:.6f})")
        return "+".join(terms)

    def _channel_pan_filter(self, channels: int) -> str:
        """Builds a ``pan`` filter that fans a mono source out to N channels.

        Gives the generated sine tone the same channel count as the source without
        relying on named layouts (works for arbitrary channel counts).

        Args:
            channels: Target channel count.

        Returns:
            An FFmpeg ``pan`` filter string.
        """
        n = max(1, channels)
        outs = "|".join(f"c{i}=c0" for i in range(n))
        return f"pan={n}c|{outs}"

    def _run_censor_ffmpeg(
        self,
        audio_path: Path,
        output_path: Path,
        windows: list[tuple[int, int]],
        sample_rate: int,
        channels: int,
        crossfade_ms: int,
        export_params: dict,
    ) -> None:
        """Runs FFmpeg to censor the given windows, streaming input to output.

        MUTE: sets ``volume=0`` inside the censored windows (source passes through
        elsewhere). TONE: mutes the source inside the windows and mixes in a sine tone
        (freq=self.tone_freq, ~-20 dBFS) gated to the same windows, fanned out to the
        source channel count. Neither path decodes the full PCM into a Python buffer.

        Args:
            audio_path: Input WAV path.
            output_path: Output path.
            windows: Disjoint censored windows (ms).
            sample_rate: Source sample rate.
            channels: Source channel count.
            crossfade_ms: Boundary fade duration in ms (applied to toned regions).
            export_params: Params from ``_build_export_params`` (format/codec/etc.).

        Raises:
            RuntimeError: If the FFmpeg invocation fails.
        """
        enable = self._build_enable_expr(windows)

        cmd = ["ffmpeg", "-y", "-i", str(audio_path)]

        if self.mode == CensorMode.MUTE:
            # Silence the source inside the censored windows; pass through elsewhere.
            audio_filter = f"volume=enable='{enable}':volume=0:eval=frame"
            cmd += ["-af", audio_filter]
        else:
            # TONE: mute source in windows, generate a gated sine, mix them.
            # ~-20 dBFS => linear amplitude ~0.1.
            tone_amplitude = 0.1
            not_enable = f"1-({enable})"
            pan = self._channel_pan_filter(channels)
            fade = self._tone_fade_expr(windows, crossfade_ms)

            filtergraph = (
                f"[0:a]volume=enable='{enable}':volume=0:eval=frame[src];"
                f"sine=frequency={self.tone_freq}:sample_rate={sample_rate},"
                f"volume={tone_amplitude},{pan},"
                # Keep the tone only inside the censored windows.
                f"volume=enable='{not_enable}':volume=0:eval=frame"
                f"{fade}[beep];"
                f"[src][beep]amix=inputs=2:normalize=0:duration=first[out]"
            )
            cmd += ["-filter_complex", filtergraph, "-map", "[out]"]

        cmd += self._export_params_to_ffmpeg_args(export_params)
        cmd.append(str(output_path))

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"FFmpeg censoring failed: {result.stderr.strip()}"
            )

    def _tone_fade_expr(
        self, windows: list[tuple[int, int]], crossfade_ms: int
    ) -> str:
        """Builds per-window boundary fades for the toned replacement.

        Preserves the previous behavior of applying a 10ms fade in/out on the
        replacement only when the window is longer than ``2 * crossfade_ms``. Windows
        at or below that threshold get no fade (matching ``_apply_crossfade``). Fades
        smooth the tone at window edges and avoid audible clicks.

        Args:
            windows: Disjoint censored windows (ms).
            crossfade_ms: Fade duration in ms.

        Returns:
            An FFmpeg filter chain fragment beginning with ``,`` (or empty string).
        """
        if crossfade_ms <= 0:
            return ""

        fade_s = crossfade_ms / 1000.0
        parts: list[str] = []
        for start_ms, end_ms in windows:
            duration_ms = end_ms - start_ms
            if duration_ms <= crossfade_ms * 2:
                continue
            start_s = start_ms / 1000.0
            fade_out_start = (end_ms - crossfade_ms) / 1000.0
            parts.append(
                f"afade=t=in:st={start_s:.6f}:d={fade_s:.6f}:curve=tri"
            )
            parts.append(
                f"afade=t=out:st={fade_out_start:.6f}:d={fade_s:.6f}:curve=tri"
            )
        if not parts:
            return ""
        return "," + ",".join(parts)

    @staticmethod
    def _export_params_to_ffmpeg_args(export_params: dict) -> list[str]:
        """Translates ``_build_export_params`` output into raw FFmpeg CLI args.

        Preserves the same codec/sample_rate/bit_depth/channel_layout/bitrate mapping
        used by the previous pydub export path, so the encoded output matches.

        Args:
            export_params: Dict from ``_build_export_params``.

        Returns:
            List of FFmpeg output-option arguments.
        """
        args: list[str] = []

        fmt = export_params.get("format")
        if fmt:
            args += ["-f", fmt]

        codec = export_params.get("codec")
        if codec:
            args += ["-c:a", codec]

        # Extra ffmpeg params (sample rate, sample format, channel layout).
        extra = export_params.get("parameters")
        if extra:
            args += list(extra)

        bitrate = export_params.get("bitrate")
        if bitrate:
            args += ["-b:a", str(bitrate)]

        return args

    def build_encoding_command(self, audio_metadata: AudioMetadata) -> dict[str, str]:
        """Builds FFmpeg encoding parameters that match the source audio metadata.

        This method constructs the encoding parameters dictionary that ensures
        the re-encoded audio uses the same codec, sample rate, bit depth,
        channel layout, and bitrate as the source.

        Args:
            audio_metadata: The source audio metadata to match.

        Returns:
            Dictionary of FFmpeg encoding parameters.
        """
        params: dict[str, str] = {}

        # Codec
        encoder = CODEC_TO_FFMPEG_ENCODER.get(audio_metadata.codec, audio_metadata.codec)
        params["codec"] = encoder

        # Sample rate
        params["ar"] = str(audio_metadata.sample_rate)

        # Bit depth (sample format)
        if audio_metadata.bit_depth in BIT_DEPTH_TO_SAMPLE_FMT:
            params["sample_fmt"] = BIT_DEPTH_TO_SAMPLE_FMT[audio_metadata.bit_depth]

        # Channel layout
        params["channel_layout"] = audio_metadata.channel_layout

        # Bitrate
        if audio_metadata.bitrate > 0:
            params["audio_bitrate"] = str(audio_metadata.bitrate)

        return params

    def _build_export_params(
        self,
        audio_metadata: AudioMetadata | None,
        actual_sample_rate: int | None = None,
    ) -> dict:
        """Builds pydub export parameters from audio metadata.

        Args:
            audio_metadata: Source audio metadata. If None, exports as WAV.
            actual_sample_rate: The sample rate PROBED from the working WAV that is
                actually being encoded. When provided and valid (> 0), it takes
                precedence over ``audio_metadata.sample_rate`` for the ``-ar`` option.
                Encoding at a rate that differs from the real PCM rate shifts
                pitch/speed, so the probed rate must win when the two disagree.

        Returns:
            Dictionary of export parameters for pydub's export method.
        """
        if audio_metadata is None:
            return {"format": "wav"}

        # The rate of the PCM actually being encoded is authoritative. Fall back to
        # the source metadata rate only when no probed rate was supplied.
        export_sample_rate = (
            actual_sample_rate
            if actual_sample_rate and actual_sample_rate > 0
            else audio_metadata.sample_rate
        )

        # Determine format from codec
        format_map = {
            "aac": "adts",
            "mp3": "mp3",
            "libmp3lame": "mp3",
            "opus": "opus",
            "vorbis": "ogg",
            "flac": "flac",
            "ac3": "ac3",
            "eac3": "eac3",
            "pcm_s16le": "wav",
            "pcm_s24le": "wav",
            "pcm_s32le": "wav",
            "pcm_f32le": "wav",
        }

        codec = audio_metadata.codec
        format_name = format_map.get(codec, "wav")

        params: dict = {"format": format_name}

        # Add codec parameter
        encoder = CODEC_TO_FFMPEG_ENCODER.get(codec, codec)
        params["codec"] = encoder

        # Build ffmpeg parameters list
        ffmpeg_params = []

        # Sample rate — use the probed rate of the WAV being encoded (see above).
        if export_sample_rate > 0:
            ffmpeg_params.extend(["-ar", str(export_sample_rate)])

        # Bit depth (sample format)
        if audio_metadata.bit_depth in BIT_DEPTH_TO_SAMPLE_FMT:
            sample_fmt = BIT_DEPTH_TO_SAMPLE_FMT[audio_metadata.bit_depth]
            ffmpeg_params.extend(["-sample_fmt", sample_fmt])

        # Channel layout
        if audio_metadata.channel_layout:
            ffmpeg_params.extend(["-channel_layout", audio_metadata.channel_layout])

        # Bitrate
        if audio_metadata.bitrate > 0:
            params["bitrate"] = str(audio_metadata.bitrate)

        if ffmpeg_params:
            params["parameters"] = ffmpeg_params

        return params

    def _generate_tone(
        self, duration_ms: int, sample_rate: int, channels: int
    ) -> AudioSegment:
        """Generates a sine wave censor tone.

        Args:
            duration_ms: Duration in milliseconds.
            sample_rate: Sample rate in Hz.
            channels: Number of audio channels.

        Returns:
            AudioSegment with the censor tone.
        """
        tone = Sine(self.tone_freq).to_audio_segment(
            duration=duration_ms,
            volume=-20,  # -20 dBFS to avoid clipping
        )
        tone = tone.set_frame_rate(sample_rate)

        if channels > 1:
            tone = tone.set_channels(channels)

        return tone

    def _apply_crossfade(
        self,
        original: AudioSegment,
        replacement: AudioSegment,
        start_ms: int,
        end_ms: int,
        crossfade_ms: int,
    ) -> AudioSegment:
        """Applies crossfade at the boundaries of a replacement segment.

        Creates a smooth transition between original audio and the replacement
        to avoid audible clicks or pops.

        Args:
            original: The full original audio.
            replacement: The replacement segment (tone or silence).
            start_ms: Start position in milliseconds.
            end_ms: End position in milliseconds.
            crossfade_ms: Duration of crossfade in milliseconds.

        Returns:
            The replacement segment with crossfade applied at boundaries.
        """
        duration_ms = end_ms - start_ms

        # If the segment is too short for crossfade, skip it
        if duration_ms <= crossfade_ms * 2:
            return replacement

        # Apply fade in at the start and fade out at the end
        replacement = replacement.fade_in(crossfade_ms).fade_out(crossfade_ms)

        return replacement
