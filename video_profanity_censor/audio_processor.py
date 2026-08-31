"""Audio processing for censoring profane segments using FFmpeg."""

import subprocess
import tempfile
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


# Crossfade applied at the boundaries of every censored window.
CROSSFADE_MS = 10

# Level of the censor tone, in dBFS.
TONE_LEVEL_DBFS = -20.0

# Peak amplitude, relative to full scale, emitted by FFmpeg's `sine` source.
# The source has no amplitude option, so the censor tone is scaled relative to
# this to land at TONE_LEVEL_DBFS.
FFMPEG_SINE_PEAK_AMPLITUDE = 0.125

# Target size of the audio frames fed to the censoring filtergraph, in
# milliseconds. FFmpeg evaluates timeline (`enable`) and `volume` expressions
# once per frame, so frames are resized to keep window boundaries accurate to
# roughly a millisecond.
GATE_FRAME_MS = 1.0

# Filtergraphs longer than this are passed to FFmpeg in a file rather than on
# the command line, which has a length limit (~32 KB on Windows).
MAX_INLINE_FILTERGRAPH_CHARS = 8000


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

        The audio is streamed through a single FFmpeg filtergraph rather than
        being decoded into memory, so peak memory usage is bounded and
        independent of the length and channel count of the track.

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

        # Probe the audio instead of decoding it — only its duration, sample
        # rate, channel count and sample format are needed
        sample_rate, channels, audio_duration_ms, sample_fmt = self._probe_audio(
            audio_path
        )

        # Sort detections by start time
        sorted_detections = sorted(
            detections, key=lambda d: d.timestamp_range.start
        )

        total_censored_duration = 0.0
        censor_windows: list[tuple[int, int]] = []

        for detection in sorted_detections:
            start_ms = int(detection.timestamp_range.start * 1000) - self.censor_buffer_ms
            end_ms = int(detection.timestamp_range.end * 1000) + self.censor_buffer_ms

            # Cap word duration to avoid overly long mutes from imprecise Whisper timestamps.
            # When Whisper assigns too-long durations, the actual word is typically at the END
            # of the range (Whisper anchors the start too early), so we trim from the start.
            if self.max_word_duration_ms > 0:
                raw_duration = end_ms - start_ms
                if raw_duration > self.max_word_duration_ms:
                    # Keep the end, trim the start — the word is spoken near the end
                    start_ms = end_ms - self.max_word_duration_ms

            # Clamp to audio boundaries
            start_ms = max(0, start_ms)
            end_ms = min(end_ms, audio_duration_ms)

            duration_ms = end_ms - start_ms

            if duration_ms <= 0:
                continue

            censor_windows.append((start_ms, end_ms))

            total_censored_duration += duration_ms / 1000.0

            if progress_callback:
                progress = (sorted_detections.index(detection) + 1) / len(sorted_detections)
                progress_callback(
                    ProcessingStage.AUDIO_CENSORING,
                    progress * 100,
                    f"Censored segment {sorted_detections.index(detection) + 1}/{len(sorted_detections)}",
                )

        # Stream the audio through FFmpeg, replacing the censored windows
        self._render_censored_audio(
            audio_path=audio_path,
            output_path=output_path,
            censor_windows=censor_windows,
            sample_rate=sample_rate,
            channels=channels,
            sample_fmt=sample_fmt,
            audio_metadata=audio_metadata,
        )

        return CensorResult(
            censored_audio_path=output_path,
            segments_censored=len(sorted_detections),
            total_censored_duration=total_censored_duration,
        )

    def _probe_audio(self, audio_path: Path) -> tuple[int, int, int, str | None]:
        """Probes the input audio without decoding it.

        Args:
            audio_path: Path to the audio file.

        Returns:
            Tuple of (sample_rate, channels, duration_ms, sample_fmt). The
            duration is rounded to the nearest millisecond, matching the length
            pydub reports for the same file. sample_fmt is None when FFmpeg does
            not report one.

        Raises:
            RuntimeError: If probing fails or the file has no audio stream.
        """
        try:
            probe = ffmpeg.probe(str(audio_path))
        except ffmpeg.Error as e:
            raise RuntimeError(
                f"Failed to probe audio file: "
                f"{e.stderr.decode() if e.stderr else str(e)}"
            ) from e

        stream = next(
            (
                s
                for s in probe.get("streams", [])
                if s.get("codec_type") == "audio"
            ),
            None,
        )
        if stream is None:
            raise RuntimeError(f"No audio stream found in: {audio_path}")

        sample_rate = int(stream.get("sample_rate", 0))
        channels = int(stream.get("channels", 0))
        if sample_rate <= 0 or channels <= 0:
            raise RuntimeError(
                f"Could not determine sample rate/channel count of: {audio_path}"
            )

        duration_seconds = float(stream.get("duration", 0.0))
        if duration_seconds == 0.0:
            duration_seconds = float(probe.get("format", {}).get("duration", 0.0))
        if duration_seconds <= 0.0:
            raise RuntimeError(
                f"Could not determine the duration of: {audio_path}"
            )

        return sample_rate, channels, round(duration_seconds * 1000), stream.get("sample_fmt")

    def _render_censored_audio(
        self,
        audio_path: Path,
        output_path: Path,
        censor_windows: list[tuple[int, int]],
        sample_rate: int,
        channels: int,
        sample_fmt: str | None,
        audio_metadata: AudioMetadata | None,
    ) -> None:
        """Streams the audio through FFmpeg, censoring the given windows.

        Args:
            audio_path: Path to the input audio file.
            output_path: Path to write the censored audio file.
            censor_windows: Windows to censor, as (start_ms, end_ms) pairs.
            sample_rate: Sample rate of the input audio in Hz.
            channels: Channel count of the input audio.
            sample_fmt: Sample format of the input audio, used when the output
                        format is not dictated by audio_metadata.
            audio_metadata: Source audio metadata for re-encoding parameters.

        Raises:
            RuntimeError: If FFmpeg fails.
        """
        filtergraph = self._build_filtergraph(censor_windows, sample_rate, channels)
        output_args = self._build_output_args(audio_metadata, sample_fmt)

        graph_file: Path | None = None
        if len(filtergraph) > MAX_INLINE_FILTERGRAPH_CHARS:
            # Too long for the command line; hand FFmpeg a script file instead
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".txt", prefix="censor_filtergraph_", delete=False
            ) as handle:
                handle.write(filtergraph)
                graph_file = Path(handle.name)

        try:
            self._run_ffmpeg(
                audio_path, output_path, filtergraph, graph_file, output_args
            )
        finally:
            if graph_file is not None:
                try:
                    graph_file.unlink()
                except OSError:
                    pass

    def _run_ffmpeg(
        self,
        audio_path: Path,
        output_path: Path,
        filtergraph: str,
        graph_file: Path | None,
        output_args: list[str],
    ) -> None:
        """Invokes FFmpeg with the censoring filtergraph.

        When the filtergraph is passed in a file, the option that carries it
        differs between FFmpeg releases (`-/filter_complex` from 7.0,
        `-filter_complex_script` before 9.0), so both spellings are tried.

        Raises:
            RuntimeError: If FFmpeg fails.
        """
        if graph_file is None:
            graph_options: list[list[str]] = [["-filter_complex", filtergraph]]
        else:
            graph_options = [
                ["-/filter_complex", str(graph_file)],
                ["-filter_complex_script", str(graph_file)],
            ]

        for graph_option in graph_options:
            result = subprocess.run(
                [
                    "ffmpeg", "-y", "-nostdin", "-loglevel", "error",
                    "-i", str(audio_path),
                    *graph_option,
                    "-map", "[out]",
                    *output_args,
                    str(output_path),
                ],
                capture_output=True,
                text=True,
            )

            if result.returncode == 0:
                return
            if "Unrecognized option" not in (result.stderr or ""):
                break

        raise RuntimeError(
            f"FFmpeg audio censoring failed: {(result.stderr or '').strip()}"
        )

    def _build_filtergraph(
        self,
        censor_windows: list[tuple[int, int]],
        sample_rate: int,
        channels: int,
    ) -> str:
        """Builds the filtergraph that censors the given windows.

        In MUTE mode the source is silenced over each window with a `volume`
        filter gated by an enable expression. In TONE mode a sine generator,
        gated by the same windows and faded in and out over CROSSFADE_MS, is
        mixed over the silenced windows.

        Args:
            censor_windows: Windows to censor, as (start_ms, end_ms) pairs.
            sample_rate: Sample rate of the input audio in Hz.
            channels: Channel count of the input audio.

        Returns:
            The filtergraph string, producing a single [out] output.
        """
        # FFmpeg evaluates the gate expressions once per frame, so shrink the
        # frames to GATE_FRAME_MS to keep the window boundaries accurate
        frame_samples = max(1, round(sample_rate * GATE_FRAME_MS / 1000))
        frame_seconds = frame_samples / sample_rate
        windows = self._gate_windows(censor_windows, frame_seconds)

        base = (
            f"[0:a]asetnsamples=n={frame_samples}:p=0,"
            f"volume=0:enable='{self._gate_expression(windows, ramped=False)}'"
        )

        if self.mode != CensorMode.TONE:
            return f"{base}[out]"

        # The sine source has no amplitude control, so scale its fixed peak to
        # the censor tone level
        tone_gain = (
            10 ** (TONE_LEVEL_DBFS / 20) / FFMPEG_SINE_PEAK_AMPLITUDE
        )
        tone = (
            f"sine=frequency={self.tone_freq}:sample_rate={sample_rate}"
            f":samples_per_frame={frame_samples},"
            f"volume=volume='{tone_gain:.6f}*"
            f"({self._gate_expression(windows, ramped=True)})':eval=frame"
        )
        if channels > 1:
            # Duplicate the mono tone across every channel
            pan_map = "|".join(f"c{index}=c0" for index in range(channels))
            tone += f",pan={channels}c|{pan_map}"

        return (
            f"{base}[base];{tone}[tone];"
            f"[base][tone]amix=inputs=2:duration=first:normalize=0[out]"
        )

    @staticmethod
    def _gate_windows(
        censor_windows: list[tuple[int, int]], frame_seconds: float
    ) -> list[tuple[float, float]]:
        """Converts censor windows to the sorted, non-overlapping gate windows.

        Each window is extended backwards by one frame so that the frame
        straddling its start is censored too, then overlapping windows are
        merged — the gate expression addresses one window at a time, and
        overlapping windows censored the union of their ranges before.

        Args:
            censor_windows: Windows to censor, as (start_ms, end_ms) pairs.
            frame_seconds: Duration of one audio frame in seconds.

        Returns:
            Sorted, non-overlapping (start, end) windows in seconds.
        """
        expanded = sorted(
            (max(0.0, start_ms / 1000.0 - frame_seconds), end_ms / 1000.0)
            for start_ms, end_ms in censor_windows
        )

        merged: list[tuple[float, float]] = []
        for start, end in expanded:
            if merged and start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))

        return merged

    @staticmethod
    def _gate_expression(
        windows: list[tuple[float, float]], ramped: bool
    ) -> str:
        """Builds the FFmpeg expression that gates the censored windows.

        The windows are laid out as a balanced tree of `if()` expressions rather
        than a flat sum of terms: FFmpeg's expression parser rejects expressions
        nested more than about a hundred levels deep, and `if()` only evaluates
        the branch it takes, so a tree stays cheap for thousands of windows.

        Args:
            windows: Sorted, non-overlapping (start, end) windows in seconds.
            ramped: When True, the expression ramps from 0 to 1 over the first
                    CROSSFADE_MS of a window and back to 0 over the last, giving
                    the censor tone its boundary crossfade. When False, the
                    expression is simply 1 inside a window and 0 outside.

        Returns:
            An FFmpeg expression in the timeline variable `t`.
        """
        crossfade_seconds = CROSSFADE_MS / 1000.0

        def value(start: float, end: float) -> str:
            # Windows too short to fade were not crossfaded before either
            if not ramped or (end - start) <= crossfade_seconds * 2:
                return "1"
            return (
                f"max(min(min((t-{start:.4f})/{crossfade_seconds},"
                f"({end:.4f}-t)/{crossfade_seconds}),1),0)"
            )

        def subtree(low: int, high: int) -> str:
            if low > high:
                return "0"
            middle = (low + high) // 2
            start, end = windows[middle]
            return (
                f"if(lt(t,{start:.4f}),{subtree(low, middle - 1)},"
                f"if(lte(t,{end:.4f}),{value(start, end)},"
                f"{subtree(middle + 1, high)}))"
            )

        return subtree(0, len(windows) - 1)

    def _build_output_args(
        self, audio_metadata: AudioMetadata | None, sample_fmt: str | None
    ) -> list[str]:
        """Translates the export parameters into FFmpeg command-line arguments.

        Args:
            audio_metadata: Source audio metadata for re-encoding parameters.
            sample_fmt: Sample format of the input audio, applied when the
                        export parameters do not specify one, so that the output
                        keeps the input's bit depth.

        Returns:
            List of FFmpeg output arguments.
        """
        params = self._build_export_params(audio_metadata)

        args: list[str] = []
        if params.get("codec"):
            args += ["-acodec", params["codec"]]
        if params.get("bitrate"):
            args += ["-b:a", params["bitrate"]]
        args += list(params.get("parameters", []))
        if "-sample_fmt" not in args and sample_fmt:
            args += ["-sample_fmt", sample_fmt]
        args += ["-f", params["format"]]

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
        self, audio_metadata: AudioMetadata | None
    ) -> dict:
        """Builds pydub export parameters from audio metadata.

        Args:
            audio_metadata: Source audio metadata. If None, exports as WAV.

        Returns:
            Dictionary of export parameters for pydub's export method.
        """
        if audio_metadata is None:
            return {"format": "wav"}

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

        # Sample rate
        if audio_metadata.sample_rate > 0:
            ffmpeg_params.extend(["-ar", str(audio_metadata.sample_rate)])

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
        """Generates a sine wave censor tone in memory.

        censor() renders its tone with FFmpeg's sine generator instead; this
        defines the tone the generator is matched against (same frequency,
        level and channel layout).

        Args:
            duration_ms: Duration in milliseconds.
            sample_rate: Sample rate in Hz.
            channels: Number of audio channels.

        Returns:
            AudioSegment with the censor tone.
        """
        tone = Sine(self.tone_freq).to_audio_segment(
            duration=duration_ms,
            volume=TONE_LEVEL_DBFS,  # -20 dBFS to avoid clipping
        )
        tone = tone.set_frame_rate(sample_rate)

        if channels > 1:
            tone = tone.set_channels(channels)

        return tone

