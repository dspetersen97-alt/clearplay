"""Audio processing for censoring profane segments using FFmpeg and pydub."""

from pathlib import Path

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

        # Load the audio file
        audio = AudioSegment.from_file(str(audio_path))

        # Sort detections by start time
        sorted_detections = sorted(
            detections, key=lambda d: d.timestamp_range.start
        )

        total_censored_duration = 0.0
        crossfade_ms = 10  # 10ms crossfade at boundaries
        audio_duration_ms = len(audio)

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

            # Generate replacement audio
            if self.mode == CensorMode.TONE:
                replacement = self._generate_tone(
                    duration_ms=duration_ms,
                    sample_rate=audio.frame_rate,
                    channels=audio.channels,
                )
            else:
                replacement = AudioSegment.silent(
                    duration=duration_ms,
                    frame_rate=audio.frame_rate,
                )
                # Match channels
                if audio.channels > 1 and replacement.channels == 1:
                    replacement = replacement.set_channels(audio.channels)

            # Apply crossfade at boundaries
            replacement = self._apply_crossfade(
                audio, replacement, start_ms, end_ms, crossfade_ms
            )

            # Replace the segment in the audio
            audio = audio[:start_ms] + replacement + audio[end_ms:]

            total_censored_duration += duration_ms / 1000.0

            if progress_callback:
                progress = (sorted_detections.index(detection) + 1) / len(sorted_detections)
                progress_callback(
                    ProcessingStage.AUDIO_CENSORING,
                    progress * 100,
                    f"Censored segment {sorted_detections.index(detection) + 1}/{len(sorted_detections)}",
                )

        # Export the censored audio
        export_params = self._build_export_params(audio_metadata)
        audio.export(str(output_path), **export_params)

        return CensorResult(
            censored_audio_path=output_path,
            segments_censored=len(sorted_detections),
            total_censored_duration=total_censored_duration,
        )

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
