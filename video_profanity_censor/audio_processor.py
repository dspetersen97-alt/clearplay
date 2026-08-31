"""Audio processing for censoring profane segments using FFmpeg and pydub."""

import gc
import shutil
import tempfile
import wave
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
        self,
        mode: CensorMode = CensorMode.MUTE,
        tone_freq: int = 1000,
        censor_buffer_ms: int = 100,
        max_word_duration_ms: int = 750,
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

        Uses streaming/in-place chunked processing to maintain minimal RAM usage
        even for multi-gigabyte audio tracks.

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
            shutil.copy2(str(audio_path), str(output_path))
            return CensorResult(
                censored_audio_path=output_path,
                segments_censored=0,
                total_censored_duration=0.0,
            )

        # Try low-memory streaming WAV processing first
        export_params = self._build_export_params(audio_metadata)
        is_wav_export = export_params.get("format") == "wav"

        try:
            target_wav_path = output_path if is_wav_export else Path(
                tempfile.mktemp(suffix=".wav", prefix="censored_stream_")
            )
            result = self._censor_wav_stream(
                audio_path=audio_path,
                detections=detections,
                output_path=target_wav_path,
                progress_callback=progress_callback,
            )

            if not is_wav_export:
                # Re-encode to requested format using pydub
                audio = AudioSegment.from_file(str(target_wav_path))
                audio.export(str(output_path), **export_params)
                del audio
                if target_wav_path.exists():
                    target_wav_path.unlink()
                gc.collect()

            return result
        except Exception:
            # Fall back to in-place bytearray processing if wave module cannot handle
            return self._censor_inplace(
                audio_path=audio_path,
                detections=detections,
                output_path=output_path,
                export_params=export_params,
                progress_callback=progress_callback,
            )

    def _censor_wav_stream(
        self,
        audio_path: Path,
        detections: list[Detection],
        output_path: Path,
        progress_callback: ProgressCallback = None,
    ) -> CensorResult:
        """Processes WAV files using streaming I/O with constant minimal RAM usage."""
        sorted_detections = sorted(
            detections, key=lambda d: d.timestamp_range.start
        )
        crossfade_ms = 10

        with wave.open(str(audio_path), "rb") as in_wav:
            params = in_wav.getparams()
            nchannels = params.nchannels
            sampwidth = params.sampwidth
            framerate = params.framerate
            nframes = params.nframes
            frame_width = nchannels * sampwidth
            audio_duration_ms = int(nframes * 1000.0 / framerate) if framerate > 0 else 0

            # Pre-compute replacement buffers and frame ranges
            replacements: list[tuple[int, int, bytes]] = []
            total_censored_duration = 0.0

            for detection in sorted_detections:
                start_ms = int(detection.timestamp_range.start * 1000) - self.censor_buffer_ms
                end_ms = int(detection.timestamp_range.end * 1000) + self.censor_buffer_ms

                if self.max_word_duration_ms > 0:
                    raw_duration = end_ms - start_ms
                    if raw_duration > self.max_word_duration_ms:
                        start_ms = end_ms - self.max_word_duration_ms

                start_ms = max(0, start_ms)
                end_ms = min(end_ms, audio_duration_ms)
                duration_ms = end_ms - start_ms

                if duration_ms <= 0:
                    continue

                if self.mode == CensorMode.TONE:
                    replacement = self._generate_tone(
                        duration_ms=duration_ms,
                        sample_rate=framerate,
                        channels=nchannels,
                    )
                else:
                    replacement = AudioSegment.silent(
                        duration=duration_ms,
                        frame_rate=framerate,
                    )
                    if nchannels > 1 and replacement.channels == 1:
                        replacement = replacement.set_channels(nchannels)

                if replacement.sample_width != sampwidth:
                    replacement = replacement.set_sample_width(sampwidth)

                replacement = self._apply_crossfade(
                    None, replacement, start_ms, end_ms, crossfade_ms
                )

                start_frame = int(start_ms * framerate / 1000.0)
                rep_data = replacement.raw_data
                rep_nframes = len(rep_data) // frame_width
                end_frame = min(nframes, start_frame + rep_nframes)
                replacements.append((start_frame, end_frame, rep_data))

                total_censored_duration += duration_ms / 1000.0

            with wave.open(str(output_path), "wb") as out_wav:
                out_wav.setparams(params)
                chunk_size_frames = 65536  # ~1.4s of 48kHz audio (~256KB-1MB buffer)
                frames_read = 0

                while frames_read < nframes:
                    to_read = min(chunk_size_frames, nframes - frames_read)
                    data = in_wav.readframes(to_read)
                    chunk_start = frames_read
                    chunk_end = frames_read + to_read

                    chunk_buf = None
                    for s_f, e_f, rep_data in replacements:
                        if s_f < chunk_end and e_f > chunk_start:
                            if chunk_buf is None:
                                chunk_buf = bytearray(data)

                            ov_start = max(chunk_start, s_f)
                            ov_end = min(chunk_end, e_f)

                            c_byte_s = (ov_start - chunk_start) * frame_width
                            c_byte_e = (ov_end - chunk_start) * frame_width

                            r_byte_s = (ov_start - s_f) * frame_width
                            r_byte_e = r_byte_s + (c_byte_e - c_byte_s)

                            chunk_buf[c_byte_s:c_byte_e] = rep_data[r_byte_s:r_byte_e]

                    if chunk_buf is not None:
                        out_wav.writeframes(chunk_buf)
                    else:
                        out_wav.writeframes(data)

                    frames_read += to_read

                    if progress_callback and nframes > 0:
                        percent = (frames_read / nframes) * 100.0
                        progress_callback(
                            ProcessingStage.AUDIO_CENSORING,
                            percent,
                            f"Censoring audio: {percent:.0f}%",
                        )

        gc.collect()
        return CensorResult(
            censored_audio_path=output_path,
            segments_censored=len(sorted_detections),
            total_censored_duration=total_censored_duration,
        )

    def _censor_inplace(
        self,
        audio_path: Path,
        detections: list[Detection],
        output_path: Path,
        export_params: dict,
        progress_callback: ProgressCallback = None,
    ) -> CensorResult:
        """In-place bytearray fallback when streaming wave is not possible."""
        audio = AudioSegment.from_file(str(audio_path))
        frame_rate = audio.frame_rate
        channels = audio.channels
        sample_width = audio.sample_width
        frame_width = audio.frame_width
        audio_duration_ms = len(audio)

        raw_data = bytearray(audio.raw_data)
        del audio
        gc.collect()

        sorted_detections = sorted(
            detections, key=lambda d: d.timestamp_range.start
        )
        total_censored_duration = 0.0
        crossfade_ms = 10

        for idx, detection in enumerate(sorted_detections):
            start_ms = int(detection.timestamp_range.start * 1000) - self.censor_buffer_ms
            end_ms = int(detection.timestamp_range.end * 1000) + self.censor_buffer_ms

            if self.max_word_duration_ms > 0:
                raw_duration = end_ms - start_ms
                if raw_duration > self.max_word_duration_ms:
                    start_ms = end_ms - self.max_word_duration_ms

            start_ms = max(0, start_ms)
            end_ms = min(end_ms, audio_duration_ms)
            duration_ms = end_ms - start_ms

            if duration_ms <= 0:
                continue

            if self.mode == CensorMode.TONE:
                replacement = self._generate_tone(
                    duration_ms=duration_ms,
                    sample_rate=frame_rate,
                    channels=channels,
                )
            else:
                replacement = AudioSegment.silent(
                    duration=duration_ms,
                    frame_rate=frame_rate,
                )
                if channels > 1 and replacement.channels == 1:
                    replacement = replacement.set_channels(channels)

            if replacement.sample_width != sample_width:
                replacement = replacement.set_sample_width(sample_width)

            replacement = self._apply_crossfade(
                None, replacement, start_ms, end_ms, crossfade_ms
            )

            start_sample = int(start_ms * frame_rate / 1000.0)
            start_byte = start_sample * frame_width
            rep_bytes = replacement.raw_data
            end_byte = min(len(raw_data), start_byte + len(rep_bytes))
            raw_data[start_byte:end_byte] = rep_bytes[:end_byte - start_byte]

            total_censored_duration += duration_ms / 1000.0

            if progress_callback:
                progress = (idx + 1) / len(sorted_detections)
                progress_callback(
                    ProcessingStage.AUDIO_CENSORING,
                    progress * 100,
                    f"Censored segment {idx + 1}/{len(sorted_detections)}",
                )

        censored_audio = AudioSegment(
            data=bytes(raw_data),
            sample_width=sample_width,
            frame_rate=frame_rate,
            channels=channels,
        )
        del raw_data
        gc.collect()

        censored_audio.export(str(output_path), **export_params)
        del censored_audio
        gc.collect()

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
        original: AudioSegment | None,
        replacement: AudioSegment,
        start_ms: int,
        end_ms: int,
        crossfade_ms: int,
    ) -> AudioSegment:
        """Applies crossfade at the boundaries of a replacement segment.

        Creates a smooth transition between original audio and the replacement
        to avoid audible clicks or pops.

        Args:
            original: The full original audio (optional, unused).
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
