"""Audio extraction from video files using FFmpeg."""

import json
import subprocess
import tempfile
from pathlib import Path

import ffmpeg

from video_profanity_censor.models import AudioExtractionResult, AudioMetadata


class AudioExtractor:
    """Extracts audio tracks from video files to WAV format for transcription."""

    def extract(
        self,
        source_path: Path,
        track_index: int = 0,
        output_path: Path | None = None,
    ) -> AudioExtractionResult:
        """Extracts audio track to WAV format for transcription.

        Args:
            source_path: Path to the source video file.
            track_index: Index of the audio track to extract (default: 0, first track).
            output_path: Path for the output WAV file. If None, a temporary file is created.

        Returns:
            AudioExtractionResult with the output WAV path and audio metadata.

        Raises:
            FileNotFoundError: If the source file does not exist.
            ValueError: If the specified track index does not exist.
            RuntimeError: If FFmpeg extraction fails.
        """
        source_path = Path(source_path)
        if not source_path.exists():
            raise FileNotFoundError(f"Source file not found: {source_path}")

        # Probe the source file for audio stream metadata
        audio_metadata = self._probe_audio_metadata(source_path, track_index)

        # Determine output path
        if output_path is None:
            output_path = Path(
                tempfile.mktemp(suffix=".wav", prefix="audio_extract_")
            )
        else:
            output_path = Path(output_path)

        # Extract the audio track to WAV
        self._extract_audio(source_path, track_index, output_path)

        return AudioExtractionResult(
            output_path=output_path,
            audio_metadata=audio_metadata,
        )

    def _probe_audio_metadata(
        self, source_path: Path, track_index: int
    ) -> AudioMetadata:
        """Probes the source file to get audio stream metadata.

        Args:
            source_path: Path to the source video file.
            track_index: Index of the audio track to probe.

        Returns:
            AudioMetadata for the specified audio track.

        Raises:
            ValueError: If the specified track index does not exist.
            RuntimeError: If probing fails.
        """
        try:
            probe = ffmpeg.probe(str(source_path))
        except ffmpeg.Error as e:
            raise RuntimeError(
                f"Failed to probe source file: {e.stderr.decode() if e.stderr else str(e)}"
            ) from e

        # Filter audio streams only
        audio_streams = [
            stream
            for stream in probe.get("streams", [])
            if stream.get("codec_type") == "audio"
        ]

        if not audio_streams:
            raise ValueError(f"No audio tracks found in: {source_path}")

        if track_index >= len(audio_streams):
            raise ValueError(
                f"Audio track index {track_index} does not exist. "
                f"File has {len(audio_streams)} audio track(s) (indices 0-{len(audio_streams) - 1})."
            )

        stream = audio_streams[track_index]

        # Extract metadata from the stream
        codec = stream.get("codec_name", "unknown")
        sample_rate = int(stream.get("sample_rate", 0))
        channels = int(stream.get("channels", 0))
        channel_layout = stream.get("channel_layout", self._default_channel_layout(channels))
        bitrate = int(stream.get("bit_rate", 0))
        duration_seconds = float(stream.get("duration", 0.0))

        # bit_depth: derive from bits_per_sample or bits_per_raw_sample
        bit_depth = int(
            stream.get("bits_per_sample", 0)
            or stream.get("bits_per_raw_sample", 0)
        )

        # If duration not in stream, try format level
        if duration_seconds == 0.0:
            format_info = probe.get("format", {})
            duration_seconds = float(format_info.get("duration", 0.0))

        # If bitrate not in stream, try to compute from format
        if bitrate == 0:
            format_info = probe.get("format", {})
            format_bitrate = int(format_info.get("bit_rate", 0))
            # Rough estimate: divide format bitrate among streams if only one audio
            if format_bitrate > 0 and len(audio_streams) == 1:
                bitrate = format_bitrate

        return AudioMetadata(
            codec=codec,
            sample_rate=sample_rate,
            bit_depth=bit_depth,
            channels=channels,
            channel_layout=channel_layout,
            bitrate=bitrate,
            duration_seconds=duration_seconds,
            track_index=track_index,
        )

    def _extract_audio(
        self, source_path: Path, track_index: int, output_path: Path
    ) -> None:
        """Extracts the specified audio track to a WAV file using FFmpeg.

        Args:
            source_path: Path to the source video file.
            track_index: Index of the audio track to extract.
            output_path: Path to write the output WAV file.

        Raises:
            RuntimeError: If FFmpeg extraction fails.
        """
        try:
            # Build the ffmpeg command to extract the specific audio track
            # Map the audio stream by finding its absolute stream index
            probe = ffmpeg.probe(str(source_path))
            audio_streams = [
                stream
                for stream in probe.get("streams", [])
                if stream.get("codec_type") == "audio"
            ]

            if track_index >= len(audio_streams):
                raise ValueError(
                    f"Audio track index {track_index} does not exist."
                )

            # Get the absolute stream index for the audio track
            absolute_index = int(audio_streams[track_index].get("index", 0))

            # Extract audio to WAV using ffmpeg-python
            stream = ffmpeg.input(str(source_path))
            stream = stream[str(absolute_index)]
            stream = ffmpeg.output(
                stream,
                str(output_path),
                acodec="pcm_s16le",
                ar=audio_streams[track_index].get("sample_rate", "44100"),
            )
            stream = ffmpeg.overwrite_output(stream)
            ffmpeg.run(stream, quiet=True)

        except ffmpeg.Error as e:
            raise RuntimeError(
                f"FFmpeg audio extraction failed: {e.stderr.decode() if e.stderr else str(e)}"
            ) from e

    @staticmethod
    def _default_channel_layout(channels: int) -> str:
        """Returns a default channel layout name based on channel count."""
        layouts = {
            1: "mono",
            2: "stereo",
            6: "5.1",
            8: "7.1",
        }
        return layouts.get(channels, f"{channels} channels")
