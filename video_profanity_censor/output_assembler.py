"""Output assembly: muxes censored audio with original video streams using FFmpeg."""

import subprocess
from pathlib import Path

import ffmpeg

from video_profanity_censor.models import AssemblyResult


# Audio codecs that FFmpeg can re-encode to (common ones supported by default)
SUPPORTED_AUDIO_CODECS = {
    "aac",
    "mp3",
    "ac3",
    "eac3",
    "flac",
    "opus",
    "vorbis",
    "pcm_s16le",
    "pcm_s24le",
    "pcm_s32le",
    "pcm_f32le",
    "libmp3lame",
    "wmav2",
    "alac",
}

# Map of container format extensions to FFmpeg format names
CONTAINER_FORMAT_MAP = {
    ".mp4": "mp4",
    ".mkv": "matroska",
    ".avi": "avi",
    ".mov": "mov",
    ".wmv": "asf",
}



# Fallback codec for unsupported source codecs (e.g., DTS, TrueHD)
# AC3 supports up to 5.1 surround at 640kbps; EAC3 supports 7.1 and higher bitrates
FALLBACK_CODEC = "ac3"
FALLBACK_CODEC_HIGH_CHANNEL = "eac3"  # Used when source has > 6 channels


class OutputAssembler:
    """Assembles the final output video by muxing censored audio with original video stream."""

    def assemble(
        self,
        source_path: Path,
        censored_audio_path: Path,
        output_path: Path,
        audio_track_index: int = 0,
    ) -> AssemblyResult:
        """Muxes censored audio with original video (stream copy for video).

        Copies the video stream without re-encoding, replaces the specified audio
        track with the censored version, and preserves all other streams (subtitles,
        additional audio tracks, metadata, chapter markers, and timecodes).

        Args:
            source_path: Path to the original source video file.
            censored_audio_path: Path to the censored audio file.
            output_path: Path for the assembled output video file.
            audio_track_index: Index of the audio track that was censored (default: 0).

        Returns:
            AssemblyResult with output path and assembly details.

        Raises:
            FileNotFoundError: If source or censored audio file does not exist.
            ValueError: If the source uses an unsupported audio codec.
            RuntimeError: If FFmpeg assembly fails.
        """
        source_path = Path(source_path)
        censored_audio_path = Path(censored_audio_path)
        output_path = Path(output_path)

        if not source_path.exists():
            raise FileNotFoundError(f"Source file not found: {source_path}")
        if not censored_audio_path.exists():
            raise FileNotFoundError(
                f"Censored audio file not found: {censored_audio_path}"
            )

        # Determine container format from source extension
        container_format = self._get_container_format(source_path)

        # Probe source to understand stream layout and validate audio codec
        probe_info = self._probe_source(source_path)
        streams = probe_info.get("streams", [])

        # Check for unsupported audio codec — fall back if needed
        fallback_codec: str | None = None
        audio_streams = [s for s in streams if s.get("codec_type") == "audio"]
        if audio_track_index < len(audio_streams):
            audio_codec = audio_streams[audio_track_index].get("codec_name", "")
            channels = audio_streams[audio_track_index].get("channels", 2)
            fallback_codec = self._validate_audio_codec(audio_codec, channels)

        # Count additional streams (non-video, non-target-audio)
        streams_preserved = self._count_preserved_streams(
            streams, audio_track_index
        )

        # Build and run the FFmpeg command
        self._run_assembly(
            source_path,
            censored_audio_path,
            output_path,
            audio_track_index,
            audio_streams,
            container_format,
            fallback_codec,
        )

        return AssemblyResult(
            output_path=output_path,
            container_format=container_format,
            video_stream_copied=True,
            streams_preserved=streams_preserved,
            codec_fallback=fallback_codec,
        )

    def _get_container_format(self, source_path: Path) -> str:
        """Determines the container format from the source file extension.

        Args:
            source_path: Path to the source video file.

        Returns:
            The container format string (e.g., "mp4", "matroska").
        """
        ext = source_path.suffix.lower()
        fmt = CONTAINER_FORMAT_MAP.get(ext)
        if fmt is None:
            # Fall back to extension without dot
            fmt = ext.lstrip(".")
        return fmt

    def _probe_source(self, source_path: Path) -> dict:
        """Probes the source file to get stream information.

        Args:
            source_path: Path to the source video file.

        Returns:
            Probe result dictionary with streams and format information.

        Raises:
            RuntimeError: If probing fails.
        """
        try:
            return ffmpeg.probe(str(source_path))
        except ffmpeg.Error as e:
            raise RuntimeError(
                f"Failed to probe source file: {e.stderr.decode() if e.stderr else str(e)}"
            ) from e

    def _validate_audio_codec(self, codec: str, channels: int = 2) -> str | None:
        """Validates that the audio codec is supported for re-encoding.

        If the codec is unsupported, returns a fallback codec (AC3 or EAC3)
        instead of raising an error.

        Args:
            codec: The audio codec name from the source file.
            channels: Number of audio channels (used to select appropriate fallback).

        Returns:
            None if the codec is supported, or the fallback codec name if unsupported.
        """
        if not codec or codec in SUPPORTED_AUDIO_CODECS:
            return None

        # Unsupported codec — fall back to AC3/EAC3
        if channels > 6:
            return FALLBACK_CODEC_HIGH_CHANNEL
        return FALLBACK_CODEC

    def _count_preserved_streams(
        self, streams: list[dict], audio_track_index: int
    ) -> int:
        """Counts additional streams that will be preserved (subtitles, extra audio).

        Args:
            streams: List of stream info dicts from probe.
            audio_track_index: Index of the audio track being replaced.

        Returns:
            Number of additional streams preserved.
        """
        audio_idx = 0
        count = 0
        for stream in streams:
            codec_type = stream.get("codec_type", "")
            if codec_type == "video":
                # Video is always preserved but counted separately
                continue
            elif codec_type == "audio":
                if audio_idx != audio_track_index:
                    # Other audio tracks are preserved
                    count += 1
                audio_idx += 1
            else:
                # Subtitles, data streams, attachments, etc.
                count += 1
        return count

    def _run_assembly(
        self,
        source_path: Path,
        censored_audio_path: Path,
        output_path: Path,
        audio_track_index: int,
        audio_streams: list[dict],
        container_format: str,
        fallback_codec: str | None = None,
    ) -> None:
        """Runs the FFmpeg command to assemble the output file.

        Uses stream mapping to:
        - Copy all video streams without re-encoding (-c:v copy)
        - Replace the target audio track with the censored version
        - Preserve all other audio tracks via stream copy
        - Preserve subtitle streams and other data streams
        - Preserve metadata and chapter markers

        Args:
            source_path: Path to the source video file.
            censored_audio_path: Path to the censored audio file.
            output_path: Path for the output video.
            audio_track_index: Index of the audio track being replaced.
            audio_streams: List of audio stream info dicts from probe.
            container_format: FFmpeg format name for the output container.
            fallback_codec: If set, re-encode the censored track to this codec
                           instead of stream copying (used when source codec is unsupported).

        Raises:
            RuntimeError: If FFmpeg command fails.
        """
        # Build FFmpeg command using subprocess for precise stream mapping control
        cmd = [
            "ffmpeg",
            "-y",  # Overwrite output
            "-i", str(source_path),  # Input 0: original video
            "-i", str(censored_audio_path),  # Input 1: censored audio
        ]

        # Map all video streams from source (stream copy)
        cmd.extend(["-map", "0:v"])

        # Map audio streams: replace the target track, keep others from source
        for idx in range(len(audio_streams)):
            if idx == audio_track_index:
                # Use censored audio from input 1
                cmd.extend(["-map", "1:a:0"])
            else:
                # Keep other audio tracks from source
                cmd.extend(["-map", f"0:a:{idx}"])

        # Map all subtitle streams from source
        cmd.extend(["-map", "0:s?"])

        # Map any data/attachment streams from source
        cmd.extend(["-map", "0:t?"])

        # Codec settings
        cmd.extend(["-c:v", "copy"])  # Copy video without re-encoding

        # For non-replaced audio tracks, stream copy them
        audio_output_idx = 0
        for idx in range(len(audio_streams)):
            if idx == audio_track_index:
                if fallback_codec:
                    # Re-encode censored audio to the fallback codec
                    cmd.extend([f"-c:a:{audio_output_idx}", fallback_codec])
                    # Preserve channel layout and use high bitrate for quality
                    source_channels = audio_streams[idx].get("channels", 2)
                    source_bitrate = audio_streams[idx].get("bit_rate")
                    if source_channels:
                        cmd.extend([f"-ac:{audio_output_idx}", str(source_channels)])
                    # Use 640k for surround, 384k for stereo (high quality)
                    if source_channels and source_channels > 2:
                        cmd.extend([f"-b:a:{audio_output_idx}", "640k"])
                    else:
                        cmd.extend([f"-b:a:{audio_output_idx}", "384k"])
                else:
                    # The censored audio is already encoded properly - stream copy it
                    cmd.extend([f"-c:a:{audio_output_idx}", "copy"])
            else:
                # Other audio tracks: stream copy
                cmd.extend([f"-c:a:{audio_output_idx}", "copy"])
            audio_output_idx += 1

        # Copy subtitle streams
        cmd.extend(["-c:s", "copy"])

        # Preserve metadata and chapter markers
        cmd.extend(["-map_metadata", "0"])
        cmd.extend(["-map_chapters", "0"])

        # Output format
        cmd.extend(["-f", container_format])

        # Output file
        cmd.append(str(output_path))

        # Run FFmpeg
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"FFmpeg output assembly failed: {e.stderr}"
            ) from e
        except FileNotFoundError as e:
            raise RuntimeError(
                "FFmpeg not found. Please ensure FFmpeg is installed and available in PATH."
            ) from e
