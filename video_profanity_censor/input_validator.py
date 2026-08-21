"""Input validation for video files before processing."""

import os
from pathlib import Path

import ffmpeg

from video_profanity_censor.models import AudioMetadata, ValidationResult


class InputValidator:
    """Validates input video files for the censoring pipeline.

    Checks file existence, format, size, container validity, and audio track presence.
    """

    SUPPORTED_FORMATS: set[str] = {".mp4", ".mkv", ".avi", ".mov", ".wmv"}
    MAX_FILE_SIZE_BYTES: int = 50 * 1024 * 1024 * 1024  # 50 GB

    def validate(self, file_path: Path) -> ValidationResult:
        """Validates the input file and returns file metadata or error.

        Args:
            file_path: Path to the video file to validate.

        Returns:
            ValidationResult with is_valid=True and metadata on success,
            or is_valid=False with an error_message on failure.
        """
        # Check file existence and accessibility
        if not file_path.exists():
            return ValidationResult(
                is_valid=False,
                error_message=f"File not found: '{file_path}'",
            )

        if not file_path.is_file():
            return ValidationResult(
                is_valid=False,
                error_message=f"Path is not a file: '{file_path}'",
            )

        if not os.access(file_path, os.R_OK):
            return ValidationResult(
                is_valid=False,
                error_message=f"File is not accessible: '{file_path}'",
            )

        # Verify file extension against supported formats
        extension = file_path.suffix.lower()
        if extension not in self.SUPPORTED_FORMATS:
            supported = ", ".join(sorted(self.SUPPORTED_FORMATS))
            return ValidationResult(
                is_valid=False,
                error_message=(
                    f"Unsupported file format: '{extension}'. "
                    f"Supported formats: {supported}"
                ),
            )

        # Validate file size
        file_size = file_path.stat().st_size
        if file_size > self.MAX_FILE_SIZE_BYTES:
            max_gb = self.MAX_FILE_SIZE_BYTES / (1024 * 1024 * 1024)
            size_gb = file_size / (1024 * 1024 * 1024)
            return ValidationResult(
                is_valid=False,
                error_message=(
                    f"File size ({size_gb:.2f} GB) exceeds maximum allowed "
                    f"size of {max_gb:.0f} GB"
                ),
                file_size_bytes=file_size,
            )

        # Probe file with FFmpeg to confirm valid video container
        try:
            probe = ffmpeg.probe(str(file_path))
        except ffmpeg.Error:
            return ValidationResult(
                is_valid=False,
                error_message=(
                    f"Cannot parse file as a valid video container: '{file_path}'"
                ),
                file_size_bytes=file_size,
            )
        except FileNotFoundError:
            return ValidationResult(
                is_valid=False,
                error_message=(
                    "FFmpeg is not installed or not found in PATH. "
                    "FFmpeg is required for video file validation."
                ),
                file_size_bytes=file_size,
            )

        # Extract container format info
        format_info = probe.get("format", {})
        container_format = format_info.get("format_name", None)

        # Check for video stream
        streams = probe.get("streams", [])
        video_streams = [s for s in streams if s.get("codec_type") == "video"]
        video_codec = video_streams[0].get("codec_name") if video_streams else None

        # Confirm at least one audio track exists
        audio_streams = [s for s in streams if s.get("codec_type") == "audio"]
        if not audio_streams:
            return ValidationResult(
                is_valid=False,
                error_message="No audio track found in the file",
                video_codec=video_codec,
                container_format=container_format,
                file_size_bytes=file_size,
            )

        # Extract audio metadata from the first audio stream
        audio_stream = audio_streams[0]
        audio_metadata = self._extract_audio_metadata(audio_stream, streams)

        return ValidationResult(
            is_valid=True,
            audio_metadata=audio_metadata,
            video_codec=video_codec,
            container_format=container_format,
            file_size_bytes=file_size,
        )

    def _extract_audio_metadata(
        self, audio_stream: dict, all_streams: list[dict]
    ) -> AudioMetadata:
        """Extracts AudioMetadata from an FFmpeg probe audio stream dict.

        Args:
            audio_stream: The audio stream dictionary from ffprobe.
            all_streams: All streams to determine the audio track index.

        Returns:
            AudioMetadata populated from the stream information.
        """
        # Determine the audio track index (among audio streams only)
        audio_streams = [s for s in all_streams if s.get("codec_type") == "audio"]
        track_index = audio_streams.index(audio_stream)

        codec = audio_stream.get("codec_name", "unknown")
        sample_rate = int(audio_stream.get("sample_rate", 0))
        channels = int(audio_stream.get("channels", 0))
        channel_layout = audio_stream.get("channel_layout", "unknown")

        # Bit depth: try bits_per_sample, then bits_per_raw_sample
        bit_depth = int(audio_stream.get("bits_per_sample", 0))
        if bit_depth == 0:
            bit_depth = int(audio_stream.get("bits_per_raw_sample", 0))

        # Bitrate: try stream-level, then fall back to format-level for single audio
        bitrate_str = audio_stream.get("bit_rate", "0")
        bitrate = int(bitrate_str) if bitrate_str else 0

        # Duration from stream or format
        duration_str = audio_stream.get("duration", "0")
        duration = float(duration_str) if duration_str else 0.0

        return AudioMetadata(
            codec=codec,
            sample_rate=sample_rate,
            bit_depth=bit_depth,
            channels=channels,
            channel_layout=channel_layout,
            bitrate=bitrate,
            duration_seconds=duration,
            track_index=track_index,
        )
