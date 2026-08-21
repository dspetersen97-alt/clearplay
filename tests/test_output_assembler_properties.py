"""Property-based tests for OutputAssembler container format matching.

**Validates: Requirements 5.4**
"""

from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
from hypothesis import given, strategies as st, settings

from video_profanity_censor.output_assembler import (
    OutputAssembler,
    CONTAINER_FORMAT_MAP,
)


# All supported extensions from the CONTAINER_FORMAT_MAP
SUPPORTED_EXTENSIONS = sorted(CONTAINER_FORMAT_MAP.keys())

# Strategy: generate supported format extensions
supported_extensions = st.sampled_from(SUPPORTED_EXTENSIONS)

# Strategy: generate file stems (valid filenames without extension)
file_stems = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyz0123456789_-",
    min_size=1,
    max_size=30,
).filter(lambda s: len(s.strip("_-")) > 0)


class TestOutputContainerFormatMatchesSource:
    """Property 8: Output Container Format Matches Source.

    For any supported input container format, the OutputAssembler SHALL produce
    an output file in the same container format as the source file.

    **Validates: Requirements 5.4**
    """

    @given(extension=supported_extensions, stem=file_stems)
    @settings(max_examples=200)
    def test_output_format_matches_source_format(
        self, extension: str, stem: str
    ):
        """The output container format always matches the source file's format."""
        assembler = OutputAssembler()

        source_path = Path(f"C:/videos/{stem}{extension}")
        expected_format = CONTAINER_FORMAT_MAP[extension]

        # _get_container_format is the method that determines output format
        result_format = assembler._get_container_format(source_path)

        assert result_format == expected_format, (
            f"Output format for source '{source_path.name}' should be "
            f"'{expected_format}', but got '{result_format}'"
        )

    @given(extension=supported_extensions)
    @settings(max_examples=100)
    def test_format_determination_is_case_insensitive(self, extension: str):
        """Container format determination works regardless of extension case."""
        assembler = OutputAssembler()

        # Test uppercase variant
        upper_path = Path(f"C:/videos/test{extension.upper()}")
        expected_format = CONTAINER_FORMAT_MAP[extension]

        result_format = assembler._get_container_format(upper_path)

        assert result_format == expected_format, (
            f"Uppercase extension '{extension.upper()}' should resolve to "
            f"'{expected_format}', but got '{result_format}'"
        )

    @given(extension=supported_extensions, stem=file_stems)
    @settings(max_examples=200)
    def test_assemble_produces_output_in_same_container_format(
        self, extension: str, stem: str
    ):
        """Full assembly produces output with the same container format as source."""
        assembler = OutputAssembler()

        source_path = Path(f"C:/videos/{stem}{extension}")
        censored_audio_path = Path("C:/tmp/censored.wav")
        output_path = Path(f"C:/output/{stem}_censored{extension}")

        expected_format = CONTAINER_FORMAT_MAP[extension]

        # Mock all external dependencies (file system, ffprobe, subprocess)
        mock_probe_result = {
            "streams": [
                {"codec_type": "video", "codec_name": "h264"},
                {"codec_type": "audio", "codec_name": "aac"},
            ],
            "format": {"format_name": "mov,mp4,m4a,3gp,3g2,mj2"},
        }

        with (
            patch.object(Path, "exists", return_value=True),
            patch("ffmpeg.probe", return_value=mock_probe_result),
            patch("subprocess.run") as mock_run,
        ):
            mock_run.return_value = MagicMock(returncode=0)
            result = assembler.assemble(
                source_path=source_path,
                censored_audio_path=censored_audio_path,
                output_path=output_path,
            )

        # Verify the assembly result uses the correct container format
        assert result.container_format == expected_format, (
            f"AssemblyResult container_format for source '{source_path.name}' "
            f"should be '{expected_format}', but got '{result.container_format}'"
        )

        # Verify the FFmpeg command was called with the correct format flag
        mock_run.assert_called_once()
        ffmpeg_cmd = mock_run.call_args[0][0]

        # Find the -f flag and verify it uses the expected format
        assert "-f" in ffmpeg_cmd, "FFmpeg command should include -f format flag"
        f_index = ffmpeg_cmd.index("-f")
        actual_format_flag = ffmpeg_cmd[f_index + 1]

        assert actual_format_flag == expected_format, (
            f"FFmpeg -f flag should be '{expected_format}' for source "
            f"extension '{extension}', but got '{actual_format_flag}'"
        )

    @given(
        src_ext=supported_extensions,
        out_ext=supported_extensions,
        stem=file_stems,
    )
    @settings(max_examples=200)
    def test_format_derived_from_source_not_output_path(
        self, src_ext: str, out_ext: str, stem: str
    ):
        """The container format is always derived from the source file extension,
        not the output file path extension."""
        assembler = OutputAssembler()

        source_path = Path(f"C:/videos/{stem}{src_ext}")
        expected_format = CONTAINER_FORMAT_MAP[src_ext]

        # Derive format from source path
        result_format = assembler._get_container_format(source_path)

        assert result_format == expected_format, (
            f"Format should be derived from source extension '{src_ext}' "
            f"('{expected_format}'), but got '{result_format}'"
        )
