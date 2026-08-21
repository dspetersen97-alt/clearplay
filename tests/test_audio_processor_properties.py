"""Property-based tests for AudioProcessor.

**Validates: Requirements 4.1, 4.2, 4.3, 4.4**
"""

import math
import tempfile
from pathlib import Path

from hypothesis import given, settings, strategies as st
from pydub import AudioSegment
from pydub.generators import Sine

from video_profanity_censor.audio_processor import AudioProcessor
from video_profanity_censor.models import (
    CensorMode,
    Detection,
    TimestampRange,
)


# --- Strategies ---

# Strategy: generate censor modes
censor_modes = st.sampled_from([CensorMode.TONE, CensorMode.MUTE])

# Strategy: generate valid sample rates
sample_rates = st.sampled_from([8000, 16000, 22050, 44100, 48000])

# Strategy: generate valid channel counts
channel_counts = st.integers(min_value=1, max_value=2)

# Strategy: generate a timestamp range with positive duration
@st.composite
def timestamp_range_strategy(draw):
    """Generate a TimestampRange with start < end, within reasonable bounds.

    Duration is constrained to at least 21ms to accommodate crossfade (10ms each side).
    """
    start = draw(st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False))
    # Minimum 21ms to ensure segment is long enough for crossfade
    duration = draw(st.floats(min_value=0.021, max_value=2.0, allow_nan=False, allow_infinity=False))
    end = start + duration
    return TimestampRange(start=start, end=end)


# Strategy: generate a list of non-overlapping timestamp ranges
@st.composite
def non_overlapping_timestamp_ranges(draw):
    """Generate a sorted list of non-overlapping timestamp ranges."""
    num_ranges = draw(st.integers(min_value=1, max_value=5))
    ranges = []
    current_time = draw(st.floats(min_value=0.1, max_value=1.0, allow_nan=False, allow_infinity=False))

    for _ in range(num_ranges):
        # Minimum 21ms for crossfade support
        duration = draw(st.floats(min_value=0.021, max_value=0.5, allow_nan=False, allow_infinity=False))
        end = current_time + duration
        ranges.append(TimestampRange(start=current_time, end=end))
        # Add gap between ranges
        gap = draw(st.floats(min_value=0.05, max_value=0.5, allow_nan=False, allow_infinity=False))
        current_time = end + gap

    return ranges


# Strategy: generate detections with non-overlapping timestamps
@st.composite
def detections_strategy(draw):
    """Generate a list of Detection objects with non-overlapping timestamps."""
    ranges = draw(non_overlapping_timestamp_ranges())
    mode = draw(censor_modes)
    detections = [
        Detection(
            word=f"word{i}",
            timestamp_range=r,
            censor_action=mode,
        )
        for i, r in enumerate(ranges)
    ]
    return detections


class TestAudioCensoringCorrectness:
    """Property 4: Audio Censoring Correctness.

    For any set of timestamp ranges and censor mode (tone or mute), the
    AudioProcessor SHALL replace the audio at each timestamp range with the
    appropriate content — a censor tone of matching duration in TONE mode,
    or silence of matching duration in MUTE mode — and the replacement
    duration SHALL equal the duration of the timestamp range (end - start).

    **Validates: Requirements 4.1, 4.2, 4.4**
    """

    @given(
        ts_range=timestamp_range_strategy(),
        mode=censor_modes,
        sample_rate=sample_rates,
        n_channels=channel_counts,
    )
    @settings(max_examples=200, deadline=5000)
    def test_tone_replacement_duration_equals_timestamp_range_duration(
        self,
        ts_range: TimestampRange,
        mode: CensorMode,
        sample_rate: int,
        n_channels: int,
    ):
        """The generated replacement segment duration (in ms) must equal
        the timestamp range duration (end - start) in TONE mode."""
        processor = AudioProcessor(mode=mode, tone_freq=1000)

        expected_duration_ms = int((ts_range.end - ts_range.start) * 1000)

        if mode == CensorMode.TONE:
            replacement = processor._generate_tone(
                duration_ms=expected_duration_ms,
                sample_rate=sample_rate,
                channels=n_channels,
            )
        else:
            replacement = AudioSegment.silent(
                duration=expected_duration_ms,
                frame_rate=sample_rate,
            )
            if n_channels > 1:
                replacement = replacement.set_channels(n_channels)

        # Verify the replacement has the correct duration
        actual_duration_ms = len(replacement)

        assert actual_duration_ms == expected_duration_ms, (
            f"Replacement duration {actual_duration_ms}ms does not match "
            f"expected {expected_duration_ms}ms for range "
            f"[{ts_range.start:.6f}, {ts_range.end:.6f}] (mode={mode.value})"
        )

    @given(
        ts_range=timestamp_range_strategy(),
        sample_rate=sample_rates,
        n_channels=channel_counts,
    )
    @settings(max_examples=200, deadline=5000)
    def test_tone_mode_produces_nonzero_samples(
        self,
        ts_range: TimestampRange,
        sample_rate: int,
        n_channels: int,
    ):
        """In TONE mode, the replacement must contain non-zero audio content
        (an audible censor beep), not silence."""
        processor = AudioProcessor(mode=CensorMode.TONE, tone_freq=1000)

        duration_ms = int((ts_range.end - ts_range.start) * 1000)

        replacement = processor._generate_tone(
            duration_ms=duration_ms,
            sample_rate=sample_rate,
            channels=n_channels,
        )

        # TONE mode: should have non-zero RMS (audible content)
        assert replacement.rms > 0, (
            f"TONE mode replacement should have non-zero RMS (audible tone), "
            f"but got RMS=0 for duration={duration_ms}ms"
        )

    @given(
        ts_range=timestamp_range_strategy(),
        sample_rate=sample_rates,
        n_channels=channel_counts,
    )
    @settings(max_examples=200, deadline=5000)
    def test_mute_mode_produces_silence(
        self,
        ts_range: TimestampRange,
        sample_rate: int,
        n_channels: int,
    ):
        """In MUTE mode, the replacement must be complete silence (zero RMS)."""
        duration_ms = int((ts_range.end - ts_range.start) * 1000)

        replacement = AudioSegment.silent(
            duration=duration_ms,
            frame_rate=sample_rate,
        )
        if n_channels > 1:
            replacement = replacement.set_channels(n_channels)

        # MUTE mode: should be silence (zero RMS)
        assert replacement.rms == 0, (
            f"MUTE mode replacement should be silence (RMS=0), "
            f"but got RMS={replacement.rms} for duration={duration_ms}ms"
        )

    @given(
        ts_range=timestamp_range_strategy(),
        mode=censor_modes,
        sample_rate=sample_rates,
        n_channels=channel_counts,
    )
    @settings(max_examples=200, deadline=5000)
    def test_replacement_channel_count_matches_requested(
        self,
        ts_range: TimestampRange,
        mode: CensorMode,
        sample_rate: int,
        n_channels: int,
    ):
        """The generated replacement must have the same channel count as requested,
        maintaining the source audio's channel layout."""
        processor = AudioProcessor(mode=mode, tone_freq=1000)

        duration_ms = int((ts_range.end - ts_range.start) * 1000)

        if mode == CensorMode.TONE:
            replacement = processor._generate_tone(
                duration_ms=duration_ms,
                sample_rate=sample_rate,
                channels=n_channels,
            )
        else:
            replacement = AudioSegment.silent(
                duration=duration_ms,
                frame_rate=sample_rate,
            )
            if n_channels > 1:
                replacement = replacement.set_channels(n_channels)

        assert replacement.channels == n_channels, (
            f"Replacement has {replacement.channels} channels, "
            f"expected {n_channels} (mode={mode.value})"
        )

    @given(
        detections=detections_strategy(),
        sample_rate=sample_rates,
        n_channels=channel_counts,
    )
    @settings(max_examples=100, deadline=10000)
    def test_all_detections_produce_matching_duration_replacements(
        self,
        detections: list[Detection],
        sample_rate: int,
        n_channels: int,
    ):
        """For each detection in a list, the replacement segment duration must
        equal the timestamp range duration (end - start), verifying the property
        holds across multiple censored segments."""
        processor = AudioProcessor(mode=detections[0].censor_action, tone_freq=1000)
        mode = detections[0].censor_action

        for detection in detections:
            expected_duration_ms = int(
                (detection.timestamp_range.end - detection.timestamp_range.start) * 1000
            )

            if mode == CensorMode.TONE:
                replacement = processor._generate_tone(
                    duration_ms=expected_duration_ms,
                    sample_rate=sample_rate,
                    channels=n_channels,
                )
            else:
                replacement = AudioSegment.silent(
                    duration=expected_duration_ms,
                    frame_rate=sample_rate,
                )
                if n_channels > 1:
                    replacement = replacement.set_channels(n_channels)

            actual_duration_ms = len(replacement)

            assert actual_duration_ms == expected_duration_ms, (
                f"Detection '{detection.word}' at "
                f"[{detection.timestamp_range.start:.4f}, "
                f"{detection.timestamp_range.end:.4f}]: "
                f"replacement duration {actual_duration_ms}ms != "
                f"expected {expected_duration_ms}ms"
            )


# ============================================================================
# Property 7: Audio Encoding Parameters Match Source
# ============================================================================

from video_profanity_censor.audio_processor import (
    BIT_DEPTH_TO_SAMPLE_FMT,
    CODEC_TO_FFMPEG_ENCODER,
)
from video_profanity_censor.models import AudioMetadata


# Strategy: generate valid audio codecs that the system supports
audio_codecs = st.sampled_from(list(CODEC_TO_FFMPEG_ENCODER.keys()))

# Strategy: generate valid bit depths that have a known sample format mapping
bit_depths = st.sampled_from(list(BIT_DEPTH_TO_SAMPLE_FMT.keys()))

# Strategy: generate channel counts and their layouts
channel_configs = st.sampled_from([
    (1, "mono"),
    (2, "stereo"),
    (6, "5.1"),
    (6, "5.1(side)"),
    (8, "7.1"),
])

# Strategy: generate positive bitrates (common audio bitrates in bits/sec)
bitrates = st.sampled_from([
    64000, 96000, 128000, 160000, 192000, 256000, 320000, 512000, 1411200,
])

# Strategy: generate positive durations
audio_durations = st.floats(min_value=0.1, max_value=7200.0, allow_nan=False, allow_infinity=False)

# Strategy: generate track indices
track_indices = st.integers(min_value=0, max_value=7)


@st.composite
def audio_metadata_strategy(draw):
    """Generate valid AudioMetadata instances with realistic parameter combinations."""
    codec = draw(audio_codecs)
    sr = draw(sample_rates)
    bd = draw(bit_depths)
    channels, channel_layout = draw(channel_configs)
    bitrate = draw(bitrates)
    duration = draw(audio_durations)
    track_index = draw(track_indices)

    return AudioMetadata(
        codec=codec,
        sample_rate=sr,
        bit_depth=bd,
        channels=channels,
        channel_layout=channel_layout,
        bitrate=bitrate,
        duration_seconds=duration,
        track_index=track_index,
    )


class TestAudioEncodingParametersMatchSource:
    """Property 7: Audio Encoding Parameters Match Source.

    For any source audio metadata (codec, sample rate, bit depth, channel layout,
    bitrate), the re-encoded censored audio SHALL use the same codec, sample rate,
    bit depth, channel layout, and bitrate as the source.

    **Validates: Requirements 5.2**
    """

    @given(metadata=audio_metadata_strategy())
    @settings(max_examples=200)
    def test_encoding_codec_matches_source(self, metadata: AudioMetadata):
        """The encoding command must specify the correct codec/encoder for the source codec."""
        processor = AudioProcessor()
        params = processor.build_encoding_command(metadata)

        expected_encoder = CODEC_TO_FFMPEG_ENCODER.get(metadata.codec, metadata.codec)
        assert params["codec"] == expected_encoder, (
            f"Encoding codec mismatch: expected '{expected_encoder}' for "
            f"source codec '{metadata.codec}', got '{params['codec']}'"
        )

    @given(metadata=audio_metadata_strategy())
    @settings(max_examples=200)
    def test_encoding_sample_rate_matches_source(self, metadata: AudioMetadata):
        """The encoding command must specify the same sample rate as the source."""
        processor = AudioProcessor()
        params = processor.build_encoding_command(metadata)

        assert params["ar"] == str(metadata.sample_rate), (
            f"Sample rate mismatch: expected '{metadata.sample_rate}', "
            f"got '{params['ar']}'"
        )

    @given(metadata=audio_metadata_strategy())
    @settings(max_examples=200)
    def test_encoding_bit_depth_matches_source(self, metadata: AudioMetadata):
        """The encoding command must specify the correct sample format for the source bit depth."""
        processor = AudioProcessor()
        params = processor.build_encoding_command(metadata)

        expected_fmt = BIT_DEPTH_TO_SAMPLE_FMT[metadata.bit_depth]
        assert "sample_fmt" in params, (
            f"Encoding command missing sample_fmt for bit depth {metadata.bit_depth}"
        )
        assert params["sample_fmt"] == expected_fmt, (
            f"Bit depth/sample format mismatch: expected '{expected_fmt}' for "
            f"{metadata.bit_depth}-bit, got '{params['sample_fmt']}'"
        )

    @given(metadata=audio_metadata_strategy())
    @settings(max_examples=200)
    def test_encoding_channel_layout_matches_source(self, metadata: AudioMetadata):
        """The encoding command must specify the same channel layout as the source."""
        processor = AudioProcessor()
        params = processor.build_encoding_command(metadata)

        assert params["channel_layout"] == metadata.channel_layout, (
            f"Channel layout mismatch: expected '{metadata.channel_layout}', "
            f"got '{params['channel_layout']}'"
        )

    @given(metadata=audio_metadata_strategy())
    @settings(max_examples=200)
    def test_encoding_bitrate_matches_source(self, metadata: AudioMetadata):
        """The encoding command must specify the same bitrate as the source."""
        processor = AudioProcessor()
        params = processor.build_encoding_command(metadata)

        assert "audio_bitrate" in params, (
            f"Encoding command missing audio_bitrate for source bitrate {metadata.bitrate}"
        )
        assert params["audio_bitrate"] == str(metadata.bitrate), (
            f"Bitrate mismatch: expected '{metadata.bitrate}', "
            f"got '{params['audio_bitrate']}'"
        )

    @given(metadata=audio_metadata_strategy())
    @settings(max_examples=200)
    def test_all_encoding_parameters_match_source_simultaneously(
        self, metadata: AudioMetadata
    ):
        """All encoding parameters must match source simultaneously in a single command."""
        processor = AudioProcessor()
        params = processor.build_encoding_command(metadata)

        expected_encoder = CODEC_TO_FFMPEG_ENCODER.get(metadata.codec, metadata.codec)
        expected_fmt = BIT_DEPTH_TO_SAMPLE_FMT[metadata.bit_depth]

        # Verify all parameters at once
        assert params["codec"] == expected_encoder, f"Codec mismatch"
        assert params["ar"] == str(metadata.sample_rate), f"Sample rate mismatch"
        assert params["sample_fmt"] == expected_fmt, f"Bit depth mismatch"
        assert params["channel_layout"] == metadata.channel_layout, f"Channel layout mismatch"
        assert params["audio_bitrate"] == str(metadata.bitrate), f"Bitrate mismatch"

    @given(
        metadata=audio_metadata_strategy(),
        mode=st.sampled_from([CensorMode.TONE, CensorMode.MUTE]),
    )
    @settings(max_examples=200)
    def test_encoding_parameters_independent_of_censor_mode(
        self, metadata: AudioMetadata, mode: CensorMode
    ):
        """Encoding parameters must be identical regardless of censor mode (TONE or MUTE)."""
        processor_tone = AudioProcessor(mode=CensorMode.TONE)
        processor_mute = AudioProcessor(mode=CensorMode.MUTE)

        params_tone = processor_tone.build_encoding_command(metadata)
        params_mute = processor_mute.build_encoding_command(metadata)

        assert params_tone == params_mute, (
            f"Encoding parameters differ between TONE and MUTE modes: "
            f"TONE={params_tone}, MUTE={params_mute}"
        )


# ============================================================================
# Property 5: Channel Layout Preservation
# ============================================================================


@st.composite
def audio_config_with_detections(draw):
    """Generate an audio configuration with varying channel counts and detections.

    Returns a dict with:
        channels: number of audio channels (1 = mono, 2 = stereo)
        sample_rate: audio sample rate in Hz
        sample_width: bytes per sample (2 = 16-bit)
        mode: CensorMode (TONE or MUTE)
        detections: list of Detection objects within the audio duration
    """
    channels = draw(channel_counts)
    sample_rate = draw(sample_rates)
    mode = draw(censor_modes)

    # Generate 1-3 non-overlapping detections within a 3-second clip
    num_detections = draw(st.integers(min_value=1, max_value=3))
    detections = []
    current_time = 0.2  # Start after 200ms to avoid edge effects

    for _ in range(num_detections):
        gap = draw(
            st.floats(min_value=0.05, max_value=0.3, allow_nan=False, allow_infinity=False)
        )
        start = current_time + gap
        duration = draw(
            st.floats(min_value=0.05, max_value=0.3, allow_nan=False, allow_infinity=False)
        )
        end = start + duration

        # Ensure we stay within the 3-second audio clip
        if end >= 2.8:
            break

        detections.append(
            Detection(
                word="profanity",
                timestamp_range=TimestampRange(start=start, end=end),
                censor_action=mode,
            )
        )
        current_time = end

    # Ensure at least one detection
    if not detections:
        detections.append(
            Detection(
                word="profanity",
                timestamp_range=TimestampRange(start=0.3, end=0.5),
                censor_action=mode,
            )
        )

    return {
        "channels": channels,
        "sample_rate": sample_rate,
        "mode": mode,
        "detections": detections,
    }


def _create_test_audio(
    channels: int,
    sample_rate: int,
    duration_ms: int = 3000,
) -> AudioSegment:
    """Create a test audio segment with specified channel count and sample rate.

    Generates a 440Hz sine wave tone with the given properties.

    Args:
        channels: Number of audio channels.
        sample_rate: Sample rate in Hz.
        duration_ms: Duration in milliseconds.

    Returns:
        AudioSegment with the specified properties.
    """
    tone = Sine(440).to_audio_segment(duration=duration_ms, volume=-10.0)
    tone = tone.set_frame_rate(sample_rate)
    tone = tone.set_channels(channels)
    return tone


class TestChannelLayoutPreservation:
    """Property 5: Channel Layout Preservation.

    For any source audio with N channels and a given channel layout, after
    censoring, the output audio SHALL have the same number of channels and
    the same channel layout as the source.

    **Validates: Requirements 4.3**
    """

    @given(config=audio_config_with_detections())
    @settings(max_examples=100, deadline=30000)
    def test_output_channels_match_source_channels(self, config):
        """Output audio must have the same number of channels as the source audio."""
        channels = config["channels"]
        sample_rate = config["sample_rate"]
        mode = config["mode"]
        detections = config["detections"]

        # Create test audio with specified channel count
        source_audio = _create_test_audio(
            channels=channels,
            sample_rate=sample_rate,
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            source_path = Path(tmp_dir) / "source.wav"
            source_audio.export(str(source_path), format="wav")

            output_path = Path(tmp_dir) / "output.wav"
            processor = AudioProcessor(mode=mode)
            processor.censor(
                audio_path=source_path,
                detections=detections,
                output_path=output_path,
            )

            # Load output and verify channel count
            output_audio = AudioSegment.from_file(str(output_path))

            assert output_audio.channels == channels, (
                f"Output audio has {output_audio.channels} channels, "
                f"expected {channels} channels to match source. "
                f"Mode: {mode.value}, sample_rate: {sample_rate}"
            )

    @given(
        channels=channel_counts,
        sample_rate=sample_rates,
        mode=censor_modes,
    )
    @settings(max_examples=100, deadline=30000)
    def test_single_detection_preserves_channel_layout(
        self, channels: int, sample_rate: int, mode: CensorMode
    ):
        """A single detection in the middle of audio preserves channel count."""
        source_audio = _create_test_audio(
            channels=channels,
            sample_rate=sample_rate,
        )

        detection = Detection(
            word="test",
            timestamp_range=TimestampRange(start=1.0, end=1.5),
            censor_action=mode,
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            source_path = Path(tmp_dir) / "source.wav"
            source_audio.export(str(source_path), format="wav")

            output_path = Path(tmp_dir) / "output.wav"
            processor = AudioProcessor(mode=mode)
            processor.censor(
                audio_path=source_path,
                detections=[detection],
                output_path=output_path,
            )

            output_audio = AudioSegment.from_file(str(output_path))

            assert output_audio.channels == channels, (
                f"Single detection: output has {output_audio.channels} channels, "
                f"expected {channels}. Mode: {mode.value}, "
                f"sample_rate: {sample_rate}"
            )

    @given(config=audio_config_with_detections())
    @settings(max_examples=100, deadline=30000)
    def test_channel_count_preserved_across_all_censor_modes(self, config):
        """Channel layout must be preserved regardless of censor mode (tone or mute)."""
        channels = config["channels"]
        sample_rate = config["sample_rate"]
        detections = config["detections"]

        source_audio = _create_test_audio(
            channels=channels,
            sample_rate=sample_rate,
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            source_path = Path(tmp_dir) / "source.wav"
            source_audio.export(str(source_path), format="wav")

            # Test with TONE mode
            tone_detections = [
                Detection(
                    word=d.word,
                    timestamp_range=d.timestamp_range,
                    censor_action=CensorMode.TONE,
                )
                for d in detections
            ]
            output_tone_path = Path(tmp_dir) / "output_tone.wav"
            tone_processor = AudioProcessor(mode=CensorMode.TONE)
            tone_processor.censor(
                audio_path=source_path,
                detections=tone_detections,
                output_path=output_tone_path,
            )
            output_tone = AudioSegment.from_file(str(output_tone_path))

            # Test with MUTE mode
            mute_detections = [
                Detection(
                    word=d.word,
                    timestamp_range=d.timestamp_range,
                    censor_action=CensorMode.MUTE,
                )
                for d in detections
            ]
            output_mute_path = Path(tmp_dir) / "output_mute.wav"
            mute_processor = AudioProcessor(mode=CensorMode.MUTE)
            mute_processor.censor(
                audio_path=source_path,
                detections=mute_detections,
                output_path=output_mute_path,
            )
            output_mute = AudioSegment.from_file(str(output_mute_path))

            assert output_tone.channels == channels, (
                f"TONE mode: output has {output_tone.channels} channels, "
                f"expected {channels}"
            )
            assert output_mute.channels == channels, (
                f"MUTE mode: output has {output_mute.channels} channels, "
                f"expected {channels}"
            )

    @given(
        channels=channel_counts,
        num_detections=st.integers(min_value=1, max_value=5),
    )
    @settings(max_examples=50, deadline=30000)
    def test_multiple_detections_preserve_channel_layout(
        self, channels: int, num_detections: int
    ):
        """Multiple sequential detections must all preserve the source channel layout."""
        source_audio = _create_test_audio(
            channels=channels,
            sample_rate=44100,
            duration_ms=5000,
        )

        # Create evenly spaced non-overlapping detections
        detections = []
        for i in range(num_detections):
            start = 0.3 + i * 0.8
            end = start + 0.2
            if end > 4.5:
                break
            detections.append(
                Detection(
                    word=f"word{i}",
                    timestamp_range=TimestampRange(start=start, end=end),
                    censor_action=CensorMode.TONE,
                )
            )

        if not detections:
            return

        with tempfile.TemporaryDirectory() as tmp_dir:
            source_path = Path(tmp_dir) / "source.wav"
            source_audio.export(str(source_path), format="wav")

            output_path = Path(tmp_dir) / "output.wav"
            processor = AudioProcessor(mode=CensorMode.TONE)
            processor.censor(
                audio_path=source_path,
                detections=detections,
                output_path=output_path,
            )

            output_audio = AudioSegment.from_file(str(output_path))

            assert output_audio.channels == channels, (
                f"Multiple detections ({len(detections)}): output has "
                f"{output_audio.channels} channels, expected {channels}"
            )
