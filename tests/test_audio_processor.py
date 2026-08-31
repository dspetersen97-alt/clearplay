"""Behavioral tests for AudioProcessor.censor().

censor() streams the audio through a single FFmpeg filtergraph instead of
decoding the whole track into an in-memory pydub AudioSegment, so its peak
memory is bounded and independent of the length and channel count of the track.

These tests pin the censoring behavior the in-memory implementation had rather
than byte-for-byte equality with its output, because the two cannot be
byte-identical by construction: the tone now comes from FFmpeg's own sine
generator and gain ramps (linear in amplitude, where pydub's fades were linear
in dB), and the timeline gate is evaluated once per ~1ms audio frame, so each
window boundary can land up to one frame later than the sample-exact slice
pydub produced. What is asserted instead is everything a caller can observe:
the output duration, channel count and sample rate; near-silent MUTE windows;
censor-tone windows at the configured frequency and level; audio outside the
windows untouched sample-for-sample; and identical CensorResult accounting,
window math and progress-callback sequence.

**Validates: Requirements 4.1, 4.2, 4.3, 4.4**
"""

from pathlib import Path

import pytest
from pydub import AudioSegment
from pydub.generators import Sine

from video_profanity_censor import audio_processor
from video_profanity_censor.audio_processor import AudioProcessor
from video_profanity_censor.models import (
    CensorMode,
    Detection,
    ProcessingStage,
    TimestampRange,
)


AUDIO_DURATION_MS = 5000
SAMPLE_RATE = 48000


def _write_audio(
    path: Path,
    channels: int = 2,
    duration_ms: int = AUDIO_DURATION_MS,
    sample_rate: int = SAMPLE_RATE,
) -> AudioSegment:
    """Writes a WAV file holding a steady 440Hz tone and returns it."""
    audio = Sine(440).to_audio_segment(duration=duration_ms, volume=-10.0)
    audio = audio.set_frame_rate(sample_rate).set_channels(channels)
    audio.export(str(path), format="wav")
    return audio


def _detection(
    start: float, end: float, mode: CensorMode = CensorMode.MUTE, word: str = "damn"
) -> Detection:
    """Builds a Detection covering the given range in seconds."""
    return Detection(
        word=word,
        timestamp_range=TimestampRange(start=start, end=end),
        censor_action=mode,
    )


def _censor(
    tmp_path: Path,
    detections: list[Detection],
    mode: CensorMode = CensorMode.MUTE,
    channels: int = 2,
    processor: AudioProcessor | None = None,
    duration_ms: int = AUDIO_DURATION_MS,
    sample_rate: int = SAMPLE_RATE,
    progress_callback=None,
):
    """Censors a freshly generated audio file.

    Returns:
        Tuple of (source AudioSegment, output AudioSegment, CensorResult).
    """
    source_path = tmp_path / "source.wav"
    output_path = tmp_path / "output.wav"
    source = _write_audio(
        source_path,
        channels=channels,
        duration_ms=duration_ms,
        sample_rate=sample_rate,
    )

    if processor is None:
        processor = AudioProcessor(mode=mode)

    result = processor.censor(
        audio_path=source_path,
        detections=detections,
        output_path=output_path,
        progress_callback=progress_callback,
    )

    return source, AudioSegment.from_file(str(output_path)), result


def _zero_crossings(segment: AudioSegment) -> int:
    """Counts sign changes in a segment, mixed down to one channel."""
    samples = segment.set_channels(1).get_array_of_samples()
    crossings = 0
    previous = 0
    for sample in samples:
        if sample == 0:
            continue
        if previous and (sample > 0) != (previous > 0):
            crossings += 1
        previous = sample
    return crossings


class TestOutputShapeIsPreserved:
    """The censored output keeps the shape of the source audio."""

    @pytest.mark.parametrize("mode", [CensorMode.MUTE, CensorMode.TONE])
    @pytest.mark.parametrize("channels", [1, 2, 6])
    def test_duration_channels_and_rate_match_the_source(
        self, tmp_path: Path, mode: CensorMode, channels: int
    ):
        """Output duration, channel count and sample rate match the source."""
        source, output, _ = _censor(
            tmp_path, [_detection(1.0, 1.4, mode)], mode=mode, channels=channels
        )

        assert len(output) == len(source)
        assert output.channels == source.channels
        assert output.frame_rate == source.frame_rate
        assert output.sample_width == source.sample_width

    def test_output_duration_is_preserved_at_every_sample_rate(self, tmp_path: Path):
        """The gate frame size adapts to the sample rate without changing length."""
        for sample_rate in (8000, 22050, 44100, 48000):
            source, output, _ = _censor(
                tmp_path,
                [_detection(1.0, 1.4)],
                mode=CensorMode.MUTE,
                sample_rate=sample_rate,
            )
            assert len(output) == len(source), f"at {sample_rate}Hz"
            assert output.frame_rate == sample_rate


class TestCensoredWindows:
    """Censored windows hold silence (MUTE) or the censor tone (TONE)."""

    def test_mute_window_is_silent(self, tmp_path: Path):
        """A MUTE window is silent, and the audio around it is not."""
        _, output, _ = _censor(tmp_path, [_detection(1.0, 1.4)], mode=CensorMode.MUTE)

        # 1.0s-1.4s plus the 100ms buffer on both sides -> 0.9s-1.5s
        assert output[920:1480].max == 0
        assert output[0:880].rms > 0
        assert output[1520:2000].rms > 0

    def test_mute_window_is_silent_in_every_channel(self, tmp_path: Path):
        """Every channel of a multichannel track is silenced."""
        _, output, _ = _censor(
            tmp_path, [_detection(1.0, 1.4)], mode=CensorMode.MUTE, channels=6
        )

        for index, channel in enumerate(output.split_to_mono()):
            assert channel[920:1480].max == 0, f"channel {index} not silenced"

    def test_tone_window_holds_the_censor_tone(self, tmp_path: Path):
        """A TONE window holds a sine wave at the configured frequency."""
        processor = AudioProcessor(mode=CensorMode.TONE, tone_freq=1000)
        _, output, _ = _censor(
            tmp_path, [_detection(1.0, 1.4, CensorMode.TONE)], processor=processor
        )

        # Sampled inside the window, clear of the 10ms boundary crossfades
        window = output[1000:1400]
        # A 1000Hz sine crosses zero twice per cycle
        assert _zero_crossings(window) == pytest.approx(2 * 1000 * 0.4, abs=4)

        # ...at the level the in-memory implementation used for its replacement
        reference = processor._generate_tone(
            duration_ms=400, sample_rate=SAMPLE_RATE, channels=output.channels
        )
        assert window.dBFS == pytest.approx(reference.dBFS, abs=1.0)

    def test_tone_frequency_follows_the_configured_frequency(self, tmp_path: Path):
        """The censor tone uses the processor's tone_freq."""
        processor = AudioProcessor(mode=CensorMode.TONE, tone_freq=500)
        _, output, _ = _censor(
            tmp_path, [_detection(1.0, 1.4, CensorMode.TONE)], processor=processor
        )

        assert _zero_crossings(output[1000:1400]) == pytest.approx(
            2 * 500 * 0.4, abs=4
        )

    def test_tone_is_mixed_into_every_channel(self, tmp_path: Path):
        """The censor tone is duplicated across all channels of the source."""
        _, output, _ = _censor(
            tmp_path,
            [_detection(1.0, 1.4, CensorMode.TONE)],
            mode=CensorMode.TONE,
            channels=6,
        )

        for index, channel in enumerate(output.split_to_mono()):
            assert channel[1000:1400].rms > 0, f"channel {index} has no tone"

    def test_tone_window_is_faded_in_and_out(self, tmp_path: Path):
        """The tone ramps up and down over the 10ms window boundaries."""
        _, output, _ = _censor(
            tmp_path,
            [_detection(1.0, 1.4, CensorMode.TONE)],
            mode=CensorMode.TONE,
        )

        # Window is 0.9s-1.5s: the first and last 10ms are quieter than the body
        assert output[900:905].rms < output[1000:1400].rms
        assert output[1495:1500].rms < output[1000:1400].rms


class TestUncensoredAudioIsUntouched:
    """Audio outside the censored windows survives unchanged."""

    @pytest.mark.parametrize("mode", [CensorMode.MUTE, CensorMode.TONE])
    def test_samples_outside_the_windows_are_unchanged(
        self, tmp_path: Path, mode: CensorMode
    ):
        """Samples before and after a censored window are identical."""
        source, output, _ = _censor(
            tmp_path, [_detection(1.0, 1.4, mode)], mode=mode
        )

        # The window covers 0.9s-1.5s; compare a millisecond clear of its edges
        assert source[0:880].raw_data == output[0:880].raw_data
        assert source[1520:5000].raw_data == output[1520:5000].raw_data

    def test_no_detections_copies_the_input_unchanged(self, tmp_path: Path):
        """With no detections the input is passed through byte for byte."""
        source_path = tmp_path / "source.wav"
        output_path = tmp_path / "output.wav"
        _write_audio(source_path)

        result = AudioProcessor(mode=CensorMode.MUTE).censor(
            audio_path=source_path, detections=[], output_path=output_path
        )

        assert output_path.read_bytes() == source_path.read_bytes()
        assert result.censored_audio_path == output_path
        assert result.segments_censored == 0
        assert result.total_censored_duration == 0.0

    def test_skipped_windows_leave_the_audio_unchanged(self, tmp_path: Path):
        """A detection entirely past the end of the audio censors nothing."""
        source, output, result = _censor(tmp_path, [_detection(6.0, 6.2)])

        assert source.raw_data == output.raw_data
        assert result.total_censored_duration == 0.0


class TestCensorResultAccounting:
    """CensorResult reports the same window math as the in-memory version."""

    def test_censor_buffer_is_included_in_the_censored_duration(self, tmp_path: Path):
        """The 100ms buffer on each side counts towards the censored duration."""
        _, output, result = _censor(tmp_path, [_detection(1.0, 1.4)])

        assert result.segments_censored == 1
        assert result.total_censored_duration == pytest.approx(0.6)
        assert output[920:1480].max == 0

    def test_long_detection_is_capped_and_trimmed_from_the_start(self, tmp_path: Path):
        """An over-long window is capped to max_word_duration_ms, keeping its end."""
        _, output, result = _censor(tmp_path, [_detection(0.5, 2.5)])

        # 0.4s-2.6s is 2.2s; capped to 750ms by trimming the start -> 1.85s-2.6s
        assert result.total_censored_duration == pytest.approx(0.75)
        assert output[0:1840].rms > 0
        assert output[1870:2590].max == 0

    def test_windows_are_clamped_to_the_audio_boundaries(self, tmp_path: Path):
        """Windows are clamped to the start and end of the track."""
        _, output, result = _censor(
            tmp_path, [_detection(0.0, 0.2), _detection(4.9, 5.2)]
        )

        # 0s-0.3s (start clamped) plus 4.8s-5.0s (end clamped)
        assert result.segments_censored == 2
        assert result.total_censored_duration == pytest.approx(0.3 + 0.2)
        assert output[0:290].max == 0
        assert output[4820:5000].max == 0

    def test_non_positive_windows_are_skipped_but_still_reported(self, tmp_path: Path):
        """A detection past the end of the audio is counted but censors nothing."""
        _, _, result = _censor(
            tmp_path, [_detection(1.0, 1.4), _detection(6.0, 6.2)]
        )

        assert result.segments_censored == 2
        assert result.total_censored_duration == pytest.approx(0.6)

    def test_overlapping_detections_are_counted_individually(self, tmp_path: Path):
        """Overlapping windows censor their union but are accounted separately."""
        _, output, result = _censor(
            tmp_path, [_detection(1.0, 1.4), _detection(1.3, 1.7)]
        )

        # Windows 0.9s-1.5s and 1.2s-1.8s: 0.6s each, union 0.9s
        assert result.segments_censored == 2
        assert result.total_censored_duration == pytest.approx(1.2)
        assert output[920:1780].max == 0
        assert output[1820:2000].rms > 0

    def test_censored_audio_path_is_the_requested_output_path(self, tmp_path: Path):
        """The result points at the requested output file."""
        source_path = tmp_path / "source.wav"
        output_path = tmp_path / "censored.wav"
        _write_audio(source_path)

        result = AudioProcessor(mode=CensorMode.MUTE).censor(
            audio_path=source_path,
            detections=[_detection(1.0, 1.4)],
            output_path=output_path,
        )

        assert result.censored_audio_path == output_path
        assert output_path.exists()


class TestProgressReporting:
    """Progress is reported once per censored segment."""

    def test_each_censored_segment_is_reported(self, tmp_path: Path):
        """One callback per censored segment, ending at 100%."""
        calls = []
        _censor(
            tmp_path,
            [_detection(1.0, 1.4), _detection(2.0, 2.4), _detection(3.0, 3.4)],
            progress_callback=lambda stage, percent, message: calls.append(
                (stage, percent, message)
            ),
        )

        assert [message for _, _, message in calls] == [
            "Censored segment 1/3",
            "Censored segment 2/3",
            "Censored segment 3/3",
        ]
        assert all(stage is ProcessingStage.AUDIO_CENSORING for stage, _, _ in calls)
        assert [percent for _, percent, _ in calls] == pytest.approx(
            [100 / 3, 200 / 3, 100.0]
        )

    def test_skipped_segments_are_not_reported(self, tmp_path: Path):
        """Detections that produce no window report no progress."""
        calls = []
        _censor(
            tmp_path,
            [_detection(1.0, 1.4), _detection(6.0, 6.2)],
            progress_callback=lambda stage, percent, message: calls.append(message),
        )

        assert calls == ["Censored segment 1/2"]


class TestFiltergraphDelivery:
    """The filtergraph reaches FFmpeg however long it is."""

    @pytest.mark.parametrize("mode", [CensorMode.MUTE, CensorMode.TONE])
    def test_long_filtergraph_is_passed_in_a_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: CensorMode
    ):
        """Filtergraphs too long for the command line are written to a file."""
        monkeypatch.setattr(
            audio_processor, "MAX_INLINE_FILTERGRAPH_CHARS", 1
        )

        source, output, result = _censor(
            tmp_path,
            [_detection(1.0, 1.4, mode), _detection(2.0, 2.4, mode)],
            mode=mode,
        )

        assert len(output) == len(source)
        assert result.total_censored_duration == pytest.approx(1.2)
        assert source[0:880].raw_data == output[0:880].raw_data
        if mode == CensorMode.MUTE:
            assert output[920:1480].max == 0
        else:
            assert output[1000:1400].rms > 0

    def test_many_detections_are_censored_in_one_pass(self, tmp_path: Path):
        """A detection count that outgrows the inline limit still censors."""
        detections = [
            _detection(0.5 + index * 0.75, 0.7 + index * 0.75) for index in range(60)
        ]

        source, output, result = _censor(
            tmp_path, detections, duration_ms=46000, sample_rate=8000
        )

        assert len(output) == len(source)
        assert result.segments_censored == 60
        assert result.total_censored_duration == pytest.approx(60 * 0.4)
        # Each window covers 0.4s starting 100ms before its detection
        assert output[420:680].max == 0
        assert output[44670:44930].max == 0
        assert output[700:1120].rms > 0


class TestErrorHandling:
    """Failures are surfaced to the caller."""

    def test_missing_input_raises_file_not_found(self, tmp_path: Path):
        """A missing input file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            AudioProcessor(mode=CensorMode.MUTE).censor(
                audio_path=tmp_path / "missing.wav",
                detections=[_detection(1.0, 1.4)],
                output_path=tmp_path / "output.wav",
            )

    def test_unencodable_output_raises_runtime_error(self, tmp_path: Path):
        """An FFmpeg failure is reported as a RuntimeError."""
        source_path = tmp_path / "source.wav"
        _write_audio(source_path)

        processor = AudioProcessor(mode=CensorMode.MUTE)
        with pytest.raises(RuntimeError, match="FFmpeg audio censoring failed"):
            processor.censor(
                audio_path=source_path,
                detections=[_detection(1.0, 1.4)],
                output_path=tmp_path / "missing_dir" / "output.wav",
            )
