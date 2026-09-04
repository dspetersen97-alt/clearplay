"""Regression tests for the "mute never un-mutes after first swear" bug.

Bug: when a detection carried a bogus end timestamp (e.g. faster-whisper anchored
the word's end to the segment/track end), the censor window was clamped to the full
file duration and MUTE mode silenced everything from the first detected word to the
end of the file. Only the tail should ever be silenced if a real detection lands
there — a single word must never mute an unbounded span.

These tests build a real audio file, run the actual censor() pipeline (which shells
out to FFmpeg), and assert that audio well past a detection is still audible.
"""

import tempfile
from pathlib import Path

from pydub import AudioSegment
from pydub.generators import Sine

from video_profanity_censor.audio_processor import AudioProcessor
from video_profanity_censor.models import (
    CensorMode,
    Detection,
    TimestampRange,
)


def _make_source(duration_ms: int, sample_rate: int = 44100, channels: int = 2) -> AudioSegment:
    """Create a steady, audible tone of the given length."""
    tone = Sine(440).to_audio_segment(duration=duration_ms, volume=-10.0)
    tone = tone.set_frame_rate(sample_rate)
    tone = tone.set_channels(channels)
    return tone


def _rms_between(audio: AudioSegment, start_ms: int, end_ms: int) -> int:
    """RMS of the audio slice in [start_ms, end_ms)."""
    return audio[start_ms:end_ms].rms


class TestMuteDoesNotRunToEndOfFile:
    """A single detection must not mute past its bounded window."""

    def test_bogus_end_at_file_duration_does_not_mute_tail(self):
        """A detection starting early with an end at the file's duration must NOT
        silence the rest of the file — the classic 'muted after first swear' bug.
        """
        duration_ms = 10_000  # 10s
        source = _make_source(duration_ms)

        # Detection begins at 1.0s but carries a bogus end at the very end of the
        # file (10.0s) — exactly the corrupt timestamp shape that caused the bug.
        detection = Detection(
            word="damn",
            timestamp_range=TimestampRange(start=1.0, end=10.0),
            censor_action=CensorMode.MUTE,
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            source_path = Path(tmp_dir) / "source.wav"
            output_path = Path(tmp_dir) / "output.wav"
            source.export(str(source_path), format="wav")

            processor = AudioProcessor(mode=CensorMode.MUTE)
            processor.censor(
                audio_path=source_path,
                detections=[detection],
                output_path=output_path,
            )

            out = AudioSegment.from_file(str(output_path))

            # Well past the detection start + cap (default 750ms), audio must remain
            # audible. If the bug were present, this region would be silent.
            tail_rms = _rms_between(out, 5_000, 9_000)
            assert tail_rms > 0, (
                "Audio after the detection window was silenced — the mute ran to the "
                f"end of the file (tail RMS={tail_rms})."
            )

    def test_single_detection_only_mutes_bounded_window(self):
        """The muted span must be bounded by max_word_duration_ms, not the raw
        (possibly bogus) detection duration."""
        duration_ms = 8_000
        source = _make_source(duration_ms)

        detection = Detection(
            word="hell",
            timestamp_range=TimestampRange(start=1.0, end=7.5),  # 6.5s raw span
            censor_action=CensorMode.MUTE,
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            source_path = Path(tmp_dir) / "source.wav"
            output_path = Path(tmp_dir) / "output.wav"
            source.export(str(source_path), format="wav")

            # Cap of 750ms; the window must not exceed start + cap (+ buffer).
            processor = AudioProcessor(mode=CensorMode.MUTE, max_word_duration_ms=750)
            processor.censor(
                audio_path=source_path,
                detections=[detection],
                output_path=output_path,
            )

            out = AudioSegment.from_file(str(output_path))

            # Region beyond start(1.0s) + cap(0.75s) + buffer(0.1s) ≈ 1.85s must be
            # audible. Check from 3.0s onward to leave generous margin.
            assert _rms_between(out, 3_000, 7_500) > 0, (
                "Audio beyond the capped window was silenced; the end cap did not bound "
                "the mute."
            )

    def test_moderately_long_detection_is_end_capped_not_repositioned(self):
        """A detection whose raw span is longer than the cap but under the drop
        threshold must be END-capped from its start — not anchored to its (possibly
        wrong) end. The muted region begins at the detection start and lasts only
        max_word_duration_ms.
        """
        duration_ms = 10_000
        source = _make_source(duration_ms)

        # 4s raw span (1.0..5.0): > 750ms cap but < 10x cap, so it is capped, not dropped.
        detection = Detection(
            word="damn",
            timestamp_range=TimestampRange(start=1.0, end=5.0),
            censor_action=CensorMode.MUTE,
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            source_path = Path(tmp_dir) / "source.wav"
            output_path = Path(tmp_dir) / "output.wav"
            source.export(str(source_path), format="wav")

            processor = AudioProcessor(mode=CensorMode.MUTE, max_word_duration_ms=750)
            result = processor.censor(
                audio_path=source_path,
                detections=[detection],
                output_path=output_path,
            )

            out = AudioSegment.from_file(str(output_path))

            # Only the cap's worth of audio is censored (not the full 4s raw span).
            assert result.total_censored_duration == 0.75

            # The window just after the start is muted...
            assert _rms_between(out, 1_100, 1_700) < source.rms * 0.5
            # ...but audio well before the (wrong) end at 5.0s is audible, proving the
            # mute was anchored to the START and capped, not run out to the raw end.
            assert _rms_between(out, 3_000, 4_800) > 0

    def test_implausibly_long_detection_is_dropped(self):
        """A detection whose raw duration is >10x the cap is treated as a corrupt
        timestamp and dropped, leaving the audio untouched."""
        duration_ms = 12_000
        source = _make_source(duration_ms)
        source_rms = source.rms

        # 9s raw span vs 750ms cap => 12x the cap => should be dropped entirely.
        detection = Detection(
            word="crap",
            timestamp_range=TimestampRange(start=1.0, end=10.0),
            censor_action=CensorMode.MUTE,
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            source_path = Path(tmp_dir) / "source.wav"
            output_path = Path(tmp_dir) / "output.wav"
            source.export(str(source_path), format="wav")

            processor = AudioProcessor(mode=CensorMode.MUTE, max_word_duration_ms=750)
            result = processor.censor(
                audio_path=source_path,
                detections=[detection],
                output_path=output_path,
            )

            out = AudioSegment.from_file(str(output_path))

            # Nothing should be muted anywhere: whole-file RMS stays close to source.
            assert out.rms >= source_rms * 0.9, (
                f"Audio was censored despite the detection being an implausible "
                f"timestamp that should be dropped (out RMS={out.rms}, "
                f"source RMS={source_rms})."
            )
            # total_censored_duration should be ~0 since the window was dropped.
            assert result.total_censored_duration == 0.0

    def test_valid_short_detection_still_mutes(self):
        """Sanity: a normal, valid detection is still muted correctly (the fix must
        not disable legitimate censoring)."""
        duration_ms = 6_000
        source = _make_source(duration_ms)

        detection = Detection(
            word="damn",
            timestamp_range=TimestampRange(start=2.0, end=2.4),
            censor_action=CensorMode.MUTE,
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            source_path = Path(tmp_dir) / "source.wav"
            output_path = Path(tmp_dir) / "output.wav"
            source.export(str(source_path), format="wav")

            processor = AudioProcessor(mode=CensorMode.MUTE)
            processor.censor(
                audio_path=source_path,
                detections=[detection],
                output_path=output_path,
            )

            out = AudioSegment.from_file(str(output_path))

            # The detection window (2.0-2.4s, +100ms buffer each side) should be quiet.
            muted_rms = _rms_between(out, 2_100, 2_300)
            # Audio outside the window stays audible.
            before_rms = _rms_between(out, 500, 1_500)
            after_rms = _rms_between(out, 4_000, 5_500)

            assert muted_rms < before_rms * 0.5, (
                f"Valid detection was not muted (window RMS={muted_rms}, "
                f"before RMS={before_rms})."
            )
            assert after_rms > 0, "Audio after a valid short mute was silenced."


class TestExportParamsUseProbedSampleRate:
    """Regression: the censored track must be re-encoded to the source codec at the
    sample rate of the WAV actually being encoded — never left as raw PCM ('araw'),
    and never re-encoded at a stale metadata rate that would shift pitch/speed.
    """

    def _md(self, codec="eac3", sample_rate=44100, channels=6):
        from video_profanity_censor.models import AudioMetadata

        return AudioMetadata(
            codec=codec,
            sample_rate=sample_rate,
            bit_depth=0,
            channels=channels,
            channel_layout="5.1(side)",
            bitrate=640000,
            duration_seconds=10.0,
            track_index=0,
        )

    def test_probed_rate_overrides_stale_metadata_rate(self):
        """When the probed WAV rate differs from the metadata rate, the export uses
        the probed rate (the audio actually being encoded)."""
        processor = AudioProcessor()
        params = processor._build_export_params(
            self._md(sample_rate=44100), actual_sample_rate=48000
        )
        assert "-ar" in params["parameters"]
        ar_value = params["parameters"][params["parameters"].index("-ar") + 1]
        assert ar_value == "48000", (
            f"Export used stale metadata rate instead of probed rate (got {ar_value})"
        )

    def test_falls_back_to_metadata_rate_when_no_probe(self):
        processor = AudioProcessor()
        params = processor._build_export_params(self._md(sample_rate=44100))
        ar_value = params["parameters"][params["parameters"].index("-ar") + 1]
        assert ar_value == "44100"

    def test_eac3_source_reencodes_to_eac3_not_raw(self):
        """An E-AC3 source must produce eac3 export params, not a bare WAV. This is
        what stops the output track from being raw PCM ('araw')."""
        processor = AudioProcessor()
        params = processor._build_export_params(self._md(codec="eac3"), actual_sample_rate=48000)
        assert params["format"] == "eac3"
        assert params["codec"] == "eac3"

    def test_none_metadata_still_exports_wav(self):
        """With no metadata (e.g. standalone use), the WAV fallback is preserved."""
        processor = AudioProcessor()
        params = processor._build_export_params(None, actual_sample_rate=48000)
        assert params == {"format": "wav"}
