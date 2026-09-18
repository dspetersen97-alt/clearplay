"""Tests for the detection report parser (edit-and-rerun workflow)."""

import pytest

from video_profanity_censor.models import CensorMode, Detection, TimestampRange
from video_profanity_censor.report_generator import ReportGenerator
from video_profanity_censor.report_parser import (
    ReportParseError,
    _parse_timestamp,
    parse_report,
)


def _write(tmp_path, content: str):
    p = tmp_path / "report.txt"
    p.write_text(content, encoding="utf-8")
    return p


class TestParseTimestamp:
    def test_hms(self):
        assert _parse_timestamp("00:00:19.711") == pytest.approx(19.711)

    def test_hms_with_hours(self):
        assert _parse_timestamp("1:02:03.5") == pytest.approx(3723.5)

    def test_mmss_is_minutes_not_hours(self):
        # Two components fill from the right: MM:SS, not HH:SS.
        assert _parse_timestamp("02:19.711") == pytest.approx(139.711)

    def test_bare_seconds(self):
        assert _parse_timestamp("19.711") == pytest.approx(19.711)
        assert _parse_timestamp("19") == pytest.approx(19.0)

    def test_comma_decimal(self):
        assert _parse_timestamp("10,5") == pytest.approx(10.5)

    def test_invalid(self):
        assert _parse_timestamp("abc") is None
        assert _parse_timestamp("1:2:3:4") is None
        assert _parse_timestamp("") is None


class TestRoundTrip:
    def test_generated_report_parses_back_to_same_detections(self, tmp_path):
        dets = [
            Detection("damn", TimestampRange(19.711, 20.231), CensorMode.MUTE),
            Detection("hell", TimestampRange(65.0, 65.4), CensorMode.TONE),
        ]
        report_path = tmp_path / "r_report.txt"
        ReportGenerator().generate(
            detections=dets, output_path=tmp_path / "r.mkv", report_path=report_path
        )

        parsed = parse_report(report_path)

        assert len(parsed) == 2
        assert parsed[0].word == "damn"
        assert parsed[0].timestamp_range.start == pytest.approx(19.711)
        assert parsed[0].timestamp_range.end == pytest.approx(20.231)
        assert parsed[0].censor_action == CensorMode.MUTE
        assert parsed[1].censor_action == CensorMode.TONE


class TestTolerantParsing:
    def test_skips_header_separators_and_summary(self, tmp_path):
        p = _write(tmp_path, (
            "Profanity Detection Report\n"
            "========================================\n"
            "Total detections: 1\n"
            "Word                 Start          End            Action\n"
            "-----------------------------------------------------------------\n"
            "damn                 00:00:01.000   00:00:01.500   muted\n"
        ))
        parsed = parse_report(p)
        assert len(parsed) == 1
        assert parsed[0].word == "damn"

    def test_loose_formats_and_default_action(self, tmp_path):
        p = _write(tmp_path, (
            "hell   02:19.711   02:20.231   tone\n"
            "oops   10,5   11,25\n"  # no action -> default
        ))
        parsed = parse_report(p, default_mode=CensorMode.MUTE)
        assert parsed[0].timestamp_range.start == pytest.approx(139.711)
        assert parsed[0].censor_action == CensorMode.TONE
        assert parsed[1].timestamp_range.start == pytest.approx(10.5)
        assert parsed[1].censor_action == CensorMode.MUTE  # default fallback

    def test_action_aliases(self, tmp_path):
        p = _write(tmp_path, (
            "a 1 2 mute\n"
            "b 3 4 beep\n"
            "c 5 6 tone_replaced\n"
            "d 7 8 muted\n"
        ))
        parsed = parse_report(p, default_mode=CensorMode.MUTE)
        assert [d.censor_action for d in parsed] == [
            CensorMode.MUTE, CensorMode.TONE, CensorMode.TONE, CensorMode.MUTE
        ]

    def test_extra_whitespace_tolerated(self, tmp_path):
        p = _write(tmp_path, "damn      00:00:01.000        00:00:01.500     muted\n")
        parsed = parse_report(p)
        assert len(parsed) == 1


class TestErrors:
    def test_missing_file(self, tmp_path):
        with pytest.raises(ReportParseError, match="not found"):
            parse_report(tmp_path / "nope.txt")

    def test_no_detections(self, tmp_path):
        p = _write(tmp_path, (
            "Profanity Detection Report\n"
            "========\n"
            "No profanity was found in the source file.\n"
        ))
        with pytest.raises(ReportParseError, match="No detections"):
            parse_report(p)

    def test_single_timestamp_row_errors(self, tmp_path):
        p = _write(tmp_path, "bad   00:00:19.711   muted\n")
        with pytest.raises(ReportParseError, match="one valid timestamp"):
            parse_report(p)

    def test_end_before_start_errors(self, tmp_path):
        p = _write(tmp_path, "bad   00:00:20.000   00:00:19.000   muted\n")
        with pytest.raises(ReportParseError, match="after start"):
            parse_report(p)
