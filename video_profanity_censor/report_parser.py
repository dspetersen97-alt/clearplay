"""Parser for detection report files.

Reads the plain-text report written by ReportGenerator back into a list of
Detection objects, so a user can run the censor once, hand-edit the timings in
the generated ``*_report.txt``, and re-run using those edited timestamps instead
of transcribing again.

The parser is intentionally tolerant of hand edits:
- Header and separator lines are ignored.
- Extra whitespace between columns is fine.
- Timestamps accept ``HH:MM:SS.mmm``, ``MM:SS.mmm``, ``SS.mmm`` or a bare number
  of seconds (with ``.`` or ``,`` as the decimal separator).
- The action column maps ``muted``/``mute`` -> MUTE and ``tone_replaced``/``tone``
  -> TONE. Anything missing or unrecognized falls back to a supplied default.
- A word may be omitted; it defaults to a placeholder.

Rows whose timestamps cannot be parsed raise ReportParseError with the offending
line, rather than being silently dropped.
"""

import re
from pathlib import Path

from video_profanity_censor.models import CensorMode, Detection, TimestampRange


class ReportParseError(Exception):
    """Raised when a report file cannot be parsed into detections."""


# Lines that are structural (header/separators/summary) and carry no detection.
_SKIP_PREFIXES = (
    "profanity detection report",
    "total detections",
    "no profanity",
    "word ",  # the column header row "Word   Start   End   Action"
)

# One colon-separated component: digits with an optional [.,] fractional part.
# The number of components decides units, filled from the RIGHT (see _parse_timestamp):
#   1 component  -> SS[.fff]
#   2 components -> MM:SS[.fff]
#   3 components -> HH:MM:SS[.fff]
# Examples: 00:00:19.711, 1:02:03.5, 02:19.711, 19.711, 19
_TS_COMPONENT = re.compile(r"^\d+(?:[.,]\d+)?$")

# Map report action strings (and friendlier aliases) to CensorMode.
_ACTION_TO_MODE = {
    CensorMode.MUTE.value: CensorMode.MUTE,        # "muted"
    CensorMode.TONE.value: CensorMode.TONE,        # "tone_replaced"
    "mute": CensorMode.MUTE,
    "muted": CensorMode.MUTE,
    "tone": CensorMode.TONE,
    "tone_replaced": CensorMode.TONE,
    "beep": CensorMode.TONE,
}


def _parse_timestamp(token: str) -> float | None:
    """Parse a single timestamp token into seconds, or None if unparseable.

    Colon-separated components are interpreted from the RIGHT: the last is
    seconds (may have a fraction), the next is minutes, the next is hours. This
    makes ``02:19.711`` unambiguously 2 minutes 19.711 seconds, not 2 hours.
    """
    token = token.strip()
    if not token:
        return None

    parts = token.split(":")
    if len(parts) > 3:
        return None
    for p in parts:
        if not _TS_COMPONENT.match(p):
            return None

    # Only the seconds (rightmost) component may carry a fraction; normalize comma.
    for p in parts[:-1]:
        if "." in p or "," in p:
            return None

    seconds = float(parts[-1].replace(",", "."))
    minutes = int(parts[-2]) if len(parts) >= 2 else 0
    hours = int(parts[-3]) if len(parts) == 3 else 0

    return hours * 3600.0 + minutes * 60.0 + seconds


def _resolve_action(token: str | None, default_mode: CensorMode) -> CensorMode:
    """Map an action token to a CensorMode, falling back to default_mode."""
    if not token:
        return default_mode
    return _ACTION_TO_MODE.get(token.strip().lower(), default_mode)


def parse_report(
    file_path: Path,
    default_mode: CensorMode = CensorMode.MUTE,
) -> list[Detection]:
    """Parse a detection report file into a list of Detection objects.

    Args:
        file_path: Path to the report .txt file.
        default_mode: Censor mode to use for rows whose action column is missing
            or unrecognized.

    Returns:
        A list of Detection objects, in file order.

    Raises:
        ReportParseError: If the file cannot be read, contains no usable
            detections, or has a row whose timestamps cannot be parsed.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise ReportParseError(f"Report file not found: {file_path}")

    try:
        text = file_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as e:
        raise ReportParseError(f"Cannot read report file: {file_path} ({e})") from e

    detections: list[Detection] = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        # Skip structural lines (header, separators, summary).
        lowered = line.lower()
        if lowered.startswith(_SKIP_PREFIXES):
            continue
        if set(line) <= {"-", "=", " "}:  # separator rules like ----- or =====
            continue

        detection = _parse_row(line, default_mode)
        if detection is not None:
            detections.append(detection)

    if not detections:
        raise ReportParseError(
            f"No detections found in report file: {file_path}. Expected rows like "
            "'word   00:00:19.711   00:00:20.231   muted'."
        )

    return detections


def _parse_row(line: str, default_mode: CensorMode) -> Detection | None:
    """Parse one detection row. Returns None for lines that hold no timestamps.

    Raises ReportParseError if a line looks like a row but has bad timestamps.
    """
    tokens = line.split()

    # Find the first two tokens that parse as timestamps; those are start and end.
    # This tolerates the word column containing spaces is not supported, but the
    # word is a single column in the generated report, so this is safe.
    ts_indices = [i for i, t in enumerate(tokens) if _parse_timestamp(t) is not None]

    if len(ts_indices) < 2:
        # Not a detection row (e.g. stray prose). A row that clearly intends to be a
        # detection but has only one timestamp is a likely edit mistake — flag it.
        if len(ts_indices) == 1:
            raise ReportParseError(
                f"Row has only one valid timestamp (need start AND end): {line!r}"
            )
        return None

    start_idx, end_idx = ts_indices[0], ts_indices[1]
    start = _parse_timestamp(tokens[start_idx])
    end = _parse_timestamp(tokens[end_idx])

    if start is None or end is None:  # defensive; indices came from successful parses
        raise ReportParseError(f"Unparseable timestamps in row: {line!r}")

    if end <= start:
        raise ReportParseError(
            f"End timestamp must be after start in row: {line!r} "
            f"(start={start:.3f}s, end={end:.3f}s)"
        )

    # Word: everything before the start timestamp (usually a single token).
    word = " ".join(tokens[:start_idx]).strip() or "censored"

    # Action: the first non-timestamp token after the end timestamp, if any.
    action_token: str | None = None
    for t in tokens[end_idx + 1:]:
        if _parse_timestamp(t) is None:
            action_token = t
            break
    censor_action = _resolve_action(action_token, default_mode)

    return Detection(
        word=word,
        timestamp_range=TimestampRange(start=start, end=end),
        censor_action=censor_action,
    )
