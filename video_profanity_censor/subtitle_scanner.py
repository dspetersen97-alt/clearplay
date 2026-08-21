"""Subtitle scanner component for pre-filtering profanity regions.

This module implements the SubtitleScanner which checks for embedded or external
subtitle tracks, scans them for profanity using the same matching rules as
ProfanityDetector, and identifies time regions for targeted speech recognition.

The scanner expands profanity cue timings by a configurable buffer (default ±2s)
and merges overlapping/adjacent regions into single continuous regions.
"""

import json
import re
import subprocess
from pathlib import Path

from nltk.stem import SnowballStemmer

from video_profanity_censor.models import (
    ProfanityList,
    ProfanityRegion,
    SubtitleCue,
    SubtitleScanResult,
    SubtitleTrackInfo,
)
from video_profanity_censor.profanity_list import load_profanity_list

# Shared stemmer instance (same as used in profanity_detector.py and profanity_list.py)
_stemmer = SnowballStemmer("english")

# Pattern to strip leading/trailing punctuation from words
_PUNCTUATION_PATTERN = re.compile(r"^[^\w]+|[^\w]+$")

# SRT timestamp pattern: HH:MM:SS,mmm --> HH:MM:SS,mmm
_SRT_TIMESTAMP_PATTERN = re.compile(
    r"(\d{1,2}):(\d{2}):(\d{2})[,.](\d{3})\s*-->\s*(\d{1,2}):(\d{2}):(\d{2})[,.](\d{3})"
)

# ASS/SSA Dialogue line pattern
_ASS_DIALOGUE_PATTERN = re.compile(
    r"^Dialogue:\s*\d+,(\d+):(\d{2}):(\d{2})\.(\d{2}),(\d+):(\d{2}):(\d{2})\.(\d{2}),",
    re.IGNORECASE,
)

# ASS/SSA style override tags to strip from text
_ASS_TAG_PATTERN = re.compile(r"\{[^}]*\}")


class SubtitleScanner:
    """Scans subtitles for profanity and identifies time regions for targeted transcription.

    The scanner uses the same matching rules as ProfanityDetector:
    1. Case-insensitive exact whole-word matching
    2. Morphological variant matching via NLTK Snowball stemming

    It avoids substring false positives by comparing only the entire word
    (stripped of punctuation) against list entries.
    """

    def __init__(
        self,
        profanity_list: ProfanityList | None = None,
        buffer_seconds: float = 2.0,
    ):
        """Initialize with profanity list and buffer duration for region expansion.

        Args:
            profanity_list: A pre-loaded ProfanityList. If None, loads the
                           default bundled profanity list.
            buffer_seconds: Buffer in seconds to expand each profanity cue timing
                          (applied before and after). Default is 2.0 seconds.
        """
        if profanity_list is not None:
            self._profanity_list = profanity_list
        else:
            self._profanity_list = load_profanity_list()
        self._buffer_seconds = buffer_seconds

    def scan(
        self,
        source_path: Path,
        external_subtitle_path: Path | None = None,
    ) -> SubtitleScanResult:
        """Scans subtitles for profanity and returns identified regions.

        Prefers external subtitle file over embedded tracks when specified (Req 8.8).
        If no subtitles are available, returns a result indicating no subtitles found.

        Args:
            source_path: Path to the source video file.
            external_subtitle_path: Optional path to an external subtitle file
                                   (SRT, ASS, or SSA format).

        Returns:
            A SubtitleScanResult with profanity regions, or indication that
            no subtitles are available.
        """
        cues: list[SubtitleCue] = []
        subtitle_source: str | None = None

        # Prefer external subtitle file over embedded tracks (Req 8.8)
        if external_subtitle_path is not None:
            cues = self._parse_subtitle_file(external_subtitle_path)
            subtitle_source = str(external_subtitle_path)
        else:
            # Probe for embedded subtitle tracks
            tracks = self._find_subtitle_tracks(source_path)
            if tracks:
                # Use the first available embedded track
                track = tracks[0]
                cues = self._extract_embedded_subtitles(source_path, track)
                subtitle_source = f"embedded:{track.track_index}"

        # No subtitles available
        if not cues:
            return SubtitleScanResult(
                has_subtitles=subtitle_source is not None,
                subtitle_source=subtitle_source,
                total_cues_scanned=0,
                profane_cues_found=0,
            )

        # Scan cues for profanity
        profane_cues: list[SubtitleCue] = []
        for cue in cues:
            if self._cue_contains_profanity(cue):
                profane_cues.append(cue)

        # If no profanity found, indicate we can skip speech recognition
        if not profane_cues:
            return SubtitleScanResult(
                has_subtitles=True,
                subtitle_source=subtitle_source,
                profanity_regions=[],
                total_cues_scanned=len(cues),
                profane_cues_found=0,
                skipped_speech_recognition=True,
            )

        # Expand and merge profanity regions
        regions = self._expand_and_merge_regions(profane_cues, self._buffer_seconds)

        return SubtitleScanResult(
            has_subtitles=True,
            subtitle_source=subtitle_source,
            profanity_regions=regions,
            total_cues_scanned=len(cues),
            profane_cues_found=len(profane_cues),
            skipped_speech_recognition=False,
        )

    def _find_subtitle_tracks(self, source_path: Path) -> list[SubtitleTrackInfo]:
        """Probes source file for embedded subtitle tracks using FFprobe.

        Args:
            source_path: Path to the video file to probe.

        Returns:
            List of SubtitleTrackInfo for each detected subtitle track.
        """
        try:
            result = subprocess.run(
                [
                    "ffprobe",
                    "-v", "quiet",
                    "-print_format", "json",
                    "-show_streams",
                    "-select_streams", "s",
                    str(source_path),
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode != 0:
                return []

            data = json.loads(result.stdout)
            streams = data.get("streams", [])

            tracks: list[SubtitleTrackInfo] = []
            for stream in streams:
                codec = stream.get("codec_name", "unknown")
                language = stream.get("tags", {}).get("language")
                track_index = stream.get("index", 0)

                tracks.append(
                    SubtitleTrackInfo(
                        track_index=track_index,
                        language=language,
                        codec=codec,
                        is_embedded=True,
                    )
                )

            return tracks

        except (subprocess.TimeoutExpired, subprocess.SubprocessError, json.JSONDecodeError):
            return []

    def _extract_embedded_subtitles(
        self, source_path: Path, track: SubtitleTrackInfo
    ) -> list[SubtitleCue]:
        """Extracts embedded subtitle track content using FFmpeg.

        Args:
            source_path: Path to the video file.
            track: The subtitle track info to extract.

        Returns:
            List of parsed SubtitleCue objects.
        """
        try:
            # Extract subtitle track to SRT format via FFmpeg
            result = subprocess.run(
                [
                    "ffmpeg",
                    "-v", "quiet",
                    "-i", str(source_path),
                    "-map", f"0:{track.track_index}",
                    "-f", "srt",
                    "-",
                ],
                capture_output=True,
                text=True,
                timeout=60,
            )

            if result.returncode != 0:
                return []

            return self._parse_srt_content(result.stdout)

        except (subprocess.TimeoutExpired, subprocess.SubprocessError):
            return []

    def _parse_subtitle_file(self, subtitle_path: Path) -> list[SubtitleCue]:
        """Parses SRT, ASS, or SSA subtitle file into cue objects.

        Detects format based on file extension and content.

        Args:
            subtitle_path: Path to the subtitle file.

        Returns:
            List of SubtitleCue objects parsed from the file.
        """
        if not subtitle_path.exists():
            return []

        try:
            content = subtitle_path.read_text(encoding="utf-8-sig")
        except (OSError, UnicodeDecodeError):
            try:
                content = subtitle_path.read_text(encoding="latin-1")
            except (OSError, UnicodeDecodeError):
                return []

        suffix = subtitle_path.suffix.lower()

        if suffix == ".srt":
            return self._parse_srt_content(content)
        elif suffix in (".ass", ".ssa"):
            return self._parse_ass_content(content)
        else:
            # Try SRT first, then ASS/SSA
            cues = self._parse_srt_content(content)
            if cues:
                return cues
            return self._parse_ass_content(content)

    def _parse_srt_content(self, content: str) -> list[SubtitleCue]:
        """Parses SRT format subtitle content.

        SRT format:
            1
            00:01:23,456 --> 00:01:25,789
            Subtitle text here

        Args:
            content: Raw SRT file content.

        Returns:
            List of SubtitleCue objects.
        """
        cues: list[SubtitleCue] = []
        blocks = re.split(r"\n\s*\n", content.strip())

        index = 1
        for block in blocks:
            lines = block.strip().splitlines()
            if len(lines) < 2:
                continue

            # Find timestamp line
            timestamp_line_idx = -1
            for i, line in enumerate(lines):
                if _SRT_TIMESTAMP_PATTERN.search(line):
                    timestamp_line_idx = i
                    break

            if timestamp_line_idx < 0:
                continue

            match = _SRT_TIMESTAMP_PATTERN.search(lines[timestamp_line_idx])
            if not match:
                continue

            start = self._srt_time_to_seconds(
                int(match.group(1)), int(match.group(2)),
                int(match.group(3)), int(match.group(4))
            )
            end = self._srt_time_to_seconds(
                int(match.group(5)), int(match.group(6)),
                int(match.group(7)), int(match.group(8))
            )

            # Text is everything after the timestamp line
            text_lines = lines[timestamp_line_idx + 1:]
            text = "\n".join(text_lines).strip()

            if text:
                cues.append(SubtitleCue(index=index, start=start, end=end, text=text))
                index += 1

        return cues

    def _parse_ass_content(self, content: str) -> list[SubtitleCue]:
        """Parses ASS/SSA format subtitle content.

        ASS format dialogue lines:
            Dialogue: 0,0:01:23.45,0:01:25.78,Default,,0,0,0,,Subtitle text here

        Args:
            content: Raw ASS/SSA file content.

        Returns:
            List of SubtitleCue objects.
        """
        cues: list[SubtitleCue] = []
        index = 1

        for line in content.splitlines():
            match = _ASS_DIALOGUE_PATTERN.match(line)
            if not match:
                continue

            start = self._ass_time_to_seconds(
                int(match.group(1)), int(match.group(2)),
                int(match.group(3)), int(match.group(4))
            )
            end = self._ass_time_to_seconds(
                int(match.group(5)), int(match.group(6)),
                int(match.group(7)), int(match.group(8))
            )

            # Extract text: everything after the 9th comma in the Dialogue line
            parts = line.split(",", 9)
            if len(parts) < 10:
                continue

            text = parts[9].strip()
            # Strip ASS style override tags like {\b1}, {\i0}, etc.
            text = _ASS_TAG_PATTERN.sub("", text)
            # Replace \N with newline (ASS line break)
            text = text.replace("\\N", "\n").replace("\\n", "\n")
            text = text.strip()

            if text:
                cues.append(SubtitleCue(index=index, start=start, end=end, text=text))
                index += 1

        return cues

    def _expand_and_merge_regions(
        self,
        profane_cues: list[SubtitleCue],
        buffer_seconds: float,
    ) -> list[ProfanityRegion]:
        """Expands cue timings by buffer and merges overlapping/adjacent regions.

        For each profane cue:
          - region start = max(0, cue.start - buffer_seconds)
          - region end = cue.end + buffer_seconds

        Then merge any overlapping or adjacent regions into single continuous regions.

        Args:
            profane_cues: List of subtitle cues containing profanity.
            buffer_seconds: Buffer in seconds to expand each cue.

        Returns:
            List of merged ProfanityRegion objects, sorted by start time.
        """
        if not profane_cues:
            return []

        # Create expanded regions for each cue
        expanded: list[tuple[float, float, SubtitleCue]] = []
        for cue in profane_cues:
            start = max(0.0, cue.start - buffer_seconds)
            end = cue.end + buffer_seconds
            expanded.append((start, end, cue))

        # Sort by start time
        expanded.sort(key=lambda x: x[0])

        # Merge overlapping/adjacent regions
        merged_regions: list[ProfanityRegion] = []
        current_start = expanded[0][0]
        current_end = expanded[0][1]
        current_cues: list[SubtitleCue] = [expanded[0][2]]

        for start, end, cue in expanded[1:]:
            if start <= current_end:
                # Overlapping or adjacent — merge
                current_end = max(current_end, end)
                current_cues.append(cue)
            else:
                # Non-overlapping — finalize current region and start new one
                merged_regions.append(
                    ProfanityRegion(
                        start=current_start,
                        end=current_end,
                        source_cues=current_cues,
                    )
                )
                current_start = start
                current_end = end
                current_cues = [cue]

        # Don't forget the last region
        merged_regions.append(
            ProfanityRegion(
                start=current_start,
                end=current_end,
                source_cues=current_cues,
            )
        )

        return merged_regions

    def _cue_contains_profanity(self, cue: SubtitleCue) -> bool:
        """Check if a subtitle cue contains any profane words.

        Uses the same matching rules as ProfanityDetector:
        - Case-insensitive whole-word matching
        - Morphological variant matching via stemming

        Args:
            cue: The subtitle cue to check.

        Returns:
            True if any word in the cue text is profane.
        """
        # Split text into words
        words = cue.text.split()
        for word in words:
            if self._is_profane(word):
                return True
        return False

    def _is_profane(self, word: str) -> bool:
        """Check if a word matches the profanity list.

        Uses identical logic to ProfanityDetector._is_profane():
        - Strip leading/trailing punctuation
        - Case-insensitive whole-word comparison
        - Morphological variant via stemming

        Args:
            word: The word to check.

        Returns:
            True if the word is profane, False otherwise.
        """
        # Strip leading/trailing punctuation and normalize to lowercase
        cleaned = _PUNCTUATION_PATTERN.sub("", word).lower()

        # Empty string after cleaning is not profane
        if not cleaned:
            return False

        # Check 1: Exact whole-word match (case-insensitive)
        if cleaned in self._profanity_list.words:
            return True

        # Check 2: Morphological variant via stemming
        word_stem = _stemmer.stem(cleaned)
        if word_stem in self._profanity_list.stems:
            return True

        return False

    @staticmethod
    def _srt_time_to_seconds(hours: int, minutes: int, seconds: int, millis: int) -> float:
        """Convert SRT timestamp components to seconds."""
        return hours * 3600 + minutes * 60 + seconds + millis / 1000.0

    @staticmethod
    def _ass_time_to_seconds(hours: int, minutes: int, seconds: int, centis: int) -> float:
        """Convert ASS/SSA timestamp components to seconds.

        ASS uses centiseconds (hundredths of a second) for the fractional part.
        """
        return hours * 3600 + minutes * 60 + seconds + centis / 100.0
