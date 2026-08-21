"""Data models and enums for the Video Profanity Censor pipeline."""

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class AccelerationBackend(Enum):
    """Available hardware acceleration backends for speech recognition."""

    DIRECTML = "directml"  # AMD GPU on Windows via ONNX Runtime
    CPU = "cpu"  # Default fallback via CTranslate2/faster-whisper


@dataclass
class BackendInfo:
    """Information about a detected acceleration backend."""

    backend: AccelerationBackend
    is_available: bool
    device_name: str | None = None  # e.g., "AMD Radeon RX 9060 XT"
    vram_mb: int | None = None  # GPU VRAM in MB, None for CPU
    driver_version: str | None = None
    reason_unavailable: str | None = None  # Why this backend can't be used


@dataclass
class BackendDetectionResult:
    """Result of backend auto-detection."""

    selected_backend: AccelerationBackend
    selected_model_size: str  # Whisper model size based on VRAM
    available_backends: list[BackendInfo] = field(default_factory=list)
    gpu_vram_mb: int | None = None
    detection_warnings: list[str] = field(default_factory=list)
    error_message: str | None = None  # Set when user-specified backend is unavailable


class CensorMode(Enum):
    """Censoring mode for profane audio segments."""

    TONE = "tone_replaced"
    MUTE = "muted"


class ProcessingStage(Enum):
    """Stages of the video processing pipeline."""

    VALIDATION = "validation"
    BACKEND_DETECTION = "backend_detection"
    SUBTITLE_SCANNING = "subtitle_scanning"
    TRANSCRIPTION = "transcription"
    PROFANITY_DETECTION = "profanity_detection"
    AUDIO_CENSORING = "audio_censoring"
    OUTPUT_WRITING = "output_writing"


@dataclass
class SubtitleCue:
    """A single timed subtitle entry."""

    index: int
    start: float  # seconds
    end: float  # seconds
    text: str


@dataclass
class SubtitleTrackInfo:
    """Metadata about an embedded subtitle track."""

    track_index: int
    language: str | None
    codec: str  # subrip, ass, ssa
    is_embedded: bool


@dataclass
class ProfanityRegion:
    """A time region identified as containing potential profanity, with buffer applied."""

    start: float  # seconds (after buffer expansion, clamped to >= 0)
    end: float  # seconds (after buffer expansion)
    source_cues: list[SubtitleCue]  # the original cues that contributed to this region


@dataclass
class SubtitleScanResult:
    """Result of the subtitle pre-filtering scan."""

    has_subtitles: bool
    subtitle_source: str | None = None  # path or "embedded:track_index"
    profanity_regions: list[ProfanityRegion] = field(default_factory=list)
    total_cues_scanned: int = 0
    profane_cues_found: int = 0
    skipped_speech_recognition: bool = False


@dataclass
class AudioMetadata:
    """Metadata about an audio track."""

    codec: str
    sample_rate: int
    bit_depth: int
    channels: int
    channel_layout: str
    bitrate: int
    duration_seconds: float
    track_index: int


@dataclass
class ValidationResult:
    """Result of input file validation."""

    is_valid: bool
    error_message: str | None = None
    audio_metadata: AudioMetadata | None = None
    video_codec: str | None = None
    container_format: str | None = None
    file_size_bytes: int = 0


@dataclass
class TimestampRange:
    """A time range with millisecond precision."""

    start: float  # seconds with millisecond precision
    end: float


@dataclass
class TranscribedWord:
    """A single word from speech recognition with timing and confidence."""

    word: str
    start: float
    end: float
    confidence: float


@dataclass
class TranscriptionResult:
    """Result of speech-to-text transcription."""

    words: list[TranscribedWord] = field(default_factory=list)
    has_speech: bool = True
    warnings: list[str] = field(default_factory=list)
    skipped_ranges: list[TimestampRange] = field(default_factory=list)


@dataclass
class Detection:
    """A single profanity detection with timestamp and action."""

    word: str
    timestamp_range: TimestampRange
    censor_action: CensorMode


@dataclass
class DetectionResult:
    """Result of profanity detection scan."""

    detections: list[Detection] = field(default_factory=list)
    total_words_scanned: int = 0


@dataclass
class ProfanityList:
    """A profanity word list with pre-computed stems for matching."""

    words: set[str]
    stems: set[str]  # Pre-computed stems for morphological matching
    source_path: Path | None = None

    @property
    def is_valid(self) -> bool:
        return len(self.words) > 0


@dataclass
class AudioExtractionResult:
    """Result of audio extraction from a video file."""

    output_path: Path  # Path to the extracted WAV file
    audio_metadata: AudioMetadata  # Metadata of the original audio track


@dataclass
class CensorResult:
    """Result of audio censoring."""

    censored_audio_path: Path
    segments_censored: int
    total_censored_duration: float  # seconds


@dataclass
class AssemblyResult:
    """Result of output assembly (muxing censored audio with original video)."""

    output_path: Path
    container_format: str  # e.g., "mp4", "mkv"
    video_stream_copied: bool = True  # True if video was stream-copied (no re-encode)
    streams_preserved: int = 0  # Number of additional streams preserved
    error_message: str | None = None
    codec_fallback: str | None = None  # Set when source codec was unsupported and a fallback was used


@dataclass
class ProcessingResult:
    """Final result of the complete processing pipeline."""

    success: bool
    output_path: Path | None = None
    report_path: Path | None = None
    total_elapsed_seconds: float = 0.0
    profane_instances_detected: int = 0
    profane_instances_censored: int = 0
    active_backend: AccelerationBackend | None = None  # Backend used for transcription
    model_size_used: str | None = None  # Whisper model size used
    error_message: str | None = None
    error_stage: ProcessingStage | None = None


# Type alias for progress callbacks used throughout the pipeline
ProgressCallback = Callable[[ProcessingStage, float, str], None] | None
