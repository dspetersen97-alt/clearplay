"""Command-line interface for the Video Profanity Censor.

Provides a user-friendly CLI using argparse that wires command-line arguments
to CensorEngine.process() and displays progress and summary to stdout.
"""

import argparse
import sys
from pathlib import Path

from video_profanity_censor.censor_engine import CensorEngine
from video_profanity_censor.input_validator import InputValidator
from video_profanity_censor.models import CensorMode

# Name of the subfolder that batch (folder) mode writes censored output into.
FILTERED_DIR_NAME = "filtered"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Optional list of arguments. Defaults to sys.argv[1:].

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        prog="video-profanity-censor",
        description="Detect and censor profanity in video files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  %(prog)s movie.mp4\n"
            "  %(prog)s movie.mp4 --output censored.mp4 --mode mute\n"
            "  %(prog)s movie.mp4 --model-size large\n"
            "  %(prog)s movie.mp4 --subtitle-path subs.srt --profanity-list custom.txt\n"
            "  %(prog)s /home/daniel/Movies   # batch: process every video into Movies/filtered/\n"
        ),
    )

    parser.add_argument(
        "input",
        type=str,
        help=(
            "Path to the input video file, OR a folder. When a folder is given, every "
            "supported video in it is processed into a 'filtered' subfolder."
        ),
    )

    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Path for the output video file (default: input_censored.ext)",
    )

    parser.add_argument(
        "--audio-track",
        type=int,
        default=0,
        help="Index of the audio track to process (default: 0)",
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["tone", "mute"],
        default="mute",
        help="Censoring mode: 'tone' for beep replacement, 'mute' for silence (default: mute)",
    )

    parser.add_argument(
        "--profanity-list",
        type=str,
        default=None,
        help="Path to a custom profanity list file (one word per line)",
    )

    parser.add_argument(
        "--report-path",
        type=str,
        default=None,
        help="Path for the detection report (default: output_report.txt)",
    )

    parser.add_argument(
        "--subtitle-path",
        type=str,
        default=None,
        help="Path to an external subtitle file (SRT, ASS, SSA) for pre-filtering",
    )

    parser.add_argument(
        "--disable-subtitle-prefilter",
        action="store_true",
        default=False,
        help="Disable subtitle pre-filtering and use full audio transcription",
    )

    parser.add_argument(
        "--disable-subtitle-fallback",
        action="store_true",
        default=False,
        help=(
            "Disable the subtitle fallback. By default, profane words found in the "
            "subtitles but missed by audio transcription are still censored using the "
            "subtitle timing."
        ),
    )

    parser.add_argument(
        "--model-size",
        type=str,
        choices=["tiny", "base", "small", "medium", "large"],
        default=None,
        help="Whisper model size (default: auto-select based on system RAM)",
    )

    return parser.parse_args(argv)


def _resolve_censor_mode(mode_str: str) -> CensorMode:
    """Convert CLI mode string to CensorMode enum.

    Args:
        mode_str: The mode string from argparse ('tone' or 'mute').

    Returns:
        Corresponding CensorMode enum value.
    """
    if mode_str == "mute":
        return CensorMode.MUTE
    return CensorMode.TONE


def discover_videos(folder: Path) -> list[Path]:
    """Find supported video files directly inside a folder (non-recursive).

    The set of supported extensions is sourced from InputValidator so batch mode
    and single-file validation always agree. The output 'filtered' subfolder is
    skipped so re-running a batch never tries to re-censor its own output.

    Args:
        folder: Directory to scan.

    Returns:
        A sorted list of video file paths (may be empty).
    """
    supported = {ext.lower() for ext in InputValidator.SUPPORTED_FORMATS}
    videos: list[Path] = []
    for entry in folder.iterdir():
        if not entry.is_file():
            continue
        if entry.suffix.lower() in supported:
            videos.append(entry)
    return sorted(videos)


def _print_result(result) -> None:
    """Print a per-file processing summary block."""
    if result.success:
        print("--- Processing Summary ---")
        print(f"Elapsed time: {_format_duration(result.total_elapsed_seconds)}")
        print(f"Profane instances found: {result.profane_instances_detected}")
        print(f"Profane instances censored: {result.profane_instances_censored}")
        if result.active_backend:
            print(f"Active backend: {result.active_backend.value}")
        if result.model_size_used:
            print(f"Model size used: {result.model_size_used}")
        if result.output_path:
            print(f"Output: {result.output_path}")
        else:
            print("No profanity found — no output file created.")
        if result.report_path:
            print(f"Report: {result.report_path}")
        print("--------------------------")
    else:
        print("--- Processing Failed ---")
        if result.error_stage:
            print(f"Failed at stage: {result.error_stage.value}")
        if result.error_message:
            print(f"Error: {result.error_message}")
        if result.active_backend:
            print(f"Active backend: {result.active_backend.value}")
        print("-------------------------")


def _process_one_file(
    engine: CensorEngine,
    input_path: Path,
    output_path: Path | None,
    report_path: Path | None,
    subtitle_path: Path | None,
    args: argparse.Namespace,
    censor_mode: CensorMode,
):
    """Run the censoring pipeline for a single input file and return the result."""
    profanity_list_path = Path(args.profanity_list) if args.profanity_list else None
    return engine.process(
        input_path=input_path,
        output_path=output_path,
        audio_track=args.audio_track,
        censor_mode=censor_mode,
        profanity_list_path=profanity_list_path,
        report_path=report_path,
        subtitle_path=subtitle_path,
        disable_subtitle_prefilter=args.disable_subtitle_prefilter,
        subtitle_fallback=not args.disable_subtitle_fallback,
        model_size=args.model_size,
    )


def _run_single(args: argparse.Namespace, input_path: Path, censor_mode: CensorMode) -> int:
    """Handle a single-file input. Returns a process exit code."""
    output_path = Path(args.output) if args.output else None
    report_path = Path(args.report_path) if args.report_path else None
    subtitle_path = Path(args.subtitle_path) if args.subtitle_path else None

    print("Video Profanity Censor")
    print(f"Input: {input_path}")
    if output_path:
        print(f"Output: {output_path}")
    print(f"Model size: {args.model_size or 'auto-select'}")
    print(f"Censor mode: {args.mode}")
    print()

    engine = CensorEngine()
    result = _process_one_file(
        engine, input_path, output_path, report_path, subtitle_path, args, censor_mode
    )

    print()
    _print_result(result)
    return 0 if result.success else 1


def _run_batch(args: argparse.Namespace, folder: Path, censor_mode: CensorMode) -> int:
    """Handle a folder input: process every supported video into <folder>/filtered/.

    Continues past per-file failures so one bad file (e.g. an undecodable codec)
    does not abort the whole batch. Returns a process exit code: 0 when at least
    one file succeeded and none failed, 1 if any file failed or none were found.

    Args:
        args: Parsed CLI arguments.
        folder: The input directory.
        censor_mode: Resolved censor mode.

    Returns:
        Exit code (0 success, 1 failure/no files).
    """
    videos = discover_videos(folder)

    print("Video Profanity Censor (batch mode)")
    print(f"Input folder: {folder}")
    print(f"Model size: {args.model_size or 'auto-select'}")
    print(f"Censor mode: {args.mode}")

    if not videos:
        print(f"\nNo supported video files found in: {folder}")
        supported = ", ".join(sorted(InputValidator.SUPPORTED_FORMATS))
        print(f"Supported formats: {supported}")
        return 1

    filtered_dir = folder / FILTERED_DIR_NAME
    filtered_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output folder: {filtered_dir}")
    print(f"Found {len(videos)} video file(s) to process.\n")

    engine = CensorEngine()
    succeeded: list[str] = []
    failed: list[tuple[str, str]] = []

    for i, video in enumerate(videos, start=1):
        print("=" * 60)
        print(f"[{i}/{len(videos)}] {video.name}")
        print("=" * 60)

        # Output and report land in the filtered/ subfolder, keyed off the source name.
        output_path = filtered_dir / f"{video.stem}_censored{video.suffix}"
        report_path = filtered_dir / f"{video.stem}_censored_report.txt"

        try:
            result = _process_one_file(
                engine, video, output_path, report_path, None, args, censor_mode
            )
        except Exception as e:  # noqa: BLE001 - never let one file abort the batch
            print(f"Error: unexpected failure processing {video.name}: {e}")
            failed.append((video.name, str(e)))
            print()
            continue

        _print_result(result)
        print()
        if result.success:
            succeeded.append(video.name)
        else:
            failed.append((video.name, result.error_message or "unknown error"))

    # Batch summary
    print("=" * 60)
    print("Batch Summary")
    print("=" * 60)
    print(f"Total files: {len(videos)}")
    print(f"Succeeded:   {len(succeeded)}")
    print(f"Failed:      {len(failed)}")
    if failed:
        print("\nFailed files:")
        for name, reason in failed:
            print(f"  - {name}: {reason}")
    print(f"\nCensored output written to: {filtered_dir}")

    return 0 if (succeeded and not failed) else 1


def main(argv: list[str] | None = None) -> int:
    """Main CLI entry point.

    Parses arguments and routes to single-file or batch (folder) processing.

    Args:
        argv: Optional list of arguments. Defaults to sys.argv[1:].

    Returns:
        Exit code: 0 for success, 1 for failure.
    """
    args = parse_args(argv)

    input_path = Path(args.input)
    censor_mode = _resolve_censor_mode(args.mode)

    if not input_path.exists():
        print(f"Error: input path not found: {input_path}")
        return 1

    if input_path.is_dir():
        # Folder input => batch mode. Some single-file options are incompatible.
        if args.output:
            print(
                "Error: --output names a single file and cannot be used with a folder "
                "input. In batch mode, output goes to the 'filtered' subfolder "
                "automatically."
            )
            return 1
        if args.subtitle_path:
            print(
                "Error: --subtitle-path applies to a single video and cannot be used "
                "with a folder input. Embedded subtitles are still used per-file."
            )
            return 1
        if args.report_path:
            print(
                "Error: --report-path names a single file and cannot be used with a "
                "folder input. Per-file reports are written to the 'filtered' subfolder."
            )
            return 1
        return _run_batch(args, input_path, censor_mode)

    return _run_single(args, input_path, censor_mode)


def _format_duration(seconds: float) -> str:
    """Format a duration in seconds to a human-readable string.

    Args:
        seconds: Duration in seconds.

    Returns:
        Formatted string like "1m 30s" or "5.2s".
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    remaining = seconds % 60
    if minutes < 60:
        return f"{minutes}m {remaining:.0f}s"
    hours = minutes // 60
    remaining_minutes = minutes % 60
    return f"{hours}h {remaining_minutes}m"


if __name__ == "__main__":
    sys.exit(main())
