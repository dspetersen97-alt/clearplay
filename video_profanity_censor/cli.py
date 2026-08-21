"""Command-line interface for the Video Profanity Censor.

Provides a user-friendly CLI using argparse that wires command-line arguments
to CensorEngine.process() and displays progress and summary to stdout.
"""

import argparse
import sys
from pathlib import Path

from video_profanity_censor.censor_engine import CensorEngine
from video_profanity_censor.models import CensorMode


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
            "  %(prog)s movie.mp4 --backend directml --model-size large\n"
            "  %(prog)s movie.mp4 --subtitle-path subs.srt --profanity-list custom.txt\n"
        ),
    )

    parser.add_argument(
        "input",
        type=str,
        help="Path to the input video file (required)",
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


def main(argv: list[str] | None = None) -> int:
    """Main CLI entry point.

    Parses arguments, runs the censoring pipeline via CensorEngine,
    and displays progress and summary to stdout.

    Args:
        argv: Optional list of arguments. Defaults to sys.argv[1:].

    Returns:
        Exit code: 0 for success, 1 for failure.
    """
    args = parse_args(argv)

    # Resolve paths
    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else None
    profanity_list_path = Path(args.profanity_list) if args.profanity_list else None
    report_path = Path(args.report_path) if args.report_path else None
    subtitle_path = Path(args.subtitle_path) if args.subtitle_path else None

    # Resolve enums
    censor_mode = _resolve_censor_mode(args.mode)

    # Display startup info
    print(f"Video Profanity Censor")
    print(f"Input: {input_path}")
    if output_path:
        print(f"Output: {output_path}")
    if args.model_size:
        print(f"Model size: {args.model_size}")
    else:
        print("Model size: auto-select")
    print(f"Censor mode: {args.mode}")
    print()

    # Run the pipeline
    engine = CensorEngine()
    result = engine.process(
        input_path=input_path,
        output_path=output_path,
        audio_track=args.audio_track,
        censor_mode=censor_mode,
        profanity_list_path=profanity_list_path,
        report_path=report_path,
        subtitle_path=subtitle_path,
        disable_subtitle_prefilter=args.disable_subtitle_prefilter,
        model_size=args.model_size,
    )

    # Display summary
    print()
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
        return 0
    else:
        print("--- Processing Failed ---")
        if result.error_stage:
            print(f"Failed at stage: {result.error_stage.value}")
        if result.error_message:
            print(f"Error: {result.error_message}")
        if result.active_backend:
            print(f"Active backend: {result.active_backend.value}")
        print("-------------------------")
        return 1


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
