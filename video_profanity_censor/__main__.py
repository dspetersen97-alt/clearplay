"""Allow running the video profanity censor as a module.

Usage:
    python -m video_profanity_censor [arguments]
"""

import sys

# --- TEMPORARY DIAGNOSTIC: dump all thread stacks if the process hangs ---
# Writes every running thread's stack trace to hang_trace.txt repeatedly,
# starting 30s in and then every 30s. When a job appears to hang, the most
# recent block in hang_trace.txt shows exactly where Python is stuck (e.g. a
# faster-whisper decode loop, an ffmpeg subprocess, etc.).
#
# This is a debugging aid, not a shipping feature. Remove this block (and the
# faulthandler/os imports) once the hang is confirmed fixed. Controlled by an
# env var so it is easy to leave in place but off by default:
#   set CLEARPLAY_HANG_TRACE=1   (PowerShell: $env:CLEARPLAY_HANG_TRACE=1)
import os

if os.environ.get("CLEARPLAY_HANG_TRACE") == "1":
    import faulthandler

    _hang_trace_path = os.path.join(os.getcwd(), "hang_trace.txt")
    _hang_trace_file = open(_hang_trace_path, "w", buffering=1)
    faulthandler.dump_traceback_later(
        30, repeat=True, file=_hang_trace_file, exit=False
    )
# --- END TEMPORARY DIAGNOSTIC ---

from video_profanity_censor.cli import main

sys.exit(main())
