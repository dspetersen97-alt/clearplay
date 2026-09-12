"""Allow running the video profanity censor as a module.

Usage:
    python -m video_profanity_censor [arguments]
"""

import sys

# --- TEMPORARY DIAGNOSTIC: dump all thread stacks if the process hangs ---
# Writes every running thread's stack trace to hang_trace.txt repeatedly,
# starting 30s in and then every 30s. When the job hangs at "transcription
# complete 100%", the most recent block in hang_trace.txt shows exactly where
# Python is stuck. Remove this block once the hang is diagnosed.
import faulthandler
import os

_hang_trace_path = os.path.join(os.getcwd(), "hang_trace.txt")
_hang_trace_file = open(_hang_trace_path, "w", buffering=1)
faulthandler.dump_traceback_later(
    30, repeat=True, file=_hang_trace_file, exit=False
)
# --- END TEMPORARY DIAGNOSTIC ---

from video_profanity_censor.cli import main

sys.exit(main())
