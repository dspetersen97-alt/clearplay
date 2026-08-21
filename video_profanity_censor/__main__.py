"""Allow running the video profanity censor as a module.

Usage:
    python -m video_profanity_censor [arguments]
"""

import sys

from video_profanity_censor.cli import main

sys.exit(main())
