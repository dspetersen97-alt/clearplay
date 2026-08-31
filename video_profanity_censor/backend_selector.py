"""Backend selector for model size selection based on system resources."""

import logging
import platform
import subprocess
import sys
from pathlib import Path

from video_profanity_censor.models import (
    AccelerationBackend,
    BackendDetectionResult,
    BackendInfo,
)

logger = logging.getLogger(__name__)


class BackendSelector:
    """Selects the optimal Whisper model size based on available system RAM.

    The application uses CPU inference exclusively via faster-whisper (CTranslate2).
    Model size is selected based on available system RAM:
    - >= 16 GB RAM → large model (best accuracy)
    - < 16 GB RAM → medium model (good balance of speed and accuracy)
    """

    RAM_THRESHOLD_LARGE_MODEL: int = 16 * 1024  # 16 GB in MB

    def detect(
        self,
        preferred_backend: AccelerationBackend | None = None,
    ) -> BackendDetectionResult:
        """Selects the CPU backend and determines optimal model size.

        Model size is based on available system RAM:
        - >= 16 GB → large
        - < 16 GB → medium

        The preferred_backend parameter is accepted for API compatibility
        but only CPU is supported.
        """
        warnings: list[str] = []

        # CPU is the only supported backend
        cpu_info = BackendInfo(
            backend=AccelerationBackend.CPU,
            is_available=True,
            device_name="CPU (CTranslate2/faster-whisper)",
        )
        available_backends = [cpu_info]

        # If user requested DirectML, warn them it's not supported
        if preferred_backend == AccelerationBackend.DIRECTML:
            warnings.append(
                "DirectML GPU acceleration is not supported for Whisper inference. "
                "Using CPU backend instead."
            )

        # Query system RAM for model size selection
        system_ram_mb = self._query_system_ram()
        model_size = self._select_model_size(system_ram_mb)

        return BackendDetectionResult(
            selected_backend=AccelerationBackend.CPU,
            selected_model_size=model_size,
            available_backends=available_backends,
            gpu_vram_mb=None,
            detection_warnings=warnings,
        )

    def _query_system_ram(self) -> int | None:
        """Queries total system RAM in MB.

        Returns:
            Total system RAM in MB, or None if detection fails.
        """
        if sys.platform == "win32":
            try:
                result = subprocess.run(
                    [
                        "powershell",
                        "-NoProfile",
                        "-Command",
                        "(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                stdout = result.stdout.strip()
                if stdout and stdout.isdigit():
                    ram_bytes = int(stdout)
                    return ram_bytes // (1024 * 1024)
            except (subprocess.TimeoutExpired, ValueError, OSError) as e:
                logger.warning(f"Failed to query system RAM: {e}")
        else:
            # Linux/macOS fallback
            try:
                import os
                mem_bytes = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")

                # Check for cgroup memory limits (e.g. Linux containers)
                cgroup_v2 = Path("/sys/fs/cgroup/memory.max")
                if cgroup_v2.exists():
                    try:
                        val = cgroup_v2.read_text().strip()
                        if val and val != "max" and val.isdigit():
                            mem_bytes = min(mem_bytes, int(val))
                    except (OSError, ValueError):
                        pass

                cgroup_v1 = Path("/sys/fs/cgroup/memory/memory.limit_in_bytes")
                if cgroup_v1.exists():
                    try:
                        val = cgroup_v1.read_text().strip()
                        if val and val.isdigit() and int(val) < 0x7FFFFFFFFFFFF000:
                            mem_bytes = min(mem_bytes, int(val))
                    except (OSError, ValueError):
                        pass

                return mem_bytes // (1024 * 1024)
            except (ValueError, OSError, AttributeError) as e:
                logger.warning(f"Failed to query system RAM: {e}")

        return None

    def _select_model_size(self, ram_mb: int | None) -> str:
        """Selects Whisper model size based on system RAM.

        >= 16 GB RAM → 'large' (best accuracy, ~4 GB memory usage)
        < 16 GB RAM → 'medium' (good accuracy, ~1.5 GB memory usage)
        Unknown RAM → 'medium' (safe default)
        """
        if ram_mb is None:
            return "medium"

        if ram_mb >= self.RAM_THRESHOLD_LARGE_MODEL:
            return "large"

        return "medium"
