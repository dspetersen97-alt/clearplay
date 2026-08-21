"""Backend selector for hardware acceleration and model size selection."""

import logging
import subprocess
import sys

from video_profanity_censor.models import (
    AccelerationBackend,
    BackendDetectionResult,
    BackendInfo,
)

logger = logging.getLogger(__name__)


class BackendSelector:
    """Selects the optimal acceleration backend and Whisper model size.

    On Windows, probes for AMD GPU availability via DirectML (ONNX Runtime
    with DmlExecutionProvider). Falls back to CPU (faster-whisper/CTranslate2)
    when DirectML is unavailable or on non-Windows platforms.

    Model size selection:
    - GPU (VRAM-based): >= 8 GB → large, >= 4 GB → medium, < 4 GB → small
    - CPU (RAM-based):  >= 16 GB → large, < 16 GB → medium
    """

    RAM_THRESHOLD_LARGE_MODEL: int = 16 * 1024  # 16 GB in MB
    VRAM_THRESHOLD_LARGE: int = 8 * 1024  # 8 GB in MB
    VRAM_THRESHOLD_MEDIUM: int = 4 * 1024  # 4 GB in MB

    def detect(
        self,
        preferred_backend: AccelerationBackend | None = None,
    ) -> BackendDetectionResult:
        """Detects available backends and selects the optimal one.

        When preferred_backend is None, auto-detects: tries DirectML on Windows
        first, falls back to CPU. When a specific backend is requested, validates
        its availability and returns an error if unavailable.
        """
        warnings: list[str] = []
        available_backends: list[BackendInfo] = []
        gpu_vram_mb: int | None = None

        cpu_info = BackendInfo(
            backend=AccelerationBackend.CPU,
            is_available=True,
            device_name="CPU (CTranslate2/faster-whisper)",
        )
        available_backends.append(cpu_info)

        directml_info = self._detect_directml()
        if directml_info is not None:
            available_backends.append(directml_info)
            if directml_info.is_available:
                gpu_vram_mb = directml_info.vram_mb
                logger.info(
                    "DirectML available: %s (VRAM: %s MB)",
                    directml_info.device_name,
                    directml_info.vram_mb,
                )
            else:
                logger.info(
                    "DirectML unavailable: %s", directml_info.reason_unavailable
                )
        else:
            logger.info("DirectML detection skipped (non-Windows platform).")

        if preferred_backend == AccelerationBackend.CPU:
            selected = AccelerationBackend.CPU
        elif preferred_backend == AccelerationBackend.DIRECTML:
            if directml_info is not None and directml_info.is_available:
                selected = AccelerationBackend.DIRECTML
            else:
                reason = (
                    directml_info.reason_unavailable
                    if directml_info is not None
                    else "DirectML is only supported on Windows."
                )
                return BackendDetectionResult(
                    selected_backend=AccelerationBackend.CPU,
                    selected_model_size=self._select_model_size_cpu(self._query_system_ram()),
                    available_backends=available_backends,
                    gpu_vram_mb=gpu_vram_mb,
                    error_message=f"DirectML backend unavailable: {reason}",
                )
        else:
            # Auto-detect: prefer DirectML if available
            if directml_info is not None and directml_info.is_available:
                selected = AccelerationBackend.DIRECTML
            else:
                selected = AccelerationBackend.CPU

        if selected == AccelerationBackend.DIRECTML:
            model_size = self._select_model_size_gpu(gpu_vram_mb)
        else:
            system_ram_mb = self._query_system_ram()
            model_size = self._select_model_size_cpu(system_ram_mb)

        return BackendDetectionResult(
            selected_backend=selected,
            selected_model_size=model_size,
            available_backends=available_backends,
            gpu_vram_mb=gpu_vram_mb,
            detection_warnings=warnings,
        )

    def _detect_directml(self) -> BackendInfo | None:
        """Probes for DirectML availability on Windows.

        Returns None on non-Windows platforms. On Windows, checks for:
        1. A GPU detected via Win32_VideoController
        2. onnxruntime with DmlExecutionProvider available
        """
        if sys.platform != "win32":
            return None

        device_name, vram_mb = self._query_gpu_info()
        if device_name is None:
            return BackendInfo(
                backend=AccelerationBackend.DIRECTML,
                is_available=False,
                reason_unavailable="No GPU detected via Win32_VideoController.",
            )

        if not self._check_directml_provider():
            return BackendInfo(
                backend=AccelerationBackend.DIRECTML,
                is_available=False,
                device_name=device_name,
                vram_mb=vram_mb,
                reason_unavailable=(
                    "onnxruntime with DmlExecutionProvider is not available. "
                    "Install onnxruntime-directml."
                ),
            )

        return BackendInfo(
            backend=AccelerationBackend.DIRECTML,
            is_available=True,
            device_name=device_name,
            vram_mb=vram_mb,
        )

    def _query_gpu_info(self) -> tuple[str | None, int | None]:
        """Queries GPU name and VRAM on Windows via PowerShell.

        Returns:
            Tuple of (device_name, vram_mb). Both None if detection fails.
        """
        try:
            result = subprocess.run(
                [
                    "powershell",
                    "-NoProfile",
                    "-Command",
                    "(Get-CimInstance Win32_VideoController | "
                    "Select-Object -First 1).Name + '|' + "
                    "(Get-CimInstance Win32_VideoController | "
                    "Select-Object -First 1).AdapterRAM",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            stdout = result.stdout.strip()
            if "|" not in stdout:
                return None, None

            parts = stdout.split("|", 1)
            device_name = parts[0].strip()
            if not device_name:
                return None, None

            vram_mb = None
            vram_str = parts[1].strip()
            if vram_str and vram_str.isdigit():
                vram_mb = int(vram_str) // (1024 * 1024)

            return device_name, vram_mb
        except (subprocess.TimeoutExpired, OSError) as e:
            logger.warning(f"Failed to query GPU info: {e}")
            return None, None

    def _check_directml_provider(self) -> bool:
        """Checks whether onnxruntime has DmlExecutionProvider available."""
        try:
            import onnxruntime as ort
            providers = ort.get_available_providers()
            logger.info("ONNX Runtime providers: %s", providers)
            return "DmlExecutionProvider" in providers
        except ImportError:
            logger.info("onnxruntime is not installed.")
            return False

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
            try:
                import os
                mem_bytes = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
                return mem_bytes // (1024 * 1024)
            except (ValueError, OSError, AttributeError) as e:
                logger.warning(f"Failed to query system RAM: {e}")

        return None

    def _select_model_size_gpu(self, vram_mb: int | None) -> str:
        """Selects Whisper model size based on GPU VRAM.

        >= 8 GB VRAM → 'large'
        >= 4 GB VRAM → 'medium'
        < 4 GB VRAM → 'small'
        Unknown VRAM → 'medium'
        """
        if vram_mb is None:
            return "medium"
        if vram_mb >= self.VRAM_THRESHOLD_LARGE:
            return "large"
        if vram_mb >= self.VRAM_THRESHOLD_MEDIUM:
            return "medium"
        return "small"

    def _select_model_size_cpu(self, ram_mb: int | None) -> str:
        """Selects Whisper model size based on system RAM.

        >= 16 GB RAM → 'large'
        < 16 GB RAM → 'medium'
        Unknown RAM → 'medium'
        """
        if ram_mb is None:
            return "medium"
        if ram_mb >= self.RAM_THRESHOLD_LARGE_MODEL:
            return "large"
        return "medium"
