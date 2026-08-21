"""Unit tests for the BackendSelector component.

Tests cover DirectML GPU detection, VRAM-based model sizing, and CPU fallback.
"""

from unittest.mock import patch, MagicMock

import pytest

from video_profanity_censor.backend_selector import BackendSelector
from video_profanity_censor.models import AccelerationBackend


@pytest.fixture
def selector():
    return BackendSelector()


class TestCPUModelSizeSelection:
    """Tests for RAM-based model size selection (CPU path)."""

    def test_ram_16gb_or_more_selects_large(self, selector):
        with patch.object(selector, "_query_system_ram", return_value=16384):
            with patch.object(selector, "_detect_directml", return_value=None):
                result = selector.detect()
        assert result.selected_model_size == "large"

    def test_ram_exactly_16gb_selects_large(self, selector):
        with patch.object(selector, "_query_system_ram", return_value=16 * 1024):
            with patch.object(selector, "_detect_directml", return_value=None):
                result = selector.detect()
        assert result.selected_model_size == "large"

    def test_ram_below_16gb_selects_medium(self, selector):
        with patch.object(selector, "_query_system_ram", return_value=8192):
            with patch.object(selector, "_detect_directml", return_value=None):
                result = selector.detect()
        assert result.selected_model_size == "medium"

    def test_ram_query_failure_defaults_to_medium(self, selector):
        with patch.object(selector, "_query_system_ram", return_value=None):
            with patch.object(selector, "_detect_directml", return_value=None):
                result = selector.detect()
        assert result.selected_model_size == "medium"

    def test_ram_32gb_selects_large(self, selector):
        with patch.object(selector, "_query_system_ram", return_value=32768):
            with patch.object(selector, "_detect_directml", return_value=None):
                result = selector.detect()
        assert result.selected_model_size == "large"


class TestGPUModelSizeSelection:
    """Tests for VRAM-based model size selection (DirectML path)."""

    def _make_directml_info(self, vram_mb):
        from video_profanity_censor.models import BackendInfo
        return BackendInfo(
            backend=AccelerationBackend.DIRECTML,
            is_available=True,
            device_name="AMD Radeon RX 7900 XTX",
            vram_mb=vram_mb,
        )

    def test_vram_8gb_or_more_selects_large(self, selector):
        info = self._make_directml_info(8192)
        with patch.object(selector, "_detect_directml", return_value=info):
            result = selector.detect()
        assert result.selected_backend == AccelerationBackend.DIRECTML
        assert result.selected_model_size == "large"

    def test_vram_4gb_selects_medium(self, selector):
        info = self._make_directml_info(4096)
        with patch.object(selector, "_detect_directml", return_value=info):
            result = selector.detect()
        assert result.selected_backend == AccelerationBackend.DIRECTML
        assert result.selected_model_size == "medium"

    def test_vram_below_4gb_selects_small(self, selector):
        info = self._make_directml_info(2048)
        with patch.object(selector, "_detect_directml", return_value=info):
            result = selector.detect()
        assert result.selected_backend == AccelerationBackend.DIRECTML
        assert result.selected_model_size == "small"

    def test_vram_none_defaults_to_medium(self, selector):
        info = self._make_directml_info(None)
        with patch.object(selector, "_detect_directml", return_value=info):
            result = selector.detect()
        assert result.selected_model_size == "medium"

    def test_gpu_vram_reported_in_result(self, selector):
        info = self._make_directml_info(16384)
        with patch.object(selector, "_detect_directml", return_value=info):
            result = selector.detect()
        assert result.gpu_vram_mb == 16384


class TestBackendSelection:
    """Tests for backend auto-detection and explicit selection."""

    def _make_available_directml(self):
        from video_profanity_censor.models import BackendInfo
        return BackendInfo(
            backend=AccelerationBackend.DIRECTML,
            is_available=True,
            device_name="AMD Radeon RX 9060 XT",
            vram_mb=8192,
        )

    def _make_unavailable_directml(self, reason="No GPU detected."):
        from video_profanity_censor.models import BackendInfo
        return BackendInfo(
            backend=AccelerationBackend.DIRECTML,
            is_available=False,
            reason_unavailable=reason,
        )

    def test_auto_selects_directml_when_available(self, selector):
        info = self._make_available_directml()
        with patch.object(selector, "_detect_directml", return_value=info):
            result = selector.detect()
        assert result.selected_backend == AccelerationBackend.DIRECTML

    def test_auto_selects_cpu_when_directml_unavailable(self, selector):
        info = self._make_unavailable_directml()
        with patch.object(selector, "_detect_directml", return_value=info):
            with patch.object(selector, "_query_system_ram", return_value=16384):
                result = selector.detect()
        assert result.selected_backend == AccelerationBackend.CPU

    def test_auto_selects_cpu_on_non_windows(self, selector):
        with patch.object(selector, "_detect_directml", return_value=None):
            with patch.object(selector, "_query_system_ram", return_value=16384):
                result = selector.detect()
        assert result.selected_backend == AccelerationBackend.CPU

    def test_preferred_cpu_forces_cpu(self, selector):
        info = self._make_available_directml()
        with patch.object(selector, "_detect_directml", return_value=info):
            with patch.object(selector, "_query_system_ram", return_value=16384):
                result = selector.detect(preferred_backend=AccelerationBackend.CPU)
        assert result.selected_backend == AccelerationBackend.CPU

    def test_preferred_directml_when_available(self, selector):
        info = self._make_available_directml()
        with patch.object(selector, "_detect_directml", return_value=info):
            result = selector.detect(preferred_backend=AccelerationBackend.DIRECTML)
        assert result.selected_backend == AccelerationBackend.DIRECTML
        assert result.error_message is None

    def test_preferred_directml_when_unavailable_returns_error(self, selector):
        info = self._make_unavailable_directml("onnxruntime-directml not installed.")
        with patch.object(selector, "_detect_directml", return_value=info):
            with patch.object(selector, "_query_system_ram", return_value=16384):
                result = selector.detect(preferred_backend=AccelerationBackend.DIRECTML)
        assert result.selected_backend == AccelerationBackend.CPU
        assert result.error_message is not None
        assert "unavailable" in result.error_message.lower()

    def test_preferred_directml_on_non_windows_returns_error(self, selector):
        with patch.object(selector, "_detect_directml", return_value=None):
            with patch.object(selector, "_query_system_ram", return_value=16384):
                result = selector.detect(preferred_backend=AccelerationBackend.DIRECTML)
        assert result.selected_backend == AccelerationBackend.CPU
        assert result.error_message is not None
        assert "windows" in result.error_message.lower()

    def test_available_backends_includes_both_when_gpu_present(self, selector):
        info = self._make_available_directml()
        with patch.object(selector, "_detect_directml", return_value=info):
            result = selector.detect()
        backends = {b.backend for b in result.available_backends}
        assert AccelerationBackend.CPU in backends
        assert AccelerationBackend.DIRECTML in backends

    def test_available_backends_only_cpu_on_non_windows(self, selector):
        with patch.object(selector, "_detect_directml", return_value=None):
            with patch.object(selector, "_query_system_ram", return_value=16384):
                result = selector.detect()
        assert len(result.available_backends) == 1
        assert result.available_backends[0].backend == AccelerationBackend.CPU


class TestDirectMLDetection:
    """Tests for the _detect_directml method."""

    def test_returns_none_on_non_windows(self, selector):
        with patch("video_profanity_censor.backend_selector.sys") as mock_sys:
            mock_sys.platform = "linux"
            result = selector._detect_directml()
        assert result is None

    def test_returns_unavailable_when_no_gpu(self, selector):
        with patch("video_profanity_censor.backend_selector.sys") as mock_sys:
            mock_sys.platform = "win32"
            with patch.object(selector, "_query_gpu_info", return_value=(None, None)):
                result = selector._detect_directml()
        assert result is not None
        assert result.is_available is False
        assert "no gpu" in result.reason_unavailable.lower()

    def test_returns_unavailable_when_no_dml_provider(self, selector):
        with patch("video_profanity_censor.backend_selector.sys") as mock_sys:
            mock_sys.platform = "win32"
            with patch.object(selector, "_query_gpu_info", return_value=("AMD Radeon RX 7900 XTX", 24576)):
                with patch.object(selector, "_check_directml_provider", return_value=False):
                    result = selector._detect_directml()
        assert result is not None
        assert result.is_available is False
        assert "onnxruntime" in result.reason_unavailable.lower()

    def test_returns_available_with_gpu_and_provider(self, selector):
        with patch("video_profanity_censor.backend_selector.sys") as mock_sys:
            mock_sys.platform = "win32"
            with patch.object(selector, "_query_gpu_info", return_value=("AMD Radeon RX 7900 XTX", 24576)):
                with patch.object(selector, "_check_directml_provider", return_value=True):
                    result = selector._detect_directml()
        assert result is not None
        assert result.is_available is True
        assert result.device_name == "AMD Radeon RX 7900 XTX"
        assert result.vram_mb == 24576


class TestCheckDirectMLProvider:
    """Tests for the _check_directml_provider method."""

    def test_returns_true_when_dml_available(self, selector):
        mock_ort = MagicMock()
        mock_ort.get_available_providers.return_value = ["DmlExecutionProvider", "CPUExecutionProvider"]
        with patch.dict("sys.modules", {"onnxruntime": mock_ort}):
            assert selector._check_directml_provider() is True

    def test_returns_false_when_dml_not_in_providers(self, selector):
        mock_ort = MagicMock()
        mock_ort.get_available_providers.return_value = ["CPUExecutionProvider"]
        with patch.dict("sys.modules", {"onnxruntime": mock_ort}):
            assert selector._check_directml_provider() is False

    def test_returns_false_when_onnxruntime_not_installed(self, selector):
        with patch.dict("sys.modules", {"onnxruntime": None}):
            with patch("builtins.__import__", side_effect=ImportError("No module")):
                assert selector._check_directml_provider() is False
