"""Unit tests for the BackendSelector component.

Tests cover model size selection based on system RAM.
"""

from unittest.mock import patch, MagicMock

import pytest

from video_profanity_censor.backend_selector import BackendSelector
from video_profanity_censor.models import AccelerationBackend


@pytest.fixture
def selector():
    return BackendSelector()


class TestModelSizeSelection:
    """Tests for RAM-based model size selection."""

    def test_ram_16gb_or_more_selects_large(self, selector):
        """System with >= 16 GB RAM should select large model."""
        with patch.object(selector, "_query_system_ram", return_value=16384):
            result = selector.detect()
        assert result.selected_model_size == "large"

    def test_ram_exactly_16gb_selects_large(self, selector):
        """System with exactly 16 GB RAM should select large model."""
        with patch.object(selector, "_query_system_ram", return_value=16 * 1024):
            result = selector.detect()
        assert result.selected_model_size == "large"

    def test_ram_below_16gb_selects_medium(self, selector):
        """System with < 16 GB RAM should select medium model."""
        with patch.object(selector, "_query_system_ram", return_value=8192):
            result = selector.detect()
        assert result.selected_model_size == "medium"

    def test_ram_query_failure_defaults_to_medium(self, selector):
        """When RAM query fails (returns None), should default to medium."""
        with patch.object(selector, "_query_system_ram", return_value=None):
            result = selector.detect()
        assert result.selected_model_size == "medium"

    def test_ram_32gb_selects_large(self, selector):
        """System with 32 GB RAM should select large model."""
        with patch.object(selector, "_query_system_ram", return_value=32768):
            result = selector.detect()
        assert result.selected_model_size == "large"


class TestBackendSelection:
    """Tests for backend selection (always CPU)."""

    def test_always_selects_cpu(self, selector):
        """Backend should always be CPU."""
        with patch.object(selector, "_query_system_ram", return_value=16384):
            result = selector.detect()
        assert result.selected_backend == AccelerationBackend.CPU

    def test_directml_preferred_gets_warning(self, selector):
        """Requesting DirectML should produce a warning and use CPU."""
        with patch.object(selector, "_query_system_ram", return_value=16384):
            result = selector.detect(preferred_backend=AccelerationBackend.DIRECTML)
        assert result.selected_backend == AccelerationBackend.CPU
        assert len(result.detection_warnings) > 0
        assert "not supported" in result.detection_warnings[0].lower()

    def test_cpu_preferred_works(self, selector):
        """Requesting CPU explicitly should work without warnings."""
        with patch.object(selector, "_query_system_ram", return_value=16384):
            result = selector.detect(preferred_backend=AccelerationBackend.CPU)
        assert result.selected_backend == AccelerationBackend.CPU

    def test_available_backends_only_cpu(self, selector):
        """Available backends should only contain CPU."""
        with patch.object(selector, "_query_system_ram", return_value=16384):
            result = selector.detect()
        assert len(result.available_backends) == 1
        assert result.available_backends[0].backend == AccelerationBackend.CPU
        assert result.available_backends[0].is_available is True
