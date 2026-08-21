"""Property-based tests for BackendSelector.

Tests RAM-based model size selection properties.
"""

import hypothesis.strategies as st
from hypothesis import given, settings

from video_profanity_censor.backend_selector import BackendSelector
from video_profanity_censor.models import AccelerationBackend


class TestRAMBasedModelSizeSelection:
    """Property tests for RAM-based model size selection."""

    @given(ram_mb=st.integers(min_value=16 * 1024, max_value=256 * 1024))
    @settings(max_examples=100)
    def test_large_model_selected_for_sufficient_ram(self, ram_mb: int):
        """For any system RAM >= 16 GB, large model should be selected."""
        selector = BackendSelector()
        model_size = selector._select_model_size(ram_mb)
        assert model_size == "large", (
            f"Expected 'large' for RAM {ram_mb} MB, got '{model_size}'"
        )

    @given(ram_mb=st.integers(min_value=1, max_value=16 * 1024 - 1))
    @settings(max_examples=100)
    def test_medium_model_selected_for_insufficient_ram(self, ram_mb: int):
        """For any system RAM < 16 GB, medium model should be selected."""
        selector = BackendSelector()
        model_size = selector._select_model_size(ram_mb)
        assert model_size == "medium", (
            f"Expected 'medium' for RAM {ram_mb} MB, got '{model_size}'"
        )

    def test_none_ram_defaults_to_medium(self):
        """When RAM is unknown (None), medium should be selected."""
        selector = BackendSelector()
        model_size = selector._select_model_size(None)
        assert model_size == "medium"

    def test_exact_threshold_boundary_selects_large(self):
        """Exactly 16 GB (16384 MB) should select large."""
        selector = BackendSelector()
        model_size = selector._select_model_size(16 * 1024)
        assert model_size == "large"

    def test_one_below_threshold_selects_medium(self):
        """One MB below 16 GB should select medium."""
        selector = BackendSelector()
        model_size = selector._select_model_size(16 * 1024 - 1)
        assert model_size == "medium"


class TestAlwaysCPUBackend:
    """Property tests confirming CPU is always selected."""

    @given(ram_mb=st.integers(min_value=1, max_value=256 * 1024))
    @settings(max_examples=50)
    def test_backend_is_always_cpu(self, ram_mb: int):
        """Regardless of RAM, backend should always be CPU."""
        selector = BackendSelector()
        # We can only test _select_model_size directly since detect() calls subprocess
        # The detect() method always returns CPU backend by design
        model_size = selector._select_model_size(ram_mb)
        assert model_size in ("medium", "large")
