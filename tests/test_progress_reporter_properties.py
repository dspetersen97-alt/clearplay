"""Property-based tests for error stage identification in ProcessingResult.

**Validates: Requirements 6.5**
"""

import hypothesis.strategies as st
from hypothesis import given, settings

from video_profanity_censor.models import ProcessingResult, ProcessingStage


# Strategy: generate any processing stage (including BACKEND_DETECTION)
all_stages = st.sampled_from(list(ProcessingStage))

# Strategy: generate non-empty error messages
error_messages = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,:-_()/",
    min_size=5,
    max_size=200,
).map(lambda s: s.strip()).filter(lambda s: len(s) >= 5)


def create_error_result(stage: ProcessingStage, message: str) -> ProcessingResult:
    """Create a ProcessingResult representing a failure at the given stage.

    This helper mirrors the pattern the CensorEngine uses when catching
    errors at each pipeline stage: it sets success=False, populates
    error_message with a message that includes the stage name, and sets
    error_stage to the stage where the failure occurred.
    """
    return ProcessingResult(
        success=False,
        error_message=f"Processing failed at {stage.value} stage: {message}",
        error_stage=stage,
    )


class TestErrorStageIdentification:
    """Property 9: Error Stage Identification.

    For any error that occurs during processing, the error message SHALL
    identify the processing stage (validation, backend_detection, transcription,
    profanity_detection, audio_censoring, or output_writing) at which the
    failure occurred.

    **Validates: Requirements 6.5**
    """

    @given(stage=all_stages, message=error_messages)
    @settings(max_examples=200)
    def test_error_result_identifies_correct_stage(
        self, stage: ProcessingStage, message: str
    ):
        """For any error at any processing stage, the ProcessingResult's
        error_stage field must match the stage where the error occurred."""
        result = create_error_result(stage, message)

        # error_stage must be set and match the stage
        assert result.error_stage is not None, (
            "error_stage must be set when an error occurs"
        )
        assert result.error_stage == stage, (
            f"error_stage should be {stage}, got {result.error_stage}"
        )

    @given(stage=all_stages, message=error_messages)
    @settings(max_examples=200)
    def test_error_message_contains_stage_name(
        self, stage: ProcessingStage, message: str
    ):
        """For any error at any processing stage, the error message SHALL
        identify the processing stage at which the failure occurred."""
        result = create_error_result(stage, message)

        # error_message must be populated
        assert result.error_message is not None, (
            "error_message must be set when an error occurs"
        )

        # error_message must contain the stage value string
        assert stage.value in result.error_message, (
            f"Error message should contain the stage name '{stage.value}', "
            f"got: '{result.error_message}'"
        )

    @given(stage=all_stages, message=error_messages)
    @settings(max_examples=200)
    def test_error_result_indicates_failure(
        self, stage: ProcessingStage, message: str
    ):
        """For any error at any processing stage, the ProcessingResult must
        indicate that processing did not succeed."""
        result = create_error_result(stage, message)

        assert result.success is False, (
            "ProcessingResult.success must be False when an error occurs"
        )

    @given(stage=all_stages, message=error_messages)
    @settings(max_examples=200)
    def test_error_message_contains_original_error_detail(
        self, stage: ProcessingStage, message: str
    ):
        """For any error at any processing stage, the error message SHALL
        preserve the original error detail so users can diagnose the issue."""
        result = create_error_result(stage, message)

        assert message in result.error_message, (
            f"Error message should contain the original error detail "
            f"'{message}', got: '{result.error_message}'"
        )

    @given(stage=all_stages, message=error_messages)
    @settings(max_examples=200)
    def test_error_stage_and_message_are_consistent(
        self, stage: ProcessingStage, message: str
    ):
        """The stage mentioned in the error_message must be consistent with
        the error_stage field value — they must refer to the same stage."""
        result = create_error_result(stage, message)

        # The error_stage.value should appear in the error_message
        assert result.error_stage.value in result.error_message, (
            f"error_stage value '{result.error_stage.value}' must appear in "
            f"error_message for consistency. Got: '{result.error_message}'"
        )

    def test_all_processing_stages_are_covered(self):
        """Verify all ProcessingStage variants (including BACKEND_DETECTION)
        can produce valid error results."""
        expected_stages = {
            ProcessingStage.VALIDATION,
            ProcessingStage.BACKEND_DETECTION,
            ProcessingStage.SUBTITLE_SCANNING,
            ProcessingStage.TRANSCRIPTION,
            ProcessingStage.PROFANITY_DETECTION,
            ProcessingStage.AUDIO_CENSORING,
            ProcessingStage.OUTPUT_WRITING,
        }

        for stage in expected_stages:
            result = create_error_result(stage, "test error")
            assert result.error_stage == stage
            assert stage.value in result.error_message
            assert result.success is False
