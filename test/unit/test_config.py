import pytest

from caul.asr_pipeline import ASRPipeline, ASRPipelineConfig
from caul.tasks import (
    ParakeetInferenceRunner,
    ParakeetPostprocessor,
    ParakeetPreprocessor,
    WhisperCppInferenceRunner,
)

_PARAKEET = """
{
    "preprocessing": {
        "model": "parakeet"
    },
    "inference": {
        "model": "parakeet"
    },
    "postprocessing": {
        "model": "parakeet"
    }
}
"""

_PARAKEET_WITH_WHISPER_INFERENCE = """
{
    "preprocessing": {
        "model": "parakeet"
    },
    "inference": {
        "model": "whisper_cpp"
    },
    "postprocessing": {
        "model": "parakeet"
    }
}
"""


@pytest.mark.parametrize(
    ["pipeline_config_json", "expected_types"],
    [
        (
            _PARAKEET,
            [ParakeetPreprocessor, ParakeetInferenceRunner, ParakeetPostprocessor],
        ),
        (
            _PARAKEET_WITH_WHISPER_INFERENCE,
            [ParakeetPreprocessor, WhisperCppInferenceRunner, ParakeetPostprocessor],
        ),
    ],
)
def test_asr_pipeline_from_config(
    pipeline_config_json: str, expected_types: list[type]
) -> None:
    # Given
    pipeline_config = ASRPipelineConfig.model_validate_json(pipeline_config_json)
    # When
    pipeline = ASRPipeline.from_config(pipeline_config)

    # Then
    for component, expected_type in zip(pipeline.tasks, expected_types, strict=True):
        assert isinstance(component, expected_type)