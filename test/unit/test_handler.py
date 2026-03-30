import pytest
import numpy as np

from caul.handler import ASRHandler

from caul.asr_pipeline import ASRPipeline
from caul.tasks import ParakeetPreprocessorConfig
from caul.tasks.asr_task import Postprocessor, Preprocessor
from test.unit import TEST_RESOURCES_PATH
from test.unit.constant import (
    PARAKEET_TEST_TRANSCRIPTION,
    PARAKEET_TEST_CONFIDENCE,
    PARAKEET_TEST_SEGMENT_START,
    PARAKEET_TEST_SEGMENT_END,
)
from test.unit.mock import (
    MockNvidiaASRInferenceRunner,
    MockNvidiaASRInferenceRunnerConfig,
)


@pytest.mark.e2e
def test_parakeet_model_handler() -> None:
    # Given
    asr_handler = ASRHandler(models=ASRPipeline.parakeet())
    audio_path = TEST_RESOURCES_PATH / "asr_test.wav"

    # When
    with asr_handler:
        result = list(asr_handler.transcribe(str(audio_path)))

    # Then
    assert len(result) == 1
    result = result[0]
    assert len(result.transcription) == 1
    transcript = result.transcription[0]
    assert transcript[2] == PARAKEET_TEST_TRANSCRIPTION


def test__handler_with_single_parakeet_model__np_array_input():
    """Test standalone Parakeet inference_handler"""
    mocked_tasks = [
        Preprocessor.from_config(ParakeetPreprocessorConfig()),
        MockNvidiaASRInferenceRunner.from_config(MockNvidiaASRInferenceRunnerConfig()),
        Postprocessor.from_config(ParakeetPreprocessorConfig()),
    ]
    mocked_pipeline = ASRPipeline(mocked_tasks)
    handler = ASRHandler(models=mocked_pipeline)

    # load wav, drop channel dim
    with handler:
        audio = np.zeros([16000])
        result = handler.transcribe(audio)[0]

    assert result.transcription == [
        (
            PARAKEET_TEST_SEGMENT_START,
            PARAKEET_TEST_SEGMENT_END,
            PARAKEET_TEST_TRANSCRIPTION,
        )
    ]
    assert result.score == PARAKEET_TEST_CONFIDENCE


def test__handler_with_single_whisper_model():
    """Test calling out to standalone whisper.cpp model"""
