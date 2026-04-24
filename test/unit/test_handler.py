from pathlib import Path

import pytest
import numpy as np

from caul.handler import ASRHandler

from caul.asr_pipeline import ASRPipeline
from caul.tasks import ParakeetPreprocessorConfig, ParakeetPostprocessorConfig
from caul.tasks.asr_task import Postprocessor, Preprocessor
from test.unit.constant import (
    TEST_MP4_TRANSCRIPTION,
    TEST_WAV_TRANSCRIPTION,
    PARAKEET_TEST_CONFIDENCE,
    PARAKEET_TEST_SEGMENT_START,
    PARAKEET_TEST_SEGMENT_END,
    TEST_MP4_PATH,
    TEST_WAV_PATH,
)
from test.unit.mock import (
    MockNvidiaASRInferenceRunner,
    MockNvidiaASRInferenceRunnerConfig,
)


@pytest.mark.e2e
@pytest.mark.parametrize(
    "audio_path,expected_transcription",
    [(TEST_MP4_PATH, TEST_MP4_TRANSCRIPTION), (TEST_WAV_PATH, TEST_WAV_TRANSCRIPTION)],
)
def test_parakeet_model_handler(audio_path: Path, expected_transcription: str) -> None:
    # Given
    asr_handler = ASRHandler(models=ASRPipeline.parakeet())

    # When
    with asr_handler:
        result = list(asr_handler.transcribe(str(audio_path)))

    # Then
    assert len(result) == 1
    result = result[0]
    assert len(result.transcription) == 1
    transcript = result.transcription[0]
    assert transcript[2] == expected_transcription


def test__handler_with_single_parakeet_model__np_array_input():
    """Test standalone Parakeet inference_handler"""
    mocked_tasks = [
        Preprocessor.from_config(ParakeetPreprocessorConfig()),
        MockNvidiaASRInferenceRunner.from_config(MockNvidiaASRInferenceRunnerConfig()),
        Postprocessor.from_config(ParakeetPostprocessorConfig()),
    ]
    mocked_pipeline = ASRPipeline(mocked_tasks)
    handler = ASRHandler(models=mocked_pipeline)

    # load wav, drop channel dim
    with handler:
        audio = np.zeros([16000])
        result = list(handler.transcribe(audio))

    assert len(result) == 1
    result = result[0]
    assert result.transcription == [
        (
            PARAKEET_TEST_SEGMENT_START,
            PARAKEET_TEST_SEGMENT_END,
            TEST_WAV_TRANSCRIPTION,
        )
    ]
    assert result.score == PARAKEET_TEST_CONFIDENCE
