import pytest

from caul.configs.parakeet import ParakeetConfig
from caul.tasks.inference.parakeet_inference import ParakeetInferenceHandler
from test.unit import TEST_RESOURCES_PATH
from test.unit.constant import (
    PARAKEET_TEST_TRANSCRIPTION,
    PARAKEET_TEST_CONFIDENCE,
    PARAKEET_TEST_SEGMENT_START,
    PARAKEET_TEST_SEGMENT_END,
)
from test.unit.mock import MockNvidiaASRInferenceHandler

from unittest.mock import patch

import numpy as np

from caul.handler import ASRHandler


@pytest.mark.e2e
def test_parakeet_model_handler() -> None:
    # Given
    model_config = ParakeetConfig(save_to_filesystem=False, return_tensors=True)
    model_handler = model_config.handler_from_config()
    handler = ASRHandler(models=model_handler)
    audio_path = TEST_RESOURCES_PATH / "asr_test.wav"

    # When
    handler.startup()
    result = list(handler.transcribe(str(audio_path)))

    # Then
    assert len(result) == 1
    result = result[0]
    assert len(result.transcription) == 1
    transcript = result.transcription[0]
    assert transcript[2] == PARAKEET_TEST_TRANSCRIPTION


@patch.object(ParakeetInferenceHandler, "__enter__", new=lambda _: None)
def test__handler_with_single_parakeet_model__np_array_input(inference_handler=None):
    """Test standalone Parakeet inference_handler"""
    model_config = ParakeetConfig()

    model_config.save_to_filesystem = False

    model_handler = model_config.handler_from_config()

    model_handler.test_inference_handler.model = MockNvidiaASRInferenceHandler()

    handler = ASRHandler(models=model_handler)

    # load wav, drop channel dim
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
