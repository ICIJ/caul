from dataclasses import astuple

from caul.model_handlers import ParakeetModelHandler
from caul.tasks.inference.parakeet_inference import ParakeetInferenceHandler
from caul.tasks.postprocessing.parakeet_postprocessor import ParakeetPostprocessor
from caul.tasks.preprocessing.parakeet_preprocessor import ParakeetPreprocessor
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


@patch.object(ParakeetInferenceHandler, "load", new=lambda _: None)
def test__handler_with_single_parakeet_model__np_array_input(inference_handler=None):
    """Test standalone Parakeet inference_handler"""
    model_handler = ParakeetModelHandler()

    model_handler.inference_handler.model = MockNvidiaASRInferenceHandler()

    handler = ASRHandler(models=model_handler)

    handler.startup()

    # load wav, drop channel dim
    audio = np.zeros([16000])
    transcriptions, scores = zip(*[astuple(r) for r in handler.transcribe(audio)])

    assert transcriptions == (
        [
            (
                PARAKEET_TEST_SEGMENT_START,
                PARAKEET_TEST_SEGMENT_END,
                PARAKEET_TEST_TRANSCRIPTION,
            )
        ],
    )
    assert scores == (PARAKEET_TEST_CONFIDENCE,)


def test__handler_with_single_whisper_model():
    """Test calling out to standalone whisper.cpp model"""
