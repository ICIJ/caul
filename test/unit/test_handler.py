from dataclasses import astuple

from caul.model.postprocessing.parakeet_postprocessor import ParakeetPostprocessor
from caul.model.preprocessing.parakeet_preprocessor import ParakeetPreprocessor
from test.unit.constant import (
    parakeet_inference,
    PARAKEET_TEST_TRANSCRIPTION,
    PARAKEET_TEST_CONFIDENCE,
    PARAKEET_TEST_SEGMENT_START,
    PARAKEET_TEST_SEGMENT_END,
)
from test.unit.mock import MockNvidiaASRInferenceHandler

from unittest.mock import patch

import numpy as np

from caul.model.inference.parakeet_inference import ParakeetInferenceHandler
from caul import ASRHandler


@patch.object(ParakeetInferenceHandler, "load", new=lambda _: None)
def test__handler_with_single_PARAKEET_INFERENCE__np_array_input():
    """Test standalone Parakeet inference_handler"""
    preprocessor = ParakeetPreprocessor()
    inference_handler = ParakeetInferenceHandler(parakeet_inference)
    postprocessor = ParakeetPostprocessor()

    inference_handler.model = MockNvidiaASRInferenceHandler()

    handler = ASRHandler(
        preprocessor=preprocessor,
        inference_handler=inference_handler,
        postprocessor=postprocessor,
    )

    handler.startup()

    # load wav, drop channel dim
    audio = np.zeros([16000])
    transcriptions, scores = astuple(handler.transcribe(audio))

    assert transcriptions == [
        [
            (
                PARAKEET_TEST_SEGMENT_START,
                PARAKEET_TEST_SEGMENT_END,
                PARAKEET_TEST_TRANSCRIPTION,
            )
        ]
    ]
    assert scores == [PARAKEET_TEST_CONFIDENCE]


def test__handler_with_single_whisper_model():
    """Test calling out to standalone whisper.cpp model"""
