from dataclasses import astuple

from test.unit.constant import (
    PARAKEET_MODEL,
    PARAKEET_TEST_TRANSCRIPTION,
    PARAKEET_TEST_CONFIDENCE,
    PARAKEET_TEST_SEGMENT_START,
    PARAKEET_TEST_SEGMENT_END,
)
from test.unit.mock import MockNvidiaASRModelHandler

from unittest.mock import patch

import numpy as np

from caul.model import ParakeetModelHandler
from caul import ASRHandler


@patch.object(ParakeetModelHandler, "load", new=lambda _: None)
def test__handler_with_single_parakeet_model__np_array_input():
    """Test standalone Parakeet model"""
    model = ParakeetModelHandler(PARAKEET_MODEL)

    model.model = MockNvidiaASRModelHandler()

    handler = ASRHandler(models=model)

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
