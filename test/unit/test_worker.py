from test.unit.constant import (
    PARAKEET_MODEL,
    PARAKEET_TEST_TRANSCRIPTION,
    PARAKEET_TEST_CONFIDENCE,
)
from test.unit.mock import MockNvidiaASRModelHandler

from unittest.mock import patch

import numpy as np

from caul.model import ParakeetModelHandler
from caul import ASRWorker


@patch.object(ParakeetModelHandler, "load", new=lambda _: None)
def test__worker_with_single_parakeet_model__np_array_input():
    """Test standalone Parakeet model"""
    model = ParakeetModelHandler(PARAKEET_MODEL)

    model.model = MockNvidiaASRModelHandler()

    worker = ASRWorker(models=model)

    worker.startup()

    # load wav, drop channel dim
    audio = np.zeros([16000])
    transcription, score = worker.transcribe(audio)[0]

    score = round(score, 0)

    assert transcription == PARAKEET_TEST_TRANSCRIPTION
    assert score == PARAKEET_TEST_CONFIDENCE


def test__worker_with_single_whisper_model():
    """Test calling out to standalone whisper.cpp model"""
