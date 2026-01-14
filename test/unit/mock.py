from test.unit.constant import (
    PARAKEET_TEST_CONFIDENCE,
    PARAKEET_TEST_TRANSCRIPTION,
    PARAKEET_TEST_SEGMENT_START,
    PARAKEET_TEST_SEGMENT_END,
)

import torch

import numpy as np


class MockNvidiaASRInferenceHandlerResult:
    # pylint: disable=C0115,C0116,R0903

    def __init__(self):
        self.timestamp = {
            "segment": [
                {
                    "segment": PARAKEET_TEST_TRANSCRIPTION,
                    "start": PARAKEET_TEST_SEGMENT_START,
                    "end": PARAKEET_TEST_SEGMENT_END,
                }
            ]
        }
        self.text = PARAKEET_TEST_TRANSCRIPTION
        self.score = PARAKEET_TEST_CONFIDENCE


class MockNvidiaASRInferenceHandler:
    # pylint: disable=C0115,C0116,W0613,R0903

    def transcribe(
        self,
        audio: list[np.ndarray | torch.Tensor | str] | np.ndarray | torch.Tensor | str,
        timestamps: bool,
    ) -> list[MockNvidiaASRInferenceHandlerResult]:
        return [MockNvidiaASRInferenceHandlerResult()]
