from test.unit.constant import PARAKEET_TEST_CONFIDENCE, PARAKEET_TEST_TRANSCRIPTION

import torch

import numpy as np


class MockNvidiaASRModelHandlerResult:
    # pylint: disable=C0115,C0116,R0903

    def __init__(self):
        self.text = PARAKEET_TEST_TRANSCRIPTION
        self.score = PARAKEET_TEST_CONFIDENCE


class MockNvidiaASRModelHandler:
    # pylint: disable=C0115,C0116,W0613,R0903

    def transcribe(
        self,
        audio: list[np.ndarray | torch.Tensor | str] | np.ndarray | torch.Tensor | str,
        timestamps: bool,
    ) -> list[MockNvidiaASRModelHandlerResult]:
        return [MockNvidiaASRModelHandlerResult()]
