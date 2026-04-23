from typing import ClassVar

from nemo.collections.asr.parts.mixins import TranscribeConfig
from nemo.collections.asr.parts.utils import Hypothesis
from pydantic import Field

from caul.constants import TorchDevice
from caul.tasks import (
    InferenceRunner,
    ParakeetInferenceRunner,
    ParakeetInferenceRunnerConfig,
)
from test.unit.constant import (
    PARAKEET_TEST_CONFIDENCE,
    TEST_WAV_TRANSCRIPTION,
    PARAKEET_TEST_SEGMENT_START,
    PARAKEET_TEST_SEGMENT_END,
)

import torch

import numpy as np


class MockNvidiaASRInferenceHandlerResult:
    # pylint: disable=C0115,C0116,R0903

    def __init__(self):
        self.timestep = {
            "segment": [
                {
                    "segment": TEST_WAV_TRANSCRIPTION,
                    "start": PARAKEET_TEST_SEGMENT_START,
                    "end": PARAKEET_TEST_SEGMENT_END,
                }
            ]
        }
        self.text = TEST_WAV_TRANSCRIPTION
        self.score = PARAKEET_TEST_CONFIDENCE


def mock_result() -> Hypothesis:
    timestamp = {
        "segment": [
            {
                "segment": TEST_WAV_TRANSCRIPTION,
                "start": PARAKEET_TEST_SEGMENT_START,
                "end": PARAKEET_TEST_SEGMENT_END,
            }
        ]
    }
    text = TEST_WAV_TRANSCRIPTION
    score = PARAKEET_TEST_CONFIDENCE
    return Hypothesis(timestamp=timestamp, text=text, score=score, y_sequence=[])


class MockParakeetModel:
    def transcribe(
        self,
        audio: list[np.ndarray | torch.Tensor | str] | np.ndarray | torch.Tensor | str,
        timestamps: bool,
        override_config: TranscribeConfig | None = None,
    ) -> tuple[list[Hypothesis]]:
        return ([mock_result()],)


class MockNvidiaASRInferenceRunnerConfig(ParakeetInferenceRunnerConfig):
    model: ClassVar[str] = Field(frozen=True, default="mock_nvidia")


@InferenceRunner.register("mock_nvidia")
class MockNvidiaASRInferenceRunner(ParakeetInferenceRunner):
    # pylint: disable=C0115,C0116,W0613,R0903
    def __init__(
        self,
        model_name: str,
        device: TorchDevice | torch.device = TorchDevice.CPU,
        return_timestamps: bool = True,
        batch_size: int = 4,
    ):
        super().__init__(model_name, device, return_timestamps, batch_size)

    def __enter__(self):
        self._model = MockParakeetModel()
