from abc import ABC
from dataclasses import dataclass

import numpy as np
import torch

from caul.configs.asr import ASRConfig
from caul.tasks.asr_task import ASRTask


@dataclass
class ASRModelHandlerResult:
    """ASR model handler result"""

    transcriptions: list[dict]
    scores: list[float]


class ASRModelHandler(ABC):
    """ASR model handler abstract"""

    # pylint: disable=R0903

    def __init__(self, config: "ASRConfig", *args, **kwargs):
        self.config = config
        self.tasks: list[ASRTask] = []

    def process(
        self,
        inputs: list[np.ndarray | torch.Tensor | str] | np.ndarray | torch.Tensor | str,
    ) -> list[ASRModelHandlerResult]:
        """Generic sequential processing method for ASR model handlers"""

        output = inputs

        for task in self.tasks:
            output = task.process(output)

        return output
