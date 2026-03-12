from abc import ABC, abstractmethod

import numpy as np
import torch

from caul.configs.asr import ASRConfig
from caul.model_handlers.objects import ASRModelHandlerResult
from caul.tasks.asr_task import ASRTask


class ASRModelHandler(ABC):
    """ASR model handler abstract"""

    # pylint: disable=R0903

    def __init__(self, config: "ASRConfig", *args, **kwargs):
        self.config = config
        self.tasks: list[ASRTask] = []

    @abstractmethod
    def startup(self):
        """Generic method to load ASR resources"""

    @abstractmethod
    def shutdown(self):
        """Generic method to unload ASR resources"""

    def process(
        self,
        inputs: list[np.ndarray | torch.Tensor | str] | np.ndarray | torch.Tensor | str,
    ) -> list[ASRModelHandlerResult]:
        """Generic sequential processing method for ASR model handlers"""

        output = inputs

        for task in self.tasks:
            output = task.process(output)

        return output
