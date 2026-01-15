from abc import ABC

import torch
import numpy as np

from caul.constant import DEVICE_CPU


class ASRPreprocessor(ABC):
    """Abstract for ASR preprocessing task"""

    def __init__(self, device: str = DEVICE_CPU):
        self.device = device

    def process(
        self,
        audio: list[np.ndarray | torch.Tensor | str] | np.ndarray | torch.Tensor | str,
    ) -> list:
        """Generic processing task

        :param audio: List of np.ndarray or torch.Tensor or str, or singleton of same types
        :return: List of preprocessed inputs
        """
