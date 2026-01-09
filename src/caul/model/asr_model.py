import torch

import numpy as np


class ASRModelHandler:
    """ASRModelHandler abstract"""

    def transcribe(self, audio: list[np.ndarray | torch.Tensor | str] | np.ndarray | torch.Tensor | str):
        """

        :param audio: List of np.ndarray or torch.Tensor or str, or a singleton of same types
        :return:
        """
        pass

    def load(self):
        """Load model"""
        pass

    def unload(self):
        """Unload model"""
        pass