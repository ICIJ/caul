import torch

import numpy as np

from src.caul.model import ASRModel


class WhisperCPPModel(ASRModel):

    def transcribe(self, audio: list[np.ndarray | torch.Tensor | str] | np.ndarray | torch.Tensor | str):
        pass