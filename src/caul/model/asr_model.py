import torch

import numpy as np


class ASRModel:

    def transcribe(self, audio: list[np.ndarray | torch.Tensor | str] | np.ndarray | torch.Tensor | str):
        pass

    def load(self):
        pass

    def unload(self):
        pass