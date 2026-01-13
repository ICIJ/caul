import torch

import numpy as np

from src.caul.model.asr_model import ASRModelHandlerResult
from src.caul.model import ASRModelHandler


class WhisperCPPModelHandler(ASRModelHandler):
    """Handler for WhisperCPP; wrapper round subprocess calls"""

    # pylint: disable=R0903

    def transcribe(
        self,
        audio: list[np.ndarray | torch.Tensor | str] | np.ndarray | torch.Tensor | str,
    ) -> list[ASRModelHandlerResult]:
        """List of np.ndarray or torch.Tensor or str, or a singleton of same types

        :param audio: List of np.ndarray or torch.Tensor or str, or a singleton of same types
        :return:
        """
