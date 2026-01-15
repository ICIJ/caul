import torch

import numpy as np

from caul.inference.asr_inference import (
    ASRInferenceHandler,
    ASRInferenceHandlerResult,
)


class WhisperCPPInferenceHandler(ASRInferenceHandler):
    """Handler for WhisperCPP; wrapper round subprocess calls"""

    # pylint: disable=R0903

    def transcribe(
        self,
        inputs: list[np.ndarray | torch.Tensor | str] | np.ndarray | torch.Tensor | str,
    ) -> list[ASRInferenceHandlerResult]:
        """List of np.ndarray or torch.Tensor or str, or a singleton of same types

        :param inputs: List of np.ndarray or torch.Tensor or str, or a singleton of same types
        :return:
        """
