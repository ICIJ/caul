from dataclasses import dataclass

import torch

import numpy as np

from abc import ABC, abstractmethod


@dataclass
class ASRModelHandlerResult:
    """Base result class for ASR models"""

    transcription: list[tuple[float, str]] = None
    score: float = None


class ASRModelHandler(ABC):
    """ASRModelHandler abstract"""

    @abstractmethod
    def transcribe(
        self,
        audio: list[np.ndarray | torch.Tensor | str] | np.ndarray | torch.Tensor | str,
    ) -> list[ASRModelHandlerResult]:
        """

        :param audio: List of np.ndarray or torch.Tensor or str, or a singleton of same types
        :return: ASRModelHandlerResult
        """

    @abstractmethod
    def load(self):
        """Load model"""

    @abstractmethod
    def unload(self):
        """Unload model"""
