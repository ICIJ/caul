from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class ASRInferenceHandlerResult:
    """Base result class for ASR models"""

    transcription: list[tuple] = None
    score: float = None


class ASRInferenceHandler(ABC):
    """Abstract for ASR inference"""

    @abstractmethod
    def transcribe(
        self,
        inputs: list,
    ) -> list[ASRInferenceHandlerResult]:
        """

        :param inputs: List of inference inputs
        :return: ASRInferenceHandlerResult
        """

    @abstractmethod
    def load(self):
        """Load model"""

    @abstractmethod
    def unload(self):
        """Unload model"""
