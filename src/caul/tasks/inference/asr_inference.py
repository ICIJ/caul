from abc import abstractmethod
from dataclasses import dataclass

from caul.tasks.asr_task import ASRTask


@dataclass
class ASRInferenceHandlerResult:
    """Base result class for ASR models"""

    transcription: list[tuple] = None
    score: float = None


class ASRInferenceHandler(ASRTask):
    """Abstract for ASR inference"""

    @abstractmethod
    def process(self, inputs: list, *args, **kwargs) -> list[ASRInferenceHandlerResult]:
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
