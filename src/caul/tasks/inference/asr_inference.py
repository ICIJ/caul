from abc import abstractmethod

from caul.model_handlers.asr_model_handler import ASRModelHandlerResult
from caul.tasks.asr_task import ASRTask


class ASRInferenceHandler(ASRTask):
    """Abstract for ASR inference"""

    @abstractmethod
    def process(self, inputs: list, *args, **kwargs) -> list[ASRModelHandlerResult]:
        """

        :param inputs: List of inference inputs
        :return: ASRModelHandlerResult
        """

    @abstractmethod
    def load(self):
        """Load model"""

    @abstractmethod
    def unload(self):
        """Unload model"""
