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

    def __enter__(self):
        return self

    def __aexit__(self, exc_type, exc_val, exc_tb): ...
