from abc import ABC


class ASRPostprocessor(ABC):
    """Abstract for ASR postprocessing task"""

    def process(self, inference_result: list) -> list:
        """Generic processing task

        :param inference_result: List of inputs
        :return: List of postprocessed outputs
        """
