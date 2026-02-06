from caul.tasks.inference.asr_inference import (
    ASRInferenceHandler,
    ASRModelHandlerResult,
)


class WhisperCPPInferenceHandler(ASRInferenceHandler):
    """Handler for WhisperCPP; wrapper round subprocess calls"""

    # pylint: disable=R0903

    def process(
        self,
        inputs: list[str],
    ) -> list[ASRModelHandlerResult]:
        """List of np.ndarray or torch.Tensor or str, or a singleton of same types

        :param inputs: List of np.ndarray or torch.Tensor or str, or a singleton of same types
        :return:
        """
