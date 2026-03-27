from contextlib import ExitStack
from typing import TYPE_CHECKING

import torch

from caul.constant import DEVICE_CPU, PARAKEET_MODEL_REF
from caul.model_handlers.asr_model_handler import ASRModelHandler
from caul.tasks.inference.parakeet_inference import ParakeetInferenceHandler
from caul.tasks.postprocessing.parakeet_postprocessor import ParakeetPostprocessor
from caul.tasks.preprocessing.parakeet_preprocessor import ParakeetPreprocessor

if TYPE_CHECKING:
    from caul.configs import ParakeetConfig


class ParakeetModelHandler(ASRModelHandler):
    """Model handler for Parakeet family"""

    def __init__(
        self,
        config: "ParakeetConfig" = None,
        model_name: str = PARAKEET_MODEL_REF,
        device: str | torch.device = DEVICE_CPU,
    ):
        super().__init__(config=config)

        if config is not None and config.model_name is not None:
            model_name = config.model_name

        self._model_name = model_name

        if config is not None and config.device is not None:
            device = config.device

        if isinstance(device, str):
            device = torch.device(device)

        self.device = device

        # TODO: load these from config
        #  self._preprocessor = ParakeetPreprocessor.from_config(config.preprocessor)
        #  etc...
        self._preprocessor = ParakeetPreprocessor(
            save_to_filesystem=config.save_to_filesystem,
            return_tensors=config.return_tensors,
        )
        self._inference_handler = ParakeetInferenceHandler(
            model_name=config.model_name,
            device=config.device,
            return_timestamps=config.return_timestamps,
        )
        self._postprocessor = ParakeetPostprocessor()
        self._exit_stack = ExitStack()
        self._tasks = [self._preprocessor, self._inference_handler, self._postprocessor]

    # Expose subcomponents for test only
    @property
    def test_inference_handler(self) -> ParakeetInferenceHandler:
        return self._inference_handler

    def set_device(self, device: str | torch.device = DEVICE_CPU):
        """Set/change device here and on inference_handler

        :param device: device to use
        """
        if isinstance(device, str):
            device = torch.device(device)

        self.device = device

        self._inference_handler.set_device(device)

        return self

    def __enter__(self):
        """Load model"""
        self._exit_stack.enter_context(self._inference_handler)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._exit_stack.__exit__(exc_type, exc_val, exc_tb)
