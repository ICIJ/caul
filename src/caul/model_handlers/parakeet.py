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

        self.model_name = model_name

        if config is not None and config.device is not None:
            device = config.device

        if isinstance(device, str):
            device = torch.device(device)

        self.device = device

        self.preprocessor = ParakeetPreprocessor(
            save_to_filesystem=config.save_to_filesystem,
            return_tensors=config.return_tensors,
        )
        self.inference_handler = ParakeetInferenceHandler(
            model_name=config.model_name, device=config.device
        )
        self.postprocessor = ParakeetPostprocessor()

        self.tasks = [self.preprocessor, self.inference_handler, self.postprocessor]

    def set_device(self, device: str | torch.device = DEVICE_CPU):
        """Set/change device here and on inference_handler

        :param device: device to use
        """
        if isinstance(device, str):
            device = torch.device(device)

        self.device = device

        self.inference_handler.set_device(device)

        return self

    def startup(self):
        """Load model"""
        self.inference_handler.load()

    def shutdown(self):
        """Shut down"""
        self.preprocessor = None
        self.inference_handler = None
        self.postprocessor = None
        self.tasks = []
