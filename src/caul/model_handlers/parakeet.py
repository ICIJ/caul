import torch

from caul.constant import DEVICE_CPU, PARAKEET_MODEL_REF
from caul.model_handlers.asr_handler import ASRModelHandler
from caul.tasks.inference.parakeet_inference import ParakeetInferenceHandler
from caul.tasks.postprocessing.parakeet_postprocessor import ParakeetPostprocessor
from caul.tasks.preprocessing.parakeet_preprocessor import ParakeetPreprocessor


class ParakeetModelHandler(ASRModelHandler):
    """Model handler for Parakeet family"""

    def __init__(
        self,
        model_name: str = PARAKEET_MODEL_REF,
        device: str | torch.device = DEVICE_CPU,
    ):
        super().__init__()

        self.model_name = model_name
        self.device = device

        self.preprocessor = ParakeetPreprocessor()
        self.inference_handler = ParakeetInferenceHandler(
            model_name=self.model_name, device=self.device
        )
        self.postprocessor = ParakeetPostprocessor()

        self.tasks = [self.preprocessor, self.inference_handler, self.postprocessor]

    def set_device(self, device: str | torch.device = DEVICE_CPU):
        """Set/change device here and on inference_handler"""
        self.device = device

        self.inference_handler.set_device(device)

    def startup(self):
        """Load model"""
        self.inference_handler.load()

    def shutdown(self):
        """Shut down"""
        self.preprocessor = None
        self.inference_handler = None
        self.postprocessor = None
        self.tasks = []
