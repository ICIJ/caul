import torch

import nemo.collections.asr as nemo_asr

from caul.constant import DEVICE_CPU
from caul.model_handlers.helpers import ParakeetModelHandlerResult
from caul.tasks.inference.asr_inference import (
    ASRInferenceHandler,
)
from caul.tasks.preprocessing.helpers import PreprocessedInput


class ParakeetInferenceHandler(ASRInferenceHandler):
    """Inference handler for NVIDIA's Parakeet family of ASR models. Supports up to 24 minutes of
    audio (batched or unbatched) in a single pass. Assumes that audio inputs (wav files or tensors)
    are single-channel with a sample rate of 16000—this last is very important for segmenting.
    """

    def __init__(
        self,
        model_name: str,
        device: str | torch.device = DEVICE_CPU,
        return_timestamps: bool = True,
    ):
        self.model_name = model_name

        if isinstance(device, str):
            device = torch.device(device)

        self.device = device
        self.return_timestamps = return_timestamps
        self.model = None

    def load(self):
        """Load model; default to CPU where no device is present"""
        device = self.device

        if device is None:
            device = DEVICE_CPU

        self.model = nemo_asr.models.ASRModel.from_pretrained(
            self.model_name, map_location=torch.device(device)
        ).eval()

    def unload(self):
        """Unload model"""
        self.model = None

    def set_device(self, device: str | torch.device = DEVICE_CPU):
        """Set/change device"""
        if isinstance(device, str):
            device = torch.device(device)

        self.device = device

        return self

    def process(
        self,
        inputs: list[list[PreprocessedInput]] | list[PreprocessedInput],
    ) -> list[ParakeetModelHandlerResult]:
        """Transcribe a batch of audio tensors or file names of max duration <= 20 minutes

        :param inputs: List of np.ndarray or torch.Tensor or str, or singleton of same types
        :return: List of results
        """
        if len(inputs) == 0:
            return []

        if isinstance(inputs[0], PreprocessedInput):
            inputs = [inputs]

        transcriptions = []

        for input_batch in inputs:
            hypotheses = self.model.transcribe(
                [i.tensor.to(self.device) for i in input_batch],
                timestamps=self.return_timestamps,
            )
            # Get timestamped segments if available, otherwise default to whole text
            for idx, hyps in enumerate(hypotheses):
                best_hyp = hyps

                if isinstance(best_hyp, list) or isinstance(best_hyp, tuple):
                    best_hyp = hyps[0]

                input_ordering_idx = input_batch[idx].metadata.input_ordering
                model_result = ParakeetModelHandlerResult(
                    input_ordering=input_ordering_idx
                ).parse_parakeet_hypothesis(best_hyp)
                transcriptions.append(model_result)

        return transcriptions
