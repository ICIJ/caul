import torch

import nemo.collections.asr as nemo_asr
from nemo.collections.asr.parts.mixins import TranscribeConfig
from nemo.collections.asr.parts.mixins.transcription import InternalTranscribeConfig

from caul.constant import DEVICE_CPU
from caul.model_handlers.objects import ParakeetModelHandlerResult
from caul.tasks.inference.asr_inference import (
    ASRInferenceHandler,
)
from caul.tasks.preprocessing.objects import PreprocessedInput


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
        batch_size: int = 4,
    ):
        self.model_name = model_name

        if isinstance(device, str):
            device = torch.device(device)

        self.device = device
        self.return_timestamps = return_timestamps
        self.model = None
        self._batch_size = batch_size

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

        transcribe_config = TranscribeConfig(
            use_lhotse=False,
            batch_size=self._batch_size,
            timestamps=self.return_timestamps,
            return_hypotheses=True,
            _internal=InternalTranscribeConfig(device=self.device),
        )
        transcriptions = []
        for input_batch in inputs:
            if not input_batch:
                continue
            if input_batch[0].tensor is not None:
                audios = [i.tensor.to(self.device) for i in input_batch]
            else:
                audios = [i.metadata.preprocessed_file_path for i in input_batch]

            hypotheses = self.model.transcribe(
                audios, self.return_timestamps, override_config=transcribe_config
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
