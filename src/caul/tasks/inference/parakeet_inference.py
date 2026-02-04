import torch

import nemo.collections.asr as nemo_asr

from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis

from caul.constant import DEVICE_CPU
from caul.tasks.inference.asr_inference import (
    ASRInferenceHandlerResult,
    ASRInferenceHandler,
)


class ParakeetInferenceHandlerResult(ASRInferenceHandlerResult):
    """Result handler for ParakeetInferenceHandler objects"""

    def parse_parakeet_hypothesis(
        self, hypothesis: Hypothesis
    ) -> ASRInferenceHandlerResult:
        """Parse a hypothesis returned by a Parakeet RNN model

        :param hypothesis: Parakeet hypothesis
        :return: copy of self
        """
        self.transcription = (
            [
                (s["start"], s["end"], s["segment"])
                for s in hypothesis.timestamp.get("segment")
            ]
            if hypothesis.timestamp.get("segment") is not None
            else [(0.0, 0.0, hypothesis.text)]
        )
        self.score = round(hypothesis.score, 2)

        return self

    def concat(
        self, model_result: ASRInferenceHandlerResult
    ) -> ASRInferenceHandlerResult:
        """Left fold with ParakeetInferenceHandlerResult object

        :param model_result: ParakeetInferenceHandlerResult
        :return: copy of self
        """
        if model_result is None:
            return

        if self.transcription is None:
            self.transcription = []

        self.transcription += model_result.transcription

        # We have to weight by total segment len
        transcription_duration = self.transcription[-1][1]
        model_result_duration = model_result.transcription[-1][1]
        total_duration = transcription_duration + model_result_duration

        self.score = round(
            (
                self.score * transcription_duration
                + model_result.score * model_result_duration
            )
            / total_duration,
            2,
        )

        return self


class ParakeetInferenceHandler(ASRInferenceHandler):
    """Inference handler for NVIDIA's Parakeet family of ASR models. Supports up to 24 minutes of
    audio (batched or unbatched) in a single pass. Assumes that audio inputs (wav files or tensors)
    are single-channel with a sample rate of 16000—this last is very important for segmenting.
    """

    def __init__(self, model_name: str, device: str | torch.device = DEVICE_CPU):
        self.model_name = model_name

        if isinstance(device, str):
            device = torch.device(device)

        self.device = device
        self.model = None

    def load(self):
        """Load model; default to CPU where no device is present"""
        device = self.device

        if device is None:
            device = DEVICE_CPU

        self.model = nemo_asr.models.ASRModel.from_pretrained(
            self.model_name, map_location=torch.device(device)
        )

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
        inputs: list[tuple[int, torch.Tensor]],
        timestamps: bool = True,
    ) -> list[tuple[int, ParakeetInferenceHandlerResult]]:
        """Transcribe a batch of audio tensors or file names of total max length <= 24 minutes

        :param inputs: List of np.ndarray or torch.Tensor or str, or singleton of same types
        :param timestamps: Whether to include timestamps with transcriptions
        :return: List of tuples of (input_idx, transcription)
        """
        transcriptions = []

        for tensor_batch in inputs:
            prebatch_indices, segments = zip(*tensor_batch)
            # send to device
            segments = [s.to(self.device) for s in segments]
            hypotheses = self.model.transcribe(segments, timestamps=timestamps)
            # Get timestamped segments if available, otherwise default to whole text
            for idx, hyp in enumerate(hypotheses):
                model_result = (
                    ParakeetInferenceHandlerResult().parse_parakeet_hypothesis(hyp)
                )
                indexed_result = prebatch_indices[idx], model_result
                transcriptions.append(indexed_result)

        return transcriptions
