from typing import ClassVar

import gc

import torch

import nemo.collections.asr as nemo_asr
from nemo.collections.asr.parts.mixins import TranscribeConfig
from nemo.collections.asr.parts.mixins.transcription import InternalTranscribeConfig
from icij_common.registrable import FromConfig
from pydantic import Field

from caul.constant import ASRModel, PARAKEET_MODEL_REF, TorchDevice
from caul.objects import ASRResult, PreprocessorOutput
from ..asr_task import InferenceRunner
from ...config import InferenceRunnerConfig


class ParakeetInferenceRunnerConfig(InferenceRunnerConfig):
    model: ClassVar[str] = Field(frozen=True, default=ASRModel.PARAKEET)
    model_name: str = PARAKEET_MODEL_REF
    return_timestamps: bool = True


@InferenceRunner.register(ASRModel.PARAKEET)
class ParakeetInferenceRunner(InferenceRunner):
    """Inference handler for NVIDIA's Parakeet family of ASR models. Supports up to 24 minutes of
    audio (batched or unbatched) in a single pass. Assumes that audio inputs (wav files or tensors)
    are single-channel with a sample rate of 16000—this last is very important for segmenting.
    """

    def __init__(
        self,
        model_name: str,
        device: TorchDevice | torch.device = TorchDevice.CPU,
        return_timestamps: bool = True,
        batch_size: int = 4,
    ):
        self.model_name = model_name

        if isinstance(device, str):
            device = torch.device(device)

        self._device = device
        self._return_timestamps = return_timestamps
        self._model = None
        self._batch_size = batch_size

    @classmethod
    def _from_config(
        cls, config: ParakeetInferenceRunnerConfig, **extras
    ) -> FromConfig:
        return cls(
            model_name=config.model_name,
            return_timestamps=config.return_timestamps,
            **extras,
        )

    def __enter__(self):
        device = self._device
        self._model = nemo_asr.models.ASRModel.from_pretrained(
            self.model_name, map_location=torch.device(device)
        ).eval()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._model = None
        if self._device == torch.device(TorchDevice.GPU):
            torch.cuda.empty_cache()
        gc.collect()

    @property
    def device(self) -> "torch.device":
        return self._device

    def set_device(self, device: TorchDevice | torch.device = TorchDevice.CPU):
        """Set/change device"""
        if isinstance(device, TorchDevice):
            device = torch.device(device)

        self._device = device

        return self

    def process(
        self,
        inputs: list[list[PreprocessorOutput]] | list[PreprocessorOutput],
        *args,
        **kwargs,
    ) -> list[ASRResult]:
        """Transcribe a batch of audio tensors or file names of max duration <= 20 minutes

        :param inputs: List of np.ndarray or torch.Tensor or str, or singleton of same types
        :return: List of results
        """
        if len(inputs) == 0:
            return []

        if not isinstance(inputs, list):
            inputs = [inputs]

        transcribe_config = TranscribeConfig(
            use_lhotse=False,
            batch_size=self._batch_size,
            timestamps=self._return_timestamps,
            return_hypotheses=True,
            _internal=InternalTranscribeConfig(device=self._device),
        )
        transcriptions = []
        for input_batch in inputs:
            if not input_batch:
                continue
            if input_batch[0].tensor is not None:
                audios = [i.tensor.to(self._device) for i in input_batch]
            else:
                audios = [i.metadata.preprocessed_file_path for i in input_batch]

            hypotheses = self._model.transcribe(
                audios, self._return_timestamps, override_config=transcribe_config
            )
            # Get timestamped segments if available, otherwise default to whole text
            for idx, hyps in enumerate(hypotheses):
                best_hyp = hyps

                if isinstance(best_hyp, (list, tuple)):
                    best_hyp = hyps[0]

                input_ordering_idx = input_batch[idx].metadata.input_ordering
                model_result = ASRResult.from_parakeet_hypothesis(
                    best_hyp, input_ordering=input_ordering_idx
                )
                transcriptions.append(model_result)

        return transcriptions
