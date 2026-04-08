from typing import ClassVar, Iterable

import gc


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
        device: "TorchDevice | torch.device" = TorchDevice.CPU,
        return_timestamps: bool = True,
        batch_size: int = 4,
    ):
        import torch  # pylint: disable=import-outside-toplevel

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
        import nemo.collections.asr as nemo_asr  # pylint: disable=import-outside-toplevel
        import torch  # pylint: disable=import-outside-toplevel

        device = self._device
        self._model = nemo_asr.models.ASRModel.from_pretrained(
            self.model_name, map_location=torch.device(device)
        ).eval()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        import torch  # pylint: disable=import-outside-toplevel

        self._model = None
        if self._device == torch.device(TorchDevice.GPU):
            torch.cuda.empty_cache()
        gc.collect()

    @property
    def device(self) -> "torch.device":
        return self._device

    def set_device(self, device: "TorchDevice | torch.device" = TorchDevice.CPU):
        import torch  # pylint: disable=import-outside-toplevel

        if isinstance(device, TorchDevice):
            device = torch.device(device)

        self._device = device

        return self

    def process(  # pylint: disable=too-many-locals
        self,
        inputs: Iterable[list[PreprocessorOutput]],
        *args,
        **kwargs,
    ) -> Iterable[ASRResult]:
        """Transcribe a batch of audio tensors or file names of max duration <= 20 minutes

        :param inputs: List of np.ndarray or torch.Tensor or str, or singleton of same types
        :return: List of results
        """
        # pylint: disable=import-outside-toplevel
        from nemo.collections.asr.parts.mixins import TranscribeConfig
        from nemo.collections.asr.parts.mixins.transcription import (
            InternalTranscribeConfig,
        )

        if isinstance(inputs, PreprocessorOutput):
            inputs = [inputs]

        transcribe_config = TranscribeConfig(
            use_lhotse=False,
            batch_size=self._batch_size,
            timestamps=self._return_timestamps,
            return_hypotheses=True,
            _internal=InternalTranscribeConfig(device=self._device),
        )
        for input_batch in inputs:
            if not input_batch:
                continue
            if hasattr(input_batch[0], "tensor"):
                audios = [i.tensor.to(self._device) for i in input_batch]
            else:
                audios = [str(i.metadata.preprocessed_file_path) for i in input_batch]

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
                yield model_result
