import logging
from pathlib import Path, PurePosixPath
from typing import Iterable, TYPE_CHECKING

from icij_common.registrable import FromConfig

from caul_core import (
    PARAKEET_MODEL_REF,
    TorchDevice,
    ASRModel,
    ASRResult,
    PreprocessorOutput,
    ParakeetInferenceRunnerConfig,
    InferenceRunner,
)
from ...utils import cache_hf_model_file

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import torch


@InferenceRunner.register(ASRModel.PARAKEET)
class ParakeetInferenceRunner(InferenceRunner):
    """Inference handler for NVIDIA's Parakeet family of ASR models. Supports up to 24 minutes of
    audio (batched or unbatched) in a single pass. Assumes that audio inputs (wav files or tensors)
    are single-channel with a sample rate of 16000—this last is very important for segmenting.
    """

    _models = [PARAKEET_MODEL_REF]

    def __init__(
        self,
        model_name: str = PARAKEET_MODEL_REF,
        device: TorchDevice = TorchDevice.CPU,
        return_timestamps: bool = True,
        batch_size: int = 4,
    ):
        super().__init__(device)
        self.model_name = model_name
        self._return_timestamps = return_timestamps
        self._model = None
        self.__transcribe_config = None
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

    @property
    def _transcribe_config(self):
        if self.__transcribe_config is None:
            # pylint: disable=import-outside-toplevel
            from nemo.collections.asr.parts.mixins import TranscribeConfig
            from nemo.collections.asr.parts.mixins.transcription import (
                InternalTranscribeConfig,
            )

            self.__transcribe_config = TranscribeConfig(
                use_lhotse=False,
                batch_size=self._batch_size,
                timestamps=self._return_timestamps,
                return_hypotheses=True,
                # Bug in Nemo's AudioToBPEDataset—by default TranscribeConfig spawns 2
                # DataLoader workers, but AudioToBPEDataset defines a class TokenizerWrapper
                # inside __init__, meaning it can't be pickled.
                num_workers=0,
                _internal=InternalTranscribeConfig(device=self._torch_device),
            )

        return self.__transcribe_config

    def __enter__(self):
        import nemo.collections.asr as nemo_asr  # pylint: disable=import-outside-toplevel

        self._model = nemo_asr.models.ASRModel.from_pretrained(
            self.model_name, map_location=self._torch_device
        ).eval()
        return self

    @classmethod
    def cache_models(cls, cache_dir: Path | None = None) -> None:
        from huggingface_hub.constants import (
            HF_HUB_CACHE,
        )  # pylint: disable=import-outside-toplevel

        if cache_dir is not None and str(cache_dir) != HF_HUB_CACHE:
            msg = (
                f"parakeet model are sadly only loaded from the HF cache hub"
                f" ({HF_HUB_CACHE}), can't load them from elsewhere"
            )
            raise ValueError(msg)

        for m in cls._models:
            logger.info("caching parakeet model %s", m)
            filenaname = PurePosixPath(m).name + ".nemo"
            cache_hf_model_file(
                repo_id=m, filename=filenaname, library_name="nemo", cache_dir=cache_dir
            )

    def _transcribe(
        self, audio_inputs: Iterable["torch.Tensor"], **kwargs
    ) -> ASRResult:
        """Transcribe normalized audio tensors

        :param audio_inputs: input tensors
        :return: transcription results
        """
        return self._model.transcribe(
            audio_inputs,
            timestamps=self._return_timestamps,
            override_config=self._transcribe_config,
        )

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

        if isinstance(inputs, PreprocessorOutput):
            inputs = [inputs]

        for input_batch in inputs:
            if not input_batch:
                continue
            if hasattr(input_batch[0], "tensor"):
                audios = [i.tensor.to(self._torch_device) for i in input_batch]
            else:
                audios = [str(i.metadata.preprocessed_file_path) for i in input_batch]

            hypotheses = self._transcribe(audios)
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
