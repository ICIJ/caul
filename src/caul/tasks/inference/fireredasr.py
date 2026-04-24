import gc
import logging
import os
from contextlib import nullcontext
from pathlib import Path
from typing import ClassVar, Iterable, TYPE_CHECKING

from icij_common.registrable import FromConfig
from pydantic import Field
from torch._C.cpp.nn import Module

from caul.constants import (
    FireRedASR2ModelTag,
    FIREREDASR2_AED_MODEL_PATH,
    FIREREDASR2_MODEL_HUB_PREFIX,
    FIREREDASR2_USE_HALF_DEFAULT,
    FIREREDASR2_BEAM_SIZE_DEFAULT,
    FIREREDASR2_NBEST_DEFAULT,
    FIREREDASR2_DECODE_MAX_LEN_DEFAULT,
    FIREREDASR2_SOFTMAX_SMOOTHING_DEFAULT,
    FIREREDASR2_AED_LENGTH_PENALTY_DEFAULT,
    FIREREDASR2_EOS_PENALTY_DEFAULT,
    FIREREDASR2_RETURN_TIMESTAMP_DEFAULT,
    TorchDevice,
    FireRedASR2ModelRef,
)
from caul.objects import ASRResult, PreprocessorOutput, ASRModel
from caul.tasks.asr_task import InferenceRunner
from caul.config import InferenceRunnerConfig
from caul.utils import prepare_file_input_batch

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import torch


def fireredasr2_from_pretrained(
    model_ref: str,
    model_dir: str,
    model_tag: str | None = None,
    config: "FireRedAsr2Config | None" = None,
):
    """Wrapper for FireRedAsr2.from_pretrained that downloads a model from
    HuggingFace Hub if not already available in model_dir.

    :param model_ref: One of ASR2, VAD, LID, or Punc
    :param model_dir: Directory to download model to
    :param model_tag: For ASR models, can be either AED or LLM
    :param config: Configuration options for model
    """
    from huggingface_hub import (
        snapshot_download,
    )  # pylint: disable=import-outside-toplevel
    from fireredasr2s.fireredasr2 import (
        FireRedAsr2,
        FireRedAsr2Config,
    )  # pylint: disable=import-outside-toplevel

    if config is None:
        config = FireRedAsr2Config()

    if model_tag is not None:
        model_ref = f"{model_ref}-{model_tag}"

    if not os.path.isdir(model_dir):
        hub_ref = FIREREDASR2_MODEL_HUB_PREFIX + model_ref.upper()
        model_dir = snapshot_download(hub_ref, local_dir=model_dir, dry_run=False)

    if model_tag is not None:
        return FireRedAsr2.from_pretrained(model_tag, model_dir, config)

    return FireRedAsr2.from_pretrained(model_dir, config)


class FireRedASR2InferenceRunnerConfig(InferenceRunnerConfig):
    model: ClassVar[str] = Field(frozen=True, default=ASRModel.FIREREDASR2_AED)

    model_dir: str = FIREREDASR2_AED_MODEL_PATH
    use_half: bool = FIREREDASR2_USE_HALF_DEFAULT
    beam_size: int = FIREREDASR2_BEAM_SIZE_DEFAULT
    nbest: int = FIREREDASR2_NBEST_DEFAULT
    decode_max_len: int = FIREREDASR2_DECODE_MAX_LEN_DEFAULT
    softmax_smoothing: float = FIREREDASR2_SOFTMAX_SMOOTHING_DEFAULT
    aed_length_penalty: float = FIREREDASR2_AED_LENGTH_PENALTY_DEFAULT
    eos_penalty: float = FIREREDASR2_EOS_PENALTY_DEFAULT
    return_timestamp: bool = FIREREDASR2_RETURN_TIMESTAMP_DEFAULT

    def to_fire_red_asr_model(self, use_gpu: bool = True) -> Module:
        from fireredasr2s.fireredasr2 import (
            FireRedAsr2Config,
        )  # pylint: disable=import-outside-toplevel

        asr_config = FireRedAsr2Config(
            use_gpu=use_gpu,
            use_half=self.use_half,
            beam_size=self.beam_size,
            nbest=self.nbest,
            decode_max_len=self.decode_max_len,
            softmax_smoothing=self.softmax_smoothing,
            aed_length_penalty=self.aed_length_penalty,
            eos_penalty=self.eos_penalty,
            return_timestamp=self.return_timestamp,
        )

        return fireredasr2_from_pretrained(
            model_tag=FireRedASR2ModelTag.AED,
            model_dir=self.model_dir,
            model_ref=FireRedASR2ModelRef.ASR2,
            config=asr_config,
        )


@InferenceRunner.register(ASRModel.FIREREDASR2_AED)
class FireRedASR2InferenceRunner(InferenceRunner):
    """Inference runner for the FireRedASR2 AED model.

    Transcribes Chinese-language (and multilingual) audio. Expects 16 kHz
    mono wav files or tensors (segments up to 60 s). Requires fireredasr2s to
    be installed from https://github.com/FireRedTeam/FireRedASR2S.
    """

    def __init__(
        self,
        config: FireRedASR2InferenceRunnerConfig = None,
        device: "TorchDevice | torch.device" = TorchDevice.CPU,
    ):
        if config is None:
            config = FireRedASR2InferenceRunnerConfig()

        self._config = config
        self._device = device
        self._model = None

    @classmethod
    def _from_config(
        cls,
        config: FireRedASR2InferenceRunnerConfig,
        device: "TorchDevice | torch.device" = TorchDevice.CPU,
        **extras,
    ) -> FromConfig:
        return cls(
            config=config,
            device=device,
            **extras,
        )

    def __enter__(self):
        use_gpu = self._device != TorchDevice.CPU
        self._model = self._config.to_fire_red_asr_model(use_gpu)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._model = None
        gc.collect()

    def process(
        self,
        inputs: Iterable[list[PreprocessorOutput]],
        *,
        output_dir: str | Path = None,
        **kwargs,
    ) -> Iterable[ASRResult]:
        """Transcribe batches of preprocessed audio segments.

        :param inputs: batches of PreprocessorOutput (file-backed or tensor)
        :param output_dir: directory to save output files (will default to tempdir if
        not provided
        :return: ASRResult per segment, in batch order
        """
        for input_batch in inputs:
            if len(input_batch) == 0:
                continue
            inp_ids, wav_paths, inp_id_ordering_map, tmp_dir = prepare_file_input_batch(
                input_batch, output_dir, self._config.tmp_dir_fallback
            )

            with tmp_dir if tmp_dir is not None else nullcontext():
                results = self._model.transcribe(inp_ids, wav_paths)

            for result in results:
                # FireRedASR2 uses uttid to refer to a unique identifier for each
                # item in a batch; technically results should be returned in the
                # order given by uttids, but we use a map to the original index
                # just to be safe.
                input_ordering = inp_id_ordering_map[result["uttid"]]
                yield ASRResult.from_fireredasr2_result(
                    result, input_ordering=input_ordering
                )
