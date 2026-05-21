import logging
from contextlib import nullcontext
from pathlib import Path
from typing import Iterable, TYPE_CHECKING


from icij_common.registrable import FromConfig

from caul_core.constants import (
    FIREREDASR2_USE_HALF_DEFAULT,
    FIREREDASR2_BEAM_SIZE_DEFAULT,
    FIREREDASR2_NBEST_DEFAULT,
    FIREREDASR2_DECODE_MAX_LEN_DEFAULT,
    FIREREDASR2_SOFTMAX_SMOOTHING_DEFAULT,
    FIREREDASR2_AED_LENGTH_PENALTY_DEFAULT,
    FIREREDASR2_EOS_PENALTY_DEFAULT,
    FIREREDASR2_RETURN_TIMESTAMP_DEFAULT,
    FIREREDASR2_MODEL_HUB_PREFIX,
)
from caul_core.objects import (
    TorchDevice,
    ASRResult,
    PreprocessorOutput,
    ASRModel,
    FireRedASR2ModelTag,
    FireRedASR2ModelRef,
)
from caul.tasks.asr_task import InferenceRunner
from caul_core.config import InferenceRunnerConfig, FireRedASR2InferenceRunnerConfig
from caul.utils import cache_hf_repo, prepare_file_input_batch

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import torch
    from fireredasr2s.fireredasr2 import FireRedAsr2


def fireredasr2_from_pretrained(
    model_ref: FireRedASR2ModelRef,
    *,
    cache_dir: Path | None = None,
    model_tag: FireRedASR2ModelTag = FireRedASR2ModelTag.AED,
    config: "FireRedAsr2Config | None" = None,
) -> "FireRedAsr2":
    """Wrapper for FireRedAsr2.from_pretrained that downloads a model from
    HuggingFace Hub if not already available in model_dir.

    :param model_ref: One of ASR2, VAD, LID, or Punc
    :param model_dir: Directory to download model to
    :param model_tag: For ASR models, can be either AED or LLM
    :param config: Configuration options for model
    """
    from fireredasr2s.fireredasr2 import (
        FireRedAsr2,
        FireRedAsr2Config,
    )  # pylint: disable=import-outside-toplevel
    from huggingface_hub.constants import (
        HF_HUB_CACHE,
    )  # pylint: disable=import-outside-toplevel
    from huggingface_hub import (
        snapshot_download,
    )  # pylint: disable=import-outside-toplevel

    if cache_dir is None:
        cache_dir = HF_HUB_CACHE

    if config is None:
        config = FireRedAsr2Config()

    if model_tag is not None:
        model_ref = f"{model_ref}-{model_tag}"

    repo_id = FIREREDASR2_MODEL_HUB_PREFIX + model_ref.upper()

    model_dir = snapshot_download(repo_id=repo_id, cache_dir=cache_dir)
    return FireRedAsr2.from_pretrained(model_tag, model_dir, config)


def inference_config_to_fire_red_asr_model(
    inference_config: FireRedASR2InferenceRunnerConfig,
    use_gpu: bool = True,
    cache_dir: Path | None = None,
) -> "torch.Module":
    from fireredasr2s.fireredasr2 import (
        FireRedAsr2Config,
    )  # pylint: disable=import-outside-toplevel

    asr_config = FireRedAsr2Config(
        use_gpu=use_gpu,
        use_half=inference_config.use_half,
        beam_size=inference_config.beam_size,
        nbest=inference_config.nbest,
        decode_max_len=inference_config.decode_max_len,
        softmax_smoothing=inference_config.softmax_smoothing,
        aed_length_penalty=inference_config.aed_length_penalty,
        eos_penalty=inference_config.eos_penalty,
        return_timestamp=inference_config.return_timestamp,
    )

    return fireredasr2_from_pretrained(
        model_ref=FireRedASR2ModelRef.ASR2,
        model_tag=FireRedASR2ModelTag.AED,
        cache_dir=cache_dir,
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
        super().__init__(device)
        if config is None:
            config = FireRedASR2InferenceRunnerConfig()

        self._config = config
        self._model = None

    @classmethod
    def _from_config(
        cls,
        config: FireRedASR2InferenceRunnerConfig,
        device: "TorchDevice | torch.device" = TorchDevice.CPU,
        **extras,
    ) -> FromConfig:
        return cls(config=config, device=device, **extras)

    def __enter__(self):
        use_gpu = self._device.type.startswith("cuda")
        self._model = inference_config_to_fire_red_asr_model(self._config, use_gpu)
        return self

    @classmethod
    def cache_models(cls, cache_dir: Path | None = None) -> None:
        for tag in FireRedASR2ModelTag:
            logger.info("caching model FireredASR tag %s", tag)
            repo_id = (
                f"{FIREREDASR2_MODEL_HUB_PREFIX}{FireRedASR2ModelRef.ASR2}"
                f"-{tag.upper()}"
            )
            cache_hf_repo(repo_id, cache_dir=cache_dir)

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
