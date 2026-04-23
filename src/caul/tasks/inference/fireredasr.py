import gc
import logging
import tempfile
from contextlib import nullcontext
from pathlib import Path
from typing import ClassVar, Iterable, TYPE_CHECKING

from icij_common.registrable import FromConfig
from pydantic import Field


from caul.constants import (
    FIREREDASR2_AED_MODEL_TAG,
    FIREREDASR2_USE_HALF_DEFAULT,
    FIREREDASR2_BEAM_SIZE_DEFAULT,
    FIREREDASR2_NBEST_DEFAULT,
    FIREREDASR2_DECODE_MAX_LEN_DEFAULT,
    FIREREDASR2_SOFTMAX_SMOOTHING_DEFAULT,
    FIREREDASR2_AED_LENGTH_PENALTY_DEFAULT,
    FIREREDASR2_EOS_PENALTY_DEFAULT,
    FIREREDASR2_RETURN_TIMESTAMP_DEFAULT,
    FIREREDASR2_AED_MODEL_REF,
    TorchDevice,
)
from caul.filesystem import save_tensor
from caul.objects import ASRResult, PreprocessorOutput, ASRModel
from caul.tasks.asr_task import InferenceRunner
from caul.config import InferenceRunnerConfig

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import torch


class FireRedASR2InferenceRunnerConfig(InferenceRunnerConfig):
    model: ClassVar[str] = Field(frozen=True, default=ASRModel.FIREREDASR2_AED)

    model_dir: str = FIREREDASR2_AED_MODEL_REF
    use_half: bool = FIREREDASR2_USE_HALF_DEFAULT
    beam_size: int = FIREREDASR2_BEAM_SIZE_DEFAULT
    nbest: int = FIREREDASR2_NBEST_DEFAULT
    decode_max_len: int = FIREREDASR2_DECODE_MAX_LEN_DEFAULT
    softmax_smoothing: float = FIREREDASR2_SOFTMAX_SMOOTHING_DEFAULT
    aed_length_penalty: float = FIREREDASR2_AED_LENGTH_PENALTY_DEFAULT
    eos_penalty: float = FIREREDASR2_EOS_PENALTY_DEFAULT
    return_timestamp: bool = FIREREDASR2_RETURN_TIMESTAMP_DEFAULT
    tmp_dir_fallback: bool = False

    def to_fire_red_asr_model(self, use_gpu: bool = True) -> "torch.Module":
        from fireredasr2s.fireredasr2 import (  # pylint: disable=import-outside-toplevel
            FireRedAsr2,
            FireRedAsr2Config,
        )

        asr_config = FireRedAsr2Config(
            use_gpu=use_gpu,
            use_half=self._use_half,
            beam_size=self._beam_size,
            nbest=self._nbest,
            decode_max_len=self._decode_max_len,
            softmax_smoothing=self._softmax_smoothing,
            aed_length_penalty=self._aed_length_penalty,
            eos_penalty=self._eos_penalty,
            return_timestamp=self._return_timestamp,
        )

        return FireRedAsr2.from_pretrained(
            FIREREDASR2_AED_MODEL_TAG, self._model_dir, asr_config
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
        config: FireRedASR2InferenceRunnerConfig,
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
        use_gpu = self._device in TorchDevice
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
            inp_ids, wav_paths, inp_id_ordering_map, tmp_dir = self._prepare_batch(
                input_batch, output_dir
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

    def _prepare_batch(
        self, input_batch: list[PreprocessorOutput], output_dir: str | Path = None
    ) -> tuple[
        list[str], list[str], dict[str, int], tempfile.TemporaryDirectory | None
    ]:
        """Collect uuids and wav paths and write tensors to a temp directory when
        no path is available.

        :param input_batch: batch of PreprocessorOutput files
        :return: tuple of batch input ids, wav paths, map from id to input ordering,
        temporary dir (if applicable) where tensor paths are kept
        """
        tmp_dir = None
        if output_dir is None and self._config.tmp_dir_fallback:
            tmp_dir = tempfile.TemporaryDirectory()
            output_dir = tmp_dir.name

        if not isinstance(output_dir, Path):
            output_dir = Path(output_dir)

        inp_ids: list[str] = []
        wav_paths: list[str] = []
        inp_id_ordering_map: dict[str, int] = {}

        for inp in input_batch:
            inp_id = inp.metadata.uuid
            inp_ids.append(inp_id)
            inp_id_ordering_map[inp_id] = inp.metadata.input_ordering

            if inp.metadata.preprocessed_file_path is None:
                if output_dir is None:
                    logger.warning(
                        "Input {} has no preprocessed file path, no output dir is specified, and temporary dir creation is disabled. Skipping."
                    )
                    continue

                wav_path = output_dir / f"{inp_id}.wav"
                save_tensor(inp.tensor, wav_path)
                wav_paths.append(str(wav_path))
            else:
                wav_paths.append(str(inp.metadata.preprocessed_file_path))

        return inp_ids, wav_paths, inp_id_ordering_map, tmp_dir
