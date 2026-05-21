import logging
from contextlib import ExitStack
from copy import copy
from pathlib import Path
from typing import Iterable, Self, TYPE_CHECKING

from icij_common.pydantic_utils import safe_copy
from caul_core.objects import TorchDevice, ASRModel, ASRResult
from caul_core.asr_pipeline import ASRPipelineConfig

from .tasks import (
    ASRTask,
    FasterWhisperInferenceRunner,
    FireRedASR2InferenceRunner,
    InferenceRunner,
    ParakeetInferenceRunner,
    Postprocessor,
    Preprocessor,
)
from .tasks.preprocessing.asr_preprocessor import ASRPreprocessor

if TYPE_CHECKING:
    import numpy as np
    import torch


logger = logging.getLogger(__name__)


class ASRPipeline:
    def __init__(
        self, tasks: list[ASRTask], device: TorchDevice = TorchDevice.CPU
    ) -> None:
        self._device = device
        self._tasks = tasks
        self._exit_stack = ExitStack()

    @classmethod
    def from_config(cls, config: ASRPipelineConfig) -> Self:
        tasks = [
            Preprocessor.from_config(config.preprocessing),
            InferenceRunner.from_config(config.inference, device=config.device),
            Postprocessor.from_config(config.postprocessing),
        ]
        return cls(tasks, device=config.device)

    def __enter__(self):
        for t in self._tasks:
            self._exit_stack.enter_context(t)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._exit_stack.__exit__(exc_type, exc_val, exc_tb)

    @property
    def tasks(self) -> list[ASRTask]:
        return copy(self._tasks)

    @property
    def device(self) -> TorchDevice:
        return self._device

    def set_device(self, device: "TorchDevice | torch._device") -> None:
        import torch  # pylint: disable=import-outside-toplevel

        if isinstance(device, TorchDevice):
            device = torch.device(device.value)
        self._device = device
        for t in self._tasks:
            t.set_device(device)

    def process(
        self,
        inputs: "Iterable[np.ndarray | torch.Tensor | str] | np.ndarray | torch.Tensor | str",
        languages: list[str] | None = None,
        tensor_output_dir: str | Path | None = None,
    ) -> Iterable[ASRResult]:
        """Generic sequential processing method for ASR model handlers"""
        output = inputs
        for task in self._tasks:
            output = task.process(
                output, output_dir=tensor_output_dir, languages=languages
            )
        yield from output

    @classmethod
    def parakeet(cls, device: "TorchDevice | torch._device" = TorchDevice.CPU) -> Self:
        config = ASRPipelineConfig.parakeet()
        config = safe_copy(config, update={"device": device})
        return cls.from_config(config)

    @classmethod
    def fireredasr2(
        cls,
        device: "TorchDevice | torch._device" = TorchDevice.CPU,
        tmp_dir_fallback: bool = False,
    ) -> Self:
        config = ASRPipelineConfig.fireredasr2(tmp_dir_fallback=tmp_dir_fallback)
        config = safe_copy(config, update={"device": device})
        return cls.from_config(config)

    @classmethod
    def faster_whisper(
        cls, device: "TorchDevice | torch._device" = TorchDevice.CPU
    ) -> Self:
        config = ASRPipelineConfig.faster_whisper()
        config = safe_copy(config, update={"device": device})
        return cls.from_config(config)


def cache_models(asr_model: ASRModel | None, cache_dir: Path) -> None:
    if asr_model is None:
        logger.info("caching all models to %s", cache_dir)
    else:
        logger.info("caching %s models to %s", asr_model, cache_dir)
    match asr_model:
        case ASRModel.PARAKEET:
            cache_fns = [
                ASRPreprocessor.cache_models,
                ParakeetInferenceRunner.cache_models,
            ]
        case ASRModel.FASTER_WHISPER:
            cache_fns = [
                ASRPreprocessor.cache_models,
                FasterWhisperInferenceRunner.cache_models,
            ]
        case ASRModel.FIREREDASR2_AED:
            cache_fns = [
                ASRPreprocessor.cache_models,
                FireRedASR2InferenceRunner.cache_models,
            ]
        case None:
            cache_fns = [
                ASRPreprocessor.cache_models,
                ParakeetInferenceRunner.cache_models,
                FasterWhisperInferenceRunner.cache_models,
                FireRedASR2InferenceRunner.cache_models,
            ]
        case _:
            raise ValueError(f"invalid model {asr_model}")
    for cache_fn in cache_fns:
        cache_fn(cache_dir)
