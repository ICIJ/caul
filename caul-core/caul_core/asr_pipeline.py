import logging
from abc import ABC
from contextlib import ExitStack
from copy import copy
from pathlib import Path
from typing import Iterable, Self

from icij_common.pydantic_utils import safe_copy

from .constants import TorchDevice
from .config import ASRPipelineConfig
from .objects import ASRResult
from .asr_task import ASRTask, InferenceRunner, Postprocessor, Preprocessor

logger = logging.getLogger(__name__)


class ASRPipeline(ABC):
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

    @device.setter
    def device(self, device: TorchDevice = TorchDevice.CPU):
        self._device = device
        for t in self._tasks:
            t.device = device

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
    def parakeet(cls, device: TorchDevice = TorchDevice.CPU) -> Self:
        config = ASRPipelineConfig.parakeet()
        config = safe_copy(config, update={"device": device})
        return cls.from_config(config)

    @classmethod
    def parakeet_trt(
        cls,
        model_path: str | Path,
        engine_path: str | Path,
        device: TorchDevice = TorchDevice.CPU,
    ) -> Self:
        config = ASRPipelineConfig.parakeet_trt()
        config = safe_copy(
            config,
            update={
                "device": device,
                "model_path": model_path,
                "engine_path": engine_path,
            },
        )
        return cls.from_config(config=config)

    @classmethod
    def fireredasr2(
        cls,
        device: TorchDevice = TorchDevice.CPU,
        tmp_dir_fallback: bool = False,
    ) -> Self:
        config = ASRPipelineConfig.fireredasr2(tmp_dir_fallback=tmp_dir_fallback)
        config = safe_copy(config, update={"device": device})
        return cls.from_config(config)

    @classmethod
    def faster_whisper(cls, device: TorchDevice = TorchDevice.CPU) -> Self:
        config = ASRPipelineConfig.faster_whisper()
        config = safe_copy(config, update={"device": device})
        return cls.from_config(config)

    @classmethod
    def whisper_trt(cls, device: TorchDevice = TorchDevice.CPU) -> Self:
        config = ASRPipelineConfig.whisper_trt()
        config = safe_copy(config, update={"device": device})
        return cls.from_config(config=config)
