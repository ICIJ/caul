import gc
from abc import ABC, abstractmethod
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any, Iterable, TYPE_CHECKING

from icij_common.registrable import RegistrableFromConfig

from .objects import ASRResult, PreprocessorOutput
from .constants import TorchDevice


if TYPE_CHECKING:
    import torch
    import numpy as np


class ASRTask(AbstractContextManager, ABC):
    """Generic ASR task"""

    # pylint: disable=R0903
    def __init__(self, device: TorchDevice = TorchDevice.CPU) -> None:
        self._device = device

    @abstractmethod
    def process(
        self, inputs: Iterable[Any], *args, **kwargs
    ) -> Iterable[PreprocessorOutput]:
        """Generic processing task"""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb): ...

    @property
    def device(self) -> TorchDevice:
        return self._device

    @device.setter
    def device(self, device: TorchDevice) -> None:  # pylint: disable=unused-argument
        self._device = device


class Preprocessor(ASRTask, RegistrableFromConfig):
    @abstractmethod
    def process(
        self,
        inputs: "Iterable[np.ndarray | torch.Tensor | str] | np.ndarray | torch.Tensor | str",
        output_dir: Path | None = None,
        **kwargs,
    ) -> Iterable[list[PreprocessorOutput]]:
        """Generic processing task"""

    @classmethod
    @abstractmethod
    def cache_models(cls, cache_dir: Path | None = None) -> None: ...


class InferenceRunner(ASRTask, RegistrableFromConfig):
    """Abstract for ASR inference"""

    def __init__(self, device: TorchDevice = TorchDevice.CPU):
        self._device = device

    def __exit__(self, exc_type, exc_val, exc_tb):
        import torch  # pylint: disable=import-outside-toplevel

        self._model = None
        if self._device == torch.device(TorchDevice.GPU):
            torch.cuda.empty_cache()
        gc.collect()

        return self

    @abstractmethod
    def process(
        self, inputs: Iterable[list[PreprocessorOutput]], *args, **kwargs
    ) -> Iterable[ASRResult]: ...

    @classmethod
    @abstractmethod
    def cache_models(cls, cache_dir: Path | None = None) -> None: ...

    @property
    def _torch_device(self) -> "torch.device":
        import torch

        return torch.device(self.device)

    @property
    def device(self) -> TorchDevice:
        return self._device

    @device.setter
    def device(self, device: TorchDevice) -> None:  # pylint: disable=unused-argument
        self._device = device


class Postprocessor(ASRTask, RegistrableFromConfig):
    def process(
        self, inputs: Iterable[ASRResult], *args, **kwargs
    ) -> Iterable[ASRResult]: ...
