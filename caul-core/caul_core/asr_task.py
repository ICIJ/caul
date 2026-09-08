import gc
from abc import ABC, abstractmethod
from collections.abc import Iterable
from contextlib import AbstractContextManager
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeAlias

from icij_common.registrable import RegistrableFromConfig

from .constants import TorchDevice
from .objects import ASRResult, AudioSegment, PreprocessorOutput

if TYPE_CHECKING:
    import numpy as np
    import torch


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


InputItem: TypeAlias = "np.ndarray | torch.Tensor | str | Path"
ASRInput: TypeAlias = "Iterable[InputItem] | InputItem"


class Preprocessor(ASRTask, RegistrableFromConfig):
    @abstractmethod
    def process(
        self,
        inputs: ASRInput,
        input_sample_rates: Iterable[int] | int | None = None,
        output_dir: Path | None = None,
        **kwargs,
    ) -> Iterable[list[PreprocessorOutput]]:
        """Generic processing task"""


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
        self, inputs: Iterable[list[AudioSegment]], *args, **kwargs
    ) -> Iterable[ASRResult]: ...

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
