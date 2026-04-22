from abc import ABC, abstractmethod
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any, Iterable, TYPE_CHECKING

from icij_common.registrable import RegistrableFromConfig

from caul.constants import TorchDevice
from caul.objects import ASRResult, PreprocessorOutput


if TYPE_CHECKING:
    import torch
    import numpy as np


class ASRTask(AbstractContextManager, ABC):
    """Generic ASR task"""

    # pylint: disable=R0903

    @abstractmethod
    def process(
        self, inputs: Iterable[Any], *args, **kwargs
    ) -> Iterable[PreprocessorOutput]:
        """Generic processing task"""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb): ...

    def set_device(
        self, device: "TorchDevice | torch._device"
    ) -> None:  # pylint: disable=unused-argument
        pass


class Preprocessor(ASRTask, RegistrableFromConfig):
    @abstractmethod
    def process(
        self,
        inputs: "Iterable[np.ndarray | torch.Tensor | str] | np.ndarray | torch.Tensor | str",
        output_dir: Path | None = None,
        **kwargs,
    ) -> Iterable[list[PreprocessorOutput]]:
        """Generic processing task"""


class InferenceRunner(ASRTask, RegistrableFromConfig):
    """Abstract for ASR inference"""

    def __init__(self, device: "TorchDevice | torch._device" = TorchDevice.CPU):
        import torch  # pylint: disable=import-outside-toplevel

        if isinstance(device, str):
            device = torch.device(device)

        self._device = device

    @abstractmethod
    def process(
        self, inputs: Iterable[list[PreprocessorOutput]], *args, **kwargs
    ) -> Iterable[ASRResult]: ...

    @property
    def device(self) -> "torch.device":
        return self._device

    def set_device(self, device: "TorchDevice | torch.device" = TorchDevice.CPU):
        import torch  # pylint: disable=import-outside-toplevel

        if isinstance(device, TorchDevice):
            device = torch.device(device)

        self._device = device

        return self


class Postprocessor(ASRTask, RegistrableFromConfig):
    def process(
        self, inputs: Iterable[ASRResult], *args, **kwargs
    ) -> Iterable[ASRResult]: ...
