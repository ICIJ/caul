from abc import ABC, abstractmethod
from contextlib import AbstractContextManager
from typing import Any, TYPE_CHECKING

from icij_common.registrable import RegistrableFromConfig

from caul.objects import ASRResult, PreprocessorOutput


if TYPE_CHECKING:
    import torch
    import numpy as np

    from caul.constant import TorchDevice


class ASRTask(AbstractContextManager, ABC):
    """Generic ASR task"""

    # pylint: disable=R0903

    @abstractmethod
    def process(self, inputs: Any, *args, **kwargs) -> list[PreprocessorOutput]:
        """Generic processing task"""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb): ...

    def set_device(self, device: "TorchDevice | torch._device") -> None:  # pylint: disable=unused-argument
        pass


class Preprocessor(ASRTask, RegistrableFromConfig):
    @abstractmethod
    def process(
        self,
        inputs: "list[np.ndarray | torch.Tensor | str] | np.ndarray | torch.Tensor | str",
        *args,
        **kwargs,
    ) -> list[PreprocessorOutput]:
        """Generic processing task"""


class InferenceRunner(ASRTask, RegistrableFromConfig):
    """Abstract for ASR inference"""

    @abstractmethod
    def process(
        self, inputs: list[PreprocessorOutput], *args, **kwargs
    ) -> list[ASRResult]: ...


class Postprocessor(ASRTask, RegistrableFromConfig):
    def process(
        self, inputs: list[PreprocessorOutput], *args, **kwargs
    ) -> list[ASRResult]: ...
