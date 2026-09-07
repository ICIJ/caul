import gc
from abc import ABC, abstractmethod
from collections.abc import Iterable
from contextlib import AbstractContextManager
from pathlib import Path
from typing import TYPE_CHECKING, TypeAlias

from icij_common.registrable import FromConfig, RegistrableFromConfig
from overrides import final

from .config import BaseBatcherConfig
from .constants import TorchDevice
from .objects import ASRResult, Error, ProcessedAudioSegment

if TYPE_CHECKING:
    import numpy as np
    import torch


class ASRTask[I, O](AbstractContextManager, ABC):
    """Generic ASR task"""

    # pylint: disable=R0903
    def __init__(self, device: TorchDevice = TorchDevice.CPU) -> None:
        self._device = device

    @abstractmethod
    def process(self, inputs: Iterable[I], *args, **kwargs) -> Iterable[O | Error]:
        """Generic processing task"""

    @final
    def process_results(
        self, inputs: Iterable[I], *args, skip_errors: bool = False, **kwargs
    ) -> Iterable[O]:
        for r in self.process(inputs, *args, **kwargs):
            if not skip_errors and isinstance(r, Error):
                raise RuntimeError(f"error: {r}")
            yield r

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
SampleRate = Iterable[int] | int


class Preprocessor(ASRTask, RegistrableFromConfig):
    @abstractmethod
    def process(
        self,
        inputs: ASRInput,
        sample_rates: SampleRate | None = None,
        output_dir: Path | None = None,
        **kwargs,
    ) -> Iterable[tuple[ProcessedAudioSegment, ...] | Error]:
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
        self, inputs: Iterable[tuple[ProcessedAudioSegment, ...]], *args, **kwargs
    ) -> Iterable[ASRResult | Error]: ...

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
    ) -> Iterable[ASRResult | Error]: ...


class Batcher[I, C](RegistrableFromConfig):
    def __init__(self, items: Iterable[I | Error], config: C):
        self._items = iter(items)
        self._errors = []
        self._config = config

    @abstractmethod
    def results(self) -> Iterable[tuple[I, ...]]: ...

    @final
    @property
    def errors(self) -> list[Error]:
        try:
            next(self._items)
            msg = (
                f"{Batcher.results.__name__} must be fully consumed to collect"
                f" and get all errors"
            )
            raise RuntimeError(msg)
        except StopIteration:
            pass
        return self._errors

    @classmethod
    def _from_config(
        cls, config: BaseBatcherConfig, *, items: Iterable[I | Error]
    ) -> FromConfig:
        return cls(config=config, items=items)
