from contextlib import ExitStack
from copy import copy
from typing import Iterable, Self, TYPE_CHECKING

from icij_common.pydantic_utils import make_enum_discriminator, tagged_union
from pydantic import Discriminator, Field

from .config import InferenceRunnerConfig, PostprocessorConfig, PreprocessorConfig
from .constants import TorchDevice
from .objects import ASRResult, BaseModel, ASRModel
from .tasks import (
    ASRTask,
    InferenceRunner,
    ParakeetInferenceRunnerConfig,
    ParakeetPostprocessorConfig,
    ParakeetPreprocessorConfig,
    Postprocessor,
    Preprocessor,
    FireRedASR2PreprocessorConfig,
    FireRedASR2InferenceRunnerConfig,
    FireRedASR2PostprocessorConfig,
)


if TYPE_CHECKING:
    import numpy as np
    import torch


PreprocessorConfig_ = tagged_union(
    PreprocessorConfig.__subclasses__(), lambda t: t.model.default.value
)

InferenceRunnerConfig_ = tagged_union(
    InferenceRunnerConfig.__subclasses__(), lambda t: t.model.default.value
)

PostprocessorConfig_ = tagged_union(
    PostprocessorConfig.__subclasses__(), lambda t: t.model.default.value
)

model_discriminator = make_enum_discriminator("model", ASRModel)


class ASRPipelineConfig(BaseModel):  # pylint: disable=too-few-public-methods
    device: TorchDevice = TorchDevice.CPU
    preprocessing: PreprocessorConfig_ = Field(
        discriminator=Discriminator(model_discriminator),
    )
    inference: InferenceRunnerConfig_ = Field(
        default_factory=InferenceRunnerConfig,
        discriminator=Discriminator(model_discriminator),
    )
    postprocessing: PostprocessorConfig_ = Field(
        discriminator=Discriminator(model_discriminator),
    )


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
    ) -> Iterable[ASRResult]:
        """Generic sequential processing method for ASR model handlers"""
        output = inputs
        for task in self._tasks:
            output = task.process(output)
        yield from output

    @classmethod
    def parakeet(cls, device: "TorchDevice | torch._device" = TorchDevice.CPU) -> Self:
        return cls.from_config(
            ASRPipelineConfig(
                device=device,
                preprocessing=ParakeetPreprocessorConfig(),
                inference=ParakeetInferenceRunnerConfig(),
                postprocessing=ParakeetPostprocessorConfig(),
            )
        )

    @classmethod
    def fireredasr2(
        cls, device: "TorchDevice | torch._device" = TorchDevice.CPU
    ) -> Self:
        return cls.from_config(
            ASRPipelineConfig(
                device=device,
                preprocessing=FireRedASR2PreprocessorConfig(),
                inference=FireRedASR2InferenceRunnerConfig(),
                postprocessing=FireRedASR2PostprocessorConfig(),
            )
        )
