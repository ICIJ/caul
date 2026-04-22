from abc import ABC
from typing import ClassVar

from icij_common.registrable import RegistrableConfig
from pydantic import Field

from .constants import (
    DEFAULT_SAMPLE_RATE,
    DEFAULT_BATCH_SIZE,
    DEFAULT_MAX_FRAMES,
)
from .objects import BaseModel, ASRModel


class _BaseConfig(BaseModel, RegistrableConfig, ABC): ...


class PreprocessorConfig(_BaseConfig):
    registry_key: ClassVar[str] = Field(frozen=True, default="model")
    model: ClassVar[ASRModel]

    max_frames: int = DEFAULT_MAX_FRAMES
    batch_size: int = DEFAULT_BATCH_SIZE
    sample_rate: int = DEFAULT_SAMPLE_RATE


class InferenceRunnerConfig(_BaseConfig):
    registry_key: ClassVar[str] = Field(frozen=True, default="model")
    model: ClassVar[ASRModel]


class PostprocessorConfig(_BaseConfig):
    registry_key: ClassVar[str] = Field(frozen=True, default="model")
    model: ClassVar[ASRModel]
