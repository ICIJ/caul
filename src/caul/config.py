from abc import ABC
from typing import ClassVar

from icij_common.registrable import RegistrableConfig
from pydantic import Field

from .objects import ASRModel, BaseModel


class _BaseConfig(BaseModel, RegistrableConfig, ABC): ...


class PreprocessorConfig(_BaseConfig):
    registry_key: ClassVar[str] = Field(frozen=True, default="model")
    model: ClassVar[ASRModel]


class InferenceRunnerConfig(_BaseConfig):
    registry_key: ClassVar[str] = Field(frozen=True, default="model")
    model: ClassVar[ASRModel]


class PostprocessorConfig(_BaseConfig):
    registry_key: ClassVar[str] = Field(frozen=True, default="model")
    model: ClassVar[ASRModel]
