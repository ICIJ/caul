from abc import ABC
from typing import ClassVar

from icij_common.registrable import Registrable, RegistrableConfig
from pydantic import Field

from .objects import BaseModel


class _BaseConfig(BaseModel, RegistrableConfig, ABC):
    registry_key: ClassVar[str] = Field(frozen=True, default="model")


class BaseComponent(BaseModel, Registrable, ABC):
    registry_key: ClassVar[str] = Field(frozen=True, default="model")


class PreprocessorConfig(_BaseConfig): ...


class InferenceRunnerConfig(_BaseConfig): ...


class PostprocessorConfig(_BaseConfig): ...
