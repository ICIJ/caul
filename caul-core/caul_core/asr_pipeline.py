import logging
from typing import Self

from icij_common.pydantic_utils import make_enum_discriminator, tagged_union
from pydantic import Discriminator, Field

from .objects import BaseModel, ASRModel, TorchDevice
from .config import (
    PreprocessorConfig,
    InferenceRunnerConfig,
    PostprocessorConfig,
    ParakeetPreprocessorConfig,
    ParakeetInferenceRunnerConfig,
    ParakeetPostprocessorConfig,
    FireRedASR2PreprocessorConfig,
    FireRedASR2InferenceRunnerConfig,
    FireRedASR2PostprocessorConfig,
    FasterWhisperPreprocessorConfig,
    FasterWhisperInferenceRunnerConfig,
    FasterWhisperPostprocessorConfig,
    ParakeetTrtInferenceRunnerConfig,
)

logger = logging.getLogger(__name__)

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

    @classmethod
    def parakeet(cls) -> Self:
        return cls(
            preprocessing=ParakeetPreprocessorConfig(),
            inference=ParakeetInferenceRunnerConfig(),
            postprocessing=ParakeetPostprocessorConfig(),
        )

    @classmethod
    def parakeet_trt(cls) -> Self:
        return cls(
            preprocessing=ParakeetPreprocessorConfig(),
            inference=ParakeetTrtInferenceRunnerConfig(),
            postprocessing=ParakeetPostprocessorConfig(),
        )

    @classmethod
    def fireredasr2(cls, tmp_dir_fallback: bool = True) -> Self:
        return cls(
            preprocessing=FireRedASR2PreprocessorConfig(),
            inference=FireRedASR2InferenceRunnerConfig(
                tmp_dir_fallback=tmp_dir_fallback
            ),
            postprocessing=FireRedASR2PostprocessorConfig(),
        )

    @classmethod
    def faster_whisper(cls) -> Self:
        return cls(
            preprocessing=FasterWhisperPreprocessorConfig(),
            inference=FasterWhisperInferenceRunnerConfig(),
            postprocessing=FasterWhisperPostprocessorConfig(),
        )
