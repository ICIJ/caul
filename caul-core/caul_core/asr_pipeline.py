import logging
from typing import Self

from icij_common.pydantic_utils import make_enum_discriminator, tagged_union
from pydantic import Discriminator, Field

from .objects import BaseModel, ASRModel, TorchDevice
from .config import (
    BasePreprocessorConfig,
    BaseInferenceRunnerConfig,
    BasePostprocessorConfig,
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

PreprocessorConfig = tagged_union(
    BasePreprocessorConfig.__subclasses__(), lambda t: t.model.default.value
)

InferenceRunnerConfig = tagged_union(
    BaseInferenceRunnerConfig.__subclasses__(), lambda t: t.model.default.value
)

PostprocessorConfig = tagged_union(
    BasePostprocessorConfig.__subclasses__(), lambda t: t.model.default.value
)

model_discriminator = make_enum_discriminator("model", ASRModel)


class ASRPipelineConfig(BaseModel):  # pylint: disable=too-few-public-methods
    device: TorchDevice = TorchDevice.CPU
    preprocessing: PreprocessorConfig = Field(
        discriminator=Discriminator(model_discriminator),
    )
    inference: InferenceRunnerConfig = Field(
        default_factory=InferenceRunnerConfig,
        discriminator=Discriminator(model_discriminator),
    )
    postprocessing: PostprocessorConfig = Field(
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
