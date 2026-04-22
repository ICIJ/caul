from typing import Callable, ClassVar

from pydantic import Field

from caul.config import PreprocessorConfig
from caul.constants import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_SAMPLE_RATE,
    FIREREDASR2_INFERENCE_MAX_FRAMES,
)
from caul.objects import ASRModel
from caul.tasks.asr_task import Preprocessor
from caul.tasks.preprocessing.asr_preprocessor import ASRPreprocessor


class FireRedASR2PreprocessorConfig(PreprocessorConfig):
    model: ClassVar[str] = Field(default=ASRModel.FIREREDASR2_AED)

    max_frames: int = FIREREDASR2_INFERENCE_MAX_FRAMES


@Preprocessor.register(ASRModel.FIREREDASR2_AED)
class FireRedASR2Preprocessor(ASRPreprocessor):
    def __init__(
        self,
        max_frames: int = FIREREDASR2_INFERENCE_MAX_FRAMES,
        batch_size: int = DEFAULT_BATCH_SIZE,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
    ):
        super().__init__(
            max_frames=max_frames,
            batch_size=batch_size,
            sample_rate=sample_rate,
        )
