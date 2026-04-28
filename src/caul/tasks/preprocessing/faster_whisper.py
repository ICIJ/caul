from typing import ClassVar

from pydantic import Field

from caul.config import PreprocessorConfig
from caul.objects import ASRModel
from caul.tasks.asr_task import Preprocessor
from caul.tasks.preprocessing.asr_preprocessor import ASRPreprocessor


class FasterWhisperPreprocessorConfig(PreprocessorConfig):
    model: ClassVar[str] = Field(default=ASRModel.FASTER_WHISPER)


@Preprocessor.register(ASRModel.FASTER_WHISPER)
class FasterWhisperPreprocessor(ASRPreprocessor): ...
