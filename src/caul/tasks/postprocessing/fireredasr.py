from typing import ClassVar

from pydantic import Field

from caul.config import PostprocessorConfig
from caul.objects import ASRModel
from caul.tasks.asr_task import Postprocessor
from caul.tasks.postprocessing.asr_postprocessor import ASRPostprocessor


class FireRedASR2PostprocessorConfig(PostprocessorConfig):
    model: ClassVar[str] = Field(frozen=True, default=ASRModel.FIREREDASR2_AED)


@Postprocessor.register(ASRModel.FIREREDASR2_AED)
class FireRedASR2Postprocessor(ASRPostprocessor): ...
