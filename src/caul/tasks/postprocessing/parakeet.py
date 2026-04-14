from typing import ClassVar

from icij_common.registrable import FromConfig
from pydantic import Field

from caul.config import PostprocessorConfig
from caul.objects import ASRModel
from caul.tasks.asr_task import Postprocessor
from caul.tasks.postprocessing.asr_postprocessor import ASRPostprocessor


class ParakeetPostprocessorConfig(PostprocessorConfig):
    model: ClassVar[str] = Field(frozen=True, default=ASRModel.PARAKEET)


@Postprocessor.register(ASRModel.PARAKEET)
class ParakeetPostprocessor(ASRPostprocessor): ...
