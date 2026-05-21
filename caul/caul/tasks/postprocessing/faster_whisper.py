from caul_core.config import (
    FasterWhisperPostprocessorConfig,
)  # pylint: disable=unused-import
from caul_core.objects import ASRModel
from caul.tasks.asr_task import Postprocessor
from .asr_postprocessor import ASRPostprocessor


@Postprocessor.register(ASRModel.FASTER_WHISPER)
class FasterWhisperPostprocessor(ASRPostprocessor): ...
