from caul.tasks.asr_task import Postprocessor
from caul.tasks.postprocessing.asr_postprocessor import ASRPostprocessor
from caul_core.objects import ASRModel


@Postprocessor.register(ASRModel.WHISPER_TRT)
class WhisperTrtPostprocessor(ASRPostprocessor): ...
