from caul_core.objects import ASRModel
from caul.tasks.asr_task import Postprocessor
from .asr_postprocessor import ASRPostprocessor


@Postprocessor.register(ASRModel.PARAKEET)
class ParakeetPostprocessor(ASRPostprocessor): ...
