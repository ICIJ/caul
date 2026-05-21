from caul_core.objects import ASRModel
from caul.tasks.asr_task import Preprocessor
from .asr_preprocessor import ASRPreprocessor


@Preprocessor.register(ASRModel.FASTER_WHISPER)
class FasterWhisperPreprocessor(ASRPreprocessor): ...
