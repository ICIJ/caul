try:
    from .parakeet import ParakeetPostprocessor, ParakeetPostprocessorConfig
except ModuleNotFoundError:
    ParakeetPostprocessor, ParakeetPostprocessorConfig = (
        None,
        None,
    )  # pylint: disable=invalid-name
try:
    from .fireredasr import FireRedASR2Postprocessor, FireRedASR2PostprocessorConfig
except ModuleNotFoundError:
    FireRedASR2Postprocessor, FireRedASR2PostprocessorConfig = None, None
try:
    from .faster_whisper import (
        FasterWhisperPostprocessor,
        FasterWhisperPostprocessorConfig,
    )
except ModuleNotFoundError:
    FasterWhisperPostprocessor, FasterWhisperPostprocessorConfig = None, None
