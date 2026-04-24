try:
    from .parakeet import ParakeetPreprocessor, ParakeetPreprocessorConfig
except ModuleNotFoundError:
    ParakeetPreprocessor, ParakeetPreprocessorConfig = (
        None,
        None,
    )  # pylint: disable=invalid-name
try:
    from .fireredasr import FireRedASR2Preprocessor, FireRedASR2PreprocessorConfig
except ModuleNotFoundError:
    FireRedASR2Preprocessor, FireRedASR2PreprocessorConfig = (
        None,
        None,
    )  # pylint: disable=invalid-name
try:
    from .faster_whisper import (
        FasterWhisperPreprocessor,
        FasterWhisperPreprocessorConfig,
    )
except ModuleNotFoundError:
    FasterWhisperPreprocessor, FasterWhisperPreprocessorConfig = None, None
