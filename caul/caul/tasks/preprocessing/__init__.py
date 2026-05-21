try:
    from .parakeet import ParakeetPreprocessor
except ModuleNotFoundError:
    ParakeetPreprocessor = None
try:
    from .fireredasr import FireRedASR2Preprocessor
except ModuleNotFoundError:
    FireRedASR2Preprocessor = None
try:
    from .faster_whisper import (
        FasterWhisperPreprocessor,
    )
except ModuleNotFoundError:
    FasterWhisperPreprocessor = None
