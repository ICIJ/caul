# We import everything top level to trigger component registration
from .tasks import (
    ParakeetInferenceRunner,
    ParakeetTrtInferenceRunner,
    WhisperTrtInferenceRunner,
    FasterWhisperInferenceRunner,
    FireRedASR2InferenceRunner,
    ParakeetPostprocessor,
    FasterWhisperPostprocessor,
    FireRedASR2Postprocessor,
    WhisperTrtPostprocessor,
    ParakeetPreprocessor,
    FasterWhisperPreprocessor,
    FireRedASR2Preprocessor,
    WhisperTrtPreprocessor,
)
