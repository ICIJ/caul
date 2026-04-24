try:
    from .parakeet import ParakeetInferenceRunner, ParakeetInferenceRunnerConfig
except ModuleNotFoundError:
    ParakeetInferenceRunner, ParakeetInferenceRunnerConfig = None, None
try:
    from .faster_whisper import (
        FasterWhisperInferenceRunner,
        FasterWhisperInferenceRunnerConfig,
    )

    WhisperCppInferenceRunner = FasterWhisperInferenceRunner
    WhisperCppInferenceRunnerConfig = FasterWhisperInferenceRunnerConfig
except ModuleNotFoundError:
    FasterWhisperInferenceRunner, FasterWhisperInferenceRunnerConfig = None, None
    WhisperCppInferenceRunner, WhisperCppInferenceRunnerConfig = None, None
try:
    from .fireredasr import FireRedASR2InferenceRunner, FireRedASR2InferenceRunnerConfig
except ModuleNotFoundError:
    FireRedASR2InferenceRunner, FireRedASR2InferenceRunnerConfig = None, None
