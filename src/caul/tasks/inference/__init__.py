try:
    from .parakeet import ParakeetInferenceRunner, ParakeetInferenceRunnerConfig
except ModuleNotFoundError:
    ParakeetInferenceRunner, ParakeetInferenceRunnerConfig = None, None
try:
    from .whisper_cpp import WhisperCppInferenceRunner, WhisperCppInferenceRunnerConfig
except ModuleNotFoundError:
    WhisperCppInferenceRunner, ParakeetInferenceRunnerConfig = None, None
