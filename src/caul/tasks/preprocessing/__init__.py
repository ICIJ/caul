try:
    from .parakeet import ParakeetPreprocessor, ParakeetPreprocessorConfig
except ModuleNotFoundError:
    ParakeetPreprocessor, ParakeetPreprocessorConfig = None, None  # pylint: disable=invalid-name
