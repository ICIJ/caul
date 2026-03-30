try:
    from .parakeet import ParakeetPostprocessor, ParakeetPostprocessorConfig
except ModuleNotFoundError:
    ParakeetPostprocessorConfig, ParakeetPostprocessor = None, None
