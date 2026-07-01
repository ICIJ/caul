import logging
from pathlib import Path

from caul_core import ASRModel

from .tasks import (
    FasterWhisperInferenceRunner,
    FireRedASR2InferenceRunner,
    ParakeetInferenceRunner,
)
from .tasks.preprocessing.asr_preprocessor import ASRPreprocessorMixin


logger = logging.getLogger(__name__)


def cache_models(asr_model: ASRModel | None, cache_dir: Path) -> None:
    if asr_model is None:
        logger.info("caching all models to %s", cache_dir)
    else:
        logger.info("caching %s models to %s", asr_model, cache_dir)
    match asr_model:
        case ASRModel.PARAKEET:
            cache_fns = [
                ASRPreprocessorMixin.cache_models,
                ParakeetInferenceRunner.cache_models,
            ]
        case ASRModel.PARAKEET_TRT:
            cache_fns = [
                ASRPreprocessorMixin.cache_models,
            ]
        case ASRModel.FASTER_WHISPER:
            cache_fns = [
                ASRPreprocessorMixin.cache_models,
                FasterWhisperInferenceRunner.cache_models,
            ]
        case ASRModel.FIREREDASR2_AED:
            cache_fns = [
                ASRPreprocessorMixin.cache_models,
                FireRedASR2InferenceRunner.cache_models,
            ]
        case None:
            cache_fns = [
                ASRPreprocessorMixin.cache_models,
                ParakeetInferenceRunner.cache_models,
                FasterWhisperInferenceRunner.cache_models,
                FireRedASR2InferenceRunner.cache_models,
            ]
        case _:
            raise ValueError(f"invalid model {asr_model}")
    for cache_fn in cache_fns:
        cache_fn(cache_dir)
