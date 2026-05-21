from caul_core.constants import (
    FIREREDASR2_INFERENCE_MAX_FRAMES,
    DEFAULT_BATCH_SIZE,
    DEFAULT_SAMPLE_RATE,
    DEFAULT_LARGE_FILE_THRESHOLD_BYTES,
)
from caul_core.objects import ASRModel
from caul.tasks.asr_task import Preprocessor
from .asr_preprocessor import ASRPreprocessor


@Preprocessor.register(ASRModel.FIREREDASR2_AED)
class FireRedASR2Preprocessor(ASRPreprocessor):
    def __init__(
        self,
        max_frames: int = FIREREDASR2_INFERENCE_MAX_FRAMES,
        batch_size: int = DEFAULT_BATCH_SIZE,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        large_file_threshold_bytes: int = DEFAULT_LARGE_FILE_THRESHOLD_BYTES,
    ):
        super().__init__(
            max_frames=max_frames,
            batch_size=batch_size,
            sample_rate=sample_rate,
            large_file_threshold_bytes=large_file_threshold_bytes,
        )
