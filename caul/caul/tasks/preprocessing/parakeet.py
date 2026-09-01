from typing import Callable, Iterable, Self

from caul_core import (
    DEFAULT_LARGE_FILE_THRESHOLD_BYTES,
    DEFAULT_SAMPLE_RATE,
    PARAKEET_INFERENCE_MAX_DURATION_S,
    PARAKEET_INFERENCE_MAX_FRAMES,
    ASRModel,
    ParakeetPreprocessorConfig,
    PreprocessedInput,
    Preprocessor,
)

from .asr_preprocessor import ASRPreprocessorMixin


def _parakeet_batching_fn(
    preprocessed_inputs: Iterable[PreprocessedInput], *args, **kwargs
) -> Iterable[list[PreprocessedInput]]:
    """Batch audio tensors by duration, 20 minutes max per batch, preserving input ordering
    and streaming batches as they fill up rather than buffering every input in memory.

    :param preprocessed_inputs: iterable of PreprocessedInput
    :return: iterable of list[PreprocessedInput]
    """
    current_batch: list[PreprocessedInput] = []
    current_batch_duration_s = 0.0

    for preprocessed_input in preprocessed_inputs:
        input_duration_s = preprocessed_input.metadata.duration_s
        if (
            current_batch
            and current_batch_duration_s + input_duration_s
            > PARAKEET_INFERENCE_MAX_DURATION_S
        ):
            yield current_batch
            current_batch = []
            current_batch_duration_s = 0.0
        current_batch.append(preprocessed_input)
        current_batch_duration_s += input_duration_s

    if current_batch:
        yield current_batch


@Preprocessor.register(ASRModel.PARAKEET)
class ParakeetPreprocessor(ASRPreprocessorMixin):
    def __init__(
        self,
        batching_fn: Callable = _parakeet_batching_fn,
        max_frames: int = PARAKEET_INFERENCE_MAX_FRAMES,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        large_file_threshold_bytes: int = DEFAULT_LARGE_FILE_THRESHOLD_BYTES,
    ):
        super().__init__(
            batching_fn=batching_fn,
            max_frames=max_frames,
            sample_rate=sample_rate,
            large_file_threshold_bytes=large_file_threshold_bytes,
        )

    @classmethod
    def _from_config(cls, config: ParakeetPreprocessorConfig, **extras) -> Self:
        return cls(
            max_frames=config.max_frames,
            sample_rate=config.sample_rate,
            large_file_threshold_bytes=config.large_file_threshold_bytes,
        )
