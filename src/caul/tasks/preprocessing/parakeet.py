from typing import ClassVar, Callable, Iterable, Self

from pydantic import Field

from caul.config import PreprocessorConfig
from caul.constants import (
    PARAKEET_INFERENCE_MAX_DURATION_S,
    PARAKEET_INFERENCE_MAX_FRAMES,
    DEFAULT_SAMPLE_RATE,
)
from caul.objects import ASRModel
from caul.objects import PreprocessedInput
from caul.tasks.asr_task import Preprocessor
from caul.tasks.preprocessing.asr_preprocessor import ASRPreprocessor


# TODO: something approximate here would be nice to avoid loading all data in memory
def _parakeet_batching_fn(  # pylint: disable=R0914
    preprocessed_inputs: Iterable[PreprocessedInput], *args, **kwargs
) -> Iterable[list[PreprocessedInput]]:
    """Batch audio tensors by duration, 20 minutes max per batch, optimizing for tightly packed
    batches.

    :param preprocessed_inputs: list of PreprocessedInput
    :return: list of list[PreprocessedInput]
    """
    import numpy as np  # pylint: disable=import-outside-toplevel

    # Sort by duration
    preprocessed_inputs = sorted(
        preprocessed_inputs, key=lambda p: p.metadata.duration_s, reverse=True
    )

    # Now this becomes a bin-packing minimization problem. We'll use a variant of best-fit
    # decreasing.

    bins = [[]]
    bins_len = [0]

    # With each pass, choose a bin by maximizing remaining space
    for preprocessed_input in preprocessed_inputs:
        remaining_spaces = [
            PARAKEET_INFERENCE_MAX_DURATION_S - bin_len for bin_len in bins_len
        ]
        input_duration_s = preprocessed_input.metadata.duration_s
        if input_duration_s > max(remaining_spaces):
            bins.append([])
            bins_len.append(0)
            remaining_spaces.append(PARAKEET_INFERENCE_MAX_DURATION_S)
        most_empty_bin = np.argmax(remaining_spaces)
        bins[most_empty_bin].append(preprocessed_input)
        bins_len[most_empty_bin] += input_duration_s

    yield from bins


class ParakeetPreprocessorConfig(PreprocessorConfig):
    model: ClassVar[str] = Field(default=ASRModel.PARAKEET)

    max_frames: int = PARAKEET_INFERENCE_MAX_FRAMES


@Preprocessor.register(ASRModel.PARAKEET)
class ParakeetPreprocessor(ASRPreprocessor):
    def __init__(
        self,
        batching_fn: Callable = _parakeet_batching_fn,
        max_frames: int = PARAKEET_INFERENCE_MAX_FRAMES,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
    ):
        super().__init__(
            batching_fn=batching_fn,
            max_frames=max_frames,
            sample_rate=sample_rate,
        )

    @classmethod
    def _from_config(cls, config: PreprocessorConfig, **extras) -> Self:
        return cls(
            max_frames=config.max_frames,
            sample_rate=config.sample_rate,
        )
