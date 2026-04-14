from itertools import groupby
from typing import Iterable

from caul.constants import DEFAULT_BATCH_SIZE
from caul.objects import ASRResult, PreprocessorOutput


def _generic_batching_fn(
    preprocessed_inputs: Iterable[PreprocessorOutput],
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> Iterable[list[PreprocessorOutput]]:
    """Group preprocessed inputs into fixed-size batches.

    :param preprocessed_inputs: stream of preprocessed inputs
    :param batch_size: maximum number of segments per batch
    :return: batches of preprocessed inputs
    """
    batch: list[PreprocessorOutput] = []
    for inp in preprocessed_inputs:
        batch.append(inp)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _generic_unbatching_fn(batched_results: Iterable[ASRResult]) -> Iterable[ASRResult]:
    """Remap unordered and segmented tensors to original inputs for return

    :param batched_results: list of unordered results
    :return: list[ParakeetModelHandlerResult]
    """
    seen: set[int] = set()
    results_grouped_by_index = groupby(batched_results, key=lambda r: r.input_ordering)
    for input_ordering, group_results in results_grouped_by_index:
        # Drop segments with no recognized speech
        group_results = [r for r in group_results if r.transcription]
        group_results = sorted(group_results, key=lambda r: r.transcription[0])
        if input_ordering in seen:
            raise ValueError("expected contiguous batches !")
        seen.add(input_ordering)
        base = ASRResult(input_ordering=input_ordering, transcription=[], score=1.0)
        merged_results = sum(group_results, base)
        yield merged_results
