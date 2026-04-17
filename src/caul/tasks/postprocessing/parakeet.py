from itertools import groupby
from typing import ClassVar, Iterable

from icij_common.registrable import FromConfig
from pydantic import Field

from caul.config import PostprocessorConfig
from caul.objects import ASRModel, ASRResult
from caul.tasks.asr_task import Postprocessor


class ParakeetPostprocessorConfig(PostprocessorConfig):
    model: ClassVar[str] = Field(frozen=True, default=ASRModel.PARAKEET)


@Postprocessor.register(ASRModel.PARAKEET)
class ParakeetPostprocessor(Postprocessor):
    """Postprocessing logic for ParakeetInferenceHandler output"""

    @classmethod
    def _from_config(cls, config: ParakeetPostprocessorConfig, **extras) -> FromConfig:
        return cls(**extras)

    def process(
        self, inputs: Iterable[ASRResult], *args, **kwargs
    ) -> Iterable[ASRResult]:
        """Process indexed ParakeetInferenceHandler results and return them in their original
        ordering

        :param inputs: List of parakeet model results
        :return: list of parakeet model results in input ordering
        """
        yield from _map_results_to_inputs(inputs)


def _map_results_to_inputs(batched_results: Iterable[ASRResult]) -> Iterable[ASRResult]:
    """Remap unordered and segmented tensors to original inputs for return

    :param batched_results: list of unordered ParakeetModelHandlerResult, still
    segmented
    :return: list[ParakeetModelHandlerResult]
    """
    # Concat segmented tensors
    seen = set()
    results_grouped_by_index = groupby(batched_results, key=lambda r: r.input_ordering)
    for input_ordering, group_results in results_grouped_by_index:
        group_results = sorted(group_results, key=lambda r: r.transcription[0])
        if input_ordering in seen:
            raise ValueError("expected contiguous batches !")
        seen.add(input_ordering)
        base = ASRResult(input_ordering=input_ordering, transcription=[], score=1.0)
        merged_results = sum(group_results, base)
        yield merged_results
