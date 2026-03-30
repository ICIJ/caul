from itertools import groupby
from typing import ClassVar

from icij_common.registrable import FromConfig
from pydantic import Field

from caul.config import PostprocessorConfig
from caul.constant import ASRModel
from caul.objects import ASRResult
from caul.tasks.asr_task import Postprocessor


class ParakeetPostprocessorConfig(PostprocessorConfig):
    model: ClassVar[str] = Field(frozen=True, default=ASRModel.PARAKEET)


@Postprocessor.register(ASRModel.PARAKEET)
class ParakeetPostprocessor(Postprocessor):
    """Postprocessing logic for ParakeetInferenceHandler output"""

    @classmethod
    def _from_config(cls, config: ParakeetPostprocessorConfig, **extras) -> FromConfig:
        return cls(**extras)

    def process(self, inputs: list[ASRResult], *args, **kwargs) -> list[ASRResult]:
        """Process indexed ParakeetInferenceHandler results and return them in their original
        ordering

        :param inputs: List of parakeet model results
        :return: list of parakeet model results in input ordering
        """

        return self.map_results_to_inputs(inputs)

    @staticmethod
    def map_results_to_inputs(
        batched_results: list[ASRResult],
    ) -> list[ASRResult]:
        """Remap unordered and segmented tensors to original inputs for return

        :param batched_results: list of unordered ParakeetModelHandlerResult, still
        segmented
        :return: list[ParakeetModelHandlerResult]
        """
        unbatched_results = []

        # Sort in order before batching
        batched_results = sorted(batched_results, key=lambda r: r.input_ordering)

        # Concat segmented tensors
        results_grouped_by_index = groupby(
            batched_results, key=lambda r: r.input_ordering
        )

        for input_ordering, group_results in results_grouped_by_index:
            base = ASRResult(input_ordering=input_ordering, transcription=[], score=1.0)
            group_results = list(group_results)
            merged_results = sum(group_results, base)
            unbatched_results.append(merged_results)

        # TODO: Drop index from result

        return unbatched_results
