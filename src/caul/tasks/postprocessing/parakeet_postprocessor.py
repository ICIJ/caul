from functools import reduce
from itertools import groupby

from caul.tasks.asr_task import ASRTask
from caul.tasks.inference.parakeet_inference import ParakeetInferenceHandlerResult


class ParakeetPostprocessor(ASRTask):
    """Postprocessing logic for ParakeetInferenceHandler output"""

    def process(
        self, inputs: list[ParakeetInferenceHandlerResult]
    ) -> list[ParakeetInferenceHandlerResult]:
        """Process indexed ParakeetInferenceHandler results and return them in their original
        ordering

        :param inputs: List of parakeet model results
        :return: list of parakeet model results in input ordering
        """

        return self.map_results_to_inputs(inputs)

    @staticmethod
    def map_results_to_inputs(
        batched_results: list[ParakeetInferenceHandlerResult],
    ) -> list[ParakeetInferenceHandlerResult]:
        """Remap unordered and segmented tensors to original inputs for return

        :param batched_results: list of unordered ParakeetInferenceHandlerResult, still
        segmented
        :return: list[ParakeetInferenceHandlerResult]
        """
        unbatched_results = []

        # Sort in order before batching
        batched_results = sorted(batched_results, key=lambda r: r.input_ordering)

        # Concat segmented tensors
        results_grouped_by_index = groupby(
            batched_results, key=lambda r: r.input_ordering
        )

        for _, group_results in results_grouped_by_index:
            group_results = list(group_results)

            merged_results = (
                reduce(lambda l, r: l.concat(r), group_results)
                if len(group_results) > 1
                else group_results[0]
            )
            unbatched_results.append(merged_results)

        return unbatched_results
