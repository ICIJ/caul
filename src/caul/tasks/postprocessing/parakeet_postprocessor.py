from functools import reduce
from itertools import groupby

from caul.tasks.asr_task import ASRTask
from caul.tasks.inference.parakeet_inference import ParakeetModelHandlerResult


class ParakeetPostprocessor(ASRTask):
    """Postprocessing logic for ParakeetInferenceHandler output"""

    def process(
        self, inputs: list[ParakeetModelHandlerResult]
    ) -> list[ParakeetModelHandlerResult]:
        """Process indexed ParakeetInferenceHandler results and return them in their original
        ordering

        :param inputs: List of parakeet model results
        :return: list of parakeet model results in input ordering
        """

        return self.map_results_to_inputs(inputs)

    @staticmethod
    def map_results_to_inputs(
        batched_results: list[ParakeetModelHandlerResult],
    ) -> list[ParakeetModelHandlerResult]:
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

        for _, group_results in results_grouped_by_index:
            group_results = list(group_results)

            merged_results = (
                reduce(lambda l, r: l.concat(r), group_results)
                if len(group_results) > 1
                else group_results[0]
            )
            unbatched_results.append(merged_results)

        # TODO: Drop index from result

        return unbatched_results
