from functools import reduce
from itertools import groupby

from caul.inference.parakeet_inference import ParakeetInferenceHandlerResult
from caul.postprocessing.asr_postprocessor import ASRPostprocessor


class ParakeetPostprocessor(ASRPostprocessor):
    """Postprocessing logic for ParakeetInferenceHandler output"""

    def process(
        self, inference_result: list[tuple[int, ParakeetInferenceHandlerResult]]
    ) -> list[ParakeetInferenceHandlerResult]:
        """Process indexed ParakeetInferenceHandler results and return them in their original
        ordering

        :param inference_result: List of indexed parakeet model results of form (input_idx, result)
        :return: list of parakeet model results in input ordering
        """

        return self.map_results_to_inputs(inference_result)

    @staticmethod
    def map_results_to_inputs(
        batched_results: list[tuple[int, ParakeetInferenceHandlerResult]],
    ) -> list[ParakeetInferenceHandlerResult]:
        """Remap unordered and segmented tensors to original inputs for return

        :param batched_results: list of unordered indexed ParakeetInferenceHandlerResult, still
        segmented
        :return: list[ParakeetInferenceHandlerResult]
        """
        unbatched_results = []

        # Sort in order received before batching
        batched_results = sorted(batched_results, key=lambda x: x[0])

        # Concat segmented tensors
        results_grouped_by_index = groupby(batched_results, key=lambda x: x[0])

        for _, group_results in results_grouped_by_index:
            group_results = [gr[1] for gr in list(group_results)]  # drop index
            merged_results = (
                reduce(lambda l, r: l.concat(r), group_results)
                if len(group_results) > 1
                else group_results[0]
            )
            unbatched_results.append(merged_results)

        return unbatched_results
