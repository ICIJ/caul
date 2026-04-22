from typing import Callable, Iterable

from icij_common.registrable import FromConfig

from caul.task_defaults import generic_unbatching_fn
from caul.config import PostprocessorConfig
from caul.objects import ASRResult
from caul.tasks.asr_task import Postprocessor


class ASRPostprocessor(Postprocessor):
    """Postprocessing logic"""

    def __init__(self, unbatching_fn: Callable = generic_unbatching_fn):
        self._unbatching_fn = unbatching_fn

    @classmethod
    def _from_config(cls, config: PostprocessorConfig, **extras) -> FromConfig:
        return cls(**extras)

    def process(
        self, inputs: Iterable[ASRResult], *args, **kwargs
    ) -> Iterable[ASRResult]:
        """Unbatch batched inference results

        :param inputs: List of model results
        :return: list of model results in input ordering
        """
        yield from self._unbatching_fn(inputs)
