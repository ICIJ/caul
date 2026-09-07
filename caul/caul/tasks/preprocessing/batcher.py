from collections.abc import Iterable
from typing import cast

from caul_core import (
    Batcher,
    BatcherType,
    ConstantSizeBatcherConfig,
    Error,
    MaxDurationBatcherConfig,
    ProcessedAudioSegment,
)


@Batcher.register(BatcherType.CONSTANT_SIZE)
class ConstantSizeBatcher(Batcher):
    def __init__(
        self,
        items: Iterable[ProcessedAudioSegment],
        config: ConstantSizeBatcherConfig | None = None,
    ):
        if config is None:
            config = ConstantSizeBatcherConfig()
        super().__init__(items, config)

    def results(self) -> Iterable[tuple[ProcessedAudioSegment, ...]]:
        errors = []
        batch: list[ProcessedAudioSegment] = []
        for item in self._items:
            if isinstance(item, Error):
                errors.append(item)
                continue
            item = cast(ProcessedAudioSegment, item)
            batch.append(item)
            if len(batch) >= self._config.batch_size:
                yield tuple(batch)
                batch = []
        if batch:
            yield tuple(batch)


@Batcher.register(BatcherType.MAX_DURATION)
class MaxDurationBatcher(Batcher):
    def __init__(
        self,
        items: Iterable[ProcessedAudioSegment],
        config: MaxDurationBatcherConfig | None = None,
    ):
        if config is None:
            config = MaxDurationBatcherConfig()
        super().__init__(items, config)

    def results(self) -> Iterable[tuple[ProcessedAudioSegment, ...]]:
        current_batch: list[ProcessedAudioSegment] = []
        current_batch_duration_s = 0.0

        for item in self._items:
            if isinstance(item, Error):
                self._errors.append(item)
                continue
            input_duration_s = item.metadata.duration_s
            if (
                current_batch
                and current_batch_duration_s + input_duration_s
                > self._config.max_duration_s
            ):
                yield tuple(current_batch)
                current_batch = []
                current_batch_duration_s = 0.0
            current_batch.append(item)
            current_batch_duration_s += input_duration_s

        if current_batch:
            yield tuple(current_batch)
