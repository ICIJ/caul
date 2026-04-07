import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Self, TYPE_CHECKING

import uuid

from icij_common.pydantic_utils import icij_config, merge_configs, no_enum_values_config
from pydantic import BaseModel as _BaseModel, Field


if TYPE_CHECKING:
    import torch


class BaseModel(_BaseModel):
    model_config = merge_configs(icij_config(), no_enum_values_config())


class ASRResult(BaseModel):
    """Base result class for ASR models"""

    input_ordering: int = -1
    transcription: list[tuple] = Field(default_factory=list)
    score: float = 1.0

    @property
    def duration(self) -> float:
        if not self.transcription:
            return 0.0
        total = sum(end - start for start, end, _ in self.transcription)
        return total

    @classmethod
    def from_parakeet_hypothesis(cls, hypothesis: "Hypothesis", **extra) -> Self:
        """Parse a hypothesis returned by a Parakeet RNN model

        :param hypothesis: Parakeet hypothesis
        :return: copy of self
        """

        # there's some weird inconsistency here between nemo versions
        timestamps = hypothesis.timestamp
        transcription = [
            (s["start"], s["end"], s["segment"]) for s in timestamps["segment"]
        ]
        score = round(hypothesis.score, 2)
        return cls(transcription=transcription, score=score, **extra)

    def __add__(self, other: Self) -> Self:
        if not isinstance(other, ASRResult):
            msg = f"expected {ASRResult.__class__.__class__} but found: {type(other)}"
            raise TypeError(msg)

        if other.input_ordering != self.input_ordering:
            raise ValueError("can't merge transcriptions from different inputs")

        transcription = self.transcription + other.transcription
        # We have to weight by total segment len
        total_duration = self.duration + other.duration
        score = 1.0
        if total_duration:
            score = self.score * self.duration + other.score * other.duration
            score /= total_duration
        return ASRResult(
            input_ordering=self.input_ordering, transcription=transcription, score=score
        )


def _utc_now() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0)


def _uuid() -> str:
    return uuid.uuid4().hex


class InputMetadata(BaseModel):
    """Preprocessed input metadata"""

    duration_s: float
    input_ordering: int = -1
    preprocessed_at: datetime.datetime = Field(default_factory=_utc_now)
    uuid: str = Field(default_factory=_uuid)
    input_format: str | None = None
    input_file_path: Path | None = None
    preprocessed_file_path: Path | None = None


class PreprocessedInput(BaseModel):
    metadata: InputMetadata


@dataclass(frozen=True)
class PreprocessedInputWithTensor:
    # Avoid importing torch when importing objects

    metadata: InputMetadata
    tensor: "torch.Tensor | list | None" = None


PreprocessorOutput = PreprocessedInput | PreprocessedInputWithTensor
