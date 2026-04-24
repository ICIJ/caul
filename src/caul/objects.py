import datetime
import math
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Self, TYPE_CHECKING

import uuid

import langcodes
from icij_common.pydantic_utils import icij_config, merge_configs, no_enum_values_config
from pydantic import BaseModel as _BaseModel, Field, GetCoreSchemaHandler, TypeAdapter
from pydantic_core import core_schema
from pydantic_extra_types.language_code import LanguageAlpha2

from caul.constants import PARAKEET_TDT_0_6B_V3_LANGUAGES, FIREREDASR2_LANGUAGES

if TYPE_CHECKING:
    import torch


class BaseModel(_BaseModel):
    model_config = merge_configs(icij_config(), no_enum_values_config())


class IETFLanguage(str):
    @classmethod
    def __get_pydantic_core_schema__(cls, source, handler: GetCoreSchemaHandler):
        return core_schema.no_info_plain_validator_function(cls.validate)

    @classmethod
    def validate(cls, v):
        tag = langcodes.get(str(v))
        if not tag.is_valid():
            raise ValueError(f"Invalid IETF language: {v}")
        return cls(v)


ASRLanguage = IETFLanguage | LanguageAlpha2
_LANGUAGE_TYPE_ADAPTER = TypeAdapter(ASRLanguage)

_VALIDATED_PARAKEET_LANGUAGES = {
    _LANGUAGE_TYPE_ADAPTER.validate_python(lang)
    for lang in PARAKEET_TDT_0_6B_V3_LANGUAGES
}

_VALIDATED_FIREREDASR2_LANGUAGES = {
    _LANGUAGE_TYPE_ADAPTER.validate_python(lang) for lang in FIREREDASR2_LANGUAGES
}


class ASRModel(StrEnum):
    PARAKEET = "parakeet"
    FASTER_WHISPER = "whisper_cpp"
    FIREREDASR2_AED = "fireredasr2_aed"

    def supported_languages(self) -> set[ASRLanguage]:
        match self:
            case ASRModel.PARAKEET:
                return _VALIDATED_PARAKEET_LANGUAGES
            case ASRModel.FASTER_WHISPER:
                return set()
            case ASRModel.FIREREDASR2_AED:
                return _VALIDATED_FIREREDASR2_LANGUAGES
            case _:
                msg = f"model {self} should expose supported languages"
                raise NotImplementedError(msg)


class VadModel(StrEnum):
    SILERO_MODEL = "silero_vad"
    PYANNOTE_MODEL = "pyannote/voice-activity-detection"


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
        return cls(transcription=transcription, score=hypothesis.score, **extra)

    @classmethod
    def from_fireredasr2_result(cls, result: dict, **extra) -> Self:
        """Parse a result dict returned by a FireRedASR2 AED model

        :param result: dict with keys 'text', 'confidence', 'dur_s', 'timestamp'
        :return: ASRResult
        """
        text = result.get("text") or ""
        timestamp = result.get("timestamp") or []
        if timestamp:
            start_s = float(timestamp[0][1])
            end_s = float(timestamp[-1][2])
        else:
            start_s = 0.0
            end_s = float(result.get("dur_s") or 0.0)
        transcription = [(start_s, end_s, text)] if text.strip() else []
        confidence = result.get("confidence")
        score = float(confidence) if confidence is not None else -1.0
        return cls(transcription=transcription, score=score, **extra)

    @classmethod
    def from_faster_whisper_result(cls, segments, **extra) -> Self:
        """Parse segments returned by a faster-whisper model

        :param segments: iterable of Segment objects with start, end, text, avg_logprob
        :return: ASRResult
        """
        transcription = []
        weighted_logprob = 0.0
        total_duration = 0.0
        for segment in segments:
            duration = segment.end - segment.start
            transcription.append((segment.start, segment.end, segment.text))
            weighted_logprob += segment.avg_logprob * duration
            total_duration += duration
        score = math.exp(weighted_logprob / total_duration) if total_duration else -1.0
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
