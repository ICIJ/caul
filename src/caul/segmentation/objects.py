from dataclasses import dataclass
from enum import StrEnum
from typing import ClassVar, TYPE_CHECKING

from icij_common.registrable import RegistrableConfig
from pydantic import Field

from caul.constant import (
    DEFAULT_SAMPLE_RATE,
    PARAKEET_INFERENCE_MAX_DURATION_S
)

if TYPE_CHECKING:
    import torch


@dataclass(frozen=True)
class TensorSegment:
    """An audio tensor segment with position and duration metadata"""
    tensor: "torch.Tensor"
    segment_start: int
    segment_end: int
    # Unique identifier linking segments to source tensor
    tensor_id: str
    sample_rate: int = DEFAULT_SAMPLE_RATE

    @property
    def duration(self) -> float:
        """Duration of segment in seconds"""
        return (self.segment_end - self.segment_start) / self.sample_rate


class SegmentationStrategy(StrEnum):
    FIXED = "fixed"
    SILENCE = "silence"
    VOICE = "voice"


class SegmentationConfig(RegistrableConfig):
    registry_key: ClassVar[str] = Field(frozen=True, default="strategy")
    strategy: ClassVar[SegmentationStrategy]
    # TODO: parakeet specific variables shouldn't pop in here, rename or define global
    #  naming
    max_segment_len_s: float = PARAKEET_INFERENCE_MAX_DURATION_S


class FixedSegmentationConfig(SegmentationConfig):
    strategy: ClassVar[SegmentationStrategy] = Field(
        frozen=True, default=SegmentationStrategy.FIXED
    )



class SilenceSegmentationConfig(SegmentationConfig):
    """Silence segmentation parameters; see librosa for explanation of parameters"""

    strategy: ClassVar[SegmentationStrategy] = Field(
        frozen=True, default=SegmentationStrategy.SILENCE
    )
    frame_len: int = 2048
    silence_thresh_db: int = 35
    hop_len: int = 512
    kept_silence_len_s: float = 0.15
    min_silence_len_s: float = 0.5


class SileroVoiceSegmentationConfig(SegmentationConfig):
    """Voice segmentation parameters using silero VAD;
    see silero for explanation of parameters"""

    strategy: ClassVar[SegmentationStrategy] = Field(
        frozen=True, default=SegmentationStrategy.VOICE
    )
    threshold: float = 0.5
    min_speech_duration_ms: int = 250
    min_silence_duration_ms: int = 100
    speech_pad_ms: int = 30
