from enum import StrEnum

import torch
from pydantic import BaseModel, Field, computed_field

from caul.constant import (
    EXPECTED_SAMPLE_RATE,
    FIXED_SEGMENT_DEFAULT_LENGTH_SECS,
    PARAKEET_INFERENCE_MAX_DURATION_SECS,
)


class TensorSegment(BaseModel):
    """An audio tensor segment with position and duration metadata"""

    model_config = {"arbitrary_types_allowed": True}

    tensor: torch.Tensor
    segment_start: int
    segment_end: int
    sample_rate: int = EXPECTED_SAMPLE_RATE
    tensor_id: str = Field(
        description="Unique identifier linking segments to source tensor"
    )

    @computed_field
    @property
    def duration(self) -> float:
        """Duration of segment in seconds"""
        return (self.segment_end - self.segment_start) / self.sample_rate


class SegmentationStrategyEnum(StrEnum):
    FIXED = "fixed"
    SILENCE = "silence"
    VOICE = "voice"


class SegmentationConfig(BaseModel):
    segmentation_strategy: SegmentationStrategyEnum = SegmentationStrategyEnum.FIXED
    sample_rate: int = EXPECTED_SAMPLE_RATE


class FixedSegmentationConfig(SegmentationConfig):
    segmentation_strategy: SegmentationStrategyEnum = SegmentationStrategyEnum.FIXED
    segment_duration_secs: float = FIXED_SEGMENT_DEFAULT_LENGTH_SECS


class SilenceSegmentationConfig(SegmentationConfig):
    """Silence segmentation parameters; see librosa for explanation of parameters"""

    segmentation_strategy: SegmentationStrategyEnum = SegmentationStrategyEnum.SILENCE
    frame_len: int = 2048
    silence_thresh_db: int = 35
    hop_len: int = 512
    kept_silence_len_secs: float = 0.15
    min_silence_len_secs: float = 0.5
    max_segment_len_secs: float = PARAKEET_INFERENCE_MAX_DURATION_SECS


class SileroVoiceSegmentationConfig(SegmentationConfig):
    """Voice segmentation parameters using silero VAD;
    see silero for explanation of parameters"""

    segmentation_strategy: SegmentationStrategyEnum = SegmentationStrategyEnum.VOICE
    threshold: float = 0.5
    min_speech_duration_ms: int = 250
    min_silence_duration_ms: int = 100
    speech_pad_ms: int = 30
