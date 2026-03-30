from typing import Callable

import torch

from caul.constant import SILERO_TORCH_HUB_REPO, SILERO_MODEL
from caul.segmentation.methods import (
    segment_by_silence,
    segment_fixed,
    segment_by_silero_vad,
)
from caul.segmentation.objects import (
    TensorSegment,
    SegmentationConfig,
    FixedSegmentationConfig,
    SegmentationStrategyEnum,
    SilenceSegmentationConfig,
    SileroVoiceSegmentationConfig,
)


def _load_vad_model() -> tuple[torch.nn.Module, Callable]:
    """Load silero VAD from torch.hub.

    :return: tuple of silero VAD model with VAD parsing function
    """
    model, utils = torch.hub.load(
        SILERO_TORCH_HUB_REPO,
        SILERO_MODEL,
        trust_repo=True,
    )
    return model, utils[0]


class AudioSegmenter:
    """Segments an audio tensor using the provided segmentation strategy.

    Will default to fixed-length segments in the absence of an explicit config."""

    def __init__(self):
        self._vad_model: torch.nn.Module | None = None
        self._vad_parser_fn: Callable | None = None

    def __enter__(self) -> "AudioSegmenter":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self._vad_model = None
        self._vad_parser_fn = None

    def segment(
        self,
        audio_tensor: torch.Tensor,
        segmentation_config: SegmentationConfig | None = None,
    ) -> list[TensorSegment]:
        """Segment an audio tensor according to the configured strategy.

        :param audio_tensor: 1D input tensor
        :param segmentation_config: segmentation config
        :return: list of TensorSegment
        """
        # Default to fixed
        if segmentation_config is None or not isinstance(
            segmentation_config, SegmentationConfig
        ):
            segmentation_config = FixedSegmentationConfig()

        match segmentation_config.segmentation_strategy:
            case SegmentationStrategyEnum.FIXED:
                segmentation_config: FixedSegmentationConfig = segmentation_config
                return segment_fixed(
                    audio_tensor,
                    sample_rate=segmentation_config.sample_rate,
                    max_segment_len_s=segmentation_config.max_segment_len_s,
                )

            case SegmentationStrategyEnum.SILENCE:
                segmentation_config: SilenceSegmentationConfig = segmentation_config
                return segment_by_silence(
                    audio_tensor,
                    sample_rate=segmentation_config.sample_rate,
                    frame_len=segmentation_config.frame_len,
                    silence_thresh_db=segmentation_config.silence_thresh_db,
                    hop_len=segmentation_config.hop_len,
                    kept_silence_len_secs=segmentation_config.kept_silence_len_secs,
                    min_silence_len_secs=segmentation_config.min_silence_len_secs,
                    max_segment_len_s=segmentation_config.max_segment_len_s,
                )

            case SegmentationStrategyEnum.VOICE:
                segmentation_config: SileroVoiceSegmentationConfig = segmentation_config

                # Load model lazily
                if self._vad_model is None or self._vad_parser_fn is None:
                    self._vad_model, self._vad_parser_fn = _load_vad_model()

                # TODO: Strong coupling here with silero; would prefer to make this
                # agnostic, wrapping in a VAD object with standard accessor methods
                return segment_by_silero_vad(
                    audio_tensor,
                    vad_model=self._vad_model,
                    vad_parser_fn=self._vad_parser_fn,
                    sample_rate=segmentation_config.sample_rate,
                    threshold=segmentation_config.threshold,
                    min_speech_duration_ms=segmentation_config.min_speech_duration_ms,
                    min_silence_duration_ms=segmentation_config.min_silence_duration_ms,
                    speech_pad_ms=segmentation_config.speech_pad_ms,
                    max_segment_len_s=segmentation_config.max_segment_len_s,
                )

            case _:
                raise ValueError(
                    f"Unknown segmentation strategy: {segmentation_config.segmentation_strategy}"
                )
