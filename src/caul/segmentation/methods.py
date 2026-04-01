import uuid
from typing import Callable

import librosa
import torch

from caul.constant import EXPECTED_SAMPLE_RATE, PARAKEET_INFERENCE_MAX_DURATION_SEC
from caul.segmentation.objects import TensorSegment


def segment_fixed(
    audio_tensor: torch.Tensor,
    sample_rate: int = EXPECTED_SAMPLE_RATE,
    segment_duration_secs: float = PARAKEET_INFERENCE_MAX_DURATION_SEC,
) -> list[TensorSegment]:
    """Split an audio tensor into fixed-length chunks.

    :param audio_tensor: 1D input tensor
    :param sample_rate: sample rate of the audio
    :param segment_duration_secs: duration of each segment in seconds
    :return: list of TensorSegment
    """
    tensor_id = uuid.uuid4().hex
    segment_len = int(segment_duration_secs * sample_rate)
    total_samples = audio_tensor.shape[-1]
    segments = []
    start = 0

    while start < total_samples:
        end = min(start + segment_len, total_samples)
        segments.append(
            TensorSegment(
                tensor=audio_tensor[start:end],
                segment_start=start,
                segment_end=end,
                sample_rate=sample_rate,
                tensor_id=tensor_id,
            )
        )
        start = end

    return segments


def segment_by_silence(
    audio_tensor: torch.Tensor,
    sample_rate: int = EXPECTED_SAMPLE_RATE,
    frame_len: int = 2048,
    silence_thresh_db: int = 35,
    hop_len: int = 512,
    kept_silence_len_secs: float = 0.15,
    min_silence_len_secs: float = 0.5,
    max_segment_len_secs: float = PARAKEET_INFERENCE_MAX_DURATION_SEC,
) -> list[TensorSegment]:
    """Split an audio tensor on silences using librosa, falling back to fixed splits where
    merged intervals exceed the maximum segment length.

    :param audio_tensor: 1D input tensor
    :param sample_rate: sample rate of the audio
    :param frame_len: number of samples per librosa analysis frame
    :param silence_thresh_db: dB threshold below which audio is considered silent
    :param hop_len: number of samples between analysis frames
    :param kept_silence_len_secs: seconds of silence to retain at segment boundaries
    :param min_silence_len_secs: minimum silence duration to split on
    :param max_segment_len_secs: maximum segment duration in seconds
    :return: list of TensorSegment
    """
    tensor_id = uuid.uuid4().hex

    nonsilent_intervals = librosa.effects.split(
        audio_tensor.numpy(),
        top_db=silence_thresh_db,
        frame_length=frame_len,
        hop_length=hop_len,
    )

    min_silence_samples = int(min_silence_len_secs * sample_rate)
    kept_silence_samples = int(kept_silence_len_secs * sample_rate)
    max_segment_samples = int(max_segment_len_secs * sample_rate)

    # Merge intervals separated by silences shorter than the minimum
    merged: list[list[int]] = []
    for start, end in nonsilent_intervals:
        if not merged:
            merged.append([start, end])
        else:
            prev_end = merged[-1][1]
            if start - prev_end < min_silence_samples:
                merged[-1][1] = end
            else:
                merged.append([start, end])

    segments = []
    total_samples = audio_tensor.shape[-1]

    for interval_start, interval_end in merged:
        seg_start = max(0, interval_start - kept_silence_samples)
        seg_end = min(total_samples, interval_end + kept_silence_samples)

        # Split oversized intervals using fixed-length fallback
        while seg_end - seg_start > max_segment_samples:
            chunk_end = seg_start + max_segment_samples
            segments.append(
                TensorSegment(
                    tensor=audio_tensor[seg_start:chunk_end],
                    segment_start=seg_start,
                    segment_end=chunk_end,
                    sample_rate=sample_rate,
                    tensor_id=tensor_id,
                )
            )
            seg_start = chunk_end

        segments.append(
            TensorSegment(
                tensor=audio_tensor[seg_start:seg_end],
                segment_start=seg_start,
                segment_end=seg_end,
                sample_rate=sample_rate,
                tensor_id=tensor_id,
            )
        )

    return segments


def segment_by_silero_vad(
    audio_tensor: torch.Tensor,
    vad_model: torch.nn.Module,
    vad_parser_fn: Callable,
    sample_rate: int = EXPECTED_SAMPLE_RATE,
    threshold: float = 0.5,
    min_speech_duration_ms: int = 250,
    min_silence_duration_ms: int = 100,
    speech_pad_ms: int = 30,
) -> list[TensorSegment]:
    """Split an audio tensor into voiced segments using silero VAD.

    :param audio_tensor: 1D input tensor
    :param vad_model: silero VAD model
    :param vad_parser_fn: silero VAD segmentation function
    :param sample_rate: sample rate of the audio; silero-VAD supports 8000 or 16000
    :param threshold: speech probability threshold [0, 1)
    :param min_speech_duration_ms: minimum speech segment duration in ms
    :param min_silence_duration_ms: minimum silence duration to split on in ms
    :param speech_pad_ms: padding added around each speech segment in ms
    :return: list of TensorSegment
    """
    tensor_id = uuid.uuid4().hex

    speech_timestamps = vad_parser_fn(
        audio_tensor,
        vad_model,
        sample_rate=sample_rate,
        threshold=threshold,
        min_speech_duration_ms=min_speech_duration_ms,
        min_silence_duration_ms=min_silence_duration_ms,
        speech_pad_ms=speech_pad_ms,
        return_seconds=False,
    )

    return [
        TensorSegment(
            tensor=audio_tensor[ts["start"] : ts["end"]],
            segment_start=ts["start"],
            segment_end=ts["end"],
            sample_rate=sample_rate,
            tensor_id=tensor_id,
        )
        for ts in speech_timestamps
    ]
