from unittest.mock import MagicMock, patch

import pytest
import torch

from caul.segmentation.segmenter import (
    AudioSegmenter,
    FixedSegmentationConfig,
    SilenceSegmentationConfig,
    TensorSegment,
    SileroVoiceSegmentationConfig,
    segment_by_silence,
    segment_by_silero_vad,
    segment_fixed,
)
from caul.constant import EXPECTED_SAMPLE_RATE
from test.unit.constant import TEST_TIMESTAMPS


def make_silent_tensor(
    duration_secs: float, sample_rate: int = EXPECTED_SAMPLE_RATE
) -> torch.Tensor:
    """Zero tensor treated as silence by librosa.effects.split"""
    return torch.zeros(int(duration_secs * sample_rate))


def make_noisy_tensor(
    duration_secs: float, sample_rate: int = EXPECTED_SAMPLE_RATE
) -> torch.Tensor:
    """Noisy tensor with uniform random noise"""
    return torch.rand(int(duration_secs * sample_rate))


def make_voiced_tensor(
    voice_durations_secs: list[float],
    silence_between_secs: float = 1.5,
    sample_rate: int = EXPECTED_SAMPLE_RATE,
) -> torch.Tensor:
    """Build a tensor alternating noise and silent regions"""
    chunks = []
    for i, duration in enumerate(voice_durations_secs):
        chunks.append(torch.rand(int(duration * sample_rate)))
        if i < len(voice_durations_secs) - 1:
            chunks.append(torch.zeros(int(silence_between_secs * sample_rate)))
    return torch.cat(chunks)


def _make_silero_mock(speech_timestamps: list[dict]):
    vad_model = MagicMock()
    vad_parser_fn = MagicMock(return_value=speech_timestamps)
    return vad_model, vad_parser_fn


# TensorSegment


class TestTensorSegment:
    def test_duration_computed_correctly(self):
        seg = TensorSegment(
            tensor=torch.zeros(16000),
            segment_start=0,
            segment_end=16000,
            sample_rate=16000,
            tensor_id="abc",
        )
        assert seg.duration == 1

    def test_lower_sample_rate(self):
        seg = TensorSegment(
            tensor=torch.zeros(8000),
            segment_start=0,
            segment_end=8000,
            sample_rate=8000,
            tensor_id="abc",
        )
        assert seg.duration == 1


# segment_fixed


class TestSegmentFixed:
    def test_single_chunk_shorter_than_segment_duration(self):
        tensor = make_silent_tensor(5.0)
        segments = segment_fixed(tensor, segment_duration_secs=10.0)
        assert len(segments) == 1
        assert segments[0].segment_start == 0
        assert segments[0].segment_end == tensor.shape[-1]

    def test_exact_multiple_produces_correct_num_segments(self):
        tensor = make_silent_tensor(10.0)
        segments = segment_fixed(tensor, segment_duration_secs=5.0)
        assert len(segments) == 2

    def test_inexact_includes_remainder(self):
        tensor = make_silent_tensor(11.0)
        segments = segment_fixed(tensor, segment_duration_secs=5.0)
        assert len(segments) == 3

    def test_segments_are_contiguous_and_non_overlapping(self):
        tensor = make_silent_tensor(13.0)
        segments = segment_fixed(tensor, segment_duration_secs=5.0)
        for i in range(1, len(segments)):
            assert segments[i].segment_start == segments[i - 1].segment_end

    def test_tensor_content_matches_slice(self):
        tensor = torch.arange(float(EXPECTED_SAMPLE_RATE * 3))
        segments = segment_fixed(tensor, segment_duration_secs=1.0)
        for seg in segments:
            assert torch.equal(seg.tensor, tensor[seg.segment_start : seg.segment_end])

    def test_all_segments_share_tensor_id(self):
        segments = segment_fixed(make_silent_tensor(6.0), segment_duration_secs=2.0)
        tensor_ids = {s.tensor_id for s in segments}
        assert len(tensor_ids) == 1

    def test_separate_calls_produce_different_tensor_ids(self):
        tensor = make_silent_tensor(3.0)
        segments_a = segment_fixed(tensor, segment_duration_secs=1.0)
        segments_b = segment_fixed(tensor, segment_duration_secs=1.0)
        assert segments_a[0].tensor_id != segments_b[0].tensor_id


# segment_by_silence


class TestSegmentBySilence:
    def test_all_silent_returns_single_segment(self):
        """librosa.effects.split treats an all-zero tensor as one interval spanning
        the full tensor rather than as pure silence, so we get one segment back"""

        tensor = make_silent_tensor(2.0)
        segments = segment_by_silence(tensor)
        assert len(segments) == 1
        assert segments[0].segment_start == 0
        assert segments[0].segment_end == tensor.shape[-1]

    def test_segments_do_not_exceed_max_length(self):
        tensor = make_noisy_tensor(5.0)
        max_secs = 2.0
        segments = segment_by_silence(tensor, max_segment_len_secs=max_secs)
        for seg in segments:
            assert seg.duration <= max_secs

    def test_oversized_interval_splits_correctly(self):
        tensor = make_noisy_tensor(6.0)
        segments = segment_by_silence(tensor, max_segment_len_secs=2.0)
        assert len(segments) >= 3

    def test_all_segments_share_tensor_id(self):
        tensor = make_noisy_tensor(6.0)
        segments = segment_by_silence(tensor, max_segment_len_secs=2.0)
        tensor_ids = {s.tensor_id for s in segments}
        assert len(tensor_ids) == 1

    def test_separate_calls_produce_different_tensor_ids(self):
        tensor = make_noisy_tensor(2.0)
        segments_a = segment_by_silence(tensor)
        segments_b = segment_by_silence(tensor)
        assert segments_a[0].tensor_id != segments_b[0].tensor_id

    def test_splits_voiced_tensor_into_one_segment_per_voice(self):
        voice_durations = [1.0, 1.5, 0.8]
        tensor = make_voiced_tensor(voice_durations)
        segments = segment_by_silence(tensor)
        assert len(segments) == len(voice_durations)

    def test_segment_durations_approximate_voice_regions(self):
        voice_durations = [1.0, 1.5, 0.8]
        silence_secs = 1.0
        tensor = make_voiced_tensor(voice_durations, silence_between_secs=silence_secs)
        segments = segment_by_silence(tensor)
        for seg, expected_secs in zip(segments, voice_durations):
            assert seg.duration == pytest.approx(expected_secs, abs=silence_secs)


# segment_by_silero_vad


class TestSegmentBySileroVad:
    def test_returns_tensor_segment_per_timestamp(self):
        timestamps = [{"start": 0, "end": 8000}, {"start": 16000, "end": 24000}]
        model, vad_parser_fn = _make_silero_mock(timestamps)
        segments = segment_by_silero_vad(
            make_silent_tensor(2.0), vad_model=model, vad_parser_fn=vad_parser_fn
        )
        assert len(segments) == 2
        assert segments[0].segment_start == 0
        assert segments[0].segment_end == 8000
        assert segments[1].segment_start == 16000
        assert segments[1].segment_end == 24000

    def test_no_speech_returns_empty(self):
        model, vad_parser_fn = _make_silero_mock([])
        segments = segment_by_silero_vad(
            make_silent_tensor(1.0), vad_model=model, vad_parser_fn=vad_parser_fn
        )
        assert segments == []

    def test_config_params_forwarded_to_model(self):
        timestamps = [{"start": 0, "end": 4000}]
        model, vad_parser_fn = _make_silero_mock(timestamps)
        segment_by_silero_vad(
            make_silent_tensor(1.0),
            vad_model=model,
            vad_parser_fn=vad_parser_fn,
            sample_rate=8000,
            threshold=0.7,
            min_speech_duration_ms=500,
            min_silence_duration_ms=200,
            speech_pad_ms=50,
        )
        call_kwargs = vad_parser_fn.call_args.kwargs
        assert call_kwargs["sampling_rate"] == 8000
        assert call_kwargs["threshold"] == 0.7
        assert call_kwargs["min_speech_duration_ms"] == 500
        assert call_kwargs["min_silence_duration_ms"] == 200
        assert call_kwargs["speech_pad_ms"] == 50

    def test_oversized_segment_is_split(self):
        # One 4-second timestamp with a 2-second cap should produce 2 segments
        timestamps = [{"start": 0, "end": EXPECTED_SAMPLE_RATE * 4}]
        model, vad_parser_fn = _make_silero_mock(timestamps)
        segments = segment_by_silero_vad(
            make_silent_tensor(4.0),
            vad_model=model,
            vad_parser_fn=vad_parser_fn,
            max_segment_len_secs=2.0,
        )
        assert len(segments) == 2
        assert segments[0].segment_end == EXPECTED_SAMPLE_RATE * 2
        assert segments[1].segment_start == EXPECTED_SAMPLE_RATE * 2

    def test_segments_do_not_exceed_max_length(self):
        timestamps = [{"start": 0, "end": EXPECTED_SAMPLE_RATE * 10}]
        model, vad_parser_fn = _make_silero_mock(timestamps)
        segments = segment_by_silero_vad(
            make_silent_tensor(10.0),
            vad_model=model,
            vad_parser_fn=vad_parser_fn,
            max_segment_len_secs=3.0,
        )
        for seg in segments:
            assert seg.duration <= 3.0

    def test_all_segments_share_tensor_id(self):
        timestamps = [{"start": 0, "end": 8000}, {"start": 16000, "end": 24000}]
        model, vad_parser_fn = _make_silero_mock(timestamps)
        segments = segment_by_silero_vad(
            make_silent_tensor(2.0), vad_model=model, vad_parser_fn=vad_parser_fn
        )
        tensor_ids = {s.tensor_id for s in segments}
        assert len(tensor_ids) == 1

    def test_separate_calls_produce_different_tensor_ids(self):
        timestamps = [{"start": 0, "end": 8000}]
        model, vad_parser_fn = _make_silero_mock(timestamps)
        tensor = make_silent_tensor(1.0)
        segments_a = segment_by_silero_vad(
            tensor, vad_model=model, vad_parser_fn=vad_parser_fn
        )
        segments_b = segment_by_silero_vad(
            tensor, vad_model=model, vad_parser_fn=vad_parser_fn
        )
        assert segments_a[0].tensor_id != segments_b[0].tensor_id

    def test_segments_match_voiced_regions(self):
        """Build a tensor with three voiced regions and derive sample-accurate timestamps"""
        voice_durations = [1.0, 1.5, 0.8]
        silence_secs = 0.6
        tensor = make_voiced_tensor(voice_durations, silence_between_secs=silence_secs)

        silence_samples = int(silence_secs * EXPECTED_SAMPLE_RATE)
        timestamps = []
        cursor = 0
        for duration in voice_durations:
            voice_samples = int(duration * EXPECTED_SAMPLE_RATE)
            timestamps.append({"start": cursor, "end": cursor + voice_samples})
            cursor += voice_samples + silence_samples

        model, vad_parser_fn = _make_silero_mock(timestamps)
        segments = segment_by_silero_vad(
            tensor, vad_model=model, vad_parser_fn=vad_parser_fn
        )

        assert len(segments) == len(voice_durations)
        for seg, ts in zip(segments, timestamps):
            assert seg.segment_start == ts["start"]
            assert seg.segment_end == ts["end"]
            assert torch.equal(seg.tensor, tensor[ts["start"] : ts["end"]])


# AudioSegmenter dispatch


class TestAudioSegmenter:
    def __init__(self):
        self._silero_mock = _make_silero_mock(TEST_TIMESTAMPS)

    def test_dispatches_to_segment_fixed(self):
        segmenter = AudioSegmenter()
        segments = segmenter.segment(
            make_silent_tensor(5.0), FixedSegmentationConfig(segment_duration_secs=2.0)
        )
        assert len(segments) == 3

    def test_dispatches_to_segment_by_silence(self):
        segmenter = AudioSegmenter()
        tensor = make_voiced_tensor([1.0, 1.0])
        segments = segmenter.segment(tensor, SilenceSegmentationConfig())
        assert len(segments) == 2

    def test_dispatches_to_segment_by_silero_vad(self):
        with patch(
            "caul.segmentation.segmenter._load_vad_model",
            return_value=self._silero_mock,
        ):
            segmenter = AudioSegmenter()
            segments = segmenter.segment(
                make_voiced_tensor([2.0, 2.0, 2.0]), SileroVoiceSegmentationConfig()
            )

        assert len(segments) == 3

    def test_voice_model_loaded_lazily_on_first_segment(self):
        with patch(
            "caul.segmentation.segmenter._load_vad_model",
            return_value=self._silero_mock,
        ) as mock_load:
            segmenter = AudioSegmenter()
            assert mock_load.call_count == 0
            segmenter.segment(make_silent_tensor(1.0), SileroVoiceSegmentationConfig())
            assert mock_load.call_count == 1
            segmenter.segment(make_silent_tensor(1.0), SileroVoiceSegmentationConfig())
            assert mock_load.call_count == 1

    def test_voice_model_reused_across_segment_calls(self):
        model, vad_parser_fn = self._silero_mock

        with patch(
            "caul.segmentation.segmenter._load_vad_model",
            return_value=(model, vad_parser_fn),
        ):
            segmenter = AudioSegmenter()
            segmenter.segment(make_silent_tensor(1.0), SileroVoiceSegmentationConfig())
            segmenter.segment(make_silent_tensor(1.0), SileroVoiceSegmentationConfig())

        assert vad_parser_fn.call_count == 2
        for call in vad_parser_fn.call_args_list:
            assert call.args[1] is model

    def test_context_manager_clears_vad_model_on_exit(self):
        with patch(
            "caul.segmentation.segmenter._load_vad_model",
            return_value=self._silero_mock,
        ):
            segmenter = AudioSegmenter()
            with segmenter:
                segmenter.segment(
                    make_silent_tensor(1.0), SileroVoiceSegmentationConfig()
                )
                assert segmenter._vad_model is not None
                assert segmenter._vad_parser_fn is not None
            assert segmenter._vad_model is None
            assert segmenter._vad_parser_fn is None

    def test_unknown_strategy_raises(self):
        segmenter = AudioSegmenter()
        config = FixedSegmentationConfig()
        config.segmentation_strategy = "unknown"  # type: ignore[assignment]
        with pytest.raises(ValueError, match="Unknown segmentation strategy"):
            segmenter.segment(make_silent_tensor(1.0), config)
