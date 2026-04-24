import math
from abc import ABC
from collections import namedtuple

import pytest
import torch
from numpy import ndarray

from caul.objects import (
    ASRResult,
    InputMetadata,
    PreprocessedInput,
    PreprocessedInputWithTensor,
)
from caul.tasks.inference.faster_whisper import (
    FasterWhisperInferenceRunner,
    FasterWhisperInferenceRunnerConfig,
)


EN_TEXT_A = "hello"
EN_TEXT_B = "world"

SEG_START_A, SEG_END_A, LOG_PROB_A = 0.0, 2.0, -0.2
SEG_START_B, SEG_END_B, LOG_PROB_B = 2.0, 5.0, -0.3


def _mock_raw_segment(
    start,
    end,
    text,
    avg_logprob,
    seek=None,
    tokens=[],
    no_speech_prob=0.5,
    compression_ratio=0.5,
):
    return {
        "start": start,
        "end": end,
        "seek": seek,
        "text": text,
        "avg_logprob": avg_logprob,
        "tokens": tokens,
        "no_speech_prob": no_speech_prob,
        "compression_ratio": compression_ratio,
    }


MOCK_RAW_SEGMENTS = [
    [_mock_raw_segment(SEG_START_A, SEG_END_A, EN_TEXT_A, LOG_PROB_A)],
    [_mock_raw_segment(SEG_START_B, SEG_END_B, EN_TEXT_B, LOG_PROB_B)],
]


_MockSegment = namedtuple("_MockSegment", ["start", "end", "text", "avg_logprob"])

MOCK_SEGMENTS = [
    _MockSegment(s[0]["start"], s[0]["end"], s[0]["text"], s[0]["avg_logprob"])
    for s in MOCK_RAW_SEGMENTS
]

_MOCK_OUTPUT_SEGMENTS = [
    _mock_raw_segment(SEG_START_A, SEG_END_A, EN_TEXT_A, LOG_PROB_A),
    _mock_raw_segment(SEG_START_B, SEG_END_B, EN_TEXT_B, LOG_PROB_B),
]


class MockWhisperModelModel:
    def __init__(self):
        self.is_multilingual = True


class MockHfTokenizer:
    def token_to_id(self, token):
        return "task"


class _MockWhisperModel:
    def __init__(self):
        self.hf_tokenizer = MockHfTokenizer()
        self.model = MockWhisperModelModel()

    def feature_extractor(self, waveform: ndarray, padding=160, chunk_length=None):
        return waveform

    def detect_language(self, features: ndarray):
        return "en", 1.0


class _MockBatchedInferencePipeline:
    """Returns the same two-segment result for every input in the batch"""

    def __init__(self):
        self.model = _MockWhisperModel()

    def forward(self, features, tokenizer, chunks_metadata, options):
        return [_MOCK_OUTPUT_SEGMENTS for _ in range(len(chunks_metadata))]


class MockFasterWhisperInferenceRunner(FasterWhisperInferenceRunner):
    def __enter__(self):
        self._model = _MockBatchedInferencePipeline()
        return self


def _file_backed(tmp_path, name="audio.wav", input_ordering=0, duration_s=2.0):
    import torchaudio  # pylint: disable=import-outside-toplevel

    path = tmp_path / name
    torchaudio.save(str(path), torch.zeros(1, int(duration_s * 16000)), 16000)
    return PreprocessedInput(
        metadata=InputMetadata(
            duration_s=duration_s,
            input_ordering=input_ordering,
            preprocessed_file_path=path,
        )
    )


def _tensor_backed(input_ordering=0, duration_s=2.0):
    return PreprocessedInputWithTensor(
        metadata=InputMetadata(duration_s=duration_s, input_ordering=input_ordering),
        tensor=torch.zeros(int(duration_s * 16000)),
    )


class TestASRResultFromFasterWhisper:
    def test__multiple_segments_build_transcription(self):
        result = ASRResult.from_faster_whisper_result(MOCK_SEGMENTS, input_ordering=0)

        assert result.input_ordering == 0
        assert result.transcription == [
            (SEG_START_A, SEG_END_A, EN_TEXT_A),
            (SEG_START_B, SEG_END_B, EN_TEXT_B),
        ]

    def test__score_is_duration_weighted_avg_logprob(self):
        result = ASRResult.from_faster_whisper_result(MOCK_SEGMENTS, input_ordering=0)

        dur_a = SEG_END_A - SEG_START_A
        dur_b = SEG_END_B - SEG_START_B
        expected = math.exp((LOG_PROB_A * dur_a + LOG_PROB_B * dur_b) / (dur_a + dur_b))
        assert result.score == pytest.approx(expected, abs=1e-9)

    def test__single_segment(self):
        seg = _MockSegment(1.0, 4.0, "only", -0.1)
        result = ASRResult.from_faster_whisper_result([seg], input_ordering=2)

        assert result.transcription == [(1.0, 4.0, "only")]
        assert result.score == pytest.approx(math.exp(-0.1), abs=1e-9)

    def test__empty_segments_give_empty_transcription_and_unit_score(self):
        result = ASRResult.from_faster_whisper_result([], input_ordering=0)

        assert result.transcription == []
        assert result.score == -1.0


class TestFasterWhisperInferenceRunner:
    def setup_method(self):
        self._runner = MockFasterWhisperInferenceRunner()

    def test__yields_one_result_per_input_in_batch(self, tmp_path):
        inputs = [
            _file_backed(tmp_path, "a.wav", input_ordering=0),
            _file_backed(tmp_path, "b.wav", input_ordering=1),
        ]
        with self._runner:
            results = list(self._runner.process([inputs]))

        assert len(results) == 2
        assert all(isinstance(r, ASRResult) for r in results)

    def test__preserves_input_ordering(self, tmp_path):
        inputs = [
            _file_backed(tmp_path, f"{i}.wav", input_ordering=i) for i in range(3)
        ]
        with self._runner:
            results = list(self._runner.process([inputs]))

        assert [r.input_ordering for r in results] == [0, 1, 2]

    def test__results_contain_transcription(self, tmp_path):
        inp = _file_backed(tmp_path, input_ordering=0)
        with self._runner:
            results = list(self._runner.process([[inp]]))

        assert results[0].transcription == [
            (SEG_START_A, SEG_END_A, EN_TEXT_A),
            (SEG_START_B, SEG_END_B, EN_TEXT_B),
        ]

    def test__skips_empty_batches(self):
        with self._runner:
            results = list(self._runner.process([[], []]))
        assert results == []

    def test__multiple_batches_each_yield_results(self, tmp_path):
        batch_a = [_file_backed(tmp_path, "a.wav", input_ordering=0)]
        batch_b = [_file_backed(tmp_path, "b.wav", input_ordering=1)]
        with self._runner:
            results = list(self._runner.process([batch_a, batch_b]))

        assert len(results) == 2
