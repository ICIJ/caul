from pathlib import Path

import pytest
import torch

from caul.constants import DEFAULT_SAMPLE_RATE, FIREREDASR2_INFERENCE_MAX_DURATION_S
from caul.objects import ASRResult, PreprocessedInput, InputMetadata
from caul.tasks import FireRedASR2InferenceRunnerConfig
from caul.tasks.inference.fireredasr import (
    FireRedASR2InferenceRunner,
)
from caul.tasks.postprocessing.fireredasr import FireRedASR2Postprocessor
from caul.tasks.preprocessing.fireredasr import (
    FireRedASR2Preprocessor,
)


SAMPLES_PER_S = DEFAULT_SAMPLE_RATE
MAX_FRAMES = FIREREDASR2_INFERENCE_MAX_DURATION_S * SAMPLES_PER_S

ZH_TEXT = "你好世界"
ZH_CONFIDENCE = 0.97
ZH_TIMESTAMP = [("你", 0.1, 0.4), ("好", 0.4, 0.7), ("世", 0.7, 1.0), ("界", 1.0, 1.3)]


def _make_result(uttid: str, text: str = ZH_TEXT, dur_s: float = 2.0) -> dict:
    return {
        "uttid": uttid,
        "text": text,
        "confidence": ZH_CONFIDENCE,
        "dur_s": dur_s,
        "rtf": "0.05",
        "wav": "dummy.wav",
        "timestamp": ZH_TIMESTAMP,
    }


class MockFireRedAsr2:
    def transcribe(
        self, batch_uttid: list[str], batch_wav_path: list[str]
    ) -> list[dict]:
        return [_make_result(uid) for uid in batch_uttid]


class MockFireRedASR2InferenceRunner(FireRedASR2InferenceRunner):
    def __enter__(self):
        self._model = MockFireRedAsr2()
        return self


# ASRResult.from_fireredasr2_result


class TestASRResultFromFireRedASR2:
    def test__result_with_timestamps(self):
        result = _make_result("utt0")
        asr = ASRResult.from_fireredasr2_result(result, input_ordering=0)

        assert asr.input_ordering == 0
        assert len(asr.transcription) == 1
        start, end, text = asr.transcription[0]
        assert start == pytest.approx(ZH_TIMESTAMP[0][1])
        assert end == pytest.approx(ZH_TIMESTAMP[-1][2])
        assert text == ZH_TEXT
        assert asr.score == pytest.approx(ZH_CONFIDENCE, abs=1e-4)

    def test__result_without_timestamps(self):
        result = {"uttid": "utt0", "text": ZH_TEXT, "confidence": 0.9, "dur_s": 3.5}
        asr = ASRResult.from_fireredasr2_result(result, input_ordering=1)

        assert len(asr.transcription) == 1
        start, end, text = asr.transcription[0]
        assert start == pytest.approx(0.0)
        assert end == pytest.approx(3.5)
        assert text == ZH_TEXT

    def test__result_empty_text(self):
        result = {"uttid": "utt0", "text": "", "confidence": 0.5, "dur_s": 1.0}
        asr = ASRResult.from_fireredasr2_result(result, input_ordering=0)
        assert asr.transcription == []

    def test__result_whitespace_only_text(self):
        result = {"uttid": "utt0", "text": "   ", "confidence": 0.5, "dur_s": 1.0}
        asr = ASRResult.from_fireredasr2_result(result, input_ordering=0)
        assert asr.transcription == []


# preprocessing


class TestFireRedASR2Preprocessor:
    def __init__(self):
        self._preprocessor = FireRedASR2Preprocessor()

    def test__short_audio_single_segment(self):
        """Audio shorter than 60 seconds should produce exactly one segment"""
        audio = [torch.zeros(SAMPLES_PER_S * 10)]  # 10 s

        result = list(self._preprocessor.preprocess_inputs(audio))

        assert len(result) == 1
        assert result[0].metadata.input_ordering == 0

    def test__long_audio_gets_segmented(self):
        """Audio longer than 60 seconds must be split into multiple segments"""
        # 70 s of silence — segment_by_silence will fall back to fixed splits
        audio = [torch.zeros(SAMPLES_PER_S * 70)]

        result = list(self._preprocessor.preprocess_inputs(audio))

        assert len(result) > 1
        for seg in result:
            assert seg.metadata.input_ordering == 0

    def test__multiple_inputs_ordering(self):
        """input_ordering must match the original list index"""
        audio = [torch.zeros(SAMPLES_PER_S * 5), torch.zeros(SAMPLES_PER_S * 3)]

        result = list(self._preprocessor.preprocess_inputs(audio))

        orderings = [r.metadata.input_ordering for r in result]
        assert orderings == [0, 1]

    def test__write_wavs_to_fs(self, tmpdir):
        """When output_dir is provided, wav files are written to disk"""
        output_dir = Path(tmpdir)
        audio = [torch.zeros(SAMPLES_PER_S * 2)]

        result = list(
            self._preprocessor.preprocess_inputs(audio, output_dir=output_dir)
        )

        assert len(result) == 1
        saved = output_dir / result[0].metadata.preprocessed_file_path
        assert saved.exists()


# Inference


class TestFireRedASR2InferenceRunner:
    def __init__(self):
        mock_config = FireRedASR2InferenceRunnerConfig(model_dir="test")

        self._inference_runner = MockFireRedASR2InferenceRunner(config=mock_config)

    def test__yields_asr_results(self):
        """Inference runner should yield one ASRResult per segment in the batch"""
        preprocessor = FireRedASR2Preprocessor(batch_size=4)
        audio = [torch.zeros(SAMPLES_PER_S * 2), torch.zeros(SAMPLES_PER_S * 3)]

        batches = list(preprocessor.process(audio))

        with self._inference_runner:
            results = list(self._inference_runner.process(batches))

        assert len(results) == 2
        assert all(isinstance(r, ASRResult) for r in results)
        assert {r.input_ordering for r in results} == {0, 1}

    def test__skips_empty_batches(self):
        """Empty batches should not cause errors and should be silently skipped"""
        with self._inference_runner:
            results = list(self._inference_runner.process([[], []]))
        assert results == []

    def test__writes_temp_files_for_tensors(self):
        """Tensor inputs must be written to temp wav files that are cleaned up"""
        preprocessor = FireRedASR2Preprocessor()
        audio = [torch.zeros(SAMPLES_PER_S)]
        batches = list(preprocessor.process(audio))

        # Confirm inputs are tensor-backed (no output_dir was given)
        assert hasattr(batches[0][0], "tensor")

        with self._inference_runner:
            results = list(self._inference_runner.process(batches))

        assert len(results) == 1

    def test__processes_input_without_output_dir_with_tmp_dirs_enabled(self):
        mock_config = FireRedASR2InferenceRunnerConfig(
            model_dir="test", tmp_dir_fallback=True
        )
        inference_runner = MockFireRedASR2InferenceRunner(config=mock_config)
        batches = [[PreprocessedInput(metadata=InputMetadata(duration_s=1))]]

        with inference_runner:
            results = inference_runner.process(batches)

        assert len(results) == 1

    def test__does_not_process_input_without_output_dir_with_tmp_dirs_disabled(self):
        batches = [[PreprocessedInput(metadata=InputMetadata(duration_s=1))]]

        with self._inference_runner:
            results = self._inference_runner.process(batches)

        assert len(results) == 0


# Postprocessing


class TestFireRedASR2Postprocessor:
    def __init__(self):
        self._postprocessor = FireRedASR2Postprocessor()

    def test__merges_segments(self):
        """Segments belonging to the same input_ordering are merged in time order"""
        results = [
            ASRResult(input_ordering=0, transcription=[(0.0, 1.0, "你好")], score=0.9),
            ASRResult(input_ordering=0, transcription=[(1.0, 2.0, "世界")], score=0.8),
        ]

        merged = list(self._postprocessor.process(results))

        assert len(merged) == 1
        assert merged[0].input_ordering == 0
        assert merged[0].transcription == [(0.0, 1.0, "你好"), (1.0, 2.0, "世界")]

    def test__multiple_inputs(self):
        """Each unique input_ordering yields exactly one merged result"""
        results = [
            ASRResult(input_ordering=0, transcription=[(0.0, 1.0, "你好")], score=0.9),
            ASRResult(input_ordering=1, transcription=[(0.0, 2.0, "世界")], score=0.8),
        ]

        merged = list(self._postprocessor.process(results))

        assert len(merged) == 2
        assert {r.input_ordering for r in merged} == {0, 1}

    def test__drops_empty_transcription_segments(self):
        """Segments with empty transcriptions must be dropped before merging"""
        results = [
            ASRResult(input_ordering=0, transcription=[], score=1.0),  # silent segment
            ASRResult(input_ordering=0, transcription=[(0.5, 1.5, "你好")], score=0.9),
        ]

        merged = list(self._postprocessor.process(results))

        assert len(merged) == 1
        assert merged[0].transcription == [(0.5, 1.5, "你好")]

    def test__raises_for_non_contiguous(self):
        """Non-contiguous ordering (interleaved groups) should raise ValueError"""
        results = [
            ASRResult(input_ordering=0, transcription=[(0.0, 1.0, "a")], score=1.0),
            ASRResult(input_ordering=1, transcription=[(0.0, 1.0, "b")], score=1.0),
            ASRResult(input_ordering=0, transcription=[(1.0, 2.0, "c")], score=1.0),
        ]

        with pytest.raises(ValueError, match="expected contiguous batches"):
            list(self._postprocessor.process(results))

    def test__inputs_all_silent(self):
        """An input whose segments are silent should yield an empty transcription"""
        results = [
            ASRResult(input_ordering=0, transcription=[], score=1.0),
            ASRResult(input_ordering=0, transcription=[], score=1.0),
        ]

        merged = list(self._postprocessor.process(results))

        assert len(merged) == 1
        assert merged[0].transcription == []
