from pathlib import Path

import pytest
import torch

from caul.constants import FIREREDASR2_INFERENCE_MAX_FRAMES
from caul.objects import ASRResult
from caul.tasks.postprocessing.asr_postprocessor import ASRPostprocessor
from caul.tasks.preprocessing.asr_preprocessor import ASRPreprocessor


class TestASRPreprocessor:
    def setup_method(self):
        self._preprocessor = ASRPreprocessor()

    def test__short_audio_single_segment(self):
        """Audio shorter than 60 seconds should produce exactly one segment"""
        audio = [torch.zeros(FIREREDASR2_INFERENCE_MAX_FRAMES * 10)]  # 10 s

        result = list(self._preprocessor.preprocess_inputs(audio))

        assert len(result) == 1
        assert result[0].metadata.input_ordering == 0

    def test__long_audio_gets_segmented(self):
        """Audio longer than 60 seconds must be split into multiple segments"""
        # 70 s of silence — segment_by_silence will fall back to fixed splits
        audio = [torch.zeros(FIREREDASR2_INFERENCE_MAX_FRAMES * 70)]

        result = list(self._preprocessor.preprocess_inputs(audio))

        assert len(result) > 1
        for seg in result:
            assert seg.metadata.input_ordering == 0

    def test__multiple_inputs_ordering(self):
        """input_ordering must match the original list index"""
        audio = [
            torch.zeros(FIREREDASR2_INFERENCE_MAX_FRAMES * 5),
            torch.zeros(FIREREDASR2_INFERENCE_MAX_FRAMES * 3),
        ]

        result = list(self._preprocessor.preprocess_inputs(audio))

        orderings = [r.metadata.input_ordering for r in result]
        assert orderings == [0, 1]

    def test__write_wavs_to_fs(self, tmpdir):
        """When output_dir is provided, wav files are written to disk"""
        output_dir = Path(tmpdir)
        audio = [torch.zeros(FIREREDASR2_INFERENCE_MAX_FRAMES * 2)]

        result = list(
            self._preprocessor.preprocess_inputs(audio, output_dir=output_dir)
        )

        assert len(result) == 1
        saved = output_dir / result[0].metadata.preprocessed_file_path
        assert saved.exists()


class TestASRPostprocessor:
    def setup_method(self):
        self._postprocessor = ASRPostprocessor()

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
