import inspect
import math
import sys
from pathlib import Path

import pytest
import torch
from torchcodec.encoders import AudioEncoder

from caul_core import DEFAULT_SAMPLE_RATE

from caul.tasks.preprocessing.asr_preprocessor import ASRPreprocessorMixin, load_audio
from test.unit.constant import TEST_WAV_PATH


@pytest.fixture
def long_wav(tmp_path) -> Path:
    """3-second silent wav at 16 kHz, long enough for several 1-second chunks"""
    audio = torch.zeros(1, DEFAULT_SAMPLE_RATE * 3)
    path = tmp_path / "long.wav"
    AudioEncoder(audio, sample_rate=DEFAULT_SAMPLE_RATE).to_file(path)
    return path


def _chunking_preprocessor(max_frames: int = DEFAULT_SAMPLE_RATE) -> ASRPreprocessorMixin:
    """Preprocessor with threshold=1 so every file triggers the chunked path"""
    return ASRPreprocessorMixin(large_file_threshold_bytes=1, max_frames=max_frames)


def test_load_stereo_24bit_audio() -> None:
    tensor = load_audio(TEST_WAV_PATH)
    assert len(tensor.shape) == 1


class TestASRPreprocessorChunking:
    def test__below_threshold_loads_eagerly(self, long_wav):
        """Files under the byte threshold produce exactly one chunk"""
        preprocessor = ASRPreprocessorMixin(large_file_threshold_bytes=sys.maxsize)
        chunks = list(preprocessor._load_file_as_chunks(str(long_wav)))
        assert len(chunks) == 1

    def test__above_threshold_produces_multiple_chunks(self, long_wav):
        """Files over the byte threshold are split into several time-window chunks"""
        chunks = list(_chunking_preprocessor()._load_file_as_chunks(str(long_wav)))
        assert len(chunks) > 1

    def test__chunk_count_matches_expected(self, long_wav):
        """Number of chunks equals ceil(file_duration / chunk_duration)"""
        from torchcodec.decoders import AudioDecoder

        preprocessor = _chunking_preprocessor()
        duration_s = AudioDecoder(str(long_wav)).metadata.duration_seconds
        chunk_duration_s = DEFAULT_SAMPLE_RATE / DEFAULT_SAMPLE_RATE
        expected = math.ceil(duration_s / chunk_duration_s)

        chunks = list(preprocessor._load_file_as_chunks(str(long_wav)))
        assert len(chunks) == expected

    def test__each_chunk_at_most_max_frames(self, long_wav):
        """No chunk contains more than max_frames samples"""
        chunks = list(_chunking_preprocessor()._load_file_as_chunks(str(long_wav)))
        for chunk in chunks:
            # +1 tolerance for floating-point rounding in the decoder's resampler
            assert chunk.shape[-1] <= DEFAULT_SAMPLE_RATE + 1

    def test__chunks_are_1d_tensors(self, long_wav):
        """Chunks are 1D (mono, squeezed) tensors at the target sample rate"""
        chunks = list(_chunking_preprocessor()._load_file_as_chunks(str(long_wav)))
        for chunk in chunks:
            assert len(chunk.shape) == 1

    def test__load_file_as_chunks_is_lazy(self, long_wav):
        """_load_file_as_chunks must return a generator, not a materialized list"""
        result = _chunking_preprocessor()._load_file_as_chunks(str(long_wav))
        assert inspect.isgenerator(result)

    def test__chunking_triggered_by_byte_threshold(self, long_wav):
        """Threshold comparison uses byte size derived from file metadata"""
        from torchcodec.decoders import AudioDecoder

        meta = AudioDecoder(str(long_wav)).metadata
        estimated_bytes = (
            int(meta.duration_seconds * meta.sample_rate) * meta.num_channels * 4
        )

        eager = ASRPreprocessorMixin(
            large_file_threshold_bytes=estimated_bytes + 1,
            max_frames=DEFAULT_SAMPLE_RATE,
        )
        chunked = ASRPreprocessorMixin(
            large_file_threshold_bytes=estimated_bytes - 1,
            max_frames=DEFAULT_SAMPLE_RATE,
        )

        assert len(list(eager._load_file_as_chunks(str(long_wav)))) == 1
        assert len(list(chunked._load_file_as_chunks(str(long_wav)))) > 1

    def test__seg_i_contiguous_across_chunks(self, long_wav, tmp_path):
        """Segment output file indices are sequential even across chunk boundaries"""
        preprocessor = _chunking_preprocessor()
        output_dir = tmp_path / "segments"
        output_dir.mkdir()
        results = list(
            preprocessor.preprocess_inputs([str(long_wav)], output_dir=output_dir)
        )
        indices = sorted(int(f.stem.rsplit("-", 1)[-1]) for f in output_dir.iterdir())
        assert indices == list(range(len(results)))

    def test__all_segments_share_input_ordering(self, long_wav):
        """All segments produced from one file have the same input_ordering"""
        results = list(_chunking_preprocessor().preprocess_inputs([str(long_wav)]))
        assert all(r.metadata.input_ordering == 0 for r in results)

    def test__two_large_files_have_distinct_orderings(self, long_wav):
        """Two chunked files produce segments with input_ordering 0 and 1"""
        results = list(
            _chunking_preprocessor().preprocess_inputs([str(long_wav), str(long_wav)])
        )
        assert {r.metadata.input_ordering for r in results} == {0, 1}
