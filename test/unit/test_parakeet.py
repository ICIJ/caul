from pathlib import Path

import pytest

from caul.asr_pipeline import ASRPipeline
from caul.constants import DEFAULT_SAMPLE_RATE, TorchDevice
from caul.objects import ASRResult

import torch

from caul.tasks import ParakeetPostprocessor, ParakeetPreprocessor
from caul.tasks.preprocessing.parakeet import batch_audio_tensors


def test__parakeet_batching_unbatching():
    # pylint: disable=R1728
    """Test audio segmentation with batching (max length 20 minutes) and unbatching"""
    preprocessor = ParakeetPreprocessor()
    samples_per_min = DEFAULT_SAMPLE_RATE * 60

    audio = [torch.zeros([samples_per_min * i]) for i in [12, 11, 5, 4, 7, 10, 30]]

    result = batch_audio_tensors(preprocessor.preprocess_inputs(audio))

    assert [
        [(r.metadata.input_ordering, r.tensor.shape[-1] / samples_per_min) for r in re]
        for re in result
    ] == [
        [(6, 20.0)],
        [(6, 10.0)],
        [(0, 12.0), (3, 4.0)],
        [(1, 11.0), (2, 5.0)],
        [(5, 10.0), (4, 7.0)],
    ]


def test__parakeet_preprocess_inputs_to_fs(tmpdir):
    # Given
    output_dir = Path(tmpdir)
    preprocessor = ParakeetPreprocessor()
    audio = [torch.zeros([1])]
    # When
    result = list(preprocessor.preprocess_inputs(audio, output_dir=output_dir))
    # Then
    assert len(result) == 1
    result = result[0]
    save_path = output_dir / result.metadata.preprocessed_file_path
    assert save_path.exists()


def test__parakeet_unbatching_should_raise_for_unordered_inputs():
    # Given
    postprocessor = ParakeetPostprocessor()
    results = [
        ASRResult(input_ordering=2, transcription=[(1, 2, "two one")], score=2.1),
        ASRResult(input_ordering=0, transcription=[(0, 1, "zero")], score=0.0),
        ASRResult(input_ordering=2, transcription=[(2, 3, "two two")], score=2.2),
        ASRResult(input_ordering=1, transcription=[(0, 1, "one")], score=1.0),
    ]

    # When/Then
    expected_msg = "expected contiguous batches !"
    with pytest.raises(ValueError, match=expected_msg):
        list(postprocessor.process(results))


def test__parakeet_unbatching():
    """Test parakeet unbatching including reassembling segmented tensors"""
    postprocessor = ParakeetPostprocessor()

    results = [
        ASRResult(input_ordering=1, transcription=[(0, 1, "one")], score=1.0),
        ASRResult(input_ordering=0, transcription=[(0, 1, "zero")], score=0.0),
        ASRResult(input_ordering=2, transcription=[(2, 3, "two two")], score=2.2),
        ASRResult(input_ordering=2, transcription=[(1, 2, "two one")], score=2.1),
    ]

    postprocessed_result = list(postprocessor.process(results))

    expected = [
        results[0],
        results[1],
        ASRResult(
            input_ordering=2,
            transcription=[(1, 2, "two one"), (2, 3, "two two")],
            score=2.15,
        ),
    ]
    expected = [r.model_dump() for r in expected]
    postprocessed_result = [r.model_dump() for r in postprocessed_result]
    scores = [r.pop("score") for r in postprocessed_result]
    expected_scores = [expected.pop("score") for expected in expected]
    assert postprocessed_result == expected
    assert scores == pytest.approx(expected_scores, abs=1e-6)


def test__parakeet_device_setting():
    """Test parakeet device setting"""
    # Given
    pipeline = ASRPipeline.parakeet(TorchDevice.CPU)
    assert pipeline.tasks[1].device == torch.device("cpu")

    # When
    pipeline.set_device(TorchDevice.MPS)

    # Then
    assert pipeline.device == torch.device("mps")
    assert pipeline.tasks[1].device == torch.device("mps")
