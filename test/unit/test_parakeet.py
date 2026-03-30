import pytest

from caul.asr_pipeline import ASRPipeline
from caul.constant import EXPECTED_SAMPLE_MINUTE, TorchDevice
from caul.objects import ASRResult

import torch

from caul.tasks import ParakeetPostprocessor, ParakeetPreprocessor


def test__parakeet_batching_unbatching():
    # pylint: disable=R1728
    """Test audio segmentation with batching (max length 24 minutes) and unbatching"""
    preprocessor = ParakeetPreprocessor()

    audio = [
        torch.zeros([EXPECTED_SAMPLE_MINUTE * i]) for i in [12, 11, 5, 4, 7, 10, 30]
    ]

    result = preprocessor.batch_audio_tensors(preprocessor.preprocess_inputs(audio))

    assert [
        [
            (r.metadata.input_ordering, r.tensor.shape[-1] / EXPECTED_SAMPLE_MINUTE)
            for r in re
        ]
        for re in result
    ] == [[(0, 12.0), (3, 4.0)], [(1, 11.0), (2, 5.0)], [(5, 10.0), (4, 7.0)]]


def test__parakeet_unbatching():
    """Test parakeet unbatching including reassembling segmented tensors"""
    postprocessor = ParakeetPostprocessor()

    results = [
        ASRResult(input_ordering=2, transcription=[(1, 2, "two one")], score=2.1),
        ASRResult(input_ordering=0, transcription=[(0, 1, "zero")], score=0.0),
        ASRResult(input_ordering=2, transcription=[(2, 3, "two two")], score=2.2),
        ASRResult(input_ordering=1, transcription=[(0, 1, "one")], score=1.0),
    ]

    postprocessed_result = postprocessor.process(results)

    expected = [
        results[1],
        results[3],
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
