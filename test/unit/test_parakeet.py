from caul.constant import PARAKEET_SAMPLE_MINUTE
from caul.tasks.inference.parakeet_inference import ParakeetInferenceHandlerResult
from caul.tasks.postprocessing.parakeet_postprocessor import ParakeetPostprocessor
from caul.tasks.preprocessing.parakeet_preprocessor import ParakeetPreprocessor

import torch


def test__parakeet_batching_unbatching():
    # pylint: disable=R1728
    """Test audio segmentation with batching (max length 24 minutes) and unbatching"""
    preprocessor = ParakeetPreprocessor()

    audio = [
        torch.zeros([PARAKEET_SAMPLE_MINUTE * i]) for i in [12, 11, 5, 4, 7, 10, 30]
    ]

    result = preprocessor.batch_audio_tensors(
        preprocessor.segment_audio_tensors(preprocessor.load_audio_tensors(audio))
    )

    assert [
        [(r[0], r[-1].shape[-1] / PARAKEET_SAMPLE_MINUTE) for r in re] for re in result
    ] == [
        [(6, 24.0)],
        [(0, 12.0), (1, 11.0)],
        [(5, 10.0), (4, 7.0), (6, 6.0)],
        [(2, 5.0), (3, 4.0)],
    ]


def test__parakeet_unbatching():
    """Test parakeet unbatching including reassembling segmented tensors"""
    postprocessor = ParakeetPostprocessor()

    results = [
        (
            2,
            ParakeetInferenceHandlerResult(
                transcription=[(0, 1, "two one")], score=2.1
            ),
        ),
        (
            0,
            ParakeetInferenceHandlerResult(transcription=[(0, 1, "zero")], score=0.0),
        ),
        (
            2,
            ParakeetInferenceHandlerResult(
                transcription=[(1, 2, "two two")], score=2.2
            ),
        ),
        (
            1,
            ParakeetInferenceHandlerResult(transcription=[(0, 1, "one")], score=1.0),
        ),
    ]

    postprocessed_result = postprocessor.process(results)

    assert postprocessed_result == [
        ParakeetInferenceHandlerResult(transcription=[(0, 1, "zero")], score=0.0),
        ParakeetInferenceHandlerResult(transcription=[(0, 1, "one")], score=1.0),
        ParakeetInferenceHandlerResult(
            transcription=[(0, 1, "two one"), (1, 2, "two two")], score=2.15
        ),
    ]
