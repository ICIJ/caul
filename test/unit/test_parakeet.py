from caul.configs import ParakeetConfig
from caul.constant import EXPECTED_SAMPLE_MINUTE, DEVICE_CPU, DEVICE_MPS
from caul.model_handlers.objects import ParakeetModelHandlerResult
from caul.tasks.postprocessing.parakeet_postprocessor import ParakeetPostprocessor
from caul.tasks.preprocessing.parakeet_preprocessor import ParakeetPreprocessor

import torch


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
        ParakeetModelHandlerResult(
            input_ordering=2, transcription=[(0, 1, "two one")], score=2.1
        ),
        ParakeetModelHandlerResult(
            input_ordering=0, transcription=[(0, 1, "zero")], score=0.0
        ),
        ParakeetModelHandlerResult(
            input_ordering=2, transcription=[(1, 2, "two two")], score=2.2
        ),
        ParakeetModelHandlerResult(
            input_ordering=1, transcription=[(0, 1, "one")], score=1.0
        ),
    ]

    postprocessed_result = postprocessor.process(results)

    assert postprocessed_result == [
        results[1],
        results[3],
        ParakeetModelHandlerResult(
            input_ordering=2,
            transcription=[(0, 1, "two one"), (1, 2, "two two")],
            score=2.15,
        ),
    ]


def test__parakeet_device_setting():
    """Test parakeet device setting"""
    config = ParakeetConfig()
    handler = config.handler_from_config()

    assert handler.device == torch.device(DEVICE_CPU)
    assert handler.inference_handler.device == torch.device(DEVICE_CPU)

    handler.set_device(DEVICE_MPS)

    assert handler.device == torch.device(DEVICE_MPS)
    assert handler.inference_handler.device == torch.device(DEVICE_MPS)
