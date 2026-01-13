from caul.constant import PARAKEET_SAMPLE_MINUTE
from test.unit.constant import PARAKEET_MODEL

import torch

from caul.model import ParakeetModelHandler


def test__parakeet_segmentation():
    # pylint: disable=R1728
    """Test audio segmentation with batching (max length 24 minutes)"""
    model = ParakeetModelHandler(PARAKEET_MODEL)

    audio = [
        torch.zeros([PARAKEET_SAMPLE_MINUTE * i]) for i in [12, 11, 5, 4, 7, 10, 30]
    ]

    result = model.segment_audio_tensors(model.load_audio_tensors(audio))

    assert [
        [r[-1].shape[-1] / PARAKEET_SAMPLE_MINUTE for r in re] for re in result
    ] == [[24.0], [12.0, 11.0], [10.0, 7.0, 6.0], [5.0, 4.0]]
