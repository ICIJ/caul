from test.unit.constant import PARAKEET_MODEL

import torch

from caul.model import ParakeetModelHandler


def test__parakeet_segmentation():
    # pylint: disable=R1728
    """Test audio segmentation with batching (max length 24 minutes)"""
    model = ParakeetModelHandler(PARAKEET_MODEL)

    audio = [
        torch.zeros([16000 * 60 * 12]),
        torch.zeros([16000 * 60 * 11]),
        torch.zeros([16000 * 60 * 5]),
        torch.zeros([16000 * 60 * 4]),
        torch.zeros([16000 * 60 * 7]),
        torch.zeros([16000 * 60 * 10]),
    ]

    result = model.segment_batch(audio)

    assert len(result) == 2
    assert [len(r) for r in result] == [4, 2]
    assert sum([len(r) for r in result]) == len(audio)
