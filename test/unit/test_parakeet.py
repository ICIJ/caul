from dataclasses import astuple

from more_itertools import flatten

from caul.constant import PARAKEET_SAMPLE_MINUTE
from caul.model.parakeet_model import ParakeetModelHandlerResult
from test.unit.constant import PARAKEET_MODEL

import torch

from caul.model import ParakeetModelHandler


def test__parakeet_batching_and_unbatching():
    # pylint: disable=R1728
    """Test audio segmentation with batching (max length 24 minutes) and unbatching"""
    model = ParakeetModelHandler(PARAKEET_MODEL)

    audio = [
        torch.zeros([PARAKEET_SAMPLE_MINUTE * i]) for i in [12, 11, 5, 4, 7, 10, 30]
    ]

    result = model.segment_audio_tensors(model.load_audio_tensors(audio))

    assert [
        [(r[0], r[-1].shape[-1] / PARAKEET_SAMPLE_MINUTE) for r in re] for re in result
    ] == [
        [(6, 24.0)],
        [(0, 12.0), (1, 11.0)],
        [(5, 10.0), (4, 7.0), (6, 6.0)],
        [(2, 5.0), (3, 4.0)],
    ]

    # Unbatch
    flattened_results = list(flatten(result))
    transcriptions = [
        [(0, 1, "six part one")],
        [(0, 1, "zero")],
        [(0, 1, "one")],
        [(0, 1, "five")],
        [(0, 1, "four")],
        [(1, 2, "six part two")],
        [(0, 1, "two")],
        [(0, 1, "three")],
    ]
    scores = [6.0, 0.0, 1.0, 5.0, 4.0, 7.0, 2.0, 3.0]
    results = [
        (
            input_idx,
            ParakeetModelHandlerResult(
                transcription=transcriptions[result_idx], score=scores[result_idx]
            ),
        )
        for (result_idx, (input_idx, _)) in enumerate(flattened_results)
    ]
    assert [astuple(result) for result in model.map_results_to_inputs(results)] == [
        ([(0, 1, "zero")], 0.0),
        ([(0, 1, "one")], 1.0),
        ([(0, 1, "two")], 2.0),
        ([(0, 1, "three")], 3.0),
        ([(0, 1, "four")], 4.0),
        ([(0, 1, "five")], 5.0),
        ([(0, 1, "six part one"), (1, 2, "six part two")], 6.5),
    ]
