from pathlib import Path
from typing import TYPE_CHECKING

from caul.constants import DEFAULT_BIT_RATE, DEFAULT_SAMPLE_RATE

if TYPE_CHECKING:
    import torch


def save_tensor(audio: "torch.Tensor", path: Path) -> None:
    """Filesystem routine for audio tensor; defaults to wav

    :param audio: input tensor
    :return: string file uri
    """
    # TODO: Change paths to run_id + tensor uuid + pagination
    #  Allow for remote paths
    from torchcodec.encoders import AudioEncoder

    # Channel required as first dim
    audio = audio.unsqueeze(0)
    encoder = AudioEncoder(samples=audio, sample_rate=DEFAULT_SAMPLE_RATE)
    encoder.to_file(
        path, sample_rate=DEFAULT_SAMPLE_RATE, num_channels=1, bit_rate=DEFAULT_BIT_RATE
    )
