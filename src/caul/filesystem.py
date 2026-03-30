from tempfile import mkstemp
from typing import TYPE_CHECKING

from caul.constant import DEFAULT_SAMPLE_RATE, TARGET_FORMAT

if TYPE_CHECKING:
    import torch


def save_tensor(audio_tensor: "torch.Tensor") -> str:
    """Filesystem routine for audio tensor; defaults to wav

    :param audio_tensor: input tensor
    :return: string file uri
    """
    # TODO: Change paths to run_id + tensor uuid + pagination
    #  Allow for remote paths
    import torchaudio  # pylint: disable=import-outside-toplevel

    _, file_path = mkstemp()

    # torchcodec requires this
    file_path = f"{file_path}.wav"

    # Channel required as first dim
    audio_tensor = audio_tensor.unsqueeze(0)

    torchaudio.save(
        file_path, audio_tensor, sample_rate=DEFAULT_SAMPLE_RATE, format=TARGET_FORMAT
    )

    return file_path
