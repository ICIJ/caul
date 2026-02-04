from tempfile import mkstemp

import torch
import torchaudio

from caul.constant import EXPECTED_SAMPLE_RATE, EXPECTED_FORMAT


def save_tensor(audio_tensor: torch.Tensor) -> str:
    """Filesystem routine for audio tensor; defaults to wav

    :param audio_tensor: input tensor
    :return: string file uri
    """
    # TODO: Change paths to run_id + tensor uuid + pagination
    #  Allow for remote paths

    _, file_path = mkstemp()

    # torchcodec requires this
    file_path = f"{file_path}.wav"

    # Channel required as first dim
    audio_tensor = audio_tensor.unsqueeze(0)

    torchaudio.save(
        file_path,
        audio_tensor,
        sample_rate=EXPECTED_SAMPLE_RATE,
        format=EXPECTED_FORMAT,
    )

    return file_path
