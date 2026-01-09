import torch
import torchaudio

from tests.unit.constant import PARAKEET_MODEL, TEST_PATH

from caul import ASRWorker
from caul.model import ParakeetModel


def test__worker_with_single_parakeet_model_on_mps():
    """Test Parakeet on MPS; to note: tensors must be converted to float32 to work with metal, something Nemo
    doesn't do automatically if we only pass a file path, which is why we load with torchaudio and convert to
    torch."""
    model = ParakeetModel(PARAKEET_MODEL, "mps")
    worker = ASRWorker(models=model)

    worker.startup()

    # load wav, drop channel dim
    waveform, _ = torchaudio.load(TEST_PATH)
    waveform = torch.tensor(waveform.numpy())
    waveform = waveform.squeeze(0)

    transcription, score = worker.transcribe(waveform)[0]

    score = round(score, 0)

    assert transcription == "To embrace the chaos that they fought in this battle."
    assert score == -248


def test__worker_with_single_whisper_model():
    pass