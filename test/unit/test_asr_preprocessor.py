from caul.tasks.preprocessing.asr_preprocessor import load_audio
from test.unit.constant import TEST_WAV_PATH


def test_load_stereo_24bit_audio() -> None:
    tensor = load_audio(TEST_WAV_PATH)
    assert len(tensor.shape) == 1
