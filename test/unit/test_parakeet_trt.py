from unittest.mock import MagicMock, patch

import numpy as np
import torch

from caul.tasks.inference.parakeet_trt import ParakeetTrtInferenceRunner
from caul_core.constants import PARAKEET_MODEL_REF

_ENGINE_PATH = "/fake/encoder.trt"
_INFERENCE_RUNNER_PATH = "caul.tasks.inference.parakeet_trt.TrtInferenceRunner"
_BATCH_SIZE = 2
_SIGNAL_LEN = 16000
_AUDIO_INPUT = torch.zeros(_BATCH_SIZE, _SIGNAL_LEN)
_ENC_OUT = np.zeros((_BATCH_SIZE, 50, 256), dtype=np.float32)
_ENC_OUT_LEN = np.full(_BATCH_SIZE, 50, dtype=np.int32)


def _mock_inference_runner():
    runner = ParakeetTrtInferenceRunner(PARAKEET_MODEL_REF, _ENGINE_PATH)
    runner._encoder = MagicMock()
    runner._decoder = MagicMock()
    return runner


def _mock_trt_runner(enc_out: np.ndarray, enc_len: np.ndarray):
    instance = MagicMock()
    instance.infer.return_value = (enc_out, enc_len)
    instance.__enter__ = MagicMock(return_value=instance)
    instance.__exit__ = MagicMock(return_value=False)
    return MagicMock(return_value=instance)


class TestParakeetTrtInferenceRunnerTranscribe:
    def test__builds_length_tensor_from_audio_shape(self):
        mock_inference_runner = _mock_inference_runner()
        mock_trt_runner = _mock_trt_runner(_ENC_OUT, _ENC_OUT_LEN)

        with patch(_INFERENCE_RUNNER_PATH, mock_trt_runner):
            mock_inference_runner.transcribe(_AUDIO_INPUT)

        called_inputs = mock_trt_runner.return_value.infer.call_args[0][0]
        input_signal_length = called_inputs["input_signal_length"]
        assert input_signal_length.shape == (_BATCH_SIZE,)
        assert input_signal_length.tolist() == [_SIGNAL_LEN] * _BATCH_SIZE

    def test__encoder_outputs_forwarded_to_decoder(self):
        mock_inference_runner = _mock_inference_runner()
        mock_trt_runner = _mock_trt_runner(_ENC_OUT, _ENC_OUT_LEN)

        with patch(_INFERENCE_RUNNER_PATH, mock_trt_runner):
            mock_inference_runner.transcribe(_AUDIO_INPUT)

        enc_out_arg, enc_len_arg = (
            mock_inference_runner._decoder.decoding.rnnt_decoder_predictions_tensor.call_args[
                0
            ]
        )
        assert torch.equal(enc_out_arg, torch.from_numpy(_ENC_OUT))
        assert torch.equal(enc_len_arg, torch.from_numpy(_ENC_OUT_LEN))

    def test__returns_decoder_predictions(self):
        mock_inference_runner = _mock_inference_runner()
        mock_trt_runner = _mock_trt_runner(_ENC_OUT, _ENC_OUT_LEN)
        expected = [MagicMock(), MagicMock()]
        mock_inference_runner._decoder.decoding.rnnt_decoder_predictions_tensor.return_value = (
            expected
        )

        with patch(_INFERENCE_RUNNER_PATH, mock_trt_runner):
            result = mock_inference_runner.transcribe(_AUDIO_INPUT)

        assert result is expected
