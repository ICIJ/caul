import logging
from functools import lru_cache

from pathlib import Path
from typing import Iterable

from icij_common.registrable import FromConfig

from caul_core.config import ParakeetTrtInferenceRunnerConfig
from caul_core.constants import PARAKEET_MODEL_REF
from caul_core.objects import TorchDevice, ASRModel, ASRResult
from .. import ParakeetInferenceRunner
from ..asr_task import InferenceRunner
from ...trt.handler import TrtInferenceHandler

logger = logging.getLogger(__name__)


@lru_cache(maxsize=None)
def _decoder_joint_connector():
    from nemo.core.connectors.save_restore_connector import (
        SaveRestoreConnector,
    )  # pylint: disable=import-outside-toplevel

    class DecoderJointConnector(SaveRestoreConnector):

        @staticmethod
        def _load_state_dict_from_disk(
            model_weights: dict, device: "TorchDevice | torch.device"
        ) -> dict:
            """Maps model weights into virtual address space

            :param model_weights: model weights
            :param device: device to load weights to
            :return: decoder + joint weights
            """
            import torch  # pylint: disable=import-outside-toplevel

            weight_pointers = torch.load(
                model_weights,
                map_location=device,
                mmap=True,
                weights_only=True,
            )
            without_encoder = {
                k: v
                for k, v in weight_pointers.items()
                if k.startswith(("decoder.", "joint."))
            }
            del weight_pointers  # release mmap handles
            return without_encoder

    return DecoderJointConnector()


@InferenceRunner.register(ASRModel.PARAKEET_TRT)
class ParakeetTrtInferenceRunner(ParakeetInferenceRunner):
    """Inference handler for NVIDIA parakeet models converted to TRT. Expects only the
    encoder to be converted, passing its output to decoder and joint layers using Nemo.
    Note that batch_size must match the shape profile used to convert to TRT.
    """

    _models = [PARAKEET_MODEL_REF]

    def __init__(
        self,
        model_path: Path | str,
        engine_path: Path | str,
        device: "TorchDevice | torch.device" = TorchDevice.CPU,
        return_timestamps: bool = True,
        batch_size: int = 4,
    ):
        super().__init__(
            device=device, return_timestamps=return_timestamps, batch_size=batch_size
        )

        self._model_path = str(model_path)
        self._engine_path = str(engine_path)
        self._encoder = None
        self._decoder = None

    @classmethod
    def _from_config(
        cls, config: ParakeetTrtInferenceRunnerConfig, **extras
    ) -> FromConfig:
        return cls(
            return_timestamps=config.return_timestamps,
            **extras,
        )

    def __enter__(self):
        import nemo.collections.asr as nemo_asr  # pylint: disable=import-outside-toplevel
        import torch  # pylint: disable=import-outside-toplevel
        import tensorrt as trt  # pylint: disable=import-outside-toplevel

        with open(self._engine_path, "rb") as f:
            self._encoder = trt.Runtime(
                trt.Logger(trt.Logger.ERROR)
            ).deserialize_cuda_engine(f.read())

        self._decoder = nemo_asr.models.ASRModel.restore_from(
            self._model_path,
            map_location=torch.device(self._device),
            save_restore_connector=_decoder_joint_connector(),
            strict=False,
        ).eval()

        return self

    def _transcribe(
        self,
        audio_inputs: "torch.Tensor | Iterable[torch.Tensor]",
        trt_device: "TorchDevice | torch.device" = None,
        **kwargs
    ) -> Iterable[ASRResult]:
        """Transcribe audio tensors

        :param audio_inputs: audio tensor inputs
        :return: transcription results
        """
        import torch  # pylint: disable=import-outside-toplevel

        # trt only runs on cuda
        if trt_device is None:
            trt_device = torch.device("cuda")

        if not isinstance(audio_inputs, Iterable):
            audio_inputs = [audio_inputs]

        # pad to len(max(t)), setting dim[0] to batch_size
        audio_inputs = torch.nn.utils.rnn.pad_sequence(
            audio_inputs, batch_first=True
        ).to(trt_device)

        audio_inputs_len = torch.tensor(
            [torch.tensor([ai.shape[-1]]) for ai in audio_inputs]
        ).to(trt_device)

        with TrtInferenceHandler(self._encoder) as handler:
            enc_out, enc_len = handler.infer(
                {"input_signal": audio_inputs, "input_signal_length": audio_inputs_len}
            )

        enc_out = enc_out.to(self._device)
        enc_len = enc_len.to(self._device)

        with torch.no_grad():
            return self._decoder.decoding.rnnt_decoder_predictions_tensor(
                enc_out, enc_len
            )
