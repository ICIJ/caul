import logging
from collections.abc import Iterable
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING

from caul_core import (
    ASRModel,
    InferenceRunner,
    ParakeetTrtInferenceRunnerConfig,
    TorchDevice,
)
from icij_common.registrable import FromConfig
from torchaudio.models import Hypothesis

from ...trt import import_trt
from ...trt.handler import TrtInferenceHandler
from ...utils import load_audio
from ..inference.parakeet import ParakeetInferenceRunner
from .trt_inference import TrtInferenceMixin

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import torch


@cache
def _decoder_joint_connector():
    from nemo.core.connectors.save_restore_connector import (
        SaveRestoreConnector,
    )  # pylint: disable=import-outside-toplevel

    class DecoderJointConnector(SaveRestoreConnector):
        @staticmethod
        def _load_state_dict_from_disk(
            model_weights: dict, map_location: TorchDevice = "cpu"
        ) -> dict:
            """Maps model weights into virtual address space

            :param model_weights: model weights
            :param map_location: device to load weights to
            :return: decoder + joint weights
            """
            import torch  # pylint: disable=import-outside-toplevel

            weight_pointers = torch.load(
                model_weights,
                map_location=map_location,
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
class ParakeetTrtInferenceRunner(ParakeetInferenceRunner, TrtInferenceMixin):
    """Inference handler for NVIDIA parakeet models converted to TRT. Expects only the
    encoder to be converted, passing its output to decoder and joint layers using Nemo.
    Note that batch_size must match the shape profile used to convert to TRT.
    """

    def __init__(
        self,
        model_path: Path | str,
        engine_path: Path | str,
        device: TorchDevice = TorchDevice.CPU,
        return_timestamps: bool = True,
        batch_size: int = 4,
    ):
        ParakeetInferenceRunner.__init__(
            self,
            device=device,
            return_timestamps=return_timestamps,
            batch_size=batch_size,
        )
        TrtInferenceMixin.__init__(self)

        self._model_path = str(model_path)
        self._engine_path = str(engine_path)

    @classmethod
    def _from_config(
        cls, config: ParakeetTrtInferenceRunnerConfig, **extras
    ) -> FromConfig:
        return cls(
            model_path=config.model_path,
            engine_path=config.engine_path,
            return_timestamps=config.return_timestamps,
            **extras,
        )

    def __enter__(self):
        import nemo.collections.asr as nemo_asr  # pylint: disable=import-outside-toplevel

        trt = import_trt()

        with open(self._engine_path, "rb") as f:
            self._encoder = trt.Runtime(
                trt.Logger(trt.Logger.ERROR)
            ).deserialize_cuda_engine(f.read())

        self._decoder = nemo_asr.models.ASRModel.restore_from(
            self._model_path,
            map_location=self._torch_device,
            save_restore_connector=_decoder_joint_connector(),
            strict=False,
        ).eval()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # ParakeetInferenceRunner and TrtInferenceMixin both define __exit__; MRO would
        # otherwise resolve to InferenceRunner's, which never releases the TRT engine or
        # the restored decoder.
        return TrtInferenceMixin.__exit__(self, exc_type, exc_val, exc_tb)

    def _transcribe(
        self,
        audio_inputs: "torch.Tensor | str | Path | Iterable[torch.Tensor | str | Path]",
        trt_device: TorchDevice = None,
    ) -> list[Hypothesis] | list[list[Hypothesis]]:
        """Transcribe audio tensors

        :param audio_inputs: audio tensors or audio file paths
        :return: transcription results
        """
        import torch  # pylint: disable=import-outside-toplevel

        # trt only runs on cuda
        if trt_device is None:
            trt_device = torch.device("cuda")

        if isinstance(audio_inputs, torch.Tensor):
            # 1D is a single signal, 2D is a batch of signals
            audio_inputs = (
                [audio_inputs] if audio_inputs.dim() == 1 else list(audio_inputs)
            )
        elif isinstance(audio_inputs, (str, Path)) or not isinstance(
            audio_inputs, Iterable
        ):
            audio_inputs = [audio_inputs]

        # NeMo's transcribe() accepts file paths; TRT needs tensors, so load them here
        audio_inputs = [
            load_audio(ai) if isinstance(ai, (str, Path)) else ai for ai in audio_inputs
        ]

        audio_inputs_len = torch.tensor(
            [ai.shape[-1] for ai in audio_inputs], dtype=torch.int32
        ).to(trt_device)

        # pad to len(max(t)), setting dim[0] to batch_size
        audio_inputs = torch.nn.utils.rnn.pad_sequence(
            audio_inputs, batch_first=True
        ).to(trt_device)

        with TrtInferenceHandler(self._encoder) as handler:
            enc_out, enc_len = handler.infer(
                {"input_signal": audio_inputs, "input_signal_length": audio_inputs_len}
            )

        enc_out = enc_out.to(self._torch_device)
        enc_len = enc_len.to(self._torch_device)

        with torch.no_grad():
            return self._decoder.decoding.rnnt_decoder_predictions_tensor(
                enc_out, enc_len, return_hypotheses=True
            )
