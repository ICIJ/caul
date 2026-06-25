from pathlib import Path
from typing import Self, TYPE_CHECKING

from caul_core.constants import (
    WHISPER_TRT_N_FFT,
    WHISPER_TRT_HOP_LENGTH,
    WHISPER_TRT_MAX_FRAMES,
    DEFAULT_BATCH_SIZE,
    WHISPER_TRT_MAX_NEW_TOKENS,
)
from caul.tasks.asr_task import Preprocessor
from caul.tasks.preprocessing.asr_preprocessor import ASRPreprocessor, _NoneType
from caul.utils import load_mel_filters
from caul_core.config import PreprocessorConfig
from caul_core.objects import ASRModel

if TYPE_CHECKING:
    import torch


@Preprocessor.register(ASRModel.WHISPER_TRT)
class WhisperTrtPreprocessor(ASRPreprocessor):

    def __init__(
        self,
        n_mels: int = 80,
        mel_filters_dir: str = None,
        dtype: "torch.dtype | None" = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        max_frames: int = WHISPER_TRT_MAX_FRAMES,
    ) -> None:
        import torch  # pylint: disable=import-outside-toplevel

        super().__init__(batch_size=batch_size, max_frames=max_frames)
        self._n_mels = n_mels
        self._mel_filters_dir = (
            Path(mel_filters_dir) if mel_filters_dir is not None else None
        )
        self._dtype = dtype if dtype is not None else torch.float16

    @classmethod
    def _from_config(cls, config: PreprocessorConfig, **extras) -> Self:
        return cls(
            n_mels=config.n_mels,
            mel_filters_dir=config.mel_filters_dir,
            dtype=config.dtype,
            batch_size=config.batch_size,
            max_frames=config.max_frames,
        )

    def _additional_preprocessing(self, audio_tensor: "torch.Tensor") -> "torch.Tensor":
        """Map input tensor to log mel spectrogram, needed by Whisper TRT

        :audio_tensor: input audio tensor
        :returns: log mel spectrogram
        """
        import torch  # noqa

        window = torch.hann_window(WHISPER_TRT_N_FFT)
        stft = torch.stft(
            audio_tensor,
            WHISPER_TRT_N_FFT,
            WHISPER_TRT_HOP_LENGTH,
            window=window,
            return_complex=True,
        )
        magnitudes = stft[..., :-1].abs() ** 2

        filters = load_mel_filters(self._n_mels, self._mel_filters_dir)
        mel_spec = filters @ magnitudes

        audio_tensor = torch.clamp(mel_spec, min=1e-10).log10()
        audio_tensor = torch.maximum(audio_tensor, audio_tensor.max() - 8.0)
        audio_tensor = (audio_tensor + 4.0) / 4.0
        audio_tensor = audio_tensor.unsqueeze(0)

        return audio_tensor.type(self._dtype)
