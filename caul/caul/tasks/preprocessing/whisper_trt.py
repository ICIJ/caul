from pathlib import Path
from typing import Self, TYPE_CHECKING, Callable

from caul_core.constants import (
    WHISPER_TRT_N_FFT,
    WHISPER_TRT_HOP_LENGTH,
    WHISPER_TRT_MAX_FRAMES,
    DEFAULT_BATCH_SIZE,
    WHISPER_TRT_MAX_NEW_TOKENS,
    WHISPER_TRT_PREPROCESSOR_LOG_RANGE_NORMALIZER,
    WHISPER_TRT_PREPROCESSOR_CLAMP_MIN,
    WHISPER_TRT_PREPROCESSOR_LOG_RANGE_MAX_SHIFT,
)
from caul.tasks.asr_task import Preprocessor
from caul.tasks.preprocessing.asr_preprocessor import ASRPreprocessor, _NoneType
from caul.utils import load_mel_filters
from caul_core.config import PreprocessorConfig, WhisperTrtPreprocessorConfig
from caul_core.objects import ASRModel

if TYPE_CHECKING:
    import torch


def _mel_filters_factory(
    n_mels: int, mel_filters_dir: Path, device: "str | torch.Device" = "cpu"
) -> Callable[[], "torch.Tensor"]:
    def _load_mel_filters() -> "torch.Tensor":
        return load_mel_filters(n_mels, mel_filters_dir, device)

    return _load_mel_filters


@Preprocessor.register(ASRModel.WHISPER_TRT)
class WhisperTrtPreprocessor(ASRPreprocessor):

    def __init__(
        self,
        n_mels: int = 80,
        mel_filters_factory: (
            Callable[
                [int, Path | str, "str | torch.Device"], Callable[[], "torch.Tensor"]
            ]
            | None
        ) = None,
        dtype: "str | None" = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        max_frames: int = WHISPER_TRT_MAX_FRAMES,
    ) -> None:
        import torch  # pylint: disable=import-outside-toplevel

        super().__init__(batch_size=batch_size, max_frames=max_frames)
        self._n_mels = n_mels
        self._mel_filters_factory = mel_filters_factory()
        self._mel_filters = None

        self._dtype = getattr(torch, dtype if dtype is not None else "float16")

    @classmethod
    def _from_config(
        cls,
        config: WhisperTrtPreprocessorConfig,
        mel_filters_factory: (
            Callable[
                [int, Path | str, "str | torch.Device"], Callable[[], "torch.Tensor"]
            ]
            | None
        ) = None,
        **extras,
    ) -> Self:
        return cls(
            n_mels=config.n_mels,
            mel_filters_factory=(
                mel_filters_factory(config.mel_filters_dir)
                if mel_filters_factory is not None
                else None
            ),
            dtype=config.dtype,
            batch_size=config.batch_size,
            max_frames=config.max_frames,
        )

    def __enter__(self) -> Self:
        self._mel_filters = (
            self._mel_filters_factory()
            if self._mel_filters_factory is not None
            else None
        )

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._mel_filters = None

    def _additional_preprocessing(self, audio_tensor: "torch.Tensor") -> "torch.Tensor":
        """Map input tensor to log mel spectrogram, needed by Whisper TRT

        :audio_tensor: input audio tensor
        :returns: log mel spectrogram
        """
        import torch  # pylint: disable=import-outside-toplevel

        window = torch.hann_window(WHISPER_TRT_N_FFT)
        stft = torch.stft(
            audio_tensor,
            WHISPER_TRT_N_FFT,
            WHISPER_TRT_HOP_LENGTH,
            window=window,
            return_complex=True,
        )
        magnitudes = stft[..., :-1].abs() ** 2

        mel_spec = (
            self._mel_filters @ magnitudes
            if self._mel_filters is not None
            else magnitudes
        )

        audio_tensor = torch.clamp(
            mel_spec, min=WHISPER_TRT_PREPROCESSOR_CLAMP_MIN
        ).log10()
        audio_tensor = torch.maximum(
            audio_tensor,
            audio_tensor.max() - WHISPER_TRT_PREPROCESSOR_LOG_RANGE_MAX_SHIFT,
        )
        audio_tensor = (
            audio_tensor + WHISPER_TRT_PREPROCESSOR_LOG_RANGE_NORMALIZER
        ) / WHISPER_TRT_PREPROCESSOR_LOG_RANGE_NORMALIZER
        audio_tensor = audio_tensor.unsqueeze(0)

        return audio_tensor.type(self._dtype)
