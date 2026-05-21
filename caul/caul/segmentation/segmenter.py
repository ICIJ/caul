import gc
from functools import partial
from typing import Any, Callable, Protocol, Self, TYPE_CHECKING

from icij_common.registrable import RegistrableFromConfig

from caul_core.constants import SILERO_VAD_MODEL
from caul_core.objects import VadModelRef
from .methods import (
    segment_by_pyannote_vad,
    segment_by_silence,
    segment_fixed,
    segment_by_silero_vad,
)
from .objects import (
    SegmentationConfig,
    TensorSegment,
    SegmentationStrategy,
    PyannoteVoiceSegmentationConfig,
)

if TYPE_CHECKING:
    import torch


def _load_vad_model() -> tuple["torch.nn.Module", Callable]:
    """Load silero VAD from torch.hub.

    :return: tuple of silero VAD model with VAD parsing function
    """
    import torch  # pylint: disable=import-outside-toplevel

    model, utils = torch.hub.load(
        VadModelRef.SILERO_MODEL, SILERO_VAD_MODEL, trust_repo=True
    )
    return model, utils[0]


class SegmentationFn(Protocol):  # pylint: disable=too-few-public-methods
    def __call__(
        self, audio_tensor: "torch.Tensor", **kwargs
    ) -> list[TensorSegment]: ...


class AudioSegmenter(RegistrableFromConfig):
    def __init__(self, config: SegmentationConfig):
        self._config = config
        args = self._config.model_dump()
        args.pop(SegmentationConfig.registry_key.default)
        self._args = args
        self._segmentation_fn: SegmentationFn | None = None

    def __enter__(self) -> Self: ...

    def __exit__(self, exc_type, exc_val, exc_tb): ...

    @classmethod
    def _from_config(cls, config: SegmentationConfig, **extras) -> Self:
        return cls(config)

    def segment(self, audio_tensor: "torch.Tensor") -> list[TensorSegment]:
        return self._segmentation_fn(  # pylint: disable=not-callable
            audio_tensor, **self._args
        )


@AudioSegmenter.register(SegmentationStrategy.FIXED)
class FixedSizeAudioSegmenter(AudioSegmenter):
    def __enter__(self) -> Self:
        self._segmentation_fn = segment_fixed
        return self


@AudioSegmenter.register(SegmentationStrategy.SILENCE)
class SilenceAudioSegmenter(AudioSegmenter):
    def __enter__(self) -> Self:
        self._segmentation_fn = segment_by_silence
        return self


@AudioSegmenter.register(SegmentationStrategy.VOICE_SILERO)
class SileroVoiceAudioSegmenter(AudioSegmenter):
    def __init__(self, config: SegmentationConfig, device: str = "cpu"):
        import torch  # pylint: disable=import-outside-toplevel

        super().__init__(config)
        self._vad_model: torch.nn.Module | None = None
        self._vad_parser_fn: Callable | None = None
        self._device = torch.device(device)

    def __enter__(self) -> Self:
        self._vad_model, self._vad_parser_fn = self._load_vad_model()

        self._segmentation_fn = partial(
            segment_by_silero_vad,
            vad_model=self._vad_model,
            vad_parser_fn=self._vad_parser_fn,
        )
        return self

    def _load_vad_model(self) -> tuple["torch.nn.Module", Callable]:
        model, parser_fn = _load_vad_model()

        return model.to(device=self._device), parser_fn

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        import torch  # pylint: disable=import-outside-toplevel

        self._vad_model = None
        self._vad_parser_fn = None
        try:
            # Depending on the exact accelerator running might fail to empty the cache
            # but that's OK
            torch.accelerator.empty_cache()
        except RuntimeError:
            pass
        gc.collect()


@AudioSegmenter.register(SegmentationStrategy.VOICE_PYANNOTE)
class PyannoteVoiceAudioSegmenter(AudioSegmenter):
    def __init__(
        self,
        config: PyannoteVoiceSegmentationConfig,
        hf_token: str,
        device: str = "cpu",
    ):
        import torch  # pylint: disable=import-outside-toplevel

        super().__init__(config)
        self._pipeline: Any | None = None
        self._hf_token = hf_token
        self._device = torch.device(device)

    def __enter__(self) -> Self:
        self._pipeline = self._load_pipeline()
        self._segmentation_fn = partial(
            segment_by_pyannote_vad, pipeline=self._pipeline
        )
        return self

    def _load_pipeline(self) -> Any:
        from pyannote.audio import Model  # pylint: disable=import-outside-toplevel
        from pyannote.audio.pipelines import (
            VoiceActivityDetection,
        )  # pylint: disable=import-outside-toplevel

        model = Model.from_pretrained(VadModelRef.PYANNOTE_MODEL, token=self._hf_token)

        return VoiceActivityDetection(segmentation=model).to(device=self._device)

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        import torch  # pylint: disable=import-outside-toplevel

        self._pipeline = None
        try:
            torch.accelerator.empty_cache()
        except RuntimeError:
            pass
        gc.collect()
