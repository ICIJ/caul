import logging
import uuid
from collections.abc import Callable, Iterable
from hashlib import sha256
from itertools import repeat
from pathlib import Path
from typing import TYPE_CHECKING, Self

from caul_core import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_LARGE_FILE_THRESHOLD_BYTES,
    DEFAULT_MAX_FRAMES,
    DEFAULT_SAMPLE_RATE,
    BasePreprocessorConfig,
    InputMetadata,
    PreprocessedInput,
    PreprocessedInputWithTensor,
    Preprocessor,
    PreprocessorOutput,
)
from caul_core.asr_task import PreprocessorInput
from caul_core.objects import Error
from torch import Tensor

from caul.exception import UnreadableAudio
from caul.filesystem import save_tensor
from caul.segmentation import segment_by_silence
from caul.segmentation.methods import SegmentationFunction
from caul.task_defaults import generic_batching_fn

if TYPE_CHECKING:
    import numpy as np
    import torch

_NoneType = type(None)

logger = logging.getLogger(__name__)


class ASRPreprocessorMixin(Preprocessor):
    """Preprocessing logic for ASR model inputs"""

    def __init__(
        self,
        batching_fn: Callable = generic_batching_fn,
        max_frames: int = DEFAULT_MAX_FRAMES,
        batch_size: int = DEFAULT_BATCH_SIZE,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        large_file_threshold_bytes: int = DEFAULT_LARGE_FILE_THRESHOLD_BYTES,
        segmentation_fn: SegmentationFunction = segment_by_silence,
        reported_errors: tuple[type[Exception]] | None = None,
    ):
        super().__init__()
        self._batch_fn = batching_fn
        self._max_frames = max_frames
        self._batch_size = batch_size
        self._sample_rate = sample_rate
        self._large_file_threshold_bytes = large_file_threshold_bytes
        self._segmentation_fn = segmentation_fn
        if reported_errors is None:
            reported_errors = (UnreadableAudio,)
        self._reported_errors = reported_errors

    @classmethod
    def _from_config(cls, config: BasePreprocessorConfig, **extras) -> Self:
        # TODO: configure segmentation fn
        return cls(
            max_frames=config.max_frames,
            batch_size=config.batch_size,
            sample_rate=config.sample_rate,
            large_file_threshold_bytes=config.large_file_threshold_bytes,
        )

    def process(
        self,
        inputs: "Iterable[np.ndarray | torch.Tensor | str] | np.ndarray | torch.Tensor | str",
        input_sample_rates: Iterable[int] | int | None = None,
        output_dir: Path | None = None,
        **kwargs,
    ) -> Iterable[list[PreprocessorOutput]]:
        """Segment and batch audio inputs

        :param inputs: List of np.ndarray or torch.Tensor or str, or singleton of same types
        :param input_sample_rates: sample rate(s) of audio inputs
        :param output_dir: optional directory to write preprocessed wav segments
        :return: batches of indexed preprocessed audio tensors (input_idx, preprocessed_input)
        """
        preprocessed_inputs = self.preprocess_inputs(
            inputs, input_sample_rates, output_dir=output_dir
        )
        for batch in self._batch_fn(preprocessed_inputs, self._batch_size):
            for start in range(0, len(batch), self._batch_size):
                yield batch[start : start + self._batch_size]

    def preprocess_inputs(  # pylint: disable=too-many-locals
        self,
        inputs: PreprocessorInput,
        input_sample_rates: Iterable[int] | int | None = None,
        output_dir: str | Path | None = None,
    ) -> Iterable[PreprocessorOutput]:
        """Accepts audio inputs as a list of file paths, np.ndarray, or torch.Tensor, converting to
        torch.Tensor, normalizing, segmenting inputs longer than segment_max and batching segments

        :param inputs: List of np.ndarray or torch.Tensor or str, or a singleton of same types
        :param input_sample_rates: sample rate(s) of audio inputs
        :param output_dir: if provided, save segments as wav files here
        :return: List of processed inputs
        """

        if output_dir is not None and not isinstance(output_dir, Path):
            output_dir = Path(output_dir)

        if isinstance(input_sample_rates, (int, _NoneType)):
            input_sample_rates = repeat(input_sample_rates)
            inputs_and_sample_rates = zip(inputs, input_sample_rates, strict=False)
        else:
            inputs_and_sample_rates = zip(inputs, input_sample_rates, strict=True)

        audio_path, audio_format = None, None
        for input_idx, (audio_input, sample_rate) in enumerate(inputs_and_sample_rates):
            if isinstance(audio_input, str):
                audio_input = Path(audio_input)
            try:
                audio, sample_rate, audio_path, audio_format = self._load_audio(
                    audio_input, sample_rate
                )
                segments = self._segment_audio(audio, sample_rate)
                for seg_idx, (audio_segment, segment_duration) in enumerate(segments):
                    audio_segment = self._preprocess_segment(audio_segment)
                    segment_path = None
                    if output_dir is not None:
                        segment_path = _persist_segment(
                            audio_segment, seg_idx, audio_path, output_dir=output_dir
                        ).relative_to(output_dir)
                    metadata = InputMetadata(
                        input_ordering=input_idx,
                        duration_s=segment_duration,
                        input_format=audio_format,
                        input_file_path=audio_path,
                        preprocessed_file_path=segment_path,
                    )
                    if metadata.preprocessed_file_path is None:
                        yield PreprocessedInputWithTensor(
                            metadata=metadata, tensor=audio_segment
                        )
                    else:
                        yield PreprocessedInput(metadata=metadata)
                    seg_idx += 1
            # Catch expected audio processing errors, let the other stop the processing
            # (implem bug or unexpected errors which should be dealt with)
            except self._reported_errors as e:
                if audio_format is not None:
                    # We just call logger.exception which will log the full trace
                    logger.exception(
                        "error while preprocessing audio %s. Skipping !", audio_format
                    )
                    yield PreprocessedInput(
                        metadata=InputMetadata(
                            input_ordering=input_idx,
                            duration_s=0.0,
                            input_format=audio_format,
                            input_file_path=audio_path,
                            preprocessed_file_path=None,
                            error=Error.from_exception(e),
                        )
                    )

    def _segment_audio(
        self, audio_chunks: Iterable[Tensor], sample_rate: int
    ) -> Iterable[tuple[Tensor, float]]:
        for chunk in audio_chunks:
            n_frames = chunk.shape[-1]
            if n_frames <= self._max_frames:
                yield (chunk, n_frames / sample_rate)
                continue
            max_segment_len_s = self._max_frames / sample_rate
            for s in self._segmentation_fn(
                chunk, sample_rate=sample_rate, max_segment_len_s=max_segment_len_s
            ):
                yield (s.tensor, s.duration)

    def _load_audio(
        self, audio_input: "np.ndarray | torch.Tensor | Path", sample_rate: int | None
    ) -> tuple[Iterable[Tensor], int, Path | None, str | None]:
        import numpy as np

        audio_path = None
        audio_format = None
        if isinstance(audio_input, Path):
            audio_path = Path(audio_input)
            audio_format = audio_path.suffix.removeprefix(".") or None
            sample_rate = self._sample_rate
            audio_chunks = self._load_file_as_chunks(audio_path, sample_rate)
        else:
            if isinstance(audio_input, np.ndarray):
                import torch  # pylint: disable=import-outside-toplevel

                audio_input = torch.Tensor(audio_input)
            if sample_rate is None:
                sample_rate = self._sample_rate
                audio_input = self._normalize(audio_input, sample_rate)
            elif len(audio_input.shape) > 1:
                audio_input = audio_input.squeeze(0)
            audio_chunks = iter([audio_input])
        return audio_chunks, sample_rate, audio_path, audio_format

    def _preprocess_segment(self, audio_tensor: "torch.Tensor") -> "torch.Tensor":
        """Stub for subclasses that have specific audio preprocessing logic"""
        return audio_tensor

    def _load_file_as_chunks(
        self, path: Path, sample_rate: int
    ) -> "Iterable[torch.Tensor]":
        """Return a lazy iterator of normalized 1D audio chunks at self._sample_rate.

        Reads file metadata first; falls back to a single eager load for small files.

        :param path: path to audio file
        :return: Iterable of tensor segments
        """
        from torchcodec.decoders import (
            AudioDecoder,
        )  # pylint: disable=import-outside-toplevel

        try:
            native_meta = AudioDecoder(path).metadata
        except ValueError as e:
            raise UnreadableAudio(path) from e
        num_frames = int(native_meta.duration_seconds * native_meta.sample_rate)
        estimated_bytes = num_frames * native_meta.num_channels * 4  # float32

        try:
            if estimated_bytes > self._large_file_threshold_bytes:
                yield from self._iter_audio_chunks(path, native_meta.duration_seconds)
            else:
                yield load_audio(path, sample_rate=sample_rate, num_channels=1)
        except Exception as e:
            raise UnreadableAudio(path) from e

    def _iter_audio_chunks(
        self, path: Path, total_duration_s: float
    ) -> "Iterable[torch.Tensor]":
        """Lazily decode a large audio file in max-frame-sized windows.

        :param path: path to audio file
        :param total_duration_s: total duration of audio file in seconds
        :return: Iterable of tensor segments
        """
        from torchcodec.decoders import (
            AudioDecoder,
        )  # pylint: disable=import-outside-toplevel

        chunk_duration_s = self._max_frames / self._sample_rate
        decoder = AudioDecoder(path, num_channels=1, sample_rate=self._sample_rate)
        start = 0.0
        while start < total_duration_s:
            end = min(start + chunk_duration_s, total_duration_s)
            samples = decoder.get_samples_played_in_range(
                start_seconds=start, stop_seconds=end
            )
            yield samples.data.squeeze()
            start = end

    def _normalize(
        self, audio_tensor: "torch.Tensor", sample_rate: int
    ) -> "torch.Tensor":
        """Normalize audio_tensor (single channel, sample rate = 16000)

        :param audio_tensor: input tensor
        :param sample_rate: input sample rate
        :return: normalized 1D tensor at self._sample_rate
        """
        if sample_rate != self._sample_rate:
            audio_tensor = _resample_audio(
                audio_tensor, self._sample_rate, target_rate=sample_rate
            )

        # Stereo dims (channels, aud_length); need mono (aud_length)
        if len(audio_tensor.shape) > 1:
            audio_tensor = audio_tensor.squeeze(0)

        return audio_tensor


def _persist_segment(
    audio_segment: Tensor, index: int, audio_path: Path | None, *, output_dir: Path
) -> Path:
    original_file = (
        _displayable_prefix(audio_path) if audio_path is not None else uuid.uuid4().hex
    )
    segment_name = f"{original_file}-{index}.wav"
    segment_path = output_dir / segment_name
    save_tensor(audio_segment, segment_path)
    return segment_path


def _resample_audio(
    audio: "torch.Tensor", sample_rate: int, *, target_rate: int
) -> "torch.Tensor":
    from torchcodec.encoders import AudioEncoder

    encoder = AudioEncoder(samples=audio, sample_rate=sample_rate)

    return encoder.to_tensor(format="wav", num_channels=1, sample_rate=target_rate)


def _displayable_prefix(
    path: Path, component_size_limit: int = 10, deterministic: bool = False
) -> str:
    displayable_file_name = path.name[:component_size_limit].replace(".", "__")
    if deterministic:
        uid = sha256(str(path).encode()).hexdigest()[:20]
    else:
        uid = uuid.uuid4().hex[:20]
    return f"{displayable_file_name}-{uid}"


def load_audio(
    path: str | Path, sample_rate: int = DEFAULT_SAMPLE_RATE, *, num_channels: int = 1
) -> "torch.Tensor":
    from torchcodec.decoders import (
        AudioDecoder,
    )  # pylint: disable=import-outside-toplevel

    samples = AudioDecoder(path, num_channels=num_channels, sample_rate=sample_rate)
    return samples.get_all_samples().data.squeeze()
