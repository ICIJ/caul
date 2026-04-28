import uuid
from itertools import repeat
from pathlib import Path
from typing import Iterable, Self, TYPE_CHECKING, Callable

from hashlib import sha256


from caul.task_defaults import generic_batching_fn
from caul.config import PreprocessorConfig
from caul.segmentation.methods import segment_by_silence
from caul.constants import DEFAULT_SAMPLE_RATE, DEFAULT_BATCH_SIZE, DEFAULT_MAX_FRAMES
from caul.filesystem import save_tensor
from caul.objects import (
    InputMetadata,
    PreprocessedInput,
    PreprocessedInputWithTensor,
    PreprocessorOutput,
)
from caul.tasks.asr_task import Preprocessor

if TYPE_CHECKING:
    import torch
    import numpy as np

_NoneType = type(None)


class ASRPreprocessor(Preprocessor):
    """Preprocessing logic for ASR model inputs"""

    def __init__(
        self,
        batching_fn: Callable = generic_batching_fn,
        max_frames: int = DEFAULT_MAX_FRAMES,
        batch_size: int = DEFAULT_BATCH_SIZE,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
    ):
        super().__init__()
        self._batch_fn = batching_fn
        self._max_frames = max_frames
        self._batch_size = batch_size
        self._sample_rate = sample_rate

    @classmethod
    def _from_config(cls, config: PreprocessorConfig, **extras) -> Self:
        return cls(
            max_frames=config.max_frames,
            batch_size=config.batch_size,
            sample_rate=config.sample_rate,
        )

    def process(
        self,
        inputs: "Iterable[np.ndarray | torch.Tensor | str] | np.ndarray | torch.Tensor | str",
        input_sample_rates: Iterable[int] | int = None,
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
        yield from self._batch_fn(preprocessed_inputs, self._batch_size)

    def preprocess_inputs(  # pylint: disable=too-many-locals
        self,
        inputs: Iterable["np.ndarray | torch.Tensor | str"],
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
        import numpy as np  # pylint: disable=import-outside-toplevel

        if output_dir is not None and not isinstance(output_dir, Path):
            output_dir = Path(output_dir)

        if isinstance(input_sample_rates, (int, _NoneType)):
            input_sample_rates = repeat(input_sample_rates)
            inputs_and_sample_rates = zip(inputs, input_sample_rates, strict=False)
        else:
            inputs_and_sample_rates = zip(inputs, input_sample_rates, strict=True)

        # Load arrays and divide into max_length segments
        for input_idx, (audio_input, sample_rate) in enumerate(inputs_and_sample_rates):
            input_file_path = None
            input_format = None

            # Load audio files as arrays
            if isinstance(audio_input, str):
                input_file_path = audio_input
                input_format = (
                    input_file_path.split(".")[-1]
                    if len(input_file_path.split(".")) > 1
                    else None
                )
                audio_input = load_audio(
                    audio_input, sample_rate=self._sample_rate, num_channels=1
                )

            if isinstance(audio_input, np.ndarray):
                import torch  # pylint: disable=import-outside-toplevel

                audio_input = torch.Tensor(audio_input)

            if sample_rate is None:
                sample_rate = self._sample_rate

            # Normalize
            if sample_rate != self._sample_rate:
                audio_input = self._normalize(audio_input, sample_rate)

            # Segment where necessary
            n_frames = audio_input.shape[-1]
            tensor_segments = [audio_input]

            if n_frames > self._max_frames:
                max_segment_len_s = self._max_frames / self._sample_rate
                tensor_segments = [
                    s.tensor
                    for s in segment_by_silence(
                        audio_input, max_segment_len_s=max_segment_len_s
                    )
                ]

            original_file = (
                _displayable_prefix(input_file_path)
                if input_file_path is not None
                else uuid.uuid4().hex
            )
            for seg_i, tensor_segment in enumerate(tensor_segments):
                segment_path = None
                # Create temporary filesystem reference if applicable
                if output_dir is not None:
                    segment_name = f"{original_file}-{seg_i}.wav"
                    segment_path = output_dir / segment_name
                    save_tensor(tensor_segment, segment_path)
                    segment_path = segment_path.relative_to(output_dir)
                # Create preprocessed input
                metadata = InputMetadata(
                    input_ordering=input_idx,
                    duration_s=n_frames / DEFAULT_SAMPLE_RATE,
                    input_format=input_format,
                    input_file_path=input_file_path,
                    preprocessed_file_path=segment_path,
                )
                if metadata.preprocessed_file_path is None:
                    yield PreprocessedInputWithTensor(
                        metadata=metadata, tensor=tensor_segment
                    )
                else:
                    yield PreprocessedInput(metadata=metadata)

    def _normalize(
        self, audio_tensor: "torch.Tensor", sample_rate: int
    ) -> "torch.Tensor":
        """Normalize audio_tensor (single channel, sample rate = 16000)

        :param audio_tensor: input tensor
        :param sample_rate: input sample rate
        :return: normalized 1D tensor at self._sample_rate
        """
        if sample_rate != self._sample_rate:
            audio_tensor = _resample_waveform(
                audio_tensor, self._sample_rate, target_rate=sample_rate
            )

        # Stereo dims (channels, aud_length); need mono (aud_length)
        if len(audio_tensor.shape) > 1:
            audio_tensor = audio_tensor.squeeze(0)

        return audio_tensor


def _resample_waveform(
    waveform: "torch.Tensor", sample_rate: int, *, target_rate: int
) -> "torch.Tensor":
    import torchaudio  # pylint: disable=import-outside-toplevel

    transform = torchaudio.transforms.Resample(sample_rate, target_rate)
    return transform(waveform)


def _displayable_prefix(path: str, component_size_limit: int = 10) -> str:
    path = Path(path)
    displayable_file_name = path.name[:component_size_limit].replace(".", "__")
    uid = sha256(str(path).encode()).hexdigest()[:20]
    return f"{displayable_file_name}-{uid}"


def load_audio(
    path: str | Path, sample_rate: int = DEFAULT_SAMPLE_RATE, *, num_channels: int = 1
) -> "torch.Tensor":
    from torchcodec.decoders import AudioDecoder

    samples = AudioDecoder(path, num_channels=num_channels, sample_rate=sample_rate)
    return samples.get_all_samples().data.squeeze()
