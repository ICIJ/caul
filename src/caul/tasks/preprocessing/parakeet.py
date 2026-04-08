import uuid
from itertools import repeat
from pathlib import Path
from typing import ClassVar, Iterable, Self, TYPE_CHECKING

from hashlib import sha256

from pydantic import Field

from caul.config import PreprocessorConfig
from caul.segmentation.segmenter import segment_by_silence
from caul.constant import (
    ASRModel,
    PARAKEET_INFERENCE_MAX_DURATION_S,
    PARAKEET_INFERENCE_MAX_FRAMES,
    DEFAULT_SAMPLE_RATE,
)
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


class ParakeetPreprocessorConfig(PreprocessorConfig):
    model: ClassVar[str] = Field(default=ASRModel.PARAKEET)

    sample_rate: int = DEFAULT_SAMPLE_RATE


@Preprocessor.register(ASRModel.PARAKEET)
class ParakeetPreprocessor(Preprocessor):
    """Preprocessing logic for ParakeetInferenceHandler inputs"""

    def __init__(self, sample_rate: int = DEFAULT_SAMPLE_RATE):
        super().__init__()
        self._sample_rate = sample_rate

    @classmethod
    def _from_config(cls, config: ParakeetPreprocessorConfig, **extras) -> Self:
        return cls(sample_rate=config.sample_rate)

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
        :return: batches of indexed preprocessed audio tensors (input_idx, preprocessed_input)
        """
        if not isinstance(inputs, list):
            inputs = [inputs]

        preprocessed_inputs = self.preprocess_inputs(
            inputs, input_sample_rates, output_dir=output_dir
        )
        # TODO: ideally _batch_audio_tensors should stream for real
        batches = batch_audio_tensors(preprocessed_inputs)
        return batches

    def preprocess_inputs(  # pylint: disable=too-many-locals
        self,
        inputs: Iterable["np.ndarray | torch.Tensor | str"],
        input_sample_rates: Iterable[int] | int | None = None,
        output_dir: Path | None = None,
    ) -> Iterable[PreprocessorOutput]:
        """Accepts audio inputs as a list of file paths, np.ndarray, or torch.Tensor, converting to
        torch.Tensor, normalizing, segmenting inputs longer than 20 minutes (just under Parakeet's
        max) first by silences or with overlaps where not available, and batching segments

        :param inputs: List of np.ndarray or torch.Tensor or str, or a singleton of same types
        :param input_sample_rates: sample rate(s) of audio inputs
        :return: List of processed inputs
        """
        import torchaudio  # pylint: disable=import-outside-toplevel
        import numpy as np  # pylint: disable=import-outside-toplevel

        if isinstance(input_sample_rates, (int, _NoneType)):
            input_sample_rates = repeat(input_sample_rates)
            inputs_and_sample_rates = zip(inputs, input_sample_rates, strict=False)
        else:
            inputs_and_sample_rates = zip(inputs, input_sample_rates, strict=True)

        # Load arrays and divide into max_length segments
        for input_idx, (audio_input, sample_rate) in enumerate(inputs_and_sample_rates):
            input_file_path = None
            preprocessed_file_path = None
            input_format = None

            # Load audio files as arrays
            if isinstance(audio_input, str):
                input_file_path = audio_input
                input_format = (
                    input_file_path.split(".")[-1]
                    if len(input_file_path.split(".")) > 1
                    else None
                )
                audio_input, sample_rate = torchaudio.load(audio_input)

            if isinstance(audio_input, np.ndarray):
                import torch  # pylint: disable=import-outside-toplevel

                audio_input = torch.Tensor(audio_input)

            # Normalize
            if sample_rate is None:
                sample_rate = self._sample_rate

            audio_input = self._normalize(audio_input, sample_rate)

            # Segment where necessary
            n_frames = audio_input.shape[-1]
            tensor_segments = [audio_input]

            if n_frames > PARAKEET_INFERENCE_MAX_FRAMES:
                tensor_segments = [s.tensor for s in segment_by_silence(audio_input)]

            original_file = (
                _displayable_path(audio_input)
                if isinstance(audio_input, str)
                else uuid.uuid4().hex.encode()
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
                    preprocessed_input = PreprocessedInputWithTensor(
                        metadata=metadata, tensor=tensor_segment
                    )
                else:
                    preprocessed_input = PreprocessedInput(metadata=metadata)
                yield preprocessed_input

    def _normalize(
        self, audio_tensor: "torch.Tensor", sample_rate: int
    ) -> "torch.Tensor":
        """Normalize audio_tensor (single channel, sample rate = 16000)

        :param audio_tensor: input tensor
        :param sample_rate: input sample rate
        :return: normalized tensor
        """
        if sample_rate != self._sample_rate:
            audio_tensor = _resample_waveform(
                audio_tensor, self._sample_rate, target_rate=sample_rate
            )

        # Stereo dims (channels, aud_length); need mono (aud_length)
        if len(audio_tensor.shape) > 1:
            audio_tensor = audio_tensor.squeeze(0)

        return audio_tensor


def _segment_audio_tensor(  # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    audio_tensor: "torch.Tensor",
    frame_len: int = 2048,
    silence_thresh_db: int = 35,
    hop_len: int = 512,
    kept_silence_len_s: int = 0.15,
    min_silence_len_s: int = 0.5,
    max_segment_len_s: int = PARAKEET_INFERENCE_MAX_DURATION_S,
) -> list["torch.Tensor"]:
    """Splits on silences with librosa, falling back to overlaps where min segments
    are not sufficient to safely divide audio.

    :param audio_tensor: input tensor
    :param frame_len: number of samples per analysis frame
    :param silence_thresh_db: max decibel value
    :param hop_len: number of samples between analysis frames
    :param kept_silence_len_s: number of seconds to keep silence
    :param min_silence_len_s: minimum seconds to keep silence
    :param max_segment_len_s: maximum seconds to keep silence
    :return: list of tensor segments
    """
    import librosa  # pylint: disable=import-outside-toplevel

    # TODO: Implement fallback to overlaps
    tensor_segments = []

    # Intervals between silences
    nonsilent_intervals = librosa.effects.split(  # pylint: disable=duplicate-code
        audio_tensor.numpy(),
        top_db=silence_thresh_db,
        frame_length=frame_len,
        hop_length=hop_len,
    )

    merged = []
    min_silence_sample_len = int(min_silence_len_s * DEFAULT_SAMPLE_RATE)
    kept_silence_sample_len = int(kept_silence_len_s * DEFAULT_SAMPLE_RATE)
    max_segment_sample_len = int(max_segment_len_s * DEFAULT_SAMPLE_RATE)

    # Merge intervals separated by short silences
    for start, end in nonsilent_intervals:
        if len(merged) == 0:
            merged.append((start, end))
        else:
            _, prev_end = merged[-1]
            if start - prev_end < min_silence_sample_len:
                merged[-1][1] = end
            else:
                merged.append((start, end))

    # Segment controlling max length
    for start, end in merged:
        start = max(0, start - kept_silence_sample_len)
        end = min(audio_tensor.shape[-1], end + kept_silence_sample_len)

        while end - start > max_segment_sample_len:
            segment_end = start + max_segment_sample_len
            tensor_segment = audio_tensor[start:segment_end]

            tensor_segments.append(tensor_segment)

    return tensor_segments


# TODO: something approximate here would be nice to avoid loading all data in memory
def batch_audio_tensors(  # pylint: disable=R0914
    preprocessed_inputs: Iterable[PreprocessedInput],
) -> Iterable[list[PreprocessedInput]]:
    """Batch audio tensors by duration, 20 minutes max per batch, optimizing for tightly packed
    batches.

    :param preprocessed_inputs: list of PreprocessedInput
    :return: list of list[PreprocessedInput]
    """
    import numpy as np  # pylint: disable=import-outside-toplevel

    # Sort by duration
    preprocessed_inputs = sorted(
        preprocessed_inputs, key=lambda p: p.metadata.duration_s, reverse=True
    )

    # Now this becomes a bin-packing minimization problem. We'll use a variant of best-fit
    # decreasing.

    bins = [[]]
    bins_len = [0]

    # With each pass, choose a bin by maximizing remaining space
    for preprocessed_input in preprocessed_inputs:
        remaining_spaces = [
            PARAKEET_INFERENCE_MAX_DURATION_S - bin_len for bin_len in bins_len
        ]
        input_duration_s = preprocessed_input.metadata.duration_s
        if input_duration_s > max(remaining_spaces):
            bins.append([])
            bins_len.append(0)
            remaining_spaces.append(PARAKEET_INFERENCE_MAX_DURATION_S)
        most_empty_bin = np.argmax(remaining_spaces)
        bins[most_empty_bin].append(preprocessed_input)
        bins_len[most_empty_bin] += input_duration_s

    yield from bins


def _resample_waveform(
    waveform: "torch.Tensor", sample_rate: int, *, target_rate: int
) -> "torch.Tensor":
    """Resample when sample rate is not 16000

    :param waveform: torch.Tensor
    :param sample_rate: int
    :return: resampled torch.Tensor
    """
    import torchaudio  # pylint: disable=import-outside-toplevel

    transform = torchaudio.transforms.Resample(sample_rate, target_rate)
    return transform(waveform)


def _displayable_path(path: str, component_size_limit: int = 10) -> str:
    path = Path(path)
    displayable_file_name = [c[:component_size_limit] for c in path.parts]
    uuid = sha256(str(path).encode()).hexdigest()[:20]
    return f"{'__'.join(displayable_file_name)}-{uuid}"
