import librosa
import torch

import numpy as np
import torchaudio

from caul.constant import (
    PARAKEET_INFERENCE_MAX_DURATION_KHZ,
    EXPECTED_SAMPLE_MINUTE,
    EXPECTED_SAMPLE_RATE,
    PARAKEET_INFERENCE_MAX_DURATION_MIN,
)
from caul.filesystem import save_tensor
from caul.tasks.asr_task import ASRTask
from caul.tasks.preprocessing.helpers import PreprocessedInput, InputMetadata


class ParakeetPreprocessor(ASRTask):
    """Preprocessing logic for ParakeetInferenceHandler inputs"""

    def process(
        self,
        inputs: list[np.ndarray | torch.Tensor | str] | np.ndarray | torch.Tensor | str,
        input_sample_rates: list[int] | int = None,
        save_to_filesystem: bool = True,
        return_tensors=True,
    ) -> list[list[PreprocessedInput]]:
        """Segment and batch audio inputs

        :param inputs: List of np.ndarray or torch.Tensor or str, or singleton of same types
        :param input_sample_rates: sample rate(s) of audio inputs
        :param save_to_filesystem: whether to save to filesystem
        :param return_tensors: whether to keep tensors in preprocessed inputs
        :return: batches of indexed preprocessed audio tensors (input_idx, preprocessed_input)
        """
        if not isinstance(inputs, list):
            inputs = [inputs]

        preprocessed_inputs = self.preprocess_inputs(
            inputs, input_sample_rates, save_to_filesystem, return_tensors
        )
        batches = self.batch_audio_tensors(preprocessed_inputs)

        return batches

    def preprocess_inputs(
        self,
        inputs: list[np.ndarray | torch.Tensor | str],
        input_sample_rates: list[int] = None,
        save_to_filesystem: bool = True,
        return_tensors: bool = True,
    ) -> list[PreprocessedInput]:
        """Accepts audio inputs as a list of file paths, np.ndarray, or torch.Tensor, converting to
        torch.Tensor, normalizing, segmenting inputs longer than 20 minutes (just under Parakeet's
        max) first by silences or with overlaps where not available, and batching segments

        :param inputs: List of np.ndarray or torch.Tensor or str, or a singleton of same types
        :param input_sample_rates: sample rate(s) of audio inputs
        :param save_to_filesystem: whether to save to filesystem
        :param return_tensors: whether to keep tensors in preprocessed inputs
        :return: List of processed inputs
        """
        preprocessed_inputs = []

        # Load arrays and divide into max_length segments
        for input_idx, audio_input in enumerate(inputs):
            input_file_path = None
            new_file_path = None
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
                audio_input = torch.Tensor(audio_input)

            # Normalize
            if input_sample_rates is not None and len(input_sample_rates) > input_idx:
                sample_rate = input_sample_rates[input_idx]
            else:
                sample_rate = EXPECTED_SAMPLE_RATE

            audio_input = self.normalize(audio_input, sample_rate)

            # Segment where necessary
            duration_khz = audio_input.shape[-1]
            tensor_segments = [audio_input]

            if duration_khz > PARAKEET_INFERENCE_MAX_DURATION_KHZ:
                tensor_segments = self.segment_audio_tensor(audio_input)

            for tensor_segment in tensor_segments:
                # Create temporary filesystem reference if applicable
                if save_to_filesystem:
                    new_file_path = save_tensor(tensor_segment)

                if not return_tensors:
                    tensor_segment = None

                # Create preprocessed input
                metadata = InputMetadata(
                    input_ordering=input_idx,
                    duration=duration_khz / EXPECTED_SAMPLE_MINUTE,
                    input_format=input_format,
                    input_file_path=input_file_path,
                    preprocessed_file_path=new_file_path,
                )

                preprocessed_input = PreprocessedInput(
                    tensor=tensor_segment,
                    metadata=metadata,
                )

                preprocessed_inputs.append(preprocessed_input)

        return preprocessed_inputs

    def normalize(self, audio_tensor: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """Normalize audio_tensor (single channel, sample rate = 16000)

        :param audio_tensor: input tensor
        :param sample_rate: input sample rate
        :return: normalized tensor
        """
        if sample_rate != EXPECTED_SAMPLE_RATE:
            audio_tensor = self.resample_waveform(audio_tensor, sample_rate)

        # Stereo dims (channels, aud_length); need mono (aud_length)
        if len(audio_tensor.shape) > 1:
            audio_tensor = audio_tensor.squeeze(0)

        return audio_tensor

    @staticmethod
    def segment_audio_tensor(
        audio_tensor: torch.Tensor,
        frame_len: int = 2048,
        silence_thresh_db: int = 35,
        hop_len: int = 512,
        kept_silence_len_secs: int = 0.15,
        min_silence_len_secs: int = 0.5,
        max_segment_len_secs: int = EXPECTED_SAMPLE_MINUTE
        * PARAKEET_INFERENCE_MAX_DURATION_MIN,
    ) -> list[torch.Tensor]:
        """Splits on silences with librosa, falling back to overlaps where min segments
        are not sufficient to safely divide audio.

        :param audio_tensor: input tensor
        :param frame_len: number of samples per analysis frame
        :param silence_thresh_db: max decibel value
        :param hop_len: number of samples between analysis frames
        :return: list of tensor segments
        """
        # TODO: Implement fallback to overlaps
        tensor_segments = []

        # Intervals between silences
        nonsilent_intervals = librosa.effects.split(
            audio_tensor.numpy(),
            top_db=silence_thresh_db,
            frame_length=frame_len,
            hop_length=hop_len,
        )

        merged = []
        min_silence_sample_len = int(min_silence_len_secs * EXPECTED_SAMPLE_MINUTE)
        kept_silence_sample_len = int(kept_silence_len_secs * EXPECTED_SAMPLE_MINUTE)
        max_segment_sample_len = int(max_segment_len_secs * EXPECTED_SAMPLE_MINUTE)

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

    @staticmethod
    def batch_audio_tensors(  # pylint: disable=R0914
        preprocessed_inputs: list[PreprocessedInput],
    ) -> list[list[PreprocessedInput]]:
        """Batch audio tensors by duration, 20 minutes max per batch, optimizing for tightly packed
        batches.

        :param preprocessed_inputs: list of PreprocessedInput
        :return: list of list[PreprocessedInput]
        """

        # Sort by duration
        preprocessed_inputs = sorted(
            preprocessed_inputs, key=lambda p: p.metadata.duration, reverse=True
        )

        # Now this becomes a bin-packing minimization problem. We'll use a variant of best-fit
        # decreasing.

        bins = [[]]
        bins_len = [0]

        # With each pass, choose a bin by maximizing remaining space
        for preprocessed_input in preprocessed_inputs:
            bin_len_diffs = []

            for bin_len in bins_len:
                bin_len_diffs.append(PARAKEET_INFERENCE_MAX_DURATION_MIN - bin_len)

            if max(bin_len_diffs) <= preprocessed_input.metadata.duration:
                bins.append([])
                bins_len.append(0)
                bin_len_diffs.append(PARAKEET_INFERENCE_MAX_DURATION_MIN)

            max_diff_idx = np.argmax(bin_len_diffs)

            bins[max_diff_idx].append(preprocessed_input)

            bins_len[max_diff_idx] += preprocessed_input.metadata.duration

        return bins

    @staticmethod
    def resample_waveform(waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """Resample when sample rate is not 16000

        :param waveform: torch.Tensor
        :param sample_rate: int
        :return: resampled torch.Tensor
        """
        transform = torchaudio.transforms.Resample(sample_rate, EXPECTED_SAMPLE_RATE)
        return transform(waveform)
