import torch

import numpy as np
import torchaudio

from caul.constant import (
    PARAKEET_INFERENCE_MAX_DURATION_KHZ,
    PARAKEET_SAMPLE_MINUTE,
    PARAKEET_SAMPLE_RATE,
    PARAKEET_INFERENCE_MAX_DURATION_MIN,
    DEVICE_CPU,
)
from caul.tasks.asr_task import ASRTask


class ParakeetPreprocessor(ASRTask):
    """Preprocessing logic for ParakeetInferenceHandler inputs"""

    def process(
        self,
        inputs: list[np.ndarray | torch.Tensor | str] | np.ndarray | torch.Tensor | str,
    ) -> list[list[tuple[int, torch.Tensor]]]:
        """Segment and batch audio inputs

        :param inputs: List of np.ndarray or torch.Tensor or str, or singleton of same types
        :return: batches of audio tensor segments of (input_idx, audio_tensor)
        """
        if not isinstance(inputs, list):
            inputs = [inputs]

        audio_tensors = self.load_audio_tensors(inputs)
        segmented_tensors = self.segment_audio_tensors(audio_tensors)
        batches = self.batch_audio_tensors(segmented_tensors)

        return batches

    def load_audio_tensors(
        self, audio: list[np.ndarray | torch.Tensor | str]
    ) -> list[tuple[int, torch.Tensor]]:
        """Accepts audio inputs as a list of file paths, np.ndarray, or torch.Tensor, converting to
        torch.Tensor

        :param audio: List of np.ndarray or torch.Tensor or str, or a singleton of same types
        :return: List of tuples of (input_idx, audio_tensor)
        """
        audio_tensors = []

        # Load arrays and divide into max_length segments
        for input_idx, aud in enumerate(audio):
            # Load audio files as arrays
            if isinstance(aud, str):
                aud = self.load_wave(aud)

            if isinstance(aud, np.ndarray):
                aud = torch.Tensor(aud)

            audio_tensors.append((input_idx, aud))

        return audio_tensors

    def load_wave(self, wav_path: str) -> torch.Tensor:
        """Load a wav file from path as torch.Tensor and resample if needed

        :param wav_path: path to wave file
        :return: torch.Tensor
        """
        waveform, sample_rate = torchaudio.load(wav_path)

        if sample_rate != PARAKEET_SAMPLE_RATE:
            waveform = self.resample_waveform(waveform, sample_rate)

        # Default dims [channels, aud_length]; need [aud_length]
        return waveform.squeeze(0)

    @staticmethod
    def segment_audio_tensors(
        audio_tensors: list[tuple[int, torch.Tensor]],
    ) -> list[tuple[int, torch.Tensor, int]]:
        """Segment inputs greater than 24 minutes (Parakeet's max per-batch duration), surjectively
        mapping them onto their ordering as originally received with their duration in minutes

        :param audio_tensors: List of torch.Tensor
        :return: List of tuples of (input_idx, audio_tensor, duration)
        """

        indexed_audio_with_duration = []

        for input_idx, audio_tensor in audio_tensors:
            aud_segments = [audio_tensor]

            if audio_tensor.shape[-1] > PARAKEET_INFERENCE_MAX_DURATION_KHZ:
                aud_segments = torch.split(
                    audio_tensor, PARAKEET_INFERENCE_MAX_DURATION_KHZ
                )

            indexed_audio_with_duration += [
                (
                    input_idx,
                    audio_tensor,
                    audio_tensor.shape[-1] / PARAKEET_SAMPLE_MINUTE,
                )
                for audio_tensor in aud_segments
            ]

        return indexed_audio_with_duration

    @staticmethod
    def batch_audio_tensors(  # pylint: disable=R0914
        indexed_audio_with_duration: list[tuple[int, torch.Tensor, int]],
    ) -> list[list[tuple[int, torch.Tensor]]]:
        """Batch audio tensors by duration, 24 minutes max per batch, optimizing for tightly packed
        batches.

        :param indexed_audio_with_duration: List of tuples of (input_idx, audio_tensor,
                                            duration)
        :return: list of list of tuples of (input_idx, audio_tensor)
        """

        # Sort by duration
        indexed_audio_with_duration = sorted(
            indexed_audio_with_duration, key=lambda x: x[-1], reverse=True
        )

        # Now this becomes a bin-packing minimization problem. We'll use a variant of best-fit
        # decreasing.

        bins = [[]]
        bins_len = [0]

        # With each pass, choose a bin by maximizing remaining space
        for idx, segment, duration in indexed_audio_with_duration:
            bin_len_diffs = []

            for bin_len in bins_len:
                bin_len_diffs.append(PARAKEET_INFERENCE_MAX_DURATION_MIN - bin_len)

            if max(bin_len_diffs) <= duration:
                bins.append([])
                bins_len.append(0)
                bin_len_diffs.append(PARAKEET_INFERENCE_MAX_DURATION_MIN)

            max_diff_idx = np.argmax(bin_len_diffs)

            bins[max_diff_idx].append((idx, segment))

            bins_len[max_diff_idx] += duration

        return bins

    @staticmethod
    def resample_waveform(waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """Resample when sample rate is not 16000

        :param waveform: torch.Tensor
        :param sample_rate: int
        :return: resampled torch.Tensor
        """
        transform = torchaudio.transforms.Resample(sample_rate, PARAKEET_SAMPLE_RATE)
        return transform(waveform)
