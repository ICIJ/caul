from functools import reduce
from itertools import groupby

import torch

import numpy as np
import nemo.collections.asr as nemo_asr
import torchaudio

from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis

from caul.constant import (
    PARAKEET_SAMPLE_RATE,
    PARAKEET_MODEL_MAX_DURATION_KHZ,
    PARAKEET_SAMPLE_MINUTE,
    PARAKEET_MODEL_MAX_DURATION_MIN,
)
from caul.model import ASRModelHandler, ASRModelHandlerResult


class ParakeetModelHandlerResult(ASRModelHandlerResult):

    def parse_parakeet_hypothesis(
        self, hypothesis: Hypothesis
    ) -> ASRModelHandlerResult:
        self.transcription = (
            [
                (s["start"], s["end"], s["segment"])
                for s in hypothesis.timestamp.get("segment")
            ]
            if hypothesis.timestamp.get("segment") is not None
            else [(0.0, 0.0, hypothesis.text)]
        )
        self.score = hypothesis.score

        return self

    def concat(self, model_result: ASRModelHandlerResult) -> ASRModelHandlerResult:
        if model_result is None:
            return

        if self.transcription is None:
            self.transcription = []

        self.transcription += model_result.transcription

        # We have to weight by total segment len
        transcription_duration = self.transcription[-1][1]
        model_result_duration = model_result.transcription[-1][1]
        total_duration = transcription_duration + model_result_duration

        self.score = (
            self.score * transcription_duration
            + model_result.score * model_result_duration
        ) / total_duration

        return self


class ParakeetModelHandler(ASRModelHandler):
    """Model handler for NVIDIA's Parakeet family of ASR models. Supports up to 24 minutes of audio
    (batched or unbatched) in a single pass. Assumes that audio inputs (wav files or tensors) are
    single-channel with a sample rate of 16000—this last is very important for segmenting.
    """

    def __init__(self, model_name: str, device: str = "cpu", timestamps=True):
        self.model_name = model_name
        self.device = device
        self.timestamps = timestamps
        self.model = None

    def load(self):
        """Load model"""
        self.model = nemo_asr.models.ASRModel.from_pretrained(
            self.model_name, map_location=torch.device(self.device)
        )

    def unload(self):
        """Unload model"""
        self.model = None

    def transcribe(
        self,
        audio: list[np.ndarray | torch.Tensor | str] | np.ndarray | torch.Tensor | str,
    ) -> list[ParakeetModelHandlerResult]:
        """Segment and transcribe a batch of audio tensors or file names. Max length 24 minutes.

        :param audio: List of np.ndarray or torch.Tensor or str, or a singleton of same types
        :return: List of tuples of (transcription, score)
        """
        if not isinstance(audio, list):
            audio = [audio]

        audio = self.load_audio_tensors(audio)
        batches = self.segment_audio_tensors(audio)
        transcriptions = []

        for batch in batches:
            prebatch_indices, segments = zip(*batch)
            hypotheses = self.model.transcribe(segments, timestamps=self.timestamps)
            # Get timestamped segments if available, otherwise default to whole text
            for idx, hyp in enumerate(hypotheses):
                model_result = ParakeetModelHandlerResult().parse_parakeet_hypothesis(
                    hyp
                )
                indexed_result = prebatch_indices[idx], model_result
                transcriptions.append(indexed_result)

        return self.map_results_to_inputs(transcriptions)

    def load_audio_tensors(
        self, audio: list[np.ndarray | torch.Tensor | str]
    ) -> list[tuple[int, torch.Tensor, int]]:
        """Accepts audio inputs as a list of wav paths, np.ndarray, or torch.Tensor, converting to
        torch.Tensor and sending them to device where needed, segmenting for inputs greater than
         24 minutes (Parakeet's max), and returns a list indexed by original ordering.

        :param audio: List of np.ndarray or torch.Tensor or str, or a singleton of same types
        :return: List of tuples of (original_ordering, audio_tensor, duration)
        """
        indexed_audio_with_duration = []

        # Load arrays and divide into max_length segments
        for idx, aud in enumerate(audio):
            # Load audio files as arrays
            if isinstance(aud, str):
                aud = self.load_wave(aud)

            if isinstance(aud, np.ndarray):
                aud = torch.Tensor(aud)

            # Send to GPU
            if self.device != "cpu":
                aud = aud.to(self.device)

            aud_segments = [aud]

            if aud.shape[-1] > PARAKEET_MODEL_MAX_DURATION_KHZ:
                aud_segments = torch.split(aud, PARAKEET_MODEL_MAX_DURATION_KHZ)

            indexed_audio_with_duration += [
                (idx, aud, aud.shape[-1] / PARAKEET_SAMPLE_MINUTE)
                for aud in aud_segments
            ]

        return indexed_audio_with_duration

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
    def segment_audio_tensors(  # pylint: disable=R0914
        indexed_audio_with_duration: list[tuple[int, torch.Tensor, int]],
    ) -> list[list[tuple[int, torch.Tensor]]]:
        """Segment a batch of torch.Tensors by duration, 24 minutes max per batch.

        :param indexed_audio_with_duration: List of tuples of (original_ordering, audio_tensor,
                                            duration)
        :return: Batch of torch.Tensor of duration < 24 minutes each indexed by original_ordering
        """

        # Sort by duration
        indexed_audio_with_duration = sorted(
            indexed_audio_with_duration, key=lambda x: x[-1], reverse=True
        )

        # Now this becomes a bin-packing minimization problem. We'll use a variant of best-fit
        # decreasing.
        # Get min number of bins; max is simply len(aud_segments)
        # min_bins = ceil(sum([at[1] for at in audio_with_duration]) / PARAKEET_MODEL_MAX_DURATION_MIN)

        bins = [[]]
        bins_len = [0]

        # With each pass, choose a bin by maximizing remaining space
        for idx, segment, duration in indexed_audio_with_duration:
            bin_len_diffs = []

            for bin_len in bins_len:
                bin_len_diffs.append(PARAKEET_MODEL_MAX_DURATION_MIN - bin_len)

            if max(bin_len_diffs) <= duration:
                bins.append([])
                bins_len.append(0)
                bin_len_diffs.append(PARAKEET_MODEL_MAX_DURATION_MIN)

            max_diff_idx = np.argmax(bin_len_diffs)

            bins[max_diff_idx].append((idx, segment))

            bins_len[max_diff_idx] += duration

        return bins

    @staticmethod
    def map_results_to_inputs(
        batched_results: list[tuple[int, ParakeetModelHandlerResult]],
    ) -> list[ParakeetModelHandlerResult]:
        """Remap unordered and segmented tensors to original inputs for return

        :param batched_results: list of unordered indexed ParakeetModelHandlerResult, still segmented
        :return: list[ParakeetModelHandlerResult]
        """
        unbatched_results = []

        # Sort in order received before batching
        batched_results = sorted(batched_results, key=lambda x: x[0])

        # Concat segmented tensors
        results_grouped_by_index = groupby(batched_results, key=lambda x: x[0])

        for group_idx, group_results in results_grouped_by_index:
            group_results = [gr[1] for gr in list(group_results)]  # drop index
            merged_results = (
                reduce(lambda l, r: l.concat(r), group_results)
                if len(group_results) > 1
                else group_results[0]
            )
            unbatched_results.append(merged_results)

        return unbatched_results

    @staticmethod
    def resample_waveform(waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """Resample when sample rate is not 16000

        :param waveform: torch.Tensor
        :param sample_rate: int
        :return: torch.Tensor
        """
        transform = torchaudio.transforms.Resample(sample_rate, PARAKEET_SAMPLE_RATE)
        return transform(waveform)
