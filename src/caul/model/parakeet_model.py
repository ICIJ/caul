import torch

import numpy as np
import nemo.collections.asr as nemo_asr
import torchaudio

from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis

from src.caul.constant import PARAKEET_SAMPLE_RATE, PARAKEET_MODEL_MAX_DURATION
from src.caul.model import ASRModelHandler, ASRModelHandlerResult


class ParakeetModelHandlerResult(ASRModelHandlerResult):

    def parse_parakeet_hypothesis(self, hypothesis: Hypothesis):
        self.transcription = (
            [(s["start"], s["segment"]) for s in hypothesis.timestamp.get("segment")]
            if hypothesis.timestamp.get("segment") is not None
            else [(0.0, hypothesis.text)]
        )
        self.score = hypothesis.score


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

        batches = self.segment_batch(audio)
        transcriptions = []

        for batch in batches:
            prebatch_indices, segments = zip(*batch)
            hypotheses = self.model.transcribe(segments, timestamps=self.timestamps)
            # Get timestamped segments if available, otherwise default to whole text
            for idx, hyp in enumerate(hypotheses):
                model_result = ParakeetModelHandlerResult()

                model_result.parse_parakeet_hypothesis(hyp)

                indexed_result = prebatch_indices[idx], model_result
                transcriptions.append(indexed_result)

        # Sort in order received before batching
        transcriptions = sorted(transcriptions, key=lambda x: x[0])

        # Drop index
        transcriptions = [t[-1] for t in transcriptions]

        return transcriptions

    @staticmethod
    def segment_batch(  # pylint: disable=R0914
        audio: list[np.ndarray | torch.Tensor | str],
    ) -> list[list[tuple[int, torch.Tensor]]]:
        """Segment a batch of audio tensors or file names by duration; 24 minutes per batch.

        :param audio: List of np.ndarray or torch.Tensor or str
        :return: Batch of torch.Tensor of duration < 24 minutes each
        """

        audio_by_duration = []

        # Load arrays and divide into max_length segments
        for idx, aud in enumerate(audio):
            # Load audio files as arrays
            if isinstance(aud, str):
                waveform, sample_rate = torchaudio.load(aud)

                if sample_rate != PARAKEET_SAMPLE_RATE:
                    # Resample if not 16000
                    transform = torchaudio.transforms.Resample(
                        sample_rate, PARAKEET_SAMPLE_RATE
                    )
                    waveform = transform(waveform)

                # Default dims [channels, aud_length]; need [aud_length]
                aud = waveform.numpy().squeeze(0)

            aud_len = aud.shape[-1]
            aud_segments = [(idx, aud, aud_len)]

            if aud_len > PARAKEET_MODEL_MAX_DURATION:
                aud_segments = [
                    (idx, a, a.shape[-1])
                    for a in np.array_split(aud, PARAKEET_MODEL_MAX_DURATION)
                ]

            audio_by_duration += aud_segments

        # Sort by duration
        audio_by_duration = sorted(audio_by_duration, key=lambda x: x[-1], reverse=True)

        # Now this becomes a bin-packing minimization problem. We'll use a variant of best-fit
        # decreasing.
        # Get min number of bins; max is simply len(aud_segments)
        # min_bins = ceil(sum([at[-1] for at in audio_by_duration]) / PARAKEET_MODEL_MAX_DURATION)

        bins = [[]]
        bins_len = [0]

        # With each pass, choose a bin by maximizing remaining space
        for idx, segment, duration in audio_by_duration:
            if not isinstance(segment, torch.Tensor):
                segment = torch.Tensor(segment)

            bin_len_diffs = []

            for bin_len in bins_len:
                bin_len_diffs.append(PARAKEET_MODEL_MAX_DURATION - bin_len)

            if max(bin_len_diffs) < 0:
                bins.append([])
                bins_len.append(0)

            max_diff_idx = np.argmax(bin_len_diffs)

            bins[max_diff_idx].append((idx, segment))
            bins_len[max_diff_idx] += duration

        return bins
