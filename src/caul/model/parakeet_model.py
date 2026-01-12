import torch

import numpy as np
import nemo.collections.asr as nemo_asr
import torchaudio

from caul.constant import PARAKEET_SAMPLE_RATE, PARAKEET_MODEL_MAX_DURATION
from src.caul.model import ASRModelHandler


class ParakeetModelHandler(ASRModelHandler):
    """Model handler for NVIDIA's Parakeet family of ASR models. Supports up to 24 minutes of audio (batched or
    unbatched) in a single pass. Assumes that audio inputs (wav files or tensors) are single-channel with a sample rate
    of 16000—this last is very important for segmenting.
    """

    def __init__(self, model_name: str, device: str = "cpu", timestamps = True):
        self.model_name = model_name
        self.device = device
        self.timestamps = timestamps
        self.model = None

    def load(self):
        self.model = nemo_asr.models.ASRModel.from_pretrained(self.model_name, map_location=torch.device(self.device))

        self.model.freeze()

    def unload(self):
        self.model = None

    def transcribe(self, audio: list[np.ndarray | torch.Tensor | str] | np.ndarray | torch.Tensor | str) -> list[tuple[str, float]]:
        """ Segment and transcribe a batch of audio tensors or file names. Max length 24 minutes.

        :param audio: List of np.ndarray or torch.Tensor or str, or a singleton of same types
        :return: List of tuples of (transcription, score)
        """
        if not isinstance(audio, list):
            audio = [audio]

        batches = self.segment_batch(audio)
        transcriptions = []

        for batch in batches:
            indices, segments = zip(*batch)
            predictions = self.model.transcribe(segments, timestamps=self.timestamps)
            transcriptions_with_scores = [(indices[idx], p.text, p.score) for idx, p in enumerate(predictions)]

            transcriptions += transcriptions_with_scores

        # Sort in original order
        transcriptions = sorted(transcriptions, key=lambda x: x[0])

        # Drop index
        transcriptions = [(t[1], t[2]) for t in transcriptions]

        return transcriptions

    def segment_batch(self, audio: list[np.ndarray | torch.Tensor | str]) -> list[list[tuple[int, torch.Tensor]]]:
        """ Segment a batch of audio tensors or file names by duration; 24 minutes per batch.

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
                    transform = torchaudio.transforms.Resample(sample_rate, PARAKEET_SAMPLE_RATE)
                    waveform = transform(waveform)

                # Default dims [channels, aud_length]; need [aud_length]
                aud = waveform.numpy().squeeze(0)

            aud_len = aud.shape[-1]
            aud_segments = [(idx, aud, aud_len)]

            if aud_len > PARAKEET_MODEL_MAX_DURATION:
                aud_segments = [(idx, a, a.shape[-1]) for a in np.array_split(aud, PARAKEET_MODEL_MAX_DURATION)]

            audio_by_duration += aud_segments

        # Sort by duration
        audio_by_duration = sorted(audio_by_duration, key=lambda x: x[-1], reverse=True)

        # Now this because a bin-packing minimization problem. We'll use a variant of
        # best-fit decreasing.
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





