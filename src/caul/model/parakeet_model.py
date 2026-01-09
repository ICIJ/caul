import torch

import numpy as np
import nemo.collections.asr as nemo_asr

from src.caul.model import ASRModel

MAX_LENGTH = 24  # 24 minutes per batch max


class ParakeetModel(ASRModel):

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

    def transcribe(self, audio: list[np.ndarray | torch.Tensor | str] | np.ndarray | torch.Tensor | str):
        """ Transcribe a batch of audio tensors or file names. Max length 24 minutes.

        :param audio:
        :return:
        """
        print(audio)
        predictions = self.model.transcribe(audio, timestamps=self.timestamps)
        transcriptions_with_scores = [(p.text, p.score) for p in predictions]

        return transcriptions_with_scores

    def segment(self, audio: list[np.ndarray | torch.Tensor | str] | np.ndarray | torch.Tensor | str):
        """ Segment a batch of audio tensors or file names by duration; 24 minutes per batch.

        :param audio:
        :return: Batch of audio tensors of duration < 24 minutes each
        """
        pass