from collections import namedtuple

import torch

import numpy as np

from src.caul.model import ASRModelHandler


WorkerResult = namedtuple("WorkerResult", ["transcriptions", "scores"])


class ASRWorker:
    """ASRWorker class"""

    def __init__(
        self,
        models: list[ASRModelHandler] | ASRModelHandler,
        language_map: dict[str, int] = None,
    ):
        """Primary worker class. Handles transcription agnostically.

        :param models: ASRModelHandler list or singleton
        :param language_map: Map from ISO-639-3 language code to index of model in models param
        """

        if not isinstance(models, list):
            models = [models]

        self.models = models

        if language_map is None:
            language_map = {}

        self.language_map = language_map

    def startup(self):
        """Load all models into memory"""
        for model in self.models:
            model.load()

    def shutdown(self):
        """Garbage collect models"""
        self.models = []

    def get_model_by_language(self, language: str) -> ASRModelHandler:
        """Get model from language map or return first model if language is not mapped to model

        :param language: ISO-639-3 language code
        :return: ASRModelHandler
        """
        model_idx = self.language_map.get(language, None)

        if model_idx is None:
            model_idx = 0  # default to primary model when no language given

        return self.models[model_idx]

    def transcribe(
        self,
        audio: list[np.ndarray | torch.Tensor | str] | np.ndarray | torch.Tensor | str,
        languages: list[str] = None,
    ) -> WorkerResult:
        """Transcribe audio tensors or strings. Returns a tuple of (transcription, score). A list
        of languages of len(audio) may be passed to direct inputs to certain models.

        :param audio: List of np.ndarray or torch.Tensor or str, or a singleton of same types
        :param languages: List of ISO-639-3 language codes
        :return: WorkerResult
        """
        if not isinstance(audio, list):
            audio = [audio]

        audios_by_language = {}
        model_results = {}
        batch_language_ordering = []

        if languages is None:
            return self.models[0].transcribe(audio)

        # Sort by language where present, preserving original order for returning result
        for idx, aud in enumerate(audio):
            language = languages[idx]

            if language not in audios_by_language:
                audios_by_language[language] = []

            batch_language_ordering.append(language)
            audios_by_language[language].append(aud)

        # Run model on language batch
        for language, audio_list in audios_by_language.items():
            model = self.get_model_by_language(language)
            model_results[language] = model.transcribe(audio_list)

        transcriptions = []
        scores = []

        # For use with .pop()
        batch_language_ordering.reverse()

        # Reassemble
        for language in batch_language_ordering:
            transcription, score = model_results[language].pop()

            transcriptions.append(transcription)
            scores.append(score)

        return WorkerResult(transcriptions=transcriptions, scores=scores)
