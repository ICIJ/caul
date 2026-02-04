import logging

from dataclasses import dataclass, field

import torch

import numpy as np

from caul.exception import (
    MissingModelSpecificationException,
    UnsupportedModelException,
)
from caul.model_handlers import MODEL_FAMILY_HANDLER_MAP
from caul.tasks.inference.asr_inference import (
    ASRInferenceHandlerResult,
)
from caul.model_handlers.asr_model_handler import ASRModelHandler, ASRModelHandlerResult
from caul.utils import dict_key_fuzzy_match

logger = logging.getLogger(__name__)


@dataclass
class ASRHandlerResult:
    """ASRHandlerResult class"""

    # pylint: disable=R0914

    transcriptions: list[list[tuple[float, str]]] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)

    def add_transcriptions(
        self,
        handler_result: list[ASRInferenceHandlerResult] | ASRInferenceHandlerResult,
    ):
        """Parse ASRInferenceHandlerResult

        :param handler_result: List of ASRInferenceHandlerResult
        """
        if not isinstance(handler_result, list):
            handler_result = [handler_result]

        for result in handler_result:
            self.transcriptions.append(result.transcription)
            self.scores.append(result.score)

        return self


class ASRHandler:
    """ASRHandler class"""

    # pylint: disable=R0913,R0917

    def __init__(
        self,
        models: list[str | ASRModelHandler] | str | ASRModelHandler,
        device: torch.device | str = None,
        language_map: dict[str, int] = None,
    ):
        """Primary application handler class. Handles transcription agnostically.

        :param models: Model_handler(s) or string reference(s)
        :param device: cuda/cpu/mps
        :param language_map: Map from ISO-639-3 language code to index of inference_handler
        """
        self.device = device

        if language_map is None:
            language_map = {}

        self.language_map = language_map

        self.model_handlers = []

        if isinstance(models, list) and len(models) == 0:
            raise MissingModelSpecificationException(
                "At least one model name or model handler must be provided"
            )

        if not isinstance(models, list):
            models = [models]

        for model in models:
            if isinstance(model, str):
                supported_model_handler = dict_key_fuzzy_match(
                    MODEL_FAMILY_HANDLER_MAP, model
                )

                if supported_model_handler is None:
                    raise UnsupportedModelException(f"Unsupported model '{model}'")

                # Set device after instantiation
                supported_model_handler.set_device(self.device)

                self.model_handlers.append(supported_model_handler)
            elif isinstance(model, ASRModelHandler):
                self.model_handlers.append(model)
            else:
                raise UnsupportedModelException(f"Unsupported model type '{model}'")

    def __repr__(self):
        return f"<ASRHandler " f"models: {self.model_handlers} "

    def startup(self):
        """Run all model handler startup procedures"""
        for model_handler in self.model_handlers:
            model_handler.startup()

    def shutdown(self):
        """Garbage collect model handlers"""
        self.model_handlers = []

    def get_handler_by_language(self, language: str) -> ASRModelHandler:
        """Get model_handler from language map or return first reference if language is not mapped

        :param language: ISO-639-3 language code
        :return: ASRModelHandler
        """
        reference_idx = self.language_map.get(
            language, 0
        )  # default to primary inference_handler when no language given

        if len(self.model_handlers) <= reference_idx:
            raise UnsupportedModelException(
                "Language is mapped to a model index which does not exist"
            )

        return self.model_handlers[reference_idx]

    def transcribe(
        self,
        inputs: list[np.ndarray | torch.Tensor | str] | np.ndarray | torch.Tensor | str,
        languages: list[str] = None,
    ) -> list[ASRModelHandlerResult]:
        """Transcribe audio tensors or strings. Returns a tuple of (transcription, score). A list
        of languages of len(inputs) may be passed to direct inputs to certain inference_handlers.

        :param inputs: List of np.ndarray or torch.Tensor or str, or a singleton of same types
        :param languages: List of ISO-639-3 language codes
        :return: HandlerResult
        """
        if len(self.model_handlers) == 0:
            raise MissingModelSpecificationException(
                "At least one model name or model handler must be provided"
            )

        if not isinstance(inputs, list):
            inputs = [inputs]

        audios_by_language = {}
        model_handler_results_by_language = {}
        batch_language_ordering = []
        model_handler_results = []

        if languages is None:
            # Default to first model handler
            return self.model_handlers[0].process(inputs)

        # Sort by language where present, preserving original order for returning result
        for idx, aud in enumerate(inputs):
            language = languages[idx]

            if language not in audios_by_language:
                audios_by_language[language] = []

            batch_language_ordering.append(language)
            audios_by_language[language].append(aud)

        # Run inference_handler on language batch
        for language, audio_list in audios_by_language.items():
            model_handler = self.get_handler_by_language(language)
            model_handler_results_by_language[language] = model_handler.process(inputs)

        # For use with .pop()
        batch_language_ordering.reverse()

        # Reassemble and postprocess
        for language in batch_language_ordering:
            model_handler_result = model_handler_results_by_language[language].pop()

            model_handler_results.append(model_handler_result)

        return model_handler_results
