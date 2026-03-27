import gc
import logging
from contextlib import ExitStack

import torch

import numpy as np

from caul.configs.asr import ASRConfig
from caul.exception import (
    MissingModelSpecificationException,
    UnsupportedModelException,
)
from caul.configs import MODEL_FAMILY_CONFIG_MAP
from caul.model_handlers.asr_model_handler import ASRModelHandler, ASRModelHandlerResult
from caul.utils import dict_key_fuzzy_match

logger = logging.getLogger(__name__)


class ASRHandler:
    """ASRHandler class"""

    # pylint: disable=R0913,R0917

    def __init__(
        self,
        models: (
            list[str | ASRModelHandler | ASRConfig] | str | ASRModelHandler | ASRConfig
        ),
        device: torch.device | str = None,
        language_map: dict[str, int] = None,
    ):
        """Primary application handler class. Handles transcription agnostically.

        :param models: Model_handler(s) or string reference(s)
        :param device: cuda/cpu/mps
        :param language_map: Map from ISO-639-3 language code to index of inference_handler
        """
        self._device = device

        if language_map is None:
            language_map = {}

        self._language_map = language_map

        self._model_handlers = []
        self._exit_stack = ExitStack()

        if isinstance(models, list) and len(models) == 0:
            raise MissingModelSpecificationException(
                "At least one model name or model handler must be provided"
            )

        if not isinstance(models, list):
            models = [models]

        for model in models:
            if isinstance(model, str):
                supported_model_config = dict_key_fuzzy_match(
                    MODEL_FAMILY_CONFIG_MAP, model
                )

                if supported_model_config is None:
                    raise UnsupportedModelException(f"Unsupported model '{model}'")

                # Set device after instantiation
                supported_model_handler = supported_model_config.handler_from_config()
                supported_model_handler.set_device(self._device)

                self._model_handlers.append(supported_model_handler)
            elif isinstance(model, ASRModelHandler):
                self._model_handlers.append(model)
            else:
                raise UnsupportedModelException(f"Unsupported model type '{model}'")

    def __repr__(self):
        return f"<ASRHandler models: {self._model_handlers} "

    def __enter__(self):
        """Run all model handler startup procedures"""
        for model_handler in self._model_handlers:
            self._exit_stack.enter_context(model_handler)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._exit_stack.__exit__(exc_type, exc_val, exc_tb)
        gc.collect()

    def get_handler_by_language(self, language: str) -> ASRModelHandler:
        """Get model_handler from language map or return first reference if language is not mapped

        :param language: ISO-639-3 language code
        :return: ASRModelHandler
        """
        reference_idx = self._language_map.get(
            language, 0
        )  # default to primary inference_handler when no language given

        if len(self._model_handlers) <= reference_idx:
            raise UnsupportedModelException(
                "Language is mapped to a model index which does not exist"
            )

        return self._model_handlers[reference_idx]

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
        if len(self._model_handlers) == 0:
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
            return self._model_handlers[0].process(inputs)

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
            model_handler_results_by_language[language] = model_handler.process(
                audio_list
            )

        # For use with .pop()
        batch_language_ordering.reverse()

        # Reassemble and postprocess
        for language in batch_language_ordering:
            model_handler_result = model_handler_results_by_language[language].pop()

            model_handler_results.append(model_handler_result)

        return model_handler_results
