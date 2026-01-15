import logging

from dataclasses import dataclass, field

import torch

import numpy as np

from caul import MODEL_FAMILY_COMPONENTS
from caul.exception import (
    MissingModelSpecificationException,
    UnsupportedModelException,
    MissingComponentException,
)
from caul.inference.asr_inference import (
    ASRInferenceHandlerResult,
    ASRInferenceHandler,
)
from caul.postprocessing.asr_postprocessor import ASRPostprocessor
from caul.preprocessing.asr_preprocessor import ASRPreprocessor

logger = logging.getLogger(__name__)


@dataclass
class ASRHandlerResult:
    """ASRHandlerResult class"""

    # pylint: disable=R0914

    transcriptions: list[list[tuple[float, str]]] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)

    def add_transcriptions(
        self,
        inference_result: list[ASRInferenceHandlerResult] | ASRInferenceHandlerResult,
    ):
        """Parse ASRInferenceHandlerResult

        :param inference_result: List of ASRInferenceHandlerResult
        """
        if not isinstance(inference_result, list):
            inference_result = [inference_result]

        for result in inference_result:
            self.transcriptions.append(result.transcription)
            self.scores.append(result.score)

        return self


class ASRHandler:
    """ASRHandler class"""

    # pylint: disable=R0913,R0917

    def __init__(
        self,
        model: list[str] | str = None,
        preprocessor: list[ASRPreprocessor] | ASRPreprocessor = None,
        inference_handler: list[ASRInferenceHandler] | ASRInferenceHandler = None,
        postprocessor: list[ASRPostprocessor] | ASRPostprocessor = None,
        language_map: dict[str, int] = None,
    ):
        """Primary application handler class. Handles transcription agnostically.

        :param preprocessor: ASRPreprocessor list or singleton
        :param inference_handler: ASRInferenceHandler list or singleton
        :param postprocessor: ASRPostprocessor list or singleton
        :param language_map: Map from ISO-639-3 language code to index of inference_handler
        """

        if {model, preprocessor, inference_handler, postprocessor} == {None}:
            raise MissingModelSpecificationException(
                "Either a model family must be provided or a preprocessor, inference_handler, and "
                "postprocessor"
            )

        if model is None and None in [preprocessor, inference_handler, postprocessor]:
            raise MissingComponentException(
                "One of preprocessor, inference_handler, or postprocessor is missing"
            )

        if language_map is None:
            language_map = {}

        self.language_map = language_map

        self.preprocessor = []
        self.inference_handler = []
        self.postprocessor = []

        if model is not None:
            if not isinstance(model, list):
                model = [model]

            for mod in model:
                if mod.lower() not in MODEL_FAMILY_COMPONENTS:
                    raise UnsupportedModelException(f"Unsupported model '{mod}'")

                components = MODEL_FAMILY_COMPONENTS[mod]

                self.preprocessor.append(components[0])
                self.inference_handler.append(components[1])
                self.postprocessor.append(components[2])
        else:
            if not isinstance(preprocessor, list):
                preprocessor = [preprocessor]

            if not isinstance(inference_handler, list):
                inference_handler = [inference_handler]

            if not isinstance(postprocessor, list):
                postprocessor = [postprocessor]

            self.preprocessor += preprocessor
            self.inference_handler += inference_handler
            self.postprocessor += postprocessor

    def __repr__(self):
        return (
            f"<ASRHandler "
            f"preprocessor={self.preprocessor}, "
            f"inference_handler={self.inference_handler}, "
            f"postprocessor={self.postprocessor}>"
        )

    def startup(self):
        """Load all models into memory"""
        for inference_handler in self.inference_handler:
            inference_handler.load()

    def shutdown(self):
        """Garbage collect inference handlers"""
        self.inference_handler = []

    def get_resources_by_language(
        self, language: str, resource_type: list[str] | str
    ) -> (
        list[ASRPreprocessor | ASRInferenceHandler | ASRPostprocessor]
        | ASRPreprocessor
        | ASRInferenceHandler
        | ASRPostprocessor
    ):
        """Get preprocessor and inference_handler from language map or return first reference to
         both if language is not mapped

        :param language: ISO-639-3 language code
        :param resource_type: list of resource_types
        :return: ASRPreprocessor, ASRInferenceHandler
        """
        resources = []
        reference_idx = self.language_map.get(
            language, 0
        )  # default to primary inference_handler when no language given

        for r_type in resource_type:
            resource = (
                getattr(self, r_type)[reference_idx]
                if hasattr(self, r_type) and len(getattr(self, r_type)) > reference_idx
                else None
            )
            resources.append(resource)

        if len(resources) == 1:
            resources = resources[0]

        return resources

    def transcribe(
        self,
        audio: list[np.ndarray | torch.Tensor | str] | np.ndarray | torch.Tensor | str,
        languages: list[str] = None,
    ) -> ASRHandlerResult:
        """Transcribe audio tensors or strings. Returns a tuple of (transcription, score). A list
        of languages of len(audio) may be passed to direct inputs to certain inference_handlers.

        :param audio: List of np.ndarray or torch.Tensor or str, or a singleton of same types
        :param languages: List of ISO-639-3 language codes
        :return: HandlerResult
        """
        if not isinstance(audio, list):
            audio = [audio]

        handler_result = ASRHandlerResult()
        audios_by_language = {}
        inference_results = {}
        batch_language_ordering = []

        if languages is None:
            preprocessed_inputs = self.preprocessor[0].process(audio)
            inference_results = self.inference_handler[0].transcribe(
                preprocessed_inputs
            )
            return handler_result.add_transcriptions(
                self.postprocessor[0].process(inference_results)
            )

        # Sort by language where present, preserving original order for returning result
        for idx, aud in enumerate(audio):
            language = languages[idx]

            if language not in audios_by_language:
                audios_by_language[language] = []

            batch_language_ordering.append(language)
            audios_by_language[language].append(aud)

        # Run inference_handler on language batch
        for language, audio_list in audios_by_language.items():
            preprocessor, inference_handler = self.get_resources_by_language(
                language, ["preprocessor", "inference_handler"]
            )
            preprocessed_inputs = preprocessor.process(audio_list)
            inference_results[language] = inference_handler.transcribe(
                preprocessed_inputs
            )

        # For use with .pop()
        batch_language_ordering.reverse()

        # Reassemble and postprocess
        for language in batch_language_ordering:
            postprocessor = self.get_resources_by_language(language, "postprocessor")
            inference_result = inference_results[language].pop()
            postprocessed_result = postprocessor.process(inference_result)

            handler_result.add_transcriptions(postprocessed_result)

        return handler_result
