import logging
from contextlib import ExitStack
from itertools import groupby
from typing import Iterable, TYPE_CHECKING


from caul.asr_pipeline import ASRPipeline, ASRPipelineConfig
from caul.constant import ASRModel
from caul.default_configs import MODEL_FAMILY_CONFIG_MAP
from caul.objects import ASRResult
from caul.utils import fuzzy_match


from caul.exception import MissingModelSpecificationException, UnsupportedModelException

if TYPE_CHECKING:
    import torch
    import numpy as np
    from caul.constant import TorchDevice


logger = logging.getLogger(__name__)

ASR = str | ASRPipeline | ASRPipelineConfig


class ASRHandler:
    """ASRHandler class"""

    # pylint: disable=R0913,R0917

    def __init__(
        self,
        models: list[ASR] | ASR,
        device: "torch._device | TorchDevice" = None,
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

        self._pipelines = []
        self._exit_stack = ExitStack()

        if isinstance(models, list) and len(models) == 0:
            raise MissingModelSpecificationException(
                "At least one model name or model handler must be provided"
            )

        if not isinstance(models, list):
            models = [models]

        for model in models:
            if isinstance(model, str):
                matching_keys = fuzzy_match(
                    model, set(k.value for k in MODEL_FAMILY_CONFIG_MAP)
                )
                if len(matching_keys) > 1:
                    msg = (
                        f"Ambiguous model key '{model}',"
                        f" found matching keys: {matching_keys}"
                    )
                    raise UnsupportedModelException(msg)
                if not matching_keys:
                    raise UnsupportedModelException(f"Unsupported model '{model}'")
                model = ASRModel(model)
                asr_pipeline = ASRPipeline.from_config(MODEL_FAMILY_CONFIG_MAP[model])
                self._pipelines.append(asr_pipeline)
            elif isinstance(model, ASRPipelineConfig):
                self._pipelines.append(ASRPipeline.from_config(model))
            elif isinstance(model, ASRPipeline):
                self._pipelines.append(model)
            else:
                raise UnsupportedModelException(f"Unsupported model type '{model}'")

    def __repr__(self):
        return f"<ASRHandler models: {self._pipelines} "

    def __enter__(self):
        """Run all model handler startup procedures"""
        for p in self._pipelines:
            self._exit_stack.enter_context(p)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._exit_stack.__exit__(exc_type, exc_val, exc_tb)

    def _get_pipeline_by_language(self, language: str) -> ASRPipeline:
        """Get model_handler from language map or return first reference if language is not mapped

        :param language: ISO-639-3 language code
        :return: ASRModelHandler
        """
        reference_idx = self._language_map.get(
            language, 0
        )  # default to primary inference_handler when no language given

        if len(self._pipelines) <= reference_idx:
            raise UnsupportedModelException(
                "Language is mapped to a model index which does not exist"
            )

        return self._pipelines[reference_idx]

    def transcribe(
        self,
        inputs: "Iterable[np.ndarray | torch.Tensor | str] | np.ndarray | torch.Tensor | str",
        languages: Iterable[str] | None = None,
    ) -> Iterable[ASRResult]:
        """Transcribe audio tensors or strings. Returns a tuple of (transcription, score). A list
        of languages of len(inputs) may be passed to direct inputs to certain inference_handlers.

        :param inputs: List of np.ndarray or torch.Tensor or str, or a singleton of same types
        :param languages: List of ISO-639-3 language codes
        :return: HandlerResult
        """
        import numpy as np  # pylint: disable=import-outside-toplevel
        import torch  # pylint: disable=import-outside-toplevel

        if len(self._pipelines) == 0:
            raise MissingModelSpecificationException(
                "At least one model name or model pipeline must be provided"
            )

        if not isinstance(inputs, (np.ndarray, torch.Tensor, str)):
            inputs = [inputs]

        if languages is None:
            # Default to first model handler
            yield from self._pipelines[0].process(inputs)
            return

        languages_and_inputs = zip(languages, inputs, strict=True)
        inputs_by_language = groupby(languages_and_inputs, key=lambda l: l[0])
        for language, language_ins in enumerate(inputs_by_language):
            pipe = self._get_pipeline_by_language(language)
            language_ins = (l[1] for l in language_ins)
            yield from pipe.process(language_ins)
