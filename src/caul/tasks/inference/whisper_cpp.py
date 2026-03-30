from typing import ClassVar

from icij_common.registrable import FromConfig
from pydantic import Field

from caul.constant import ASRModel
from caul.objects import ASRResult, PreprocessorOutput
from ..asr_task import InferenceRunner
from ...config import InferenceRunnerConfig


class WhisperCppInferenceRunnerConfig(InferenceRunnerConfig):
    model: ClassVar[str] = Field(frozen=True, default=ASRModel.WHISPER_CPP)


@InferenceRunner.register(ASRModel.WHISPER_CPP)
class WhisperCppInferenceRunner(InferenceRunner):
    """Handler for WhisperCPP; wrapper round subprocess calls"""
    @classmethod
    def _from_config(
        cls, config: WhisperCppInferenceRunnerConfig, **extras
    ) -> FromConfig:
        return cls()

    # pylint: disable=R0903

    def process(
        self, inputs: list[PreprocessorOutput], *args, **kwargs
    ) -> list[ASRResult]:
        """List of np.ndarray or torch.Tensor or str, or a singleton of same types

        :param inputs: List of np.ndarray or torch.Tensor or str, or a singleton of same types
        :return:
        """
