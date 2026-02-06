from dataclasses import dataclass
from typing import TYPE_CHECKING

from caul.configs.asr import ASRConfig
from caul.constant import EXPECTED_SAMPLE_RATE, PARAKEET_MODEL_REF
from caul.model_handlers.parakeet import ParakeetModelHandler

if TYPE_CHECKING:
    from caul.model_handlers.asr_model_handler import ASRModelHandler


@dataclass
class ParakeetConfig(ASRConfig):

    model_name: str = PARAKEET_MODEL_REF
    model_handler: "ASRModelHandler" = ParakeetModelHandler
    save_to_filesystem: bool = True
    return_tensors: bool = True
    sample_rate: int = EXPECTED_SAMPLE_RATE
