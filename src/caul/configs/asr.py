from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from caul.constant import DEVICE_CPU

if TYPE_CHECKING:
    from caul.model_handlers.asr_model_handler import ASRModelHandler


@dataclass
class ASRConfig:
    """Base config class"""

    model_name: str
    model_handler: "ASRModelHandler"
    device: str | torch.device = DEVICE_CPU

    def handler_from_config(self) -> "ASRModelHandler":
        return (
            self.model_handler(config=self) if self.model_handler is not None else None
        )
