from abc import ABC, abstractmethod
from typing import Any


class ASRTask(ABC):
    """Generic ASR task"""

    @abstractmethod
    def process(self, inputs: Any, *args, **kwargs) -> list:
        """Generic processing task"""
