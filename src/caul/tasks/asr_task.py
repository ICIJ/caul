from abc import ABC, abstractmethod
from typing import Any


class ASRTask(ABC):
    """Generic ASR task"""

    # pylint: disable=R0903

    @abstractmethod
    def process(self, inputs: Any, *args, **kwargs) -> list:
        """Generic processing task"""
