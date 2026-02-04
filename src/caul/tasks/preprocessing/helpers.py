import datetime
import uuid

from dataclasses import dataclass

import torch


@dataclass
class InputMetadata:
    """Preprocessed input metadata"""

    input_ordering: int
    duration: int
    start_time: int = 0
    end_time: int = 0
    preprocessed_at: str = (
        datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0).isoformat()
    )
    uuid: str = uuid.uuid4().hex
    input_format: str = None
    input_file_path: str = None
    preprocessed_file_path: str = None


@dataclass
class PreprocessedInput:
    """Preprocessed input wrapper"""

    tensor: torch.Tensor
    metadata: InputMetadata
