import datetime
import uuid

from typing import Optional

import torch
from pydantic import BaseModel


class InputMetadata(BaseModel):
    """Preprocessed input metadata"""

    duration: float
    input_ordering: int = -1
    start_time: float = 0
    end_time: float = 0
    preprocessed_at: str = (
        datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0).isoformat()
    )
    uuid: str = uuid.uuid4().hex
    input_format: Optional[str] = None
    input_file_path: Optional[str] = None
    preprocessed_file_path: Optional[str] = None


class PreprocessedInput(BaseModel):
    """Preprocessed input wrapper"""

    model_config = {"arbitrary_types_allowed": True}
    metadata: InputMetadata
    tensor: Optional[torch.Tensor | list] = None
