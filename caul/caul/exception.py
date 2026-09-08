from pathlib import Path

from caul_core import AudioMetadata, SegmentMetadata


class MissingModelSpecificationException(Exception):
    """Raise if referencing a missing model"""


class UnsupportedModelException(Exception):
    """Raise if an unsupported model type is passed"""


class MissingFireRedAsr2OutputDirException(Exception):
    """Raise if no file path is available to store FireRedAsr2 inputs at"""


class LanguageInputMismatchException(Exception):
    """Raise when inputs and input languages to ASRHandler.transcribe don't align"""


class MissingTokenizerException(Exception):
    """Raise if no tokenizer or tokenizer path provided"""


class UnreadableAudio(Exception):
    def __init__(self, path: Path):
        msg = f"failed to read audio file at {path}"
        super().__init__(msg)


class UnprocessableAudio(Exception):
    def __init__(self, metadata: SegmentMetadata | AudioMetadata):
        msg = f"failed to process audio segment {metadata}"
        super().__init__(msg)
