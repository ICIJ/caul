class MissingModelSpecificationException(Exception):
    """Raise if referencing a missing model"""


class UnsupportedModelException(Exception):
    """Raise if an unsupported model type is passed"""


class MissingFireRedAsr2OutputDirException(Exception):
    """Raise if no file path is available to store FireRedAsr2 inputs at"""
