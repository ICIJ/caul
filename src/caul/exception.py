class MissingModelSpecificationException(Exception):
    """Raise if referencing a missing model"""


class UnsupportedModelException(Exception):
    """Raise if an unsupported model type is passed"""
