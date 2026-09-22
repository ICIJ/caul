def import_trt():
    """Import TensorRT, preferring the lean runtime over the full package."""
    try:
        import tensorrt_lean as trt  # pylint: disable=import-outside-toplevel
    except ImportError:
        import tensorrt as trt  # pylint: disable=import-outside-toplevel

    return trt
