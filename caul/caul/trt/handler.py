from functools import lru_cache


@lru_cache(maxsize=None)
def _dtypes_map():
    import tensorrt as trt  # pylint: disable=import-outside-toplevel
    import torch  # pylint: disable=import-outside-toplevel

    return {
        trt.float16: torch.float16,
        trt.float32: torch.float32,
        trt.int32: torch.int32,
        trt.int64: torch.int64,
        trt.bool: torch.bool,
    }


class TrtInferenceHandler:
    """Context wrapper for inference with TRT engines"""

    def __init__(self, engine: "trt.ICudaEngine"):
        self._engine = engine
        self._context = None

    def __enter__(self):
        self._context = self._engine.create_execution_context()
        return self

    def __exit__(self, *args):
        del self._context
        self._context = None

    def infer(self, inputs: dict) -> tuple:
        import tensorrt as trt  # pylint: disable=import-outside-toplevel
        import torch  # pylint: disable=import-outside-toplevel

        trt_to_torch_dtypes = _dtypes_map()

        device = torch.device("cuda")

        for tensor_name, tensor in inputs.items():
            inputs[tensor_name] = tensor.contiguous().to(device)
            self._context.set_input_shape(tensor_name, tuple(inputs[tensor_name].shape))

        outputs = {}
        for i in range(self._engine.num_io_tensors):
            out_name = self._engine.get_tensor_name(i)
            if self._engine.get_tensor_mode(out_name) == trt.TensorIOMode.OUTPUT:
                out_dtype = trt_to_torch_dtypes[self._engine.get_tensor_dtype(out_name)]
                out_shape = tuple(self._context.get_tensor_shape(out_name))
                outputs[out_name] = torch.empty(
                    out_shape, dtype=out_dtype, device=device
                )

        for io_name, tensor in {**inputs, **outputs}.items():
            self._context.set_tensor_address(io_name, tensor.data_ptr())

        stream = torch.cuda.current_stream()
        self._context.execute_async_v3(stream.cuda_stream)
        stream.synchronize()

        return tuple(outputs.values())
