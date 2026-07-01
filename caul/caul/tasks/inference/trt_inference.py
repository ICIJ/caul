import gc

from caul_core import TorchDevice


class TrtInferenceMixin:

    def __init__(self):
        self._encoder = None
        self._decoder = None

    def __exit__(self, exc_type, exc_val, exc_tb):
        import torch  # pylint: disable=import-outside-toplevel

        self._encoder = None
        self._decoder = None
        if self._device == torch.device(TorchDevice.GPU):
            torch.cuda.empty_cache()
        gc.collect()

        return False
