from caul.constant import PARAKEET
from caul.inference.parakeet_inference import ParakeetInferenceHandler
from caul.postprocessing.parakeet_postprocessor import ParakeetPostprocessor
from caul.preprocessing.parakeet_preprocessor import ParakeetPreprocessor

MODEL_FAMILY_COMPONENTS = {
    PARAKEET: (ParakeetPreprocessor, ParakeetInferenceHandler, ParakeetPostprocessor),
}
