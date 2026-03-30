from enum import StrEnum


class ASRModel(StrEnum):
    PARAKEET = "parakeet"
    WHISPER_CPP = "whisper_cpp"


class VadModel(StrEnum):
    SILERO_MODEL = "silero_vad"


class TorchDevice(StrEnum):
    CPU = "cpu"
    GPU = "cuda"
    MPS = "mps"


DEFAULT_SAMPLE_RATE = 16000

EXPECTED_FORMAT = "wav"

EXPECTED_SAMPLE_MINUTE = DEFAULT_SAMPLE_RATE * 60

# Parakeet
PARAKEET_MODEL_REF = "nvidia/parakeet-tdt-0.6b-v3"
PARAKEET_INFERENCE_MAX_DURATION_MIN = (
    20  # actually 24, but we want to give ourselves some room to avoid CUDA OOMs
)
PARAKEET_INFERENCE_MAX_DURATION_KHZ = (
    PARAKEET_INFERENCE_MAX_DURATION_MIN * EXPECTED_SAMPLE_MINUTE
)
PARAKEET_INFERENCE_MAX_DURATION_S = (
    PARAKEET_INFERENCE_MAX_DURATION_KHZ / DEFAULT_SAMPLE_RATE
)

# Segmenter
FIXED_SEGMENT_DEFAULT_LENGTH_SECS = 25

# Silero
SILERO_TORCH_HUB_REPO = "snakers4/silero-vad"
