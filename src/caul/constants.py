from enum import StrEnum


class TorchDevice(StrEnum):
    CPU = "cpu"
    GPU = "cuda"
    MPS = "mps"


DEFAULT_SAMPLE_RATE = 16000
TARGET_FORMAT = "wav"

# Parakeet
PARAKEET_MODEL_REF = "nvidia/parakeet-tdt-0.6b-v3"
PARAKEET_INFERENCE_MAX_DURATION_S = (
    20 * 60  # actually 24, but we want to give ourselves some room to avoid CUDA OOMs
)
PARAKEET_INFERENCE_MAX_FRAMES = PARAKEET_INFERENCE_MAX_DURATION_S * DEFAULT_SAMPLE_RATE

# Silero
SILERO_TORCH_HUB_REPO = "snakers4/silero-vad"

PARAKEET_TDT_0_6B_V3_LANGUAGES = {
    "bg",
    "hr",
    "cs",
    "da",
    "nl",
    "en",
    "et",
    "fi",
    "fr",
    "de",
    "el",
    "hu",
    "it",
    "lv",
    "lt",
    "mt",
    "pl",
    "pt",
    "ro",
    "sk",
    "sl",
    "es",
    "sv",
    "ru",
    "uk",
}
