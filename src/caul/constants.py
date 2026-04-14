from enum import StrEnum


class VadModel(StrEnum):
    SILERO_MODEL = "silero_vad"
    PYANNOTE_MODEL = "pyannote/voice-activity-detection"


class TorchDevice(StrEnum):
    CPU = "cpu"
    GPU = "cuda"
    MPS = "mps"


DEFAULT_SAMPLE_RATE = 16000
DEFAULT_BATCH_SIZE = 32
DEFAULT_MAX_FRAMES = 25 * 60 * DEFAULT_SAMPLE_RATE  # twenty-five minutes
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

# FireRedASR2
FIREREDASR2_LANGUAGES = {"zh"}
FIREREDASR2_AED_MODEL_TAG = "aed"
FIREREDASR2_AED_MODEL_REF = (
    "FireRedTeam/FireRedASR2-" + FIREREDASR2_AED_MODEL_TAG.upper()
)
FIREREDASR2_INFERENCE_MAX_DURATION_S = 60
FIREREDASR2_INFERENCE_MAX_FRAMES = (
    FIREREDASR2_INFERENCE_MAX_DURATION_S * DEFAULT_SAMPLE_RATE
)
FIREREDASR2_USE_HALF_DEFAULT = False
FIREREDASR2_BEAM_SIZE_DEFAULT = 3
FIREREDASR2_NBEST_DEFAULT = 1
FIREREDASR2_DECODE_MAX_LEN_DEFAULT = 0
FIREREDASR2_SOFTMAX_SMOOTHING_DEFAULT = 1.25
FIREREDASR2_AED_LENGTH_PENALTY_DEFAULT = 0.6
FIREREDASR2_EOS_PENALTY_DEFAULT = 1.0
FIREREDASR2_RETURN_TIMESTAMP_DEFAULT = True
