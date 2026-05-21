from caul_core.objects import ASRModel
from .asr_pipeline import ASRPipeline

MODEL_FAMILY_CONFIG_MAP = {
    ASRModel.PARAKEET: ASRPipeline.parakeet(),
    ASRModel.FIREREDASR2_AED: ASRPipeline.fireredasr2(),
    ASRModel.FASTER_WHISPER: ASRPipeline.faster_whisper(),
}
