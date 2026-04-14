from caul.asr_pipeline import ASRPipeline
from caul.objects import ASRModel

MODEL_FAMILY_CONFIG_MAP = {
    ASRModel.PARAKEET: ASRPipeline.parakeet(),
    ASRModel.FIREREDASR2_AED: ASRPipeline.fireredasr2(),
}
