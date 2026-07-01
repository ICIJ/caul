from functools import partial

from caul_core import ASRModel, ASRPipeline

MODEL_DEFAULT_FACTORIES = {
    # Use factories here since we don't want to systematically initialize components
    # in here + some of them might have conflicting deps
    ASRModel.PARAKEET: partial(ASRPipeline.parakeet),
    ASRModel.FIREREDASR2_AED: partial(ASRPipeline.fireredasr2),
    ASRModel.FASTER_WHISPER: partial(ASRPipeline.faster_whisper),
}
