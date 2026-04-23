import pytest

from caul.asr_pipeline import ASRPipelineConfig


@pytest.mark.parametrize(
    "model",
    ["parakeet", "fireredasr2"],
)
def test_asr_pipeline_config(model: str):
    config = getattr(ASRPipelineConfig, model)
    assert isinstance(config, ASRPipelineConfig)
