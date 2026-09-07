from pathlib import Path
from unittest.mock import call, patch

from caul.utils import cache_hf_model

_HF_HUB_DOWNLOAD_PATH = "huggingface_hub.hf_hub_download"
_GET_TOKEN_PATH = "huggingface_hub.get_token"
_HF_HUB_CACHE_PATH = "huggingface_hub.constants.HF_HUB_CACHE"


class TestCacheHfModel:
    def test__forwards_args_through_cache_hf_model_file_to_hf_hub_download(self):
        cache_dir = Path("/fake/cache")
        models = ["org/model-a", "org/model-b"]
        token = "fake-token"

        with (
            patch(_HF_HUB_CACHE_PATH, str(cache_dir)),
            patch(_GET_TOKEN_PATH, return_value=token),
            patch(_HF_HUB_DOWNLOAD_PATH) as mock_hf_hub_download,
        ):
            cache_hf_model(
                model_family="parakeet",
                models=models,
                model_ext=".engine",
                library_name="nemo",
                cache_dir=cache_dir,
            )

        assert mock_hf_hub_download.call_args_list == [
            call(
                repo_id="org/model-a",
                filename="model-a.engine",
                cache_dir=cache_dir,
                library_name="nemo",
                library_version=None,
                force_download=False,
                token=token,
            ),
            call(
                repo_id="org/model-b",
                filename="model-b.engine",
                cache_dir=cache_dir,
                library_name="nemo",
                library_version=None,
                force_download=False,
                token=token,
            ),
        ]
