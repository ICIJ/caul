import logging
import tempfile
from pathlib import Path

from caul_core.objects import PreprocessorOutput
from .filesystem import save_tensor

logger = logging.getLogger(__name__)


def fuzzy_match(key: str, candidates: set[str]) -> set[str]:
    if key in candidates:
        return {key}
    fuzzy_matches = set(k for k in candidates if key in k or k in key)
    return fuzzy_matches


def prepare_file_input_batch(
    input_batch: list[PreprocessorOutput],
    output_dir: str | Path = None,
    tmp_dir_fallback: bool = False,
) -> tuple[list[str], list[str], dict[str, int], tempfile.TemporaryDirectory | None]:
    """Collect input ids and file paths and write tensors to a temp directory when
    no path is available.

    :param input_batch: batch of PreprocessorOutput files
    :return: tuple of batch input ids, wav paths, map from id to input ordering,
    temporary dir (if applicable) where tensor paths are kept
    """
    from torch import Tensor

    tmp_dir = None
    if output_dir is None and tmp_dir_fallback:
        tmp_dir = tempfile.TemporaryDirectory()
        output_dir = tmp_dir.name

    if output_dir is not None and not isinstance(output_dir, Path):
        output_dir = Path(output_dir)

    inp_ids: list[str] = []
    wav_paths: list[str] = []
    inp_id_ordering_map: dict[str, int] = {}

    for inp in input_batch:
        inp_id = inp.metadata.uuid

        if inp.metadata.preprocessed_file_path is None and output_dir is None:
            logger.warning(
                "Input %s has no preprocessed file path, no output dir is specified, "
                "and temporary dir creation is disabled. Skipping.",
                inp_id,
            )
            continue

        if (
            inp.metadata.preprocessed_file_path is None
            and not hasattr(inp, "tensor")
            and (
                inp.tensor is None or not isinstance(inp.tensor, (Tensor, list[Tensor]))
            )
        ):
            logger.warning(
                "Input %s does not have a preprocessed file path or a valid tensor"
                "to save to disk. Skipping",
                inp_id,
            )
            continue

        inp_ids.append(inp_id)
        inp_id_ordering_map[inp_id] = inp.metadata.input_ordering

        if inp.metadata.preprocessed_file_path is not None:
            wav_paths.append(str(inp.metadata.preprocessed_file_path))
        elif output_dir is not None:
            wav_path = output_dir / f"{inp_id}.wav"
            save_tensor(inp.tensor, wav_path)
            wav_paths.append(str(wav_path))

    return inp_ids, wav_paths, inp_id_ordering_map, tmp_dir


def cache_hf_model_file(
    repo_id: str,
    *,
    filename: str,
    library_name: str | None = None,
    library_version: str | None = None,
    cache_dir: Path | None = None,
) -> None:
    from huggingface_hub import (
        hf_hub_download,
        get_token,
    )  # pylint: disable=import-outside-toplevel

    hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        cache_dir=cache_dir,
        library_name=library_name,
        library_version=library_version,
        force_download=False,
        token=get_token(),
    )


def cache_hf_repo(
    repo_id: str,
    *,
    library_name: str | None = None,
    library_version: str | None = None,
    cache_dir: Path | None = None,
) -> None:
    from huggingface_hub import (
        snapshot_download,
        get_token,
    )  # pylint: disable=import-outside-toplevel

    snapshot_download(
        repo_id=repo_id,
        cache_dir=cache_dir,
        library_name=library_name,
        library_version=library_version,
        force_download=False,
        token=get_token(),
    )
