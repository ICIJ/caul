import pytest

from caul_core import ASRModel, ASRResult


def test_all_asr_models_expose_languages() -> None:
    for model in ASRModel:
        try:
            model.supported_languages()
        except NotImplementedError:
            pytest.fail(f"{model} does not expose supported languages")


def _result(uuid: str, start: float, end: float) -> ASRResult:
    return ASRResult(
        input_ordering=0,
        transcription=[(start, end, uuid)],
        score=1.0,
        preprocessed_input_uuids=[uuid],
    )


def test_add_merges_preprocessed_input_uuids() -> None:
    """Merging two results concatenates their preprocessed_input_uuids"""
    a = _result("uuid-a", 0.0, 1.0)
    b = _result("uuid-b", 1.0, 2.0)

    merged = a + b

    assert merged.preprocessed_input_uuids == ["uuid-a", "uuid-b"]


def test_add_preserves_duplicate_uuids() -> None:
    """Merging results from the same PreprocessedInput keeps both entries"""
    a = _result("uuid-a", 0.0, 1.0)
    b = _result("uuid-a", 1.0, 2.0)

    merged = a + b

    assert merged.preprocessed_input_uuids == ["uuid-a", "uuid-a"]


def test_summing_from_base_result_collects_all_uuids() -> None:
    """The base ASRResult used by generic_unbatching_fn has no uuids of its own, so
    summing into it should yield exactly the uuids of the merged segments"""
    base = ASRResult(input_ordering=0, transcription=[], score=1.0)
    segments = [
        _result("uuid-a", 0.0, 1.0),
        _result("uuid-b", 1.0, 2.0),
        _result("uuid-c", 2.0, 3.0),
    ]

    merged = sum(segments, base)

    assert merged.preprocessed_input_uuids == ["uuid-a", "uuid-b", "uuid-c"]


def test_asr_result_defaults_to_empty_uuid_list() -> None:
    assert ASRResult(input_ordering=0).preprocessed_input_uuids == []
