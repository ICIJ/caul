from abc import ABC
from pathlib import Path
from typing import ClassVar

from icij_common.registrable import RegistrableConfig
from pydantic import Field

from .constants import (
    DEFAULT_SAMPLE_RATE,
    DEFAULT_BATCH_SIZE,
    DEFAULT_MAX_FRAMES,
    DEFAULT_LARGE_FILE_THRESHOLD_BYTES,
    PARAKEET_INFERENCE_MAX_FRAMES,
    PARAKEET_MODEL_REF,
    FIREREDASR2_USE_HALF_DEFAULT,
    FIREREDASR2_BEAM_SIZE_DEFAULT,
    FIREREDASR2_NBEST_DEFAULT,
    FIREREDASR2_DECODE_MAX_LEN_DEFAULT,
    FIREREDASR2_SOFTMAX_SMOOTHING_DEFAULT,
    FIREREDASR2_AED_LENGTH_PENALTY_DEFAULT,
    FIREREDASR2_EOS_PENALTY_DEFAULT,
    FIREREDASR2_RETURN_TIMESTAMP_DEFAULT,
    FIREREDASR2_INFERENCE_MAX_FRAMES,
    FASTER_WHISPER_COMPUTE_TYPE_DEFAULT,
    FASTER_WHISPER_WORD_TIMESTAMPS_DEFAULT,
    FASTER_WHISPER_BEAM_SIZE_DEFAULT,
    FASTER_WHISPER_BEST_OF_DEFAULT,
    FASTER_WHISPER_PATIENCE_DEFAULT,
    FASTER_WHISPER_LENGTH_PENALTY_DEFAULT,
    FASTER_WHISPER_REPETITION_PENALTY_DEFAULT,
    FASTER_WHISPER_NO_REPEAT_NGRAM_SIZE_DEFAULT,
    FASTER_WHISPER_LOG_PROB_THRESHOLD_DEFAULT,
    FASTER_WHISPER_NO_SPEECH_THRESHOLD_DEFAULT,
    FASTER_WHISPER_COMPRESSION_RATIO_THRESHOLD_DEFAULT,
    FASTER_WHISPER_CONDITION_ON_PREVIOUS_TEXT_DEFAULT,
    FASTER_WHISPER_PROMPT_RESET_ON_TEMPERATURE_DEFAULT,
    FASTER_WHISPER_TEMPERATURES_DEFAULT,
    FASTER_WHISPER_SUPPRESS_BLANK_DEFAULT,
    FASTER_WHISPER_LARGE_V3_TURBO_SUPPRESSED_TOKENS,
    FASTER_WHISPER_WITHOUT_TIMESTAMPS_DEFAULT,
    FASTER_WHISPER_MAX_INITIAL_TIMESTAMP_DEFAULT,
    FASTER_WHISPER_PREPEND_PUNCTUATIONS_DEFAULT,
    FASTER_WHISPER_APPEND_PUNCTUATIONS_DEFAULT,
    FASTER_WHISPER_MULTILINGUAL_DEFAULT,
    FASTER_WHISPER_CLIP_TIMESTAMPS_DEFAULT,
)
from .objects import BaseModel, ASRModel, FasterWhisperModel


class _BaseConfig(BaseModel, RegistrableConfig, ABC): ...


class BasePreprocessorConfig(_BaseConfig):
    registry_key: ClassVar[str] = Field(frozen=True, default="model")
    model: ClassVar[ASRModel]

    max_frames: int = DEFAULT_MAX_FRAMES
    batch_size: int = DEFAULT_BATCH_SIZE
    sample_rate: int = DEFAULT_SAMPLE_RATE
    large_file_threshold_bytes: int = DEFAULT_LARGE_FILE_THRESHOLD_BYTES


class BaseInferenceRunnerConfig(_BaseConfig):
    registry_key: ClassVar[str] = Field(frozen=True, default="model")
    model: ClassVar[ASRModel]

    tmp_dir_fallback: bool = False


class BasePostprocessorConfig(_BaseConfig):
    registry_key: ClassVar[str] = Field(frozen=True, default="model")
    model: ClassVar[ASRModel]


# Parakeet


class ParakeetPreprocessorConfig(BasePreprocessorConfig):
    model: ClassVar[str] = Field(default=ASRModel.PARAKEET)

    max_frames: int = PARAKEET_INFERENCE_MAX_FRAMES


class ParakeetInferenceRunnerConfig(BaseInferenceRunnerConfig):
    model: ClassVar[str] = Field(frozen=True, default=ASRModel.PARAKEET)
    model_name: str = PARAKEET_MODEL_REF
    return_timestamps: bool = True


class ParakeetTrtInferenceRunnerConfig(BaseInferenceRunnerConfig):
    model_path: Path | str = None
    engine_path: Path | str = None
    model: ClassVar[str] = Field(frozen=True, default=ASRModel.PARAKEET_TRT)
    return_timestamps: bool = True


class ParakeetPostprocessorConfig(BasePostprocessorConfig):
    model: ClassVar[str] = Field(frozen=True, default=ASRModel.PARAKEET)


# Faster Whisper


class FasterWhisperPreprocessorConfig(BasePreprocessorConfig):
    model: ClassVar[str] = Field(default=ASRModel.FASTER_WHISPER)


class FasterWhisperInferenceRunnerConfig(BaseInferenceRunnerConfig):
    model: ClassVar[str] = Field(frozen=True, default=ASRModel.FASTER_WHISPER)

    whisper_model_name: FasterWhisperModel = FasterWhisperModel.MEDIUM

    compute_type: str = FASTER_WHISPER_COMPUTE_TYPE_DEFAULT
    word_timestamps: bool = FASTER_WHISPER_WORD_TIMESTAMPS_DEFAULT
    beam_size: int = FASTER_WHISPER_BEAM_SIZE_DEFAULT
    best_of: int = FASTER_WHISPER_BEST_OF_DEFAULT
    patience: int = FASTER_WHISPER_PATIENCE_DEFAULT
    length_penalty: int = FASTER_WHISPER_LENGTH_PENALTY_DEFAULT
    repetition_penalty: int = FASTER_WHISPER_REPETITION_PENALTY_DEFAULT
    no_repeat_ngram_size: int = FASTER_WHISPER_NO_REPEAT_NGRAM_SIZE_DEFAULT
    log_prob_threshold: float = FASTER_WHISPER_LOG_PROB_THRESHOLD_DEFAULT
    no_speech_threshold: float = FASTER_WHISPER_NO_SPEECH_THRESHOLD_DEFAULT
    compression_ratio_threshold: float = (
        FASTER_WHISPER_COMPRESSION_RATIO_THRESHOLD_DEFAULT
    )
    condition_on_previous_text: bool = FASTER_WHISPER_CONDITION_ON_PREVIOUS_TEXT_DEFAULT
    prompt_reset_on_temperature: float = (
        FASTER_WHISPER_PROMPT_RESET_ON_TEMPERATURE_DEFAULT
    )
    temperatures: list[float] = FASTER_WHISPER_TEMPERATURES_DEFAULT
    initial_prompt: str | None = None
    prefix: str | None = None
    suppress_blank: bool = FASTER_WHISPER_SUPPRESS_BLANK_DEFAULT
    suppress_tokens: list[int] = FASTER_WHISPER_LARGE_V3_TURBO_SUPPRESSED_TOKENS
    without_timestamps: bool = FASTER_WHISPER_WITHOUT_TIMESTAMPS_DEFAULT
    max_initial_timestamp: float = FASTER_WHISPER_MAX_INITIAL_TIMESTAMP_DEFAULT
    prepend_punctuations: str = FASTER_WHISPER_PREPEND_PUNCTUATIONS_DEFAULT
    append_punctuations: str = FASTER_WHISPER_APPEND_PUNCTUATIONS_DEFAULT
    multilingual: bool = FASTER_WHISPER_MULTILINGUAL_DEFAULT
    max_new_tokens: int | None = None
    clip_timestamps: list[float] = FASTER_WHISPER_CLIP_TIMESTAMPS_DEFAULT
    hallucination_silence_threshold: float | None = None
    hotwords: str | None = None


class FasterWhisperPostprocessorConfig(BasePostprocessorConfig):
    model: ClassVar[str] = Field(frozen=True, default=ASRModel.FASTER_WHISPER)


# FireRedASR2s


class FireRedASR2PreprocessorConfig(BasePreprocessorConfig):
    model: ClassVar[str] = Field(default=ASRModel.FIREREDASR2_AED)

    max_frames: int = FIREREDASR2_INFERENCE_MAX_FRAMES


class FireRedASR2InferenceRunnerConfig(BaseInferenceRunnerConfig):
    model: ClassVar[str] = Field(frozen=True, default=ASRModel.FIREREDASR2_AED)

    use_half: bool = FIREREDASR2_USE_HALF_DEFAULT
    beam_size: int = FIREREDASR2_BEAM_SIZE_DEFAULT
    nbest: int = FIREREDASR2_NBEST_DEFAULT
    decode_max_len: int = FIREREDASR2_DECODE_MAX_LEN_DEFAULT
    softmax_smoothing: float = FIREREDASR2_SOFTMAX_SMOOTHING_DEFAULT
    aed_length_penalty: float = FIREREDASR2_AED_LENGTH_PENALTY_DEFAULT
    eos_penalty: float = FIREREDASR2_EOS_PENALTY_DEFAULT
    return_timestamp: bool = FIREREDASR2_RETURN_TIMESTAMP_DEFAULT


class FireRedASR2PostprocessorConfig(BasePostprocessorConfig):
    model: ClassVar[str] = Field(frozen=True, default=ASRModel.FIREREDASR2_AED)
