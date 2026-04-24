import gc
from pathlib import Path
from typing import ClassVar, Iterable, TYPE_CHECKING

from faster_whisper.transcribe import TranscriptionOptions
from icij_common.registrable import FromConfig
from pydantic import Field

from caul.constants import (
    FASTER_WHISPER_APPEND_PUNCTUATIONS_DEFAULT,
    FASTER_WHISPER_BEAM_SIZE_DEFAULT,
    FASTER_WHISPER_BEST_OF_DEFAULT,
    FASTER_WHISPER_CLIP_TIMESTAMPS_DEFAULT,
    FASTER_WHISPER_COMPRESSION_RATIO_THRESHOLD_DEFAULT,
    FASTER_WHISPER_COMPUTE_TYPE_DEFAULT,
    FASTER_WHISPER_CONDITION_ON_PREVIOUS_TEXT_DEFAULT,
    FASTER_WHISPER_LARGE_V3_TURBO_SUPPRESSED_TOKENS,
    FASTER_WHISPER_LENGTH_PENALTY_DEFAULT,
    FASTER_WHISPER_LOG_PROB_THRESHOLD_DEFAULT,
    FASTER_WHISPER_MAX_INITIAL_TIMESTAMP_DEFAULT,
    FASTER_WHISPER_MODEL_NAME,
    FASTER_WHISPER_MULTILINGUAL_DEFAULT,
    FASTER_WHISPER_NO_REPEAT_NGRAM_SIZE_DEFAULT,
    FASTER_WHISPER_NO_SPEECH_THRESHOLD_DEFAULT,
    FASTER_WHISPER_PATIENCE_DEFAULT,
    FASTER_WHISPER_PREPEND_PUNCTUATIONS_DEFAULT,
    FASTER_WHISPER_PROMPT_RESET_ON_TEMPERATURE_DEFAULT,
    FASTER_WHISPER_REPETITION_PENALTY_DEFAULT,
    FASTER_WHISPER_SUPPRESS_BLANK_DEFAULT,
    FASTER_WHISPER_TEMPERATURES_DEFAULT,
    FASTER_WHISPER_WITHOUT_TIMESTAMPS_DEFAULT,
    FASTER_WHISPER_WORD_TIMESTAMPS_DEFAULT,
    TorchDevice,
    DEFAULT_SAMPLE_RATE,
)
from caul.objects import (
    ASRModel,
    ASRResult,
    PreprocessorOutput,
    PreprocessedInput,
    PreprocessedInputWithTensor,
)
from ..asr_task import InferenceRunner
from ...config import InferenceRunnerConfig

if TYPE_CHECKING:
    import torch


class FasterWhisperInferenceRunnerConfig(InferenceRunnerConfig):
    model: ClassVar[str] = Field(frozen=True, default=ASRModel.FASTER_WHISPER)
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

    def to_transcription_options(self) -> TranscriptionOptions:
        return TranscriptionOptions(
            beam_size=self.beam_size,
            best_of=self.best_of,
            patience=self.patience,
            length_penalty=self.length_penalty,
            repetition_penalty=self.repetition_penalty,
            no_repeat_ngram_size=self.no_repeat_ngram_size,
            log_prob_threshold=self.log_prob_threshold,
            no_speech_threshold=self.no_speech_threshold,
            compression_ratio_threshold=self.compression_ratio_threshold,
            condition_on_previous_text=self.condition_on_previous_text,
            prompt_reset_on_temperature=self.prompt_reset_on_temperature,
            temperatures=self.temperatures,
            initial_prompt=self.initial_prompt,
            prefix=self.prefix,
            suppress_blank=self.suppress_blank,
            suppress_tokens=self.suppress_tokens,
            without_timestamps=self.without_timestamps,
            max_initial_timestamp=self.max_initial_timestamp,
            word_timestamps=self.word_timestamps,
            prepend_punctuations=self.prepend_punctuations,
            append_punctuations=self.append_punctuations,
            multilingual=self.multilingual,
            max_new_tokens=self.max_new_tokens,
            clip_timestamps=self.clip_timestamps,
            hallucination_silence_threshold=self.hallucination_silence_threshold,
            hotwords=self.hotwords,
        )


@InferenceRunner.register(ASRModel.FASTER_WHISPER)
class FasterWhisperInferenceRunner(InferenceRunner):
    """Inference runner for faster-whisper large-v3-turbo.

    Transcribes multilingual audio. Expects 16 kHz mono wav files or tensors.
    Requires faster-whisper from https://github.com/SYSTRAN/faster-whisper.
    """

    def __init__(
        self,
        config: FasterWhisperInferenceRunnerConfig = None,
        device: "TorchDevice | torch.device" = TorchDevice.CPU,
    ):
        if config is None:
            config = FasterWhisperInferenceRunnerConfig()
        self._config = config
        self._device = device
        self._model = None

    @classmethod
    def _from_config(
        cls,
        config: FasterWhisperInferenceRunnerConfig,
        device: "TorchDevice | torch.device" = TorchDevice.CPU,
        **extras,
    ) -> FromConfig:
        return cls(config=config, device=device, **extras)

    def __enter__(self):
        import faster_whisper  # pylint: disable=import-outside-toplevel

        self._model = faster_whisper.BatchedInferencePipeline(
            faster_whisper.WhisperModel(
                FASTER_WHISPER_MODEL_NAME,
                device=str(self._device),
                compute_type=self._config.compute_type,
            )
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._model = None
        gc.collect()

    def process(
        self,
        inputs: Iterable[list[PreprocessorOutput]],
        *,
        languages: list[str] | None = None,
        **kwargs,
    ) -> Iterable[ASRResult]:
        """Transcribe batches of preprocessed audio segments using faster_whisper.generate_segment_batched.

        :param inputs: batches of PreprocessorOutput (file-backed or tensor)
        :param languages: ISO-639-1 code or list of codes
        :return: ASRResult per input, in batch order
        """
        import numpy as np  # pylint: disable=import-outside-toplevel
        from faster_whisper.audio import (
            pad_or_trim,
        )  # pylint: disable=import-outside-toplevel
        from faster_whisper.tokenizer import (
            Tokenizer,
        )  # pylint: disable=import-outside-toplevel
        from faster_whisper.transcribe import (
            Segment,
        )  # pylint: disable=import-outside-toplevel

        options = self._config.to_transcription_options()

        for idx, input_batch in enumerate(inputs):
            if len(input_batch) == 0:
                continue

            if isinstance(input_batch[0], PreprocessedInputWithTensor):
                tensors = [inp.tensor.detach().cpu().numpy() for inp in input_batch]
            elif (
                isinstance(input_batch[0], PreprocessedInput)
                and input_batch[0].metadata.preprocessed_file_path is not None
            ):
                import torchaudio  # pylint: disable=import-outside-toplevel

                tensors = [
                    torchaudio.load(inp.metadata.preprocessed_file_path)[0]
                    .squeeze(0)
                    .numpy()
                    for inp in input_batch
                ]
            else:
                continue

            raw_features = [
                self._model.model.feature_extractor(tensor)[..., :-1]
                for tensor in tensors
            ]

            del tensors

            features = np.stack([pad_or_trim(f) for f in raw_features])
            segment_metadata = [
                {
                    "offset": 0.0,
                    "duration": inp.metadata.duration_s,
                    "segments": [
                        {
                            "start": 0,
                            "end": int(inp.metadata.duration_s * DEFAULT_SAMPLE_RATE),
                        }
                    ],
                }
                for inp in input_batch
            ]

            if languages is None:
                language = self._model.model.detect_language(features=raw_features[0])[
                    0
                ]
            else:
                language = languages[idx]

            tokenizer = Tokenizer(
                self._model.model.hf_tokenizer,
                self._model.model.model.is_multilingual,
                task="transcribe",
                language=language,
            )

            outputs = self._model.forward(
                features, tokenizer, segment_metadata, options
            )

            for inp, output in zip(input_batch, outputs):
                segments = [
                    Segment(
                        id=seg_idx,
                        seek=out["seek"],
                        start=out["start"],
                        end=out["end"],
                        text=out["text"],
                        tokens=out["tokens"],
                        avg_logprob=out["avg_logprob"],
                        no_speech_prob=out["no_speech_prob"],
                        compression_ratio=out["compression_ratio"],
                        words=None,
                        temperature=options.temperatures[0],
                    )
                    for seg_idx, out in enumerate(output)
                ]
                yield ASRResult.from_faster_whisper_result(
                    segments, input_ordering=inp.metadata.input_ordering
                )
