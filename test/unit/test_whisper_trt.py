import math
import sys
from typing import Callable
from unittest.mock import MagicMock, patch

import pytest
import torch

for _mod in [
    "tensorrt_llm",
    "tensorrt_llm.runtime",
    "tensorrt_llm.runtime.session",
    "tensorrt_llm._utils",
    "tensorrt_llm.llmapi",
    "tensorrt_llm.llmapi.kv_cache_type",
    "tiktoken",
]:
    sys.modules.setdefault(_mod, MagicMock())

sys.modules["tensorrt_llm._utils"].trt_dtype_to_torch = lambda _: torch.float32
sys.modules["tensorrt_llm._utils"].torch_dtype_to_trt = MagicMock()

from caul_core.constants import (
    WHISPER_TRT_HOP_LENGTH,
    WHISPER_TRT_N_FFT,
    WHISPER_TRT_N_MELS,
    WHISPER_TRT_ENCODER_DOWNSAMPLING_FACTOR,
    DEFAULT_SAMPLE_RATE,
    WHISPER_TRT_MAX_MEL_PADDING_LEN,
)
from caul.tasks.inference.whisper_trt import WhisperTrtInferenceRunner
from caul.tasks.preprocessing.whisper_trt import WhisperTrtPreprocessor
from caul_core.objects import InputMetadata, PreprocessedInputWithTensor


_BATCH_SIZE = 2
_PROMPT_IDS = [50258, 50259, 50360, 50363]  # fake prompt token ids
_EOT_ID = 50257
_MAX_NEW_TOKENS = 8
_TEST_1D_TENSOR_A = 60
_TEST_1D_TENSOR_B = 50
_TEST_SIGNAL_LEN = 512
_ENC_LEN_VAL = 25

_MEL_FILTER_PATH = "caul.tasks.preprocessing.whisper_trt.load_mel_filters"

_STREAM_MOCK = MagicMock(cuda_stream=0)


def _mock_mel_filters_factory(
    n_mels=WHISPER_TRT_N_MELS, mel_filters_dir="/mel/filters", device="cpu"
) -> Callable[[], torch.Tensor]:
    def _load_mel_filters() -> torch.Tensor:
        return torch.ones(
            WHISPER_TRT_N_MELS, WHISPER_TRT_N_FFT // 2 + 1, dtype=torch.float32
        )

    return _load_mel_filters


def _make_tokenizer_mock() -> MagicMock:
    tokenizer = MagicMock()
    tokenizer.encode.side_effect = lambda text, **kw: (
        _PROMPT_IDS if "startoftranscript" in text else [_EOT_ID]
    )
    tokenizer.decode.return_value = "return value"
    tokenizer.special_tokens_set = set()
    return tokenizer


def _make_runner(
    encoder_factory=None, decoder_factory=None, max_new_tokens: int = _MAX_NEW_TOKENS
) -> WhisperTrtInferenceRunner:
    if encoder_factory is None:
        encoder_factory = MagicMock()

    if decoder_factory is None:
        decoder_factory = MagicMock()

    runner = WhisperTrtInferenceRunner(
        encoder_factory=encoder_factory,
        decoder_factory=decoder_factory,
        tokenizer=_make_tokenizer_mock(),
        max_new_tokens=max_new_tokens,
    )
    runner._encoder = MagicMock()
    runner._decoder = MagicMock()
    runner._device = torch.device("cpu")
    return runner


def _make_batch(
    batch_size: int = _BATCH_SIZE, t: int = 100
) -> list[PreprocessedInputWithTensor]:
    return [
        PreprocessedInputWithTensor(
            metadata=InputMetadata(input_ordering=i, duration_s=1.0),
            tensor=torch.zeros(1, WHISPER_TRT_N_MELS, t),
        )
        for i in range(batch_size)
    ]


class TestWhisperTrtPreprocessor:

    @property
    def _preprocessor(self):
        return WhisperTrtPreprocessor(
            n_mels=WHISPER_TRT_N_MELS,
            mel_filters_factory=_mock_mel_filters_factory,
            dtype="float32",
        )

    def test__output_shape_has_correct_mel_dim(self):
        preprocessor = self._preprocessor
        with preprocessor:
            out = preprocessor._additional_preprocessing(
                torch.zeros(DEFAULT_SAMPLE_RATE)
            )
        assert out.ndim == 3
        assert out.shape[1] == WHISPER_TRT_N_MELS

    def test__time_dim_consistent_with_stft_frame_count(self):
        """T must equal the number of STFT frames minus the last dropped frame."""
        audio = torch.zeros(DEFAULT_SAMPLE_RATE)
        stft = torch.stft(
            audio,
            WHISPER_TRT_N_FFT,
            WHISPER_TRT_HOP_LENGTH,
            window=torch.hann_window(WHISPER_TRT_N_FFT),
            return_complex=True,
        )
        expected_t = stft.shape[-1] - 1
        out = self._preprocessor._additional_preprocessing(audio)
        assert out.shape[2] == expected_t

    def test__normalized_values_in_bounded_range(self):
        """After (log10 + 4) / 4, values for real audio stay within bounded range"""
        out = self._preprocessor._additional_preprocessing(
            torch.randn(DEFAULT_SAMPLE_RATE)
        )
        assert out.min() >= -2.0
        assert out.max() <= 3.0


class TestInferenceRunnerSetup:
    def test__sets_encoder_and_decoder_in__init__(self):
        encoder_factory = MagicMock()
        decoder_factory = MagicMock()

        runner = _make_runner(
            encoder_factory=encoder_factory, decoder_factory=decoder_factory
        )

        assert runner._encoder_factory is encoder_factory
        assert runner._decoder_factory is decoder_factory

    def test__sets_encoder_and_decoder_in_from_config(self):
        encoder_factory = MagicMock()
        decoder_factory = MagicMock()
        config = MagicMock()
        config.registry_key.default = "model"
        config.model = "whisper_trt"

        config.encoder_path = "encoder/path"
        config.decoder_path = "decoder/path"

        runner = WhisperTrtInferenceRunner.from_config(
            config=config,
            encoder_factory=encoder_factory,
            decoder_factory=decoder_factory,
            tokenizer=_make_tokenizer_mock(),
        )

        assert runner._encoder_factory is encoder_factory.return_value
        assert runner._decoder_factory is decoder_factory.return_value


class TestInferenceRunnerRunEncoder:
    @staticmethod
    def _make_inputs():
        audio = torch.zeros(_BATCH_SIZE, WHISPER_TRT_N_MELS, _TEST_1D_TENSOR_A)
        lens = torch.tensor([_TEST_1D_TENSOR_A, _TEST_1D_TENSOR_B], dtype=torch.int32)
        positions = (
            torch.arange(
                math.ceil(_TEST_1D_TENSOR_A / WHISPER_TRT_ENCODER_DOWNSAMPLING_FACTOR),
                dtype=torch.int32,
            )
            .expand(_BATCH_SIZE, -1)
            .contiguous()
        )
        return audio, lens, positions

    def test__encoder_run_receives_all_three_input_keys(self):
        runner = _make_runner()
        audio, lens, positions = self._make_inputs()
        runner._encoder.infer_shapes.return_value = []
        try:
            runner._run_encoder(audio, lens, positions, stream=_STREAM_MOCK)
        except Exception:
            pass

        inputs = runner._encoder.run.call_args[1]["inputs"]
        assert set(inputs.keys()) == {"input_features", "input_lengths", "position_ids"}

    def test__output_lengths_equal_input_divided_by_downsampling_factor(
        self,
    ):
        runner = _make_runner()
        audio, lens, positions = self._make_inputs()
        fake_encoder_output = MagicMock()
        fake_encoder_output.name = "encoder_output"
        fake_encoder_output.shape = (
            audio.shape[0],
            audio.shape[-1] // WHISPER_TRT_ENCODER_DOWNSAMPLING_FACTOR,
            256,
        )
        runner._encoder.infer_shapes.return_value = [fake_encoder_output]

        _, output_lens = runner._run_encoder(
            audio, lens, positions, stream=_STREAM_MOCK
        )

        assert torch.equal(output_lens, lens // WHISPER_TRT_ENCODER_DOWNSAMPLING_FACTOR)

    def test__audio_inputs_len_are_padded_to_max_mel_padding_len(self):
        runner = _make_runner()
        t_a, t_b = _TEST_1D_TENSOR_A, _TEST_1D_TENSOR_B
        inputs_a = torch.zeros(1, WHISPER_TRT_N_MELS, t_a)
        inputs_b = torch.zeros(1, WHISPER_TRT_N_MELS, t_b)
        batch = [
            PreprocessedInputWithTensor(
                metadata=InputMetadata(input_ordering=0, duration_s=1.0),
                tensor=inputs_a,
            ),
            PreprocessedInputWithTensor(
                metadata=InputMetadata(input_ordering=1, duration_s=1.0),
                tensor=inputs_b,
            ),
        ]

        enc_out = torch.zeros(
            _BATCH_SIZE,
            t_a // WHISPER_TRT_ENCODER_DOWNSAMPLING_FACTOR,
            _TEST_SIGNAL_LEN,
        )
        enc_len = torch.tensor(
            [
                t_a // WHISPER_TRT_ENCODER_DOWNSAMPLING_FACTOR,
                t_b // WHISPER_TRT_ENCODER_DOWNSAMPLING_FACTOR,
            ],
            dtype=torch.int32,
        )
        runner._run_encoder = MagicMock(return_value=(enc_out, enc_len))
        runner._run_decoder = MagicMock(
            return_value=torch.zeros(_BATCH_SIZE, 1, _MAX_NEW_TOKENS, dtype=torch.int32)
        )
        list(runner.process([batch]))

        actual_lens = runner._run_encoder.call_args[0][1]
        assert actual_lens.tolist() == [
            WHISPER_TRT_MAX_MEL_PADDING_LEN,
            WHISPER_TRT_MAX_MEL_PADDING_LEN,
        ]


class TestInferenceRunnerRunDecoder:
    def _run_decoder(self, runner, enc_out, enc_len):
        runner._decoder.decode.return_value = torch.zeros(
            enc_len.shape[0], WHISPER_TRT_N_MELS, _MAX_NEW_TOKENS, dtype=torch.int32
        )
        with (patch.object(torch.Tensor, "cuda", lambda self: self),):
            return runner._run_decoder(enc_out, enc_len)

    @staticmethod
    def _encoder():
        enc_out = torch.zeros(_BATCH_SIZE, _ENC_LEN_VAL, _TEST_SIGNAL_LEN)
        enc_len = torch.tensor([_ENC_LEN_VAL] * _BATCH_SIZE, dtype=torch.int32)
        return enc_out, enc_len

    def test__setup_called_with_correct_args(self):
        runner = _make_runner(max_new_tokens=_MAX_NEW_TOKENS)
        self._run_decoder(runner, *self._encoder())
        args = runner._decoder.setup.call_args[0]
        assert args[0] == _BATCH_SIZE
        assert args[2] == _MAX_NEW_TOKENS

    def test__prompt_inputs_shape_is_batch_size_x_prompt_len(self):
        runner = _make_runner()
        self._run_decoder(runner, *self._encoder())

        prompt_inputs = runner._decoder.decode.call_args[0][0]
        prompt_len = len(_PROMPT_IDS)
        assert prompt_inputs.shape == (_BATCH_SIZE, prompt_len)

    def test__cross_attention_mask_third_dim_is_encoder_length_only(self):
        runner = _make_runner()
        self._run_decoder(runner, *self._encoder())

        mask = runner._decoder.decode.call_args[1]["cross_attention_mask"]
        prompt_len = len(_PROMPT_IDS)
        expected = (_BATCH_SIZE, prompt_len + _MAX_NEW_TOKENS, _ENC_LEN_VAL)
        assert mask.shape == expected

    def test__encoder_output_forwarded_to_decode(self):
        runner = _make_runner()
        enc_out, enc_len = self._encoder()
        self._run_decoder(runner, enc_out, enc_len)

        kw = runner._decoder.decode.call_args[1]
        assert torch.equal(kw["encoder_output"], enc_out)
        assert torch.equal(kw["encoder_input_lengths"], enc_len)

    def test__sampling_config_uses_eot_as_end_id(self):
        from tensorrt_llm.runtime import SamplingConfig  # mocked

        runner = _make_runner()
        self._run_decoder(runner, *self._encoder())

        # SamplingConfig is mocked; verify it was called with the right end_id
        SamplingConfig.assert_called_with(
            end_id=_EOT_ID,
            pad_id=_EOT_ID,
            num_beams=runner._decoder_config.beam_width,
        )


class TestInferenceRunnerProcess:
    @staticmethod
    def _inference_runner() -> WhisperTrtInferenceRunner:
        runner = _make_runner()
        enc_out = torch.zeros(_BATCH_SIZE, _TEST_1D_TENSOR_B, _TEST_SIGNAL_LEN)
        enc_len = torch.tensor([_TEST_1D_TENSOR_B] * _BATCH_SIZE, dtype=torch.int32)
        runner._run_encoder = MagicMock(return_value=(enc_out, enc_len))
        runner._run_decoder = MagicMock(
            return_value=torch.zeros(_BATCH_SIZE, 1, _MAX_NEW_TOKENS, dtype=torch.int32)
        )
        return runner

    def test__empty_batch_size_yields_no_results(self):
        results = list(self._inference_runner().process([[]]))
        assert results == []

    def test__yields_one_result_per_batch_item(self):
        results = list(self._inference_runner().process([_make_batch(_BATCH_SIZE)]))
        assert len(results) == _BATCH_SIZE

    def test__input_ordering_preserved_in_results(self):
        results = list(self._inference_runner().process([_make_batch(_BATCH_SIZE)]))
        assert [r.input_ordering for r in results] == list(range(_BATCH_SIZE))

    def test__transcription_entries_are_start_end_text_tuples(self):
        results = list(self._inference_runner().process([_make_batch(_BATCH_SIZE)]))
        for result in results:
            for entry in result.transcription:
                assert isinstance(entry, tuple) and len(entry) == 3
                assert isinstance(entry[2], str)

    def test__run_encoder_called_once_per_batch(self):
        runner = self._inference_runner()
        list(runner.process([_make_batch(_BATCH_SIZE)]))
        runner._run_encoder.assert_called_once()

    def test__run_decoder_receives_encoder_output(self):
        runner = self._inference_runner()
        list(runner.process([_make_batch(_BATCH_SIZE)]))
        enc_out_arg = runner._run_decoder.call_args[0][0]
        assert enc_out_arg.shape[0] == _BATCH_SIZE

    def test__multiple_batches_each_produce_results(self):
        runner = _make_runner()
        enc_out = torch.zeros(1, _TEST_1D_TENSOR_B, _TEST_SIGNAL_LEN)
        enc_len = torch.tensor([_TEST_1D_TENSOR_B], dtype=torch.int32)
        runner._run_encoder = MagicMock(return_value=(enc_out, enc_len))
        runner._run_decoder = MagicMock(
            return_value=torch.zeros(1, 1, _MAX_NEW_TOKENS, dtype=torch.int32)
        )
        results = list(runner.process([_make_batch(1), _make_batch(1)]))
        assert len(results) == 2
