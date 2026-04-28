import string
from pathlib import Path

import pytest

from caul.asr_pipeline import ASRPipeline
from caul.constants import TorchDevice
from caul.handler import ASRHandler
from caul.objects import ASRModel

RESOURCES = Path(__file__).parents[1] / "resources"

RU_TEXT = "Привет, мир! Меня зовут Милена. Сегодня хорошая погода."
ZH_TEXT = "你好, 世界! 今天天气很好，我很开心."
AR_TEXT = "مرحبًا بالعالم. أنا سعيد اليوم."


def _full_text(result) -> str:
    return "".join(seg[2] for seg in result.transcription)


class TestASRHandler:
    def setup_method(self):
        models = [
            ASRModel.PARAKEET,
            ASRModel.FASTER_WHISPER,
            ASRPipeline.fireredasr2(tmp_dir_fallback=True),
        ]
        language_map = {"ru": 0, "ar": 1, "zh": 2}
        self._handler = ASRHandler(
            models=models, device=TorchDevice.CPU, language_map=language_map
        )

    @pytest.mark.integration
    def test__parakeet_transcribes_russian(self):
        """Parakeet (parakeet-tdt-0.6b-v3) should transcribe Russian speech"""
        with self._handler:
            results = list(
                self._handler.transcribe(str(RESOURCES / "ru_test.wav"), languages="ru")
            )

        assert len(results) == 1
        text = _full_text(results[0]).lower()
        assert results[0].transcription
        for word in [
            token.strip(string.punctuation).lower() for token in RU_TEXT.split(" ")
        ]:
            assert word in text

    @pytest.mark.integration
    def test__fireredasr2_transcribes_chinese(self):
        """FireRedASR2-AED should transcribe Mandarin Chinese speech"""
        with self._handler:
            results = list(
                self._handler.transcribe(str(RESOURCES / "zh_test.wav"), languages="zh")
            )

        assert len(results) == 1
        text = _full_text(results[0])
        print(text)
        assert results[0].transcription
        assert "你好" in text
        assert "世界" in text

    @pytest.mark.integration
    def test__faster_whisper_transcribes_arabic(self):
        """Faster Whisper large-v3-turbo should transcribe Arabic speech"""
        with self._handler:
            results = list(
                self._handler.transcribe(str(RESOURCES / "ar_test.wav"), languages="ar")
            )

        assert len(results) == 1
        text = _full_text(results[0])
        assert results[0].transcription
        assert "مرحب" in text
        assert "العالم" in text
