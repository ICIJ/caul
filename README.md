[![Lint](https://github.com/ICIJ/caul/actions/workflows/pylint.yml/badge.svg)](https://github.com/ICIJ/caul/actions/workflows/pylint.yml)
[![Unit tests](https://github.com/ICIJ/caul/actions/workflows/test-unit.yml/badge.svg)](https://github.com/ICIJ/caul/actions/workflows/test-unit.yml)
![Supported Python versions](https://img.shields.io/badge/Python-=%3D%203.10-blue)
![Version](https://img.shields.io/badge/Version-%200.1.0-red)

# caul
**Automatic speech recognition in Python**

***"Here's to Harry ... the best, bar none."***

Audiofile transcription using NVIDIA's [Parakeet](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) family of multilingual models with fallback to [Whisper.CPP](https://https://github.com/ggml-org/whisper.cpp) for languages outside Parakeet's scope. Built with `uv` for package and project management. Installation's as simple as
```aiignore
uv python install 3.10
uv sync --dev
```

A handler object can be instantiated and run on one or more WAV files or directly on NumPy/Torch tensors, returning a list of `ASRHandlerResult` mapped to each input with attributes `transcription`, containing a list of tuples of the form `(start_time, text_segment)`, and `scores`:
```aiignore
>>> from src.caul import ASRHandler
>>> from src.caul.model import ParakeetModelHandler
>>> model = ParakeetModelHandler("nvidia/parakeet-tdt-0.6b-v3")
>>> handler = ASRHandler(models=model)
>>> handler.startup()
>>> results = handler.transcribe("<...path to some audio file...>")
>>> print(results)
ASRHandlerResult(transcriptions=[[(0.0, "Gr-r-r--there go, my heart's abhorrence! Water your damned flower-pots, do!"), ...], ...], scores=[-250.0, ...])
```