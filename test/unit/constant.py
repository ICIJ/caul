from pathlib import Path

TEST_PATH = Path(__file__).parent.parent

TEST_WAV_PATH = TEST_PATH.joinpath("resources", "asr_test.wav")

PARAKEET_TEST_SEGMENT_START = 0.08

PARAKEET_TEST_SEGMENT_END = 1.26

PARAKEET_TEST_TRANSCRIPTION = "To embrace the chaos that they fought in this battle."

PARAKEET_TEST_CONFIDENCE = -248

TEST_TIMESTAMPS = [{"start": 0, "end": 8000}]
