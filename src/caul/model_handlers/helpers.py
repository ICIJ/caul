from dataclasses import dataclass

from nemo.collections.asr.parts.utils import Hypothesis


@dataclass
class ASRModelHandlerResult:
    """Base result class for ASR models"""

    input_ordering: int = -1
    transcription: list[tuple] = None
    score: float = None


@dataclass
class ParakeetModelHandlerResult(ASRModelHandlerResult):
    """Result handler for ParakeetInferenceHandler objects"""

    def parse_parakeet_hypothesis(
        self, hypothesis: Hypothesis
    ) -> ASRModelHandlerResult:
        """Parse a hypothesis returned by a Parakeet RNN model

        :param hypothesis: Parakeet hypothesis
        :return: copy of self
        """
        self.transcription = (
            [
                (s["start"], s["end"], s["segment"])
                for s in hypothesis.timestamp.get("segment")
            ]
            if hypothesis.timestamp.get("segment") is not None
            else [(0.0, 0.0, hypothesis.text)]
        )
        self.score = round(hypothesis.score, 2)

        return self

    def concat(self, model_result: ASRModelHandlerResult) -> ASRModelHandlerResult:
        """Left fold with ParakeetModelHandlerResult object

        :param model_result: ParakeetModelHandlerResult
        :return: copy of self
        """
        if model_result is None:
            return self

        if self.transcription is None:
            self.transcription = []

        self.transcription += model_result.transcription

        # We have to weight by total segment len
        transcription_duration = self.transcription[-1][1]
        model_result_duration = model_result.transcription[-1][1]
        total_duration = transcription_duration + model_result_duration

        self.score = round(
            (
                self.score * transcription_duration
                + model_result.score * model_result_duration
            )
            / total_duration,
            2,
        )

        return self
