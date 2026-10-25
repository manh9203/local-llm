from dataclasses import dataclass


@dataclass(frozen=True)
class SummaryStats:
    input_tokens: int
    output_tokens: int
    first_token_seconds: float
    elapsed_seconds: float
    decode_tps: float
    peak_memory_gb: float | None
    finish_reason: str | None

    @property
    def prefill_tps(self) -> float:
        if self.first_token_seconds <= 0:
            return 0.0
        return self.input_tokens / self.first_token_seconds
