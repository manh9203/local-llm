from local_llm.conversation.metrics import SummaryStats


def test_summary_prefill_rate():
    stats = SummaryStats(
        input_tokens=1000,
        output_tokens=200,
        first_token_seconds=10.0,
        elapsed_seconds=25.0,
        decode_tps=13.0,
        peak_memory_gb=5.2,
        finish_reason="stop",
    )

    assert stats.prefill_tps == 100.0


def test_summary_zero_first_token_time():
    stats = SummaryStats(
        input_tokens=1000,
        output_tokens=0,
        first_token_seconds=0.0,
        elapsed_seconds=0.0,
        decode_tps=0.0,
        peak_memory_gb=None,
        finish_reason=None,
    )

    assert stats.prefill_tps == 0.0
