"""Smoke test for simulation mode metrics.

Run with: python3 slop/test_sim.py
"""
import os

os.environ["TRTLLM_LOG_LEVEL"] = "WARNING"

AIC_SYSTEMS_DIR = "/code/slop/aiconfigurator/src/aiconfigurator/systems"
MODEL_PATH = "/code/slop/models/TinyLlama-1.1B-Chat-v1.0"


def test_constant_metrics():
    """Constant predictor: exact metric values verifiable."""
    from tensorrt_llm.llmapi import LLM, SamplingParams
    from tensorrt_llm.llmapi.sim_config import SimConfig, PredictorConfig

    print("\n=== Constant Predictor Metrics ===", flush=True)
    sim_config = SimConfig(predictor=PredictorConfig(
        constant_prefill_time_ms=10.0, constant_decode_time_ms=5.0))
    llm = LLM(MODEL_PATH, sim_config=sim_config)
    llm.generate(["Hello world"], sampling_params=SamplingParams(max_tokens=8))

    clock = sim_config._clock
    assert clock is not None
    m = clock.metrics

    print(f"TTFT: {m['mean_ttft_ms']:.1f}ms", flush=True)
    print(f"TPOT: {m['mean_tpot_ms']:.1f}ms", flush=True)
    print(f"E2E:  {m['mean_e2e_latency_ms']:.1f}ms", flush=True)
    print(f"Throughput: {m['output_throughput']:.1f} tok/s", flush=True)

    # TTFT = prefill (10ms) + first decode (5ms) = 15ms
    # (sampler's update_requests isn't called during prefill iteration)
    assert abs(m["mean_ttft_ms"] - 15.0) < 0.1
    assert m["mean_tpot_ms"] > 0
    assert abs(m["mean_e2e_latency_ms"] - 45.0) < 0.1
    assert m["output_throughput"] > 170
    assert m["completed"] == 1
    assert m["total_output"] == 8

    # Per-request
    assert len(clock.request_stats) == 1
    stats = list(clock.request_stats.values())[0]
    assert len(stats.gen_token_times) == 8
    assert stats.input_length > 0

    # Per-iteration
    assert len(clock.iterations) == 8

    print("CONSTANT METRICS OK", flush=True)


def test_aic_metrics():
    """AIC predictor: structural checks + cross-consistency."""
    from tensorrt_llm.llmapi import LLM, SamplingParams
    from tensorrt_llm.llmapi.sim_config import SimConfig, PredictorConfig

    print("\n=== AIC Predictor Metrics ===", flush=True)
    sim_config = SimConfig(predictor=PredictorConfig(
        name="aiconfigurator", device_name="h100_sxm",
        backend_version="1.2.0rc5", database_path=AIC_SYSTEMS_DIR))
    llm = LLM(MODEL_PATH, sim_config=sim_config)
    llm.generate(["Hello world"], sampling_params=SamplingParams(max_tokens=8))

    clock = sim_config._clock
    assert clock is not None
    m = clock.metrics

    print(f"TTFT: {m['mean_ttft_ms']:.2f}ms", flush=True)
    print(f"TPOT: {m['mean_tpot_ms']:.2f}ms", flush=True)
    print(f"E2E:  {m['mean_e2e_latency_ms']:.2f}ms", flush=True)
    print(f"Throughput: {m['output_throughput']:.1f} tok/s", flush=True)

    assert m["completed"] == 1
    assert m["total_output"] == 8
    assert m["mean_ttft_ms"] > 0
    assert m["mean_tpot_ms"] > 0
    assert m["mean_ttft_ms"] > m["mean_tpot_ms"], \
        f"Prefill should be slower than decode: TTFT={m['mean_ttft_ms']:.2f} <= TPOT={m['mean_tpot_ms']:.2f}"

    # Cross-check: e2e ≈ ttft + num_itl * mean_itl
    num_itl = m["total_output"] - 1  # 7 inter-token gaps
    expected_e2e = m["mean_ttft_ms"] + num_itl * m["mean_itl_ms"]
    assert abs(m["mean_e2e_latency_ms"] - expected_e2e) < 1.0, \
        f"E2E mismatch: {m['mean_e2e_latency_ms']:.2f} vs expected {expected_e2e:.2f}"

    # Per-iteration: prefill vs decode should differ
    iters = clock.iterations
    assert len(iters) == 8
    prefill_time = iters[0].predicted_duration_s
    decode_time = iters[1].predicted_duration_s
    assert prefill_time != decode_time, \
        "AIC should differentiate prefill vs decode"
    print(f"Prefill iter: {prefill_time*1000:.2f}ms, Decode iter: {decode_time*1000:.2f}ms",
          flush=True)

    print("AIC METRICS OK", flush=True)


def test_aic_tp2_metrics():
    """AIC TP=2: metrics structure same, values differ from TP=1."""
    from tensorrt_llm.llmapi import LLM, SamplingParams
    from tensorrt_llm.llmapi.sim_config import SimConfig, PredictorConfig

    print("\n=== AIC TP=2 Metrics ===", flush=True)
    sim_tp2 = SimConfig(predictor=PredictorConfig(
        name="aiconfigurator", device_name="h100_sxm",
        backend_version="1.2.0rc5", database_path=AIC_SYSTEMS_DIR))
    llm = LLM(MODEL_PATH, sim_config=sim_tp2, tensor_parallel_size=2)
    llm.generate(["Hello world"], sampling_params=SamplingParams(max_tokens=8))

    m = sim_tp2._clock.metrics
    assert m["completed"] == 1
    assert m["total_output"] == 8
    assert m["mean_ttft_ms"] > 0
    assert m["mean_tpot_ms"] > 0

    print(f"TP=2 TTFT: {m['mean_ttft_ms']:.2f}ms", flush=True)
    print(f"TP=2 TPOT: {m['mean_tpot_ms']:.2f}ms", flush=True)

    print("AIC TP=2 METRICS OK", flush=True)


def main():
    test_constant_metrics()
    test_aic_metrics()
    test_aic_tp2_metrics()
    print("\n=== ALL TESTS PASSED ===", flush=True)


if __name__ == "__main__":
    main()
