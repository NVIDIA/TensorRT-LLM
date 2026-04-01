"""Smoke test for simulation mode with SimConfig.

Run with: python3 slop/test_sim.py

No env var hacks needed — sim mode auto-skips estimation and forces
single-process executor.
"""
import os
import time

os.environ["TRTLLM_LOG_LEVEL"] = "WARNING"

AIC_SYSTEMS_DIR = "/code/slop/aiconfigurator/src/aiconfigurator/systems"
MODEL_PATH = "/code/slop/models/TinyLlama-1.1B-Chat-v1.0"


def test_constant_tp1():
    """TP=1 with constant predictor. Clock must be visible."""
    from tensorrt_llm.llmapi import LLM, SamplingParams
    from tensorrt_llm.llmapi.sim_config import SimConfig, PredictorConfig

    print("\n=== Constant Predictor TP=1 ===", flush=True)
    sim_config = SimConfig(predictor=PredictorConfig(
        constant_prefill_time_ms=10.0, constant_decode_time_ms=5.0))

    llm = LLM(MODEL_PATH, sim_config=sim_config)
    output = llm.generate(["Hello world"],
                          sampling_params=SamplingParams(max_tokens=8))

    token_ids = output[0].outputs[0].token_ids
    print(f"Tokens: {token_ids}", flush=True)
    assert len(token_ids) == 8
    assert output[0].outputs[0].finish_reason == "length"

    clock = sim_config._clock
    assert clock is not None, "SimClock must be visible (single-process)"
    print(f"Predicted: {clock.total_time_s * 1000:.1f}ms", flush=True)
    print(f"Iterations: {clock.num_iterations}", flush=True)

    # 1 prefill (10ms) + 7 decodes (5ms each) = 45ms, 8 iterations
    assert abs(clock.total_time_s - 0.045) < 0.001, \
        f"Expected ~45ms, got {clock.total_time_s * 1000:.1f}ms"
    assert clock.num_iterations == 8

    print("TP1 CONSTANT OK", flush=True)


def test_constant_tp2():
    """TP=2 with constant predictor. Proves TP>1 works in sim mode."""
    from tensorrt_llm.llmapi import LLM, SamplingParams
    from tensorrt_llm.llmapi.sim_config import SimConfig, PredictorConfig

    print("\n=== Constant Predictor TP=2 ===", flush=True)
    sim_config = SimConfig(predictor=PredictorConfig(
        constant_prefill_time_ms=10.0, constant_decode_time_ms=5.0))

    llm = LLM(MODEL_PATH, sim_config=sim_config, tensor_parallel_size=2)
    output = llm.generate(["Hello world"],
                          sampling_params=SamplingParams(max_tokens=8))

    token_ids = output[0].outputs[0].token_ids
    print(f"Tokens: {token_ids}", flush=True)
    assert len(token_ids) == 8
    assert output[0].outputs[0].finish_reason == "length"

    clock = sim_config._clock
    assert clock is not None, "SimClock must be visible (single-process)"
    print(f"Predicted: {clock.total_time_s * 1000:.1f}ms", flush=True)
    print(f"Iterations: {clock.num_iterations}", flush=True)

    # Same constant times regardless of TP
    assert abs(clock.total_time_s - 0.045) < 0.001
    assert clock.num_iterations == 8

    print("TP2 CONSTANT OK", flush=True)


def test_aic_tp1_vs_tp2():
    """AIC predictor: TP=1 and TP=2 should predict different times."""
    from tensorrt_llm.llmapi import LLM, SamplingParams
    from tensorrt_llm.llmapi.sim_config import SimConfig, PredictorConfig

    print("\n=== AIC TP=1 vs TP=2 ===", flush=True)

    # TP=1
    sim_aic1 = SimConfig(predictor=PredictorConfig(
        name="aiconfigurator", device_name="h100_sxm",
        backend_version="1.2.0rc5", database_path=AIC_SYSTEMS_DIR))
    llm1 = LLM(MODEL_PATH, sim_config=sim_aic1)
    llm1.generate(["Hello world"],
                  sampling_params=SamplingParams(max_tokens=8))
    clock1 = sim_aic1._clock
    assert clock1 is not None
    time_tp1 = clock1.total_time_s
    print(f"TP=1 predicted: {time_tp1 * 1000:.1f}ms ({clock1.num_iterations} iters)",
          flush=True)

    # TP=2
    sim_aic2 = SimConfig(predictor=PredictorConfig(
        name="aiconfigurator", device_name="h100_sxm",
        backend_version="1.2.0rc5", database_path=AIC_SYSTEMS_DIR))
    llm2 = LLM(MODEL_PATH, sim_config=sim_aic2, tensor_parallel_size=2)
    llm2.generate(["Hello world"],
                  sampling_params=SamplingParams(max_tokens=8))
    clock2 = sim_aic2._clock
    assert clock2 is not None
    time_tp2 = clock2.total_time_s
    print(f"TP=2 predicted: {time_tp2 * 1000:.1f}ms ({clock2.num_iterations} iters)",
          flush=True)

    # Both should complete with correct iterations
    assert clock1.num_iterations == 8
    assert clock2.num_iterations == 8

    # TP affects AIC predictions — times should differ
    assert time_tp1 != time_tp2, \
        f"TP=1 and TP=2 should predict different times, both got {time_tp1*1000:.1f}ms"
    print(f"Ratio TP1/TP2: {time_tp1/time_tp2:.2f}x", flush=True)

    print("AIC TP1 vs TP2 OK", flush=True)


def main():
    test_constant_tp1()
    test_constant_tp2()
    test_aic_tp1_vs_tp2()
    print("\n=== ALL TESTS PASSED ===", flush=True)


if __name__ == "__main__":
    main()
