"""Smoke test for simulation mode with SimConfig."""
import os
import time

os.environ["TRTLLM_LOG_LEVEL"] = "WARNING"


def main():
    from tensorrt_llm.llmapi import LLM, SamplingParams
    from tensorrt_llm.llmapi.sim_config import SimConfig, PredictorConfig

    MODEL_PATH = "/code/slop/models/TinyLlama-1.1B-Chat-v1.0"

    print("Creating LLM with SimConfig...", flush=True)
    llm = LLM(MODEL_PATH,
              sim_config=SimConfig(
                  predictor=PredictorConfig(
                      constant_prefill_time_ms=10.0,
                      constant_decode_time_ms=5.0)))
    print("LLM created. Generating 8 tokens...", flush=True)

    start = time.monotonic()
    output = llm.generate(["Hello world"],
                          sampling_params=SamplingParams(max_tokens=8))
    elapsed_ms = (time.monotonic() - start) * 1000

    token_ids = output[0].outputs[0].token_ids
    print(f"Output tokens: {token_ids}", flush=True)
    print(f"Wall-clock: {elapsed_ms:.0f}ms", flush=True)
    print(f"Expected: ~50ms (10ms prefill + 8*5ms decode)", flush=True)

    assert len(token_ids) == 8, f"Expected 8 tokens, got {len(token_ids)}"
    assert output[0].outputs[0].finish_reason == "length"
    # Should take at least 40ms (some scheduling overhead may vary)
    assert elapsed_ms > 30, f"Too fast ({elapsed_ms:.0f}ms) — predictor not working?"
    print("SUCCESS", flush=True)


if __name__ == "__main__":
    main()
