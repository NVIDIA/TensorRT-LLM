"""Smoke test for simulation mode POC."""
import os
os.environ["TRTLLM_LOG_LEVEL"] = "WARNING"


def main():
    from tensorrt_llm.llmapi import LLM, SamplingParams

    MODEL_PATH = "/code/slop/models/TinyLlama-1.1B-Chat-v1.0"

    print("Creating LLM in simulation mode...", flush=True)
    llm = LLM(MODEL_PATH, simulation_mode=True)
    print("LLM created. Generating...", flush=True)
    output = llm.generate(["Hello world"],
                          sampling_params=SamplingParams(max_tokens=8))
    print(f"Output: {output}", flush=True)
    print("SUCCESS", flush=True)


if __name__ == "__main__":
    main()
