"""Direct test of sim mode — bypass LLM API, test the executor loop directly."""
import os
os.environ["TRTLLM_LOG_LEVEL"] = "INFO"
os.environ["TRTLLM_ALLOW_MISSING_OPS"] = "1"
os.environ["TRTLLM_SKIP_KV_CACHE_ESTIMATION"] = "1"

import sys
import traceback


def main():
    from tensorrt_llm.llmapi.llm_args import TorchLlmArgs
    from tensorrt_llm._torch.pyexecutor.py_executor_creator import (
        _create_sim_py_executor, _construct_checkpoint_loader, ModelLoader)

    MODEL_PATH = "/code/slop/models/TinyLlama-1.1B-Chat-v1.0"

    print("[test] Creating TorchLlmArgs...", flush=True)
    llm_args = TorchLlmArgs(model=MODEL_PATH, simulation_mode=True)

    print("[test] Constructing checkpoint loader...", flush=True)
    checkpoint_loader = _construct_checkpoint_loader(
        llm_args.backend, llm_args.checkpoint_loader, llm_args.checkpoint_format)

    print("[test] Loading config and applying defaults...", flush=True)
    llm_args = ModelLoader.load_config_and_apply_defaults(
        MODEL_PATH, llm_args, checkpoint_loader)

    print("[test] Creating sim PyExecutor...", flush=True)
    try:
        py_executor = _create_sim_py_executor(llm_args, MODEL_PATH,
                                               checkpoint_loader)
        print(f"[test] PyExecutor created: {py_executor}", flush=True)
        print(f"[test] Scheduler: {type(py_executor.scheduler).__name__}", flush=True)
        print(f"[test] Model engine: {type(py_executor.model_engine).__name__}", flush=True)
        print(f"[test] Sampler: {type(py_executor.sampler).__name__}", flush=True)
        print("[test] SUCCESS — PyExecutor constructed!", flush=True)
    except Exception as e:
        print(f"[test] FAILED: {e}", flush=True)
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
