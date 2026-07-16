# Confirm that the default backend is changed
import os

from defs.common import venv_check_output

from ..conftest import llm_models_root

model_path = os.path.join(
    llm_models_root(),
    "llama-models-v3",
    "llama-v3-8b-instruct-hf",
)


class TestLlmDefaultBackend:
    """
    Check that the default backend is PyTorch for v1.0 breaking change
    """

    def test_llm_args_type_default(self, llm_root, llm_venv):
        # Keep the complete example code here
        from tensorrt_llm.llmapi import LLM, KvCacheConfig, TorchLlmArgs

        kv_cache_config = KvCacheConfig(free_gpu_memory_fraction=0.4)
        llm = LLM(model=model_path, kv_cache_config=kv_cache_config)

        # The default backend should be PyTorch
        assert llm.args.backend == "pytorch"
        assert isinstance(llm.args, TorchLlmArgs)

        for output in llm.generate(["Hello, world!"]):
            print(output)

    def test_llm_args_logging(self, llm_root, llm_venv):
        # It should print the backend in the log
        script_path = os.path.join(os.path.dirname(__file__),
                                   "_run_llmapi_llm.py")
        print(f"script_path: {script_path}")

        # Test with pytorch backend
        pytorch_cmd = [
            script_path, "--model_dir", model_path, "--backend", "pytorch"
        ]

        pytorch_output = venv_check_output(llm_venv, pytorch_cmd)

        # Check that pytorch backend keyword appears in logs
        assert "Using LLM with PyTorch backend" in pytorch_output, f"Expected 'pytorch' in logs, got: {pytorch_output}"
