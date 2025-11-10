from utils.llm_data import llm_models_root
from utils.util import get_current_process_gpu_memory

from tensorrt_llm import LLM
from tensorrt_llm._torch.virtual_memory import ExecutorMemoryType
from tensorrt_llm.llmapi import KvCacheConfig, SamplingParams


def test_llm_sleep(process_gpu_memory_info_available):
    llama_model_path = str(llm_models_root() / "llama-models-v2/TinyLlama-1.1B-Chat-v1.0")
    kv_cache_config = KvCacheConfig(enable_block_reuse=False, max_tokens=4096)

    llm = LLM(
        model=llama_model_path,
        enable_sleep=True,
        cuda_graph_config=None,  # CUDA Graph unsupported yet
        kv_cache_config=kv_cache_config,
    )

    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]

    sampling_params = SamplingParams(temperature=0)

    with llm:
        outputs = llm.generate(prompts, sampling_params)
        generated_before_sleep = [output.outputs[0].text for output in outputs]

        memory_usage_active = get_current_process_gpu_memory(True)

        llm._collective_rpc("sleep", ([ExecutorMemoryType.MODEL_ENGINE_MAIN],))

        memory_usage_sleep = get_current_process_gpu_memory(True)
        if process_gpu_memory_info_available:
            assert memory_usage_sleep < memory_usage_active

        llm._collective_rpc("wakeup", ([ExecutorMemoryType.MODEL_ENGINE_MAIN],))

        memory_usage_wakeup = get_current_process_gpu_memory(True)
        if process_gpu_memory_info_available:
            assert memory_usage_wakeup > memory_usage_sleep

        outputs = llm.generate(prompts, sampling_params)
        generated_after_sleep = [output.outputs[0].text for output in outputs]

    for before, after in zip(generated_before_sleep, generated_after_sleep, strict=True):
        assert before == after, "Generated result mismatch before and after sleep"
