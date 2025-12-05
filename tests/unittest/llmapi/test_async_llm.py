import pytest
from utils.llm_data import llm_models_root
from utils.util import get_current_process_gpu_memory

from tensorrt_llm import AsyncLLM
from tensorrt_llm._torch.virtual_memory import ExecutorMemoryType
from tensorrt_llm.llmapi import KvCacheConfig, SamplingParams


@pytest.mark.ray
@pytest.mark.asyncio
async def test_async_llm_awaitable():
    llama_model_path = str(llm_models_root() / "llama-models-v2/TinyLlama-1.1B-Chat-v1.0")
    kv_cache_config = KvCacheConfig(enable_block_reuse=False)

    prompt = "The future of AI is"
    sampling_params = SamplingParams(temperature=0, max_tokens=12)

    llm = await AsyncLLM(
        model=llama_model_path,
        enable_sleep=True,
        cuda_graph_config=None,
        kv_cache_config=kv_cache_config,
    )

    output = await llm.generate_async(prompt, sampling_params)
    assert output.outputs[0].text
    print("Output text:", output.outputs[0].text)

    del llm


@pytest.mark.ray
@pytest.mark.asyncio
@pytest.mark.parametrize("num_cycles", [3], ids=lambda x: f"{x}_cycle")
async def test_async_llm_release_resume(process_gpu_memory_info_available, num_cycles):
    llama_model_path = str(llm_models_root() / "llama-models-v2/TinyLlama-1.1B-Chat-v1.0")
    kv_cache_config = KvCacheConfig(enable_block_reuse=False, max_tokens=4096)

    prompt = "The future of AI is"
    sampling_params = SamplingParams(temperature=0, max_tokens=12)
    tags = [tag.value for tag in ExecutorMemoryType]

    async with AsyncLLM(
        model=llama_model_path,
        enable_sleep=True,
        cuda_graph_config=None,
        kv_cache_config=kv_cache_config,
    ) as llm:
        # Generate baseline
        output_before = await llm.generate_async(prompt, sampling_params)
        baseline_text = output_before.outputs[0].text

        for cycle in range(num_cycles):
            memory_usage_active = get_current_process_gpu_memory(True) / 1024**3
            print(f"[Cycle {cycle + 1}] Memory usage before release: {memory_usage_active:.2f} GB")

            await llm.release(tags)
            memory_usage_released = get_current_process_gpu_memory(True) / 1024**3

            if process_gpu_memory_info_available:
                print(
                    f"[Cycle {cycle + 1}] Memory usage after release: {memory_usage_released:.2f} GB"
                )
                assert memory_usage_released < memory_usage_active, (
                    f"Released memory ({memory_usage_released:.2f} GB) should be < "
                    f"active memory ({memory_usage_active:.2f} GB)"
                )

            await llm.resume(tags)
            memory_usage_resumed = get_current_process_gpu_memory(True) / 1024**3
            print(f"[Cycle {cycle + 1}] Memory usage after resume: {memory_usage_resumed:.2f} GB")
            if process_gpu_memory_info_available:
                assert memory_usage_resumed > memory_usage_released, (
                    f"Resumed memory ({memory_usage_resumed:.2f} GB) should be > "
                    f"released memory ({memory_usage_released:.2f} GB)"
                )

        output_after = await llm.generate_async(prompt, sampling_params)
        text_after = output_after.outputs[0].text

        print(f"[Cycle {num_cycles}] Generated text after release/resume: {text_after}")
        assert baseline_text == text_after, (
            f"Generated text mismatch after {num_cycles} cycle(s): "
            f"'{baseline_text}' != '{text_after}'"
        )
