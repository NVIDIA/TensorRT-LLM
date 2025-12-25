import os

import pytest
import ray
from ray.util.placement_group import placement_group, remove_placement_group
from utils.llm_data import llm_models_root
from utils.util import get_current_process_gpu_memory

from tensorrt_llm import AsyncLLM
from tensorrt_llm._torch.utils import get_device_uuid
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
@pytest.mark.gpu2
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
        tensor_parallel_size=2,
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


@pytest.mark.ray
@pytest.mark.gpu4
@pytest.mark.asyncio
async def test_async_llm_placement_api(setup_ray_cluster, monkeypatch):
    port = setup_ray_cluster
    monkeypatch.setenv("RAY_ADDRESS", f"localhost:{port}")
    monkeypatch.setenv("TLLM_RAY_FORCE_LOCAL_CLUSTER", "0")

    n_gpus = 4
    bundle_indices = [2, 3]
    tp_size = len(bundle_indices)

    pg = None
    try:
        pg = placement_group([{"GPU": 1, "CPU": 1}] * n_gpus)
        ray.get(pg.ready())
        print(f"Placement group ready with bundles {pg.bundle_specs}")

        llm = await AsyncLLM(
            model=os.path.join(
                str(llm_models_root()), "llama-models-v2", "TinyLlama-1.1B-Chat-v1.0"
            ),
            kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0.1),
            tensor_parallel_size=tp_size,
            placement_groups=[pg],
            placement_bundle_indices=[bundle_indices],
            per_worker_gpu_share=0.8,
        )

        inference_actor_uuids = await llm.collective_rpc("report_device_id")
        expected_uuids = [get_device_uuid(idx) for idx in bundle_indices]

        print(f"{inference_actor_uuids=}, all_uuids={[get_device_uuid(i) for i in range(n_gpus)]}")

        assert sorted(inference_actor_uuids) == sorted(expected_uuids), (
            f"Workers not placed on expected GPUs. Expected: {expected_uuids}, Got: {inference_actor_uuids}"
        )

    finally:
        llm.shutdown()
        if pg is not None:
            remove_placement_group(pg)
