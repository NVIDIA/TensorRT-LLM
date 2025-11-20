import os

import pytest
import ray
from ray.util.placement_group import (PlacementGroupSchedulingStrategy,
                                      placement_group, remove_placement_group)
from utils.llm_data import llm_models_root

from tensorrt_llm import LLM
from tensorrt_llm._torch.utils import get_device_uuid
from tensorrt_llm.llmapi import KvCacheConfig


class DummyWorkerExtension:

    def additional_method(self):
        return "SUCCESS"


def test_worker_extension():
    llm = LLM(model=llm_models_root() /
              "llama-models-v2/TinyLlama-1.1B-Chat-v1.0",
              ray_worker_extension_cls="test_executor.DummyWorkerExtension",
              orchestrator_type="ray")
    result = llm._collective_rpc("additional_method")
    assert result[0] == "SUCCESS"


@pytest.mark.gpu4
def test_bundle_indices(monkeypatch):
    """Placement via bundle indices"""

    monkeypatch.setenv("RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES", "1")
    monkeypatch.setenv("TLLM_RAY_USE_RPC", "1")

    pg = None
    try:
        ray.init()
        pg = placement_group([{"GPU": 1, "CPU": 1}] * 4)
        ray.get(pg.ready())
        print(f"Placement group ready with bundles {pg.bundle_specs}")

        bundle_indices = [2, 3]
        runtime_env = {
            "env_vars": {
                "TRTLLM_RAY_PER_WORKER_GPUS": "0.8",
                "TRTLLM_RAY_BUNDLE_INDICES": ",".join(map(str, bundle_indices))
            }
        }

        llm = ray.remote(
            num_cpus=0,
            num_gpus=0,
            runtime_env=runtime_env,
            scheduling_strategy=PlacementGroupSchedulingStrategy(
                placement_group=pg,
                placement_group_capture_child_tasks=True,
            ),
        )(LLM).remote(
            model=os.path.join(llm_models_root(), "llama-models-v2",
                               "TinyLlama-1.1B-Chat-v1.0"),
            kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0.1),
            tensor_parallel_size=2,
            orchestrator_type="ray",
        )

        inference_actor_uuids = ray.get(
            llm._collective_rpc.remote("report_device_id"))

        expected_uuids = [get_device_uuid(idx) for idx in bundle_indices]

        assert sorted(inference_actor_uuids) == sorted(expected_uuids), \
            f"Workers not placed on expected GPUs. Expected UUIDs: {expected_uuids}, Got: {inference_actor_uuids}"

    finally:
        if pg is not None:
            remove_placement_group(pg)
        ray.shutdown()


@pytest.mark.gpu2
def test_cuda_visible_device(monkeypatch):
    """Placement via cuda_visible_device"""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")

    llm = LLM(model=llm_models_root() /
              "llama-models-v2/TinyLlama-1.1B-Chat-v1.0",
              orchestrator_type="ray")

    infer_actor_uuids = llm._collective_rpc("report_device_id")

    assert infer_actor_uuids[0] == get_device_uuid(1)
    print(f"{infer_actor_uuids=}")
