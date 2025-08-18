import os

import pytest
import ray
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from utils.llm_data import llm_models_root

from tensorrt_llm import LLM
from tensorrt_llm._torch.utils import get_device_uuid
from tensorrt_llm.llmapi import KvCacheConfig


# TODO: fix regression
@pytest.mark.gpu4
def test_bundle_indices():
    """Placement via bundle indices"""
    os.environ["RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES"] = "1"
    pg = placement_group([{"GPU": 1, "CPU": 2}] * 4)
    ray.get(pg.ready())

    bundle_indices = [2, 3]
    runtime_env = {
        "env_vars": {
            "TRTLLM_RAY_PER_WORKER_GPUS": "0.8",
            "TRTLLM_RAY_BUNDLE_INDICES": ",".join(map(str, bundle_indices))
        }
    }

    llm = ray.remote(
        num_cpus=0,  # leave it to RayExecutor to decide within assigned bundles
        num_gpus=0,
        runtime_env=runtime_env,
        scheduling_strategy=PlacementGroupSchedulingStrategy(
            placement_group=pg,
            placement_group_capture_child_tasks=
            True,  # to keep all sub tasks inside the same reserved bundle(s).
        ),
    )(LLM).remote(
        model=os.path.join(llm_models_root(), "llama-models-v2",
                           "TinyLlama-1.1B-Chat-v1.0"),
        kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0.1),
        tensor_parallel_size=2,
        executor_type="ray",
    )

    infer_worker_uuids = ray.get(llm.collective_rpc.remote("report_device_id"))
    print(f"{infer_worker_uuids=}")


@pytest.mark.gpu2
def test_cuda_visible_device():
    """Placement via cuda_visible_device"""
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    llm = LLM(model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", executor_type="ray")

    infer_actor_uuids = llm.collective_rpc("report_device_id")

    del os.environ["CUDA_VISIBLE_DEVICES"]
    assert infer_actor_uuids[0] == get_device_uuid(1)
    # TODO: to remove. now has thread leak
    print(f"test_cuda_visible_device passed, {infer_actor_uuids=}")
