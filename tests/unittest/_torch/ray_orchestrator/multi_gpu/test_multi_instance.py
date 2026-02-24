import os

import pytest
import ray
import torch
from ray.util.placement_group import (
    PlacementGroupSchedulingStrategy,
    placement_group,
    remove_placement_group,
)
from utils.llm_data import llm_models_root

from tensorrt_llm import AsyncLLM
from tensorrt_llm.llmapi import KvCacheConfig


@ray.remote
class TRTLLMInstance:
    def __init__(self, async_llm_kwargs: dict):
        self.llm = AsyncLLM(
            model=async_llm_kwargs["model"],
            backend="pytorch",
            orchestrator_type=async_llm_kwargs["orchestrator_type"],
            kv_cache_config=KvCacheConfig(**async_llm_kwargs["kv_cache_config"]),
            tensor_parallel_size=async_llm_kwargs["tensor_parallel_size"],
            placement_groups=async_llm_kwargs["placement_groups"],
            placement_bundle_indices=async_llm_kwargs["placement_bundle_indices"],
            per_worker_gpu_share=async_llm_kwargs["per_worker_gpu_share"],
        )

    async def init_llm(self):
        await self.llm.setup_async()

    def shutdown_llm(self):
        self.llm.shutdown()
        self.llm = None


@pytest.mark.gpu4
@pytest.mark.parametrize(
    "tp_size, num_instances", [(2, 2), (1, 4)], ids=["tp2_2instances", "tp1_4instances"]
)
def test_multi_instance(setup_ray_cluster, tp_size, num_instances):
    """Test that multiple TRTLLMInstance actors can be started without port conflicts.

    This test guards against port conflict failures when launching multiple
    TensorRT-LLM instances concurrently. It runs multiple iterations to ensure
    reliable instance creation and teardown.
    """
    port = setup_ray_cluster
    num_gpus = tp_size * num_instances
    available_gpus = torch.cuda.device_count()
    if num_gpus > 8:
        raise ValueError(
            f"Number of GPUs ({num_gpus}) is greater than 8. This script only supports single node."
        )
    if available_gpus < num_gpus:
        raise ValueError(
            f"Number of GPUs ({available_gpus}) is less than number of GPUs required ({num_gpus})."
        )
    runtime_env = ray.runtime_env.RuntimeEnv()
    runtime_env["env_vars"] = os.environ.copy()
    runtime_env["env_vars"].update(
        {
            "TLLM_RAY_FORCE_LOCAL_CLUSTER": "0",
            "RAY_ADDRESS": f"localhost:{port}",
        }
    )

    # Run multiple iterations to guard against port conflict issues
    execution_times = 5
    for iteration in range(execution_times):
        pg = None
        llm_instances = []
        try:
            pg = placement_group(
                [{"GPU": 1, "CPU": 2} for _ in range(num_gpus)], strategy="STRICT_PACK"
            )

            ray.get(pg.ready())

            placement_group_list = [[pg] for _ in range(num_instances)]
            placement_bundle_indices_list = [
                [list(range(i * tp_size, (i + 1) * tp_size))] for i in range(num_instances)
            ]

            for i in range(num_instances):
                llm_instances.append(
                    TRTLLMInstance.options(
                        num_cpus=0,
                        num_gpus=0,
                        scheduling_strategy=PlacementGroupSchedulingStrategy(
                            placement_group=pg,
                            placement_group_capture_child_tasks=True,
                        ),
                        runtime_env=runtime_env,
                    ).remote(
                        async_llm_kwargs={
                            "model": os.path.join(
                                llm_models_root(), "llama-models-v2", "TinyLlama-1.1B-Chat-v1.0"
                            ),
                            "kv_cache_config": {
                                "free_gpu_memory_fraction": 0.1,
                            },
                            "tensor_parallel_size": tp_size,
                            "orchestrator_type": "ray",
                            "placement_groups": placement_group_list[i],
                            "placement_bundle_indices": placement_bundle_indices_list[i],
                            "per_worker_gpu_share": 0.5,
                        }
                    )
                )
            ray.get([llm.__ray_ready__.remote() for llm in llm_instances])
            ray.get([llm.init_llm.remote() for llm in llm_instances])
        finally:
            # Clean up actors before removing placement group
            for llm in llm_instances:
                ray.get(llm.shutdown_llm.remote())
            if pg is not None:
                remove_placement_group(pg)
