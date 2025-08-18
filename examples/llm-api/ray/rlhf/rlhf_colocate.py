"""
A simple demo to show how to co-locate TRT-LLM inference actors with training actors
on the same GPUs for RLHF applications. This example requires 4 GPUs 80GB.

This demo has 1 GPU per bundle and does the following placement:
┌─────┬──────────────┬───────────────┬──────────┐
│ GPU │ TrainActor   │ InferActor    │ TP Rank  │
├─────┼──────────────┼───────────────┼──────────┤
│  0  │ train-0      │ infer-0       │    0     │
│  1  │ train-1      │ infer-0       │    1     │
│  2  │ train-2      │ infer-1       │    0     │
│  3  │ train-3      │ infer-1       │    1     │
└─────┴──────────────┴───────────────┴──────────┘

Key points:
1. Control the placement of the TRT-LLM inference actors by pre-defining placement group
  with global view of GPUs and setting TRTLLM_RAY_PER_WORKER_GPUS and TRTLLM_RAY_BUNDLE_INDICES properly.
2. [WIP] Use CUDA IPC to do dummy weight updates.

Colocate is typically useful when the model is small enough that GPU memory can be shared.
It avoids weight-transfer overhead but constrains you to specific TP choices.
e.g., larger TP to fit memory, which can slow throughput.

"""
import os

import ray
import torch
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from tensorrt_llm import LLM
from tensorrt_llm.llmapi import KvCacheConfig

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


class RayTrainingActor:

    def __init__(self):
        from transformers import AutoModelForCausalLM
        self.gpu = int(ray.get_gpu_ids()[0])
        torch.cuda.set_device(self.gpu)

        self.model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(
            torch.cuda.current_device())
        for name, p in self.model.named_parameters():
            p.data.zero_()
        torch.cuda.synchronize()

    def report_device_id(self) -> int:
        return self.gpu

    def get_weight_ipc_handles(self):
        from torch.multiprocessing.reductions import reduce_tensor

        data = {}
        for name, p in self.model.named_parameters():
            # the training actor might only have a subset of the weights
            # and need to all-gather the weights from all the actors.
            # for demonstration, here we assume all training actors have
            # the full weights.
            data[name] = reduce_tensor(p.detach())
        return {self.device_uuid: data}


def main():
    assert torch.cuda.device_count(
    ) >= 4, "This co-locate example requires 4 GPUs."

    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    os.environ["RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES"] = "1"
    os.environ["RAY_DEDUP_LOGS"] = "0"
    ray.init()

    pg = placement_group([{"GPU": 1, "CPU": 1}] * 4)
    ray.get(pg.ready())
    print(f"placement group has bundles {pg.bundle_specs=}")

    training_actors = []
    training_actor_device_ids = []
    inference_actors = []
    inference_actors_device_ids = []

    # ==== 1. Start one training actor per GPU ====
    for bundle_index in [0, 1, 2, 3]:
        training_actor = ray.remote(
            num_cpus=0,
            num_gpus=0.2,
            scheduling_strategy=PlacementGroupSchedulingStrategy(
                placement_group=pg,
                placement_group_capture_child_tasks=True,
                placement_group_bundle_index=bundle_index,
            ),
        )(RayTrainingActor).remote()
        training_actors.append(training_actor)

    # check training actors placement
    for bundle_index, training_actor in enumerate(training_actors):
        device_id = ray.get(training_actor.report_device_id.remote())
        print(f"training actor {bundle_index} is on {device_id}")
        training_actor_device_ids.append(device_id)

    # ==== 2. Start two inference actors (TP=2), each uses two GPUs ====
    # , [2, 3]
    for _, bundle_indices in enumerate([[0, 1], [2, 3]]):
        runtime_env = {
            "env_vars": {
                "TRTLLM_RAY_PER_WORKER_GPUS": "0.6",
                "TRTLLM_RAY_BUNDLE_INDICES": ",".join(map(str, bundle_indices))
            }
        }

        # Lower the GPU memory fraction that will be allocated for the KV cache to avoid OOM
        kv_cache_config = KvCacheConfig(free_gpu_memory_fraction=0.1)

        # In this case, placement is controlled at driver script level,
        # and then TRTLLM's RayExeuctor controls RayWorker assignments within given GPUs.
        llm = ray.remote(
            num_cpus=0,
            num_gpus=0,
            runtime_env=runtime_env,
            scheduling_strategy=PlacementGroupSchedulingStrategy(
                placement_group=pg,
                placement_group_capture_child_tasks=
                True,  # to keep all sub tasks inside the same reserved bundle(s).
            ),
        )(LLM).remote(
            model=MODEL_NAME,
            worker_extension_cls="rlhf_utils.ColocateWorkerExtension",
            kv_cache_config=kv_cache_config,
            tensor_parallel_size=2,
            executor_type="ray",
        )

        print(
            f"Created an LLM inference actor with bundle_indices={bundle_indices}"
        )
        inference_actors.append(llm)

    # check inference actors placement
    for i, llm in enumerate(inference_actors):
        device_ids = ray.get(llm.collective_rpc.remote("report_device_id"))
        ray.get(llm.collective_rpc.remote("additional_method"))
        inference_actors_device_ids.append(device_ids)
        print(f"Inference actor {i}: {device_ids=}")

        # Sanity check: run a trivial generation call
        # TODO: need more tests on LLM() being wrapped as a Ray actor.
        # result = ray.get(llm.generate.remote("The future of AI is"))
        # print("Sample output:", result[0].outputs[0].text)

    # ==== 3. Verify the placement ====
    # the first two training actors should be
    # on the same GPUs as the first inference actor
    assert training_actor_device_ids[:2] == inference_actors_device_ids[0]
    # the last two training actors should be
    # on the same GPUs as the second inference actor
    assert training_actor_device_ids[2:] == inference_actors_device_ids[1]

    # ==== 4. Demonstrate weight updates via IPC handles ====
    # TODO: commented out as weight update function is WIP
    # print("gather all the IPC handles from the training actors")
    # ipc_handles = {}
    # for actor in training_actors:
    #     ipc_handles.update(ray.get(actor.get_weight_ipc_handles.remote()))

    # print("update the weights of the inference actors")
    # for llm in inference_actors:
    #     ray.get(
    #         llm.collective_rpc.remote(
    #             "update_weights_from_ipc_handles", args=(ipc_handles,)
    #         )
    #     )
    # print("check if the weights are updated")
    # for llm in inference_actors:
    #     assert ray.get(llm.collective_rpc.remote("check_weights_changed", args=tuple()))


if __name__ == '__main__':
    main()
