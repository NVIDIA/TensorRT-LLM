import argparse
import asyncio
import json
import os
import time
from pathlib import Path

import ray
import torch
from ray.util.placement_group import (
    PlacementGroupSchedulingStrategy,
    placement_group,
    remove_placement_group,
)
from transformers import AutoConfig

from tensorrt_llm import AsyncLLM
from tensorrt_llm.llmapi import CudaGraphConfig, KvCacheConfig, SamplingParams


@ray.remote
class TRTLLMInstance:
    """Ray actor wrapping an AsyncLLM instance for distributed RL workloads.

    This actor manages a single TensorRT-LLM instance that can be scheduled
    on specific GPUs using Ray placement groups. Multiple instances can run
    in parallel for high-throughput RL generation.

    Attributes:
        async_llm_kwargs: Configuration dict for AsyncLLM initialization
        sampling_kwargs: Configuration dict for SamplingParams
        llm: The underlying AsyncLLM instance (initialized via init_llm)
        sampling_params: SamplingParams object for generation
    """

    def __init__(self, async_llm_kwargs: dict, sampling_kwargs: dict):
        self.async_llm_kwargs = async_llm_kwargs
        self.sampling_kwargs = sampling_kwargs
        self.llm = None
        self.sampling_params = None

    async def init_llm(self):
        """Initialize the AsyncLLM instance with configured parameters."""
        self.llm = await AsyncLLM(
            model=self.async_llm_kwargs["model"],
            backend="pytorch",
            orchestrator_type=self.async_llm_kwargs["orchestrator_type"],
            ray_worker_extension_cls=self.async_llm_kwargs["ray_worker_extension_cls"],
            kv_cache_config=KvCacheConfig(**self.async_llm_kwargs["kv_cache_config"]),
            cuda_graph_config=CudaGraphConfig(**self.async_llm_kwargs["cuda_graph_config"]),
            max_seq_len=self.async_llm_kwargs["max_seq_len"],
            max_batch_size=self.async_llm_kwargs["max_batch_size"],
            max_num_tokens=self.async_llm_kwargs["max_num_tokens"],
            tensor_parallel_size=self.async_llm_kwargs["tensor_parallel_size"],
            trust_remote_code=self.async_llm_kwargs["trust_remote_code"],
            enable_sleep=True,
            sampler_type=self.async_llm_kwargs["sampler_type"],
            placement_groups=self.async_llm_kwargs["placement_groups"],
            placement_bundle_indices=self.async_llm_kwargs["placement_bundle_indices"],
            per_worker_gpu_share=self.async_llm_kwargs["per_worker_gpu_share"],
            batch_wait_timeout_iters=32,
            batch_wait_max_tokens_ratio=0.5,
        )
        self.sampling_params = SamplingParams(
            temperature=self.sampling_kwargs["temperature"],
            top_p=self.sampling_kwargs["top_p"],
            top_k=self.sampling_kwargs["top_k"],
            max_tokens=self.sampling_kwargs["max_tokens"],
            logprobs=self.sampling_kwargs["logprobs"],
            detokenize=self.sampling_kwargs["detokenize"],
            end_id=self.sampling_kwargs["end_id"],
            pad_id=self.sampling_kwargs["pad_id"],
            stop_token_ids=self.sampling_kwargs["stop_token_ids"],
            include_stop_str_in_output=self.sampling_kwargs["include_stop_str_in_output"],
        )

    async def generate(self, prompt: list[int]):
        """Generate output tokens for a single prompt.

        Args:
            prompt: List of input token IDs

        Returns:
            Tuple of (token_ids, log_probs):
                - token_ids: List of generated token IDs
                - log_probs: List of log probabilities (if logprobs enabled, else None)
        """
        outputs = await self.llm.generate_async(inputs=prompt, sampling_params=self.sampling_params)
        token_ids = outputs.outputs[0].token_ids
        log_probs = None
        if self.sampling_kwargs["logprobs"] is not None:
            log_probs = [list(d.values())[0].logprob for d in outputs.outputs[0].logprobs]
        return token_ids, log_probs


async def setup_rl_llm(args):
    """Main setup and execution function for RL LLM workloads.

    This function:
    1. Loads prompts from the input JSON file
    2. Initializes Ray with placement groups for GPU allocation
    3. Creates multiple TRTLLMInstance actors distributed across GPUs
    4. Distributes prompts round-robin across instances
    5. Runs async generation and reports throughput metrics

    Args:
        args: Parsed command-line arguments
    """
    # Load prompts from JSON file (expected format: list of token ID lists)
    data_path = Path(args.data_path)
    with open(data_path, "r") as f:
        prompts = json.load(f)

    hf_config = AutoConfig.from_pretrained(args.model_dir, trust_remote_code=args.trust_remote_code)

    num_instances = args.num_instances
    num_gpus = args.tp_size * num_instances
    available_gpus = torch.cuda.device_count()
    if num_gpus > 8:
        raise ValueError(
            f"Number of GPUs ({num_gpus}) is greater than 8. This script only supports single node."
        )
    if available_gpus < num_gpus:
        raise ValueError(
            f"Number of GPUs ({available_gpus}) is less than number of GPUs required ({num_gpus})."
        )

    # Prevent Ray from setting CUDA_VISIBLE_DEVICES automatically
    # This allows TensorRT-LLM to manage GPU visibility internally
    os.environ["RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES"] = "1"
    runtime_env = {"env_vars": {"RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1"}}
    pg = None

    try:
        ray.init(address="local")
        gcs_addr = ray.get_runtime_context().gcs_address
        port = int(gcs_addr.split(":")[1])
        # Force ray.init("auto") to attach to a specific cluster via RAY_ADDRESS
        os.environ["RAY_ADDRESS"] = f"localhost:{port}"

        # Create placement group with one bundle per GPU
        # STRICT_PACK ensures all bundles are on the same node
        pg = placement_group(
            [{"GPU": 1, "CPU": 2} for _ in range(num_gpus)], strategy="STRICT_PACK"
        )

        # Wait for placement group to be ready
        ray.get(pg.ready())

        # Configure placement groups for each instance
        # Each instance gets a contiguous range of GPU bundles for tensor parallelism
        # Example with num_instances=2, tp_size=2:
        #   Instance 0: bundles [0, 1] -> GPUs 0, 1
        #   Instance 1: bundles [2, 3] -> GPUs 2, 3
        tp_size = args.tp_size
        placement_group_list = [[pg] for _ in range(num_instances)]
        placement_bundle_indices_list = [
            [list(range(i * tp_size, (i + 1) * tp_size))] for i in range(num_instances)
        ]

        # Create TRTLLMInstance actors for each parallel instance
        llm_instances = []
        for i in range(num_instances):
            llm_instances.append(
                TRTLLMInstance.options(
                    num_cpus=0,
                    num_gpus=0,
                    runtime_env=runtime_env,
                    scheduling_strategy=PlacementGroupSchedulingStrategy(
                        placement_group=pg,
                        placement_group_capture_child_tasks=True,
                    ),
                ).remote(
                    async_llm_kwargs={
                        "model": args.model_dir,
                        "backend": "pytorch",
                        "orchestrator_type": "ray",
                        "ray_worker_extension_cls": "tensorrt_llm.llmapi.rlhf_utils.WorkerExtension",
                        "kv_cache_config": {
                            "enable_block_reuse": args.enable_block_reuse,
                            "free_gpu_memory_fraction": args.kv_cache_fraction,
                        },
                        "cuda_graph_config": {
                            "enable_padding": args.enable_cuda_graph_padding,
                            "batch_sizes": args.batch_sizes,
                            "max_batch_size": 0 if args.batch_sizes else args.max_batch_size,
                        },
                        "max_seq_len": args.max_seq_len,
                        "max_batch_size": args.max_batch_size,
                        "max_num_tokens": args.max_num_tokens,
                        "tensor_parallel_size": args.tp_size,
                        "trust_remote_code": args.trust_remote_code,
                        "enable_sleep": True,
                        "sampler_type": args.sampler_type,
                        "placement_groups": placement_group_list[i],
                        "placement_bundle_indices": placement_bundle_indices_list[i],
                        "per_worker_gpu_share": 0.5,
                    },
                    sampling_kwargs={
                        "temperature": args.temperature,
                        "top_p": args.top_p,
                        "top_k": args.top_k,
                        "max_tokens": args.max_tokens,
                        "logprobs": args.logprobs,
                        "detokenize": False,
                        "end_id": -1,
                        "pad_id": hf_config.pad_token_id,
                        "stop_token_ids": [hf_config.eos_token_id],
                        "include_stop_str_in_output": True,
                    },
                )
            )
        # Wait for all Ray actors to be ready, then initialize LLM instances
        ray.get([llm.__ray_ready__.remote() for llm in llm_instances])
        ray.get([llm.init_llm.remote() for llm in llm_instances])

        total_prompts = len(prompts)

        print(
            f"Starting generation for {total_prompts} prompts across {num_instances} instances..."
        )
        start_time = time.time()

        # Helper function to wrap Ray remote call as async coroutine
        async def generate_single_prompt(instance, prompt):
            """Generate a single prompt asynchronously."""
            object_ref = instance.generate.remote(prompt=prompt)
            result = await asyncio.to_thread(ray.get, object_ref)
            return result

        # Create tasks with round-robin distribution
        tasks = [
            generate_single_prompt(llm_instances[idx % num_instances], prompt)
            for idx, prompt in enumerate(prompts)
        ]

        await asyncio.gather(*tasks)
        end_time = time.time()

        print(f"Time taken: {end_time - start_time:.2f} seconds")
        print(f"Total prompts: {total_prompts}")
        print(f"Throughput: {total_prompts / (end_time - start_time):.2f} prompts/sec")
    finally:
        if pg is not None:
            remove_placement_group(pg)
        ray.shutdown()


def add_rl_llm_args(parser):
    """Add command-line arguments for RL LLM configuration."""
    # Required arguments
    parser.add_argument("--model_dir", type=str, required=True, help="Model checkpoint directory.")
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Input data file path, expected format: list of token ID lists.",
    )
    parser.add_argument(
        "--num_instances", type=int, required=True, help="Number of TRTLLM instances."
    )

    # AsyncLLM parameters
    parser.add_argument("--tp_size", type=int, required=True, help="Tensor parallel size.")
    parser.add_argument("--max_seq_len", type=int, default=2048, help="Maximum sequence length.")
    parser.add_argument("--max_batch_size", type=int, default=384, help="Maximum batch size.")
    parser.add_argument(
        "--max_num_tokens", type=int, default=32768, help="Maximum number of tokens."
    )
    parser.add_argument(
        "--sampler_type",
        type=str,
        default="TRTLLMSampler",
        choices=["TRTLLMSampler", "TorchSampler"],
        help="Sampler type.",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        default=False,
        help="Whether to trust remote code.",
    )

    # KV Cache Config parameters
    parser.add_argument(
        "--kv_cache_fraction",
        type=float,
        default=0.6,
        help="The fraction of GPU memory to be used for KV cache.",
    )
    parser.add_argument(
        "--enable_block_reuse",
        action="store_true",
        default=False,
        help="Whether to enable block reuse for KV cache.",
    )

    # Cuda Graph Config parameters
    parser.add_argument(
        "--enable_cuda_graph_padding",
        action="store_true",
        default=False,
        help="Whether to enable padding for CUDA graphs.",
    )
    parser.add_argument(
        "--batch_sizes",
        type=int,
        nargs="+",
        default=None,
        help="The batch sizes to be used for CUDA graphs. Example: --batch_sizes 16 32 64 128 256",
    )

    # Sampling parameters
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=1)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--logprobs", type=int, default=None)

    return parser


def parse_arguments():
    parser = argparse.ArgumentParser(description="RL flow performance reproduction.")
    parser = add_rl_llm_args(parser)
    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()
    asyncio.run(setup_rl_llm(args))


if __name__ == "__main__":
    main()
