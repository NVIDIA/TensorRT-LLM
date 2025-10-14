import pytest
import torch

from tensorrt_llm._torch.pyexecutor.py_executor_creator import \
    create_py_executor
from tensorrt_llm.llmapi import (BuildConfig, CapacitySchedulerPolicy,
                                 DynamicBatchConfig, SchedulerConfig)
from tensorrt_llm.llmapi.llm_args import (CudaGraphConfig, KvCacheConfig,
                                          TorchLlmArgs)

# isort: off
from .test_llm import get_model_path
# isort: on

pytestmark = pytest.mark.threadleak(enabled=False)


def test_profile_kvcache():
    kv_cache_config = KvCacheConfig(enable_block_reuse=False,
                                    free_gpu_memory_fraction=0.9)
    cuda_graph_config = CudaGraphConfig(max_batch_size=512)
    VLM_MODEL = "Qwen2.5-VL-7B-Instruct"
    VLM_MODEL_PATH = get_model_path(VLM_MODEL)

    build_config = BuildConfig(max_beam_width=1, max_num_tokens=16384)
    dynamic_batch_config = DynamicBatchConfig(
        enable_batch_size_tuning=True,
        enable_max_num_tokens_tuning=False,
        dynamic_batch_moving_average_window=128)
    scheduler_config = SchedulerConfig(
        capacity_scheduler_policy=CapacitySchedulerPolicy.GUARANTEED_NO_EVICT,
        dynamic_batch_config=dynamic_batch_config,
    )
    backend = "pytorch"
    llm_args = {
        "model": VLM_MODEL,
        "scheduler_config": scheduler_config,
        "tokenizer": None,
        "tensor_parallel_size": 1,
        "pipeline_parallel_size": 1,
        "moe_expert_parallel_size": None,
        "gpus_per_node": 1,
        "trust_remote_code": False,
        "build_config": build_config,
        "max_batch_size": build_config.max_batch_size,
        "max_num_tokens": build_config.max_num_tokens,
        "max_beam_width": build_config.max_beam_width,
        "max_seq_len": build_config.max_seq_len,
        "kv_cache_config": kv_cache_config,
        "backend": backend,
        "num_postprocess_workers": 0,
        "postprocess_tokenizer_dir": VLM_MODEL,
        "reasoning_parser": None,
        "fail_fast_on_attention_window_too_large": False,
        "cuda_graph_config": cuda_graph_config,
    }

    torchllm_args = TorchLlmArgs(**llm_args)

    profiling_data = {"enable_mm_reqs": True}
    py_executor = create_py_executor(llm_args=torchllm_args,
                                     checkpoint_dir=VLM_MODEL_PATH,
                                     profiling_stage_data=profiling_data)
    vlm_activation_bytes_with_mm_reqs = profiling_data["activation_bytes"]
    py_executor.shutdown()
    torch.cuda.empty_cache()

    profiling_data = {"enable_mm_reqs": False}
    torchllm_args = TorchLlmArgs(**llm_args)
    py_executor_2 = create_py_executor(llm_args=torchllm_args,
                                       checkpoint_dir=VLM_MODEL_PATH,
                                       profiling_stage_data=profiling_data)
    vlm_activation_bytes_no_mm_reqs = profiling_data["activation_bytes"]
    py_executor_2.shutdown()
    torch.cuda.empty_cache()

    assert vlm_activation_bytes_with_mm_reqs > vlm_activation_bytes_no_mm_reqs, f"Activation bytes should be higher with mm reqs, but got {vlm_activation_bytes_with_mm_reqs} for mm reqs and {vlm_activation_bytes_no_mm_reqs} without mm reqs"
