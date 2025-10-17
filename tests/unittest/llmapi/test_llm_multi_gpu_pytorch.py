import pytest
from utils.util import skip_ray

from tensorrt_llm import LLM
from tensorrt_llm.executor.rpc_proxy import GenerationExecutorRpcProxy
from tensorrt_llm.llmapi import KvCacheConfig
from tensorrt_llm.lora_helper import LoraConfig
from tensorrt_llm.sampling_params import SamplingParams

from .lora_test_utils import (
    check_llama_7b_multi_lora_from_request_test_harness,
    check_phi3_lora_fused_modules_output_tp2_identical_to_tp1)
from .test_llm import (_test_llm_capture_request_error, llama_model_path,
                       tinyllama_logits_processor_test_harness)
from .test_llm_pytorch import llama_7b_lora_from_dir_test_harness

global_kv_cache_config = KvCacheConfig(free_gpu_memory_fraction=0.4)


@pytest.mark.gpu2
def test_llm_capture_request_error():
    _test_llm_capture_request_error(pytorch_backend=True, tp_size=2)


@pytest.mark.gpu4
def test_tinyllama_logits_processor_tp2pp2():
    tinyllama_logits_processor_test_harness(backend="pytorch",
                                            tensor_parallel_size=2,
                                            pipeline_parallel_size=2)


@pytest.mark.gpu2
@pytest.mark.part0
@pytest.mark.parametrize("tp_size, pp_size", [(1, 2), (2, 1)])
def test_tinyllama_logits_processor_2gpu(tp_size: int, pp_size: int):
    tinyllama_logits_processor_test_harness(backend="pytorch",
                                            tensor_parallel_size=tp_size,
                                            pipeline_parallel_size=pp_size)


@pytest.mark.gpu2
def test_llama_7b_lora_tp2():
    llama_7b_lora_from_dir_test_harness(tensor_parallel_size=2,
                                        kv_cache_config=global_kv_cache_config)


@pytest.mark.gpu2
def test_llama_7b_multi_lora_tp2():
    # For LoRA checkpoints without finetuned embedding and lm_head, we can either:
    # (1) specify lora_target_modules, or
    # (2) provide a lora_dir to infer the lora_target_modules.
    lora_config = LoraConfig(lora_target_modules=['attn_q', 'attn_k', 'attn_v'],
                             max_lora_rank=8,
                             max_loras=1,
                             max_cpu_loras=8)
    check_llama_7b_multi_lora_from_request_test_harness(
        LLM,
        lora_config=lora_config,
        tensor_parallel_size=2,
        kv_cache_config=global_kv_cache_config,
        # Disable CUDA graph
        # TODO: remove this once we have a proper fix for CUDA graph in LoRA
        cuda_graph_config=None)


@pytest.mark.gpu2
def test_phi3_lora_fused_modules_output_on_tp2_identical_to_tp1() -> None:
    check_phi3_lora_fused_modules_output_tp2_identical_to_tp1(
        LLM,
        # Disable CUDA graph
        # TODO: remove this once we have a proper fix for CUDA graph in LoRA
        cuda_graph_config=None)


@skip_ray
@pytest.mark.gpu2
def test_llm_rpc_tp2():
    with LLM(model=llama_model_path,
             kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0.4),
             orchestrator_type="rpc",
             tensor_parallel_size=2) as llm:
        assert isinstance(llm._executor, GenerationExecutorRpcProxy)

        res = llm.generate("Tell me a joke",
                           sampling_params=SamplingParams(max_tokens=10,
                                                          end_id=-1))
        print(f"get result: {res}")

        assert len(res.outputs) == 1
        assert len(res.outputs[0].token_ids) == 10


@skip_ray
@pytest.mark.gpu2
@pytest.mark.asyncio
async def test_llm_rpc_streaming_tp2():
    with LLM(model=llama_model_path,
             kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0.4),
             orchestrator_type="rpc",
             tensor_parallel_size=2) as llm:
        assert isinstance(llm._executor, GenerationExecutorRpcProxy)

        async for output in llm.generate_async("Tell me a joke",
                                               sampling_params=SamplingParams(
                                                   max_tokens=10, end_id=-1)):
            print(f"get result: {output}")
