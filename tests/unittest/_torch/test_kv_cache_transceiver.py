import pytest
import torch

import tensorrt_llm
import tensorrt_llm.bindings
import tensorrt_llm.bindings.executor as trtllm
from tensorrt_llm._torch.pyexecutor.kv_cache_transceiver import \
    create_kv_cache_transceiver
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.sampling_params import SamplingParams

AttentionTypeCpp = tensorrt_llm.bindings.internal.batch_manager.AttentionType
LlmRequestType = tensorrt_llm.bindings.internal.batch_manager.LlmRequestType
DataType = tensorrt_llm.bindings.DataType


def create_kv_cache_manager(mapping, dtype):
    return KVCacheManager(
        trtllm.KvCacheConfig(
            max_tokens=256,
            enable_block_reuse=False,
        ),
        tensorrt_llm.bindings.internal.batch_manager.CacheType.SELF,
        num_layers=1,
        num_kv_heads=1,
        head_dim=1,
        tokens_per_block=8,
        max_seq_len=256,
        max_batch_size=1,
        mapping=mapping,
        dtype=dtype)


def fill_kv_cache_buffer(kv_cache_manager):
    with torch.no_grad():
        buffer = kv_cache_manager.get_buffers(0)
        random_values = torch.rand(buffer.shape,
                                   dtype=torch.float32,
                                   device=buffer.device)
        buffer.copy_(random_values)


@pytest.mark.timeout(120)
@pytest.mark.parametrize("kv_cache_dtype",
                         [DataType.FP8, DataType.HALF, DataType.BF16],
                         ids=["fp8", "fp16", "bf16"])
@pytest.mark.parametrize("attention_type",
                         [AttentionTypeCpp.DEFAULT, AttentionTypeCpp.MLA],
                         ids=["default_attention", "mla_attention"])
def test_kv_cache_transceiver_single_process(kv_cache_dtype, attention_type):
    # Init kv_cache manager and cache transceiver
    mapping = Mapping(world_size=1, rank=0)
    kv_cache_manager_0 = create_kv_cache_manager(mapping, kv_cache_dtype)
    kv_cache_manager_1 = create_kv_cache_manager(mapping, kv_cache_dtype)

    cache_transceiver_config = trtllm.CacheTransceiverConfig(
        backend=trtllm.CacheTransceiverBackendType.DEFAULT,
        max_tokens_in_buffer=512)

    kv_cache_transceiver_0 = create_kv_cache_transceiver(
        mapping, kv_cache_manager_0, attention_type, cache_transceiver_config)

    kv_cache_transceiver_1 = create_kv_cache_transceiver(
        mapping, kv_cache_manager_1, attention_type, cache_transceiver_config)

    fill_kv_cache_buffer(kv_cache_manager_0)

    # init ctx request
    sampling_params = SamplingParams()
    ctx_request = LlmRequest(
        request_id=0,
        max_new_tokens=1,
        input_tokens=list(range(256)),
        sampling_config=tensorrt_llm.bindings.SamplingConfig(
            sampling_params._get_sampling_config()),
        is_streaming=False,
        llm_request_type=LlmRequestType.LLMREQUEST_TYPE_CONTEXT_ONLY)

    kv_cache_manager_0.impl.add_sequence(ctx_request.py_request_id,
                                         ctx_request.prompt_len, 1, ctx_request)
    # send ctx request
    kv_cache_transceiver_0.respond_and_send_async(ctx_request)

    # init gen request
    gen_request = LlmRequest(
        request_id=0,
        max_new_tokens=1,
        input_tokens=list(range(256)),
        sampling_config=tensorrt_llm.bindings.SamplingConfig(
            sampling_params._get_sampling_config()),
        is_streaming=False,
        llm_request_type=LlmRequestType.LLMREQUEST_TYPE_GENERATION_ONLY,
        context_phase_params=ctx_request.context_phase_params)

    kv_cache_manager_1.impl.add_sequence(gen_request.py_request_id,
                                         gen_request.prompt_len, 1, gen_request)
    # send gen request
    kv_cache_transceiver_1.request_and_receive_async(gen_request)

    if kv_cache_transceiver_0.check_context_transfer_status(1):
        raise Exception("Context transfer status is not 0")
    if kv_cache_transceiver_1.check_gen_transfer_status(1):
        raise Exception("Gen transfer status is not 0")

    assert torch.equal(
        kv_cache_manager_1.get_buffers(0),
        kv_cache_manager_0.get_buffers(0)), "different kv-cache values"
