import time

import pytest
import torch

import tensorrt_llm
import tensorrt_llm.bindings
import tensorrt_llm.bindings.executor as trtllm
from tensorrt_llm._torch.distributed import Distributed
from tensorrt_llm._torch.pyexecutor.kv_cache_transceiver import \
    create_kv_cache_transceiver
from tensorrt_llm._torch.pyexecutor.llm_request import (LlmRequest,
                                                        LlmRequestState)
from tensorrt_llm._torch.pyexecutor.mamba_cache_manager import \
    MambaHybridCacheManager
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
from tensorrt_llm.llmapi.llm_args import CacheTransceiverConfig, KvCacheConfig
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


@pytest.fixture(scope="function")
def ctx_gen_kv_cache_dtype(request):
    if request.param == "ctx_fp8_gen_fp8":
        return DataType.FP8, DataType.FP8
    elif request.param == "ctx_fp16_gen_fp16":
        return DataType.HALF, DataType.HALF
    elif request.param == "ctx_bf16_gen_bf16":
        return DataType.BF16, DataType.BF16
    else:
        raise ValueError(f"Invalid config: {request.param}")


@pytest.mark.timeout(120)
@pytest.mark.parametrize(
    "ctx_gen_kv_cache_dtype",
    ["ctx_fp8_gen_fp8", "ctx_fp16_gen_fp16", "ctx_bf16_gen_bf16"],
    ids=["ctx_fp8_gen_fp8", "ctx_fp16_gen_fp16", "ctx_bf16_gen_bf16"],
    indirect=True)
@pytest.mark.parametrize("attention_type",
                         [AttentionTypeCpp.DEFAULT, AttentionTypeCpp.MLA],
                         ids=["mha", "mla"])
@pytest.mark.parametrize("backend", ["NIXL", "UCX"], ids=["NIXL", "UCX"])
def test_kv_cache_transceiver_single_process(ctx_gen_kv_cache_dtype,
                                             attention_type, backend):
    # Init kv_cache manager and cache transceiver
    mapping = Mapping(world_size=1, rank=0)
    ctx_kv_cache_dtype, gen_kv_cache_dtype = ctx_gen_kv_cache_dtype
    kv_cache_manager_ctx = create_kv_cache_manager(mapping, ctx_kv_cache_dtype)
    kv_cache_manager_gen = create_kv_cache_manager(mapping, gen_kv_cache_dtype)

    cache_transceiver_config = CacheTransceiverConfig(backend=backend,
                                                      max_tokens_in_buffer=512)
    dist = Distributed.get(mapping)
    kv_cache_transceiver_ctx = create_kv_cache_transceiver(
        mapping, dist, kv_cache_manager_ctx, attention_type,
        cache_transceiver_config)

    kv_cache_transceiver_gen = create_kv_cache_transceiver(
        mapping, dist, kv_cache_manager_gen, attention_type,
        cache_transceiver_config)

    fill_kv_cache_buffer(kv_cache_manager_ctx)

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

    kv_cache_manager_ctx.impl.add_sequence(ctx_request.py_request_id,
                                           ctx_request.prompt_len, 1,
                                           ctx_request)
    # send ctx request
    kv_cache_transceiver_ctx.respond_and_send_async(ctx_request)

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

    kv_cache_manager_gen.impl.add_sequence(gen_request.py_request_id,
                                           gen_request.prompt_len, 1,
                                           gen_request)
    # send gen request
    kv_cache_transceiver_gen.request_and_receive_async(gen_request)

    kv_cache_transceiver_ctx.check_context_transfer_status(1)
    kv_cache_transceiver_gen.check_gen_transfer_status(1)

    assert torch.equal(
        kv_cache_manager_gen.get_buffers(0),
        kv_cache_manager_ctx.get_buffers(0)), "different kv-cache values"


@pytest.mark.timeout(120)
@pytest.mark.parametrize("attention_type",
                         [AttentionTypeCpp.DEFAULT, AttentionTypeCpp.MLA],
                         ids=["mha", "mla"])
def test_cancel_request_in_transmission(attention_type):
    # Init kv_cache manager and cache transceiver
    mapping = Mapping(world_size=1, rank=0)
    dist = Distributed.get(mapping)
    ctx_kv_cache_dtype, gen_kv_cache_dtype = DataType.HALF, DataType.HALF
    kv_cache_manager_ctx = create_kv_cache_manager(mapping, ctx_kv_cache_dtype)
    kv_cache_manager_gen = create_kv_cache_manager(mapping, gen_kv_cache_dtype)

    cache_transceiver_config = CacheTransceiverConfig(backend="DEFAULT",
                                                      max_tokens_in_buffer=512)

    kv_cache_transceiver_ctx = create_kv_cache_transceiver(
        mapping, dist, kv_cache_manager_ctx, attention_type,
        cache_transceiver_config)

    kv_cache_transceiver_gen = create_kv_cache_transceiver(
        mapping, dist, kv_cache_manager_gen, attention_type,
        cache_transceiver_config)

    fill_kv_cache_buffer(kv_cache_manager_ctx)

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

    kv_cache_manager_ctx.impl.add_sequence(ctx_request.py_request_id,
                                           ctx_request.prompt_len, 1,
                                           ctx_request)
    # send ctx request
    kv_cache_transceiver_ctx.respond_and_send_async(ctx_request)

    # wait for ctx request to be sent
    time.sleep(2)

    # cancel ctx request
    is_cancelled = kv_cache_transceiver_ctx.cancel_request(ctx_request)
    assert is_cancelled

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

    kv_cache_manager_gen.impl.add_sequence(gen_request.py_request_id,
                                           gen_request.prompt_len, 1,
                                           gen_request)
    # send gen request
    kv_cache_transceiver_gen.request_and_receive_async(gen_request)

    # Block the main thread due to the async operation
    time.sleep(2)
    assert gen_request.state == LlmRequestState.DISAGG_TRANS_ERROR


def create_hybrid_cache_manager(mapping,
                                dtype,
                                mamba_conv_dtype=torch.float16,
                                mamba_ssm_dtype=torch.float16):
    """
    Create a MambaHybridCacheManager for testing hybrid models.
    This manager handles both KV cache (attention layers) and Mamba cache (RNN layers).

    Args:
        mapping: The mapping configuration.
        dtype: KV cache dtype (DataType enum).
        mamba_conv_dtype: Mamba conv states dtype (torch dtype).
        mamba_ssm_dtype: Mamba SSM states dtype (torch dtype).
    """
    num_mamba_layers = 1
    num_attention_layers = 1

    # Attention at layer 0, Mamba at layer 1
    attention_layer_mask = [True, False]
    mamba_layer_mask = [False, True]

    return MambaHybridCacheManager(
        # Mamba cache parameters
        mamba_d_state=16,
        mamba_d_conv=4,
        mamba_num_heads=4,
        mamba_n_groups=1,
        mamba_head_dim=64,
        mamba_num_layers=num_mamba_layers,
        mamba_layer_mask=mamba_layer_mask,
        mamba_cache_dtype=mamba_conv_dtype,
        mamba_ssm_cache_dtype=mamba_ssm_dtype,
        kv_cache_config=KvCacheConfig(
            max_tokens=256,
            enable_block_reuse=False,
        ),
        kv_cache_type=tensorrt_llm.bindings.internal.batch_manager.CacheType.
        SELF,
        num_layers=num_attention_layers,
        layer_mask=attention_layer_mask,
        num_kv_heads=1,
        head_dim=1,
        tokens_per_block=8,
        max_seq_len=256,
        max_batch_size=1,
        mapping=mapping,
        dtype=dtype,
    )


def fill_hybrid_cache_buffers(hybrid_cache_manager):
    """Fill both KV and Mamba cache buffers with random values for testing."""
    with torch.no_grad():
        kv_buffer = hybrid_cache_manager.get_buffers(0)
        random_kv = torch.rand(kv_buffer.shape,
                               dtype=torch.float32,
                               device=kv_buffer.device)
        kv_buffer.copy_(random_kv)

        conv_buffer = hybrid_cache_manager.get_conv_states(1)
        random_conv = torch.rand(conv_buffer.shape,
                                 dtype=conv_buffer.dtype,
                                 device=conv_buffer.device)
        conv_buffer.copy_(random_conv)

        ssm_buffer = hybrid_cache_manager.get_ssm_states(1)
        random_ssm = torch.rand(ssm_buffer.shape,
                                dtype=ssm_buffer.dtype,
                                device=ssm_buffer.device)
        ssm_buffer.copy_(random_ssm)


@pytest.fixture(scope="function")
def hybrid_dtypes(request):
    """
    Returns (kv_dtype, mamba_conv_dtype, mamba_ssm_dtype) based on the parametrized string.

    KV dtype: fp8, bf16
    Conv dtype: fp8, bf16, fp32
    SSM dtype: bf16, fp32
    """
    kv_dtype_str, conv_dtype_str, ssm_dtype_str = request.param

    # Map KV dtype strings to DataType enum
    kv_dtype_map = {
        "fp8": DataType.FP8,
        "bf16": DataType.BF16,
    }

    # Map Mamba dtype strings to torch dtype
    torch_dtype_map = {
        "fp8": torch.float8_e4m3fn,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }

    return kv_dtype_map[kv_dtype_str], torch_dtype_map[
        conv_dtype_str], torch_dtype_map[ssm_dtype_str]


@pytest.mark.timeout(120)
@pytest.mark.parametrize("backend", ["UCX"], ids=["UCX"])
@pytest.mark.parametrize(
    "hybrid_dtypes",
    [
        # (kv_dtype, conv_dtype, ssm_dtype)
        ("bf16", "bf16", "bf16"),
        ("bf16", "bf16", "fp32"),
        ("bf16", "fp32", "bf16"),
        ("bf16", "fp32", "fp32"),
        ("fp8", "bf16", "bf16"),
        ("fp8", "bf16", "fp32"),
        ("fp8", "fp32", "bf16"),
        ("fp8", "fp32", "fp32"),
    ],
    ids=[
        "kv_bf16-conv_bf16-ssm_bf16",
        "kv_bf16-conv_bf16-ssm_fp32",
        "kv_bf16-conv_fp32-ssm_bf16",
        "kv_bf16-conv_fp32-ssm_fp32",
        "kv_fp8-conv_bf16-ssm_bf16",
        "kv_fp8-conv_bf16-ssm_fp32",
        "kv_fp8-conv_fp32-ssm_bf16",
        "kv_fp8-conv_fp32-ssm_fp32",
    ],
    indirect=["hybrid_dtypes"],
)
def test_hybrid_cache_transceiver_single_process(backend, hybrid_dtypes,
                                                 monkeypatch):
    monkeypatch.setenv("TRTLLM_USE_CPP_MAMBA", "1")
    mapping = Mapping(world_size=1, rank=0)
    kv_dtype, mamba_conv_dtype, mamba_ssm_dtype = hybrid_dtypes

    # Create hybrid cache managers (combines KV + Mamba) for context and generation
    hybrid_cache_manager_ctx = create_hybrid_cache_manager(
        mapping, kv_dtype, mamba_conv_dtype, mamba_ssm_dtype)
    hybrid_cache_manager_gen = create_hybrid_cache_manager(
        mapping, kv_dtype, mamba_conv_dtype, mamba_ssm_dtype)

    cache_transceiver_config = CacheTransceiverConfig(backend=backend,
                                                      max_tokens_in_buffer=512)
    dist = Distributed.get(mapping)

    # Create transceivers - MambaHybridCacheManager serves as both kv_cache_manager and mamba_cache_manager
    cache_transceiver_ctx = create_kv_cache_transceiver(
        mapping,
        dist,
        hybrid_cache_manager_ctx,
        AttentionTypeCpp.DEFAULT,
        cache_transceiver_config,
        mamba_cache_manager=hybrid_cache_manager_ctx)

    cache_transceiver_gen = create_kv_cache_transceiver(
        mapping,
        dist,
        hybrid_cache_manager_gen,
        AttentionTypeCpp.DEFAULT,
        cache_transceiver_config,
        mamba_cache_manager=hybrid_cache_manager_gen)

    # Fill both KV and Mamba cache buffers with random data
    fill_hybrid_cache_buffers(hybrid_cache_manager_ctx)

    # Init ctx request
    sampling_params = SamplingParams()
    ctx_request = LlmRequest(
        request_id=0,
        max_new_tokens=1,
        input_tokens=list(range(256)),
        sampling_config=tensorrt_llm.bindings.SamplingConfig(
            sampling_params._get_sampling_config()),
        is_streaming=False,
        llm_request_type=LlmRequestType.LLMREQUEST_TYPE_CONTEXT_ONLY)

    # Add sequence to hybrid manager (handles both KV and Mamba)
    hybrid_cache_manager_ctx.impl.add_sequence(ctx_request.py_request_id,
                                               ctx_request.prompt_len, 1,
                                               ctx_request)
    hybrid_cache_manager_ctx._impl.mamba_impl.allocate_cache_blocks(
        [ctx_request.py_request_id])

    # Send ctx request (sends both KV and Mamba states)
    cache_transceiver_ctx.respond_and_send_async(ctx_request)

    # Init gen request
    gen_request = LlmRequest(
        request_id=0,
        max_new_tokens=1,
        input_tokens=list(range(256)),
        sampling_config=tensorrt_llm.bindings.SamplingConfig(
            sampling_params._get_sampling_config()),
        is_streaming=False,
        llm_request_type=LlmRequestType.LLMREQUEST_TYPE_GENERATION_ONLY,
        context_phase_params=ctx_request.context_phase_params)

    # Add sequence to hybrid manager on gen side
    hybrid_cache_manager_gen.impl.add_sequence(gen_request.py_request_id,
                                               gen_request.prompt_len, 1,
                                               gen_request)
    hybrid_cache_manager_gen._impl.mamba_impl.allocate_cache_blocks(
        [gen_request.py_request_id])

    cache_transceiver_gen.request_and_receive_async(gen_request)

    cache_transceiver_ctx.check_context_transfer_status(1)
    cache_transceiver_gen.check_gen_transfer_status(1)

    assert torch.equal(
        hybrid_cache_manager_gen.get_buffers(0),
        hybrid_cache_manager_ctx.get_buffers(0)), "different kv-cache values"

    assert torch.equal(hybrid_cache_manager_gen.get_conv_states(1),
                       hybrid_cache_manager_ctx.get_conv_states(
                           1)), "different mamba conv states"

    assert torch.equal(hybrid_cache_manager_gen.get_ssm_states(1),
                       hybrid_cache_manager_ctx.get_ssm_states(
                           1)), "different mamba ssm states"


@pytest.mark.timeout(120)
@pytest.mark.parametrize("backend", ["UCX"], ids=["UCX"])
def test_hybrid_cache_transceiver_cancel_request(backend, monkeypatch):
    monkeypatch.setenv("TRTLLM_USE_CPP_MAMBA", "1")

    mapping = Mapping(world_size=1, rank=0)
    dtype = DataType.HALF

    hybrid_cache_manager_ctx = create_hybrid_cache_manager(mapping, dtype)
    hybrid_cache_manager_gen = create_hybrid_cache_manager(mapping, dtype)

    cache_transceiver_config = CacheTransceiverConfig(backend="DEFAULT",
                                                      max_tokens_in_buffer=512)
    dist = Distributed.get(mapping)

    cache_transceiver_ctx = create_kv_cache_transceiver(
        mapping,
        dist,
        hybrid_cache_manager_ctx,
        AttentionTypeCpp.DEFAULT,
        cache_transceiver_config,
        mamba_cache_manager=hybrid_cache_manager_ctx)

    cache_transceiver_gen = create_kv_cache_transceiver(
        mapping,
        dist,
        hybrid_cache_manager_gen,
        AttentionTypeCpp.DEFAULT,
        cache_transceiver_config,
        mamba_cache_manager=hybrid_cache_manager_gen)

    fill_hybrid_cache_buffers(hybrid_cache_manager_ctx)

    # Init ctx request
    sampling_params = SamplingParams()
    ctx_request = LlmRequest(
        request_id=0,
        max_new_tokens=1,
        input_tokens=list(range(256)),
        sampling_config=tensorrt_llm.bindings.SamplingConfig(
            sampling_params._get_sampling_config()),
        is_streaming=False,
        llm_request_type=LlmRequestType.LLMREQUEST_TYPE_CONTEXT_ONLY)

    hybrid_cache_manager_ctx.impl.add_sequence(ctx_request.py_request_id,
                                               ctx_request.prompt_len, 1,
                                               ctx_request)
    hybrid_cache_manager_ctx._impl.mamba_impl.allocate_cache_blocks(
        [ctx_request.py_request_id])

    # Send ctx request
    cache_transceiver_ctx.respond_and_send_async(ctx_request)

    # Wait for ctx request to be sent
    time.sleep(2)

    # Cancel ctx request
    is_cancelled = cache_transceiver_ctx.cancel_request(ctx_request)
    assert is_cancelled

    # Init gen request
    gen_request = LlmRequest(
        request_id=0,
        max_new_tokens=1,
        input_tokens=list(range(256)),
        sampling_config=tensorrt_llm.bindings.SamplingConfig(
            sampling_params._get_sampling_config()),
        is_streaming=False,
        llm_request_type=LlmRequestType.LLMREQUEST_TYPE_GENERATION_ONLY,
        context_phase_params=ctx_request.context_phase_params)

    hybrid_cache_manager_gen.impl.add_sequence(gen_request.py_request_id,
                                               gen_request.prompt_len, 1,
                                               gen_request)
    hybrid_cache_manager_gen._impl.mamba_impl.allocate_cache_blocks(
        [gen_request.py_request_id])

    # Try to receive gen request
    cache_transceiver_gen.request_and_receive_async(gen_request)

    # Block the main thread due to the async operation
    time.sleep(2)
    assert gen_request.state == LlmRequestState.DISAGG_TRANS_ERROR
