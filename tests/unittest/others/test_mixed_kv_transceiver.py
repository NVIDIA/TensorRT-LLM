import os
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
        tokens_per_block=2,
        max_seq_len=256,  # Match max_tokens to allow full transfer
        max_batch_size=1,
        mapping=mapping,
        dtype=dtype)


def fill_kv_cache_buffer(kv_cache_manager):
    with torch.no_grad():
        buffer = kv_cache_manager.get_buffers(0)
        vals = torch.rand(buffer.numel(),
                          dtype=torch.float32,
                          device=buffer.device)
        vals = vals.view_as(buffer).to(torch.float8_e4m3fn)
        buffer.copy_(vals)


@pytest.fixture(scope="function")
def ctx_gen_kv_cache_dtype_and_transmission(request):
    """
    Fixture that returns (ctx_dtype, gen_dtype, transmission_dtype)
    Format: ctx_<dtype>_gen_<dtype>_trans_<dtype>
    """
    print(f"FIXTURE CALLED with param: {request.param}", flush=True)
    
    # Parse the parameter string
    parts = request.param.split("_")
    if len(parts) == 6 and parts[0] == "ctx" and parts[2] == "gen" and parts[4] == "trans":
        ctx_dtype_str = parts[1]
        gen_dtype_str = parts[3]
        trans_dtype_str = parts[5]
        
        dtype_map = {
            "fp8": DataType.FP8,
            "fp16": DataType.HALF,
            "bf16": DataType.BF16,
            "fp32": DataType.FLOAT,
        }
        
        ctx_dtype = dtype_map.get(ctx_dtype_str)
        gen_dtype = dtype_map.get(gen_dtype_str)
        trans_dtype = dtype_map.get(trans_dtype_str)
        
        if ctx_dtype is None or gen_dtype is None or trans_dtype is None:
            raise ValueError(f"Invalid dtype in config: {request.param}")
        
        result = (ctx_dtype, gen_dtype, trans_dtype)
    else:
        raise ValueError(f"Invalid config format: {request.param}. Expected: ctx_<dtype>_gen_<dtype>_trans_<dtype>")
    
    print(f"FIXTURE RETURNING: ctx={ctx_dtype}, gen={gen_dtype}, transmission={trans_dtype}", flush=True)
    return result


# TODO: Test MLA TODO: Test MLA TODO: Test MLA 

@pytest.mark.parametrize(
    "ctx_gen_kv_cache_dtype_and_transmission",
    [
        # Same storage types, various transmission formats
        "ctx_fp8_gen_fp8_trans_fp16",   # FP8 storage, FP16 transmission (default)
        "ctx_fp8_gen_fp8_trans_fp32",   # FP8 storage, FP32 transmission
        "ctx_fp8_gen_fp8_trans_bf16",   # FP8 storage, BF16 transmission
        
        # Mixed storage types with FP16 transmission
        "ctx_fp16_gen_fp8_trans_fp16",  # FP16 → FP16 → FP8
        "ctx_fp8_gen_fp16_trans_fp16",  # FP8 → FP16 → FP16
        
        # Mixed storage types with FP32 transmission (higher precision)
        "ctx_fp16_gen_fp8_trans_fp32",  # FP16 → FP32 → FP8 (better precision)
        "ctx_fp8_gen_fp16_trans_fp32",  # FP8 → FP32 → FP16 (better precision)
    ],
    ids=[
        "fp8_to_fp8_via_fp16",
        "fp8_to_fp8_via_fp32",
        "fp8_to_fp8_via_bf16",
        "fp16_to_fp8_via_fp16",
        "fp8_to_fp16_via_fp16",
        "fp16_to_fp8_via_fp32",
        "fp8_to_fp16_via_fp32",
    ],
    indirect=True)
@pytest.mark.parametrize("attention_type",
                         [AttentionTypeCpp.DEFAULT],
                         ids=["mha"])
def test_kv_cache_transceiver_single_process(ctx_gen_kv_cache_dtype_and_transmission,
                                             attention_type):
    ctx_kv_cache_dtype, gen_kv_cache_dtype, transmission_dtype = ctx_gen_kv_cache_dtype_and_transmission
    print(f"TEST FUNCTION ENTERED with ctx={ctx_kv_cache_dtype}, gen={gen_kv_cache_dtype}, "
          f"transmission={transmission_dtype}, attention={attention_type}", flush=True)
    # Init kv_cache manager and cache transceiver
    print("A", flush=True)
    
    mapping = Mapping(world_size=1, rank=0)
    kv_cache_manager_ctx = create_kv_cache_manager(mapping, ctx_kv_cache_dtype)
    kv_cache_manager_gen = create_kv_cache_manager(mapping, gen_kv_cache_dtype)
    
    print("B", flush=True)
    
    # Configure with explicit transmission data type
    cache_transceiver_config = trtllm.CacheTransceiverConfig(
        backend=trtllm.CacheTransceiverBackendType.DEFAULT,
        max_tokens_in_buffer=256,
        transmission_data_type=transmission_dtype)
    
    print(f"Created CacheTransceiverConfig with transmission_data_type={transmission_dtype}", flush=True)
    print("C", flush=True)
    
    try:
        kv_cache_transceiver_ctx = create_kv_cache_transceiver(
            mapping, kv_cache_manager_ctx, attention_type, cache_transceiver_config)

        kv_cache_transceiver_gen = create_kv_cache_transceiver(
            mapping, kv_cache_manager_gen, attention_type, cache_transceiver_config)

        print("Filling in cache buffer", flush=True)
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
        
        print(f"ctx_request: {ctx_request}", flush=True)

        kv_cache_manager_ctx.impl.add_sequence(ctx_request.py_request_id,
                                               ctx_request.prompt_len, 1,
                                               ctx_request)
        
        print(f"kv_cache_manager_ctx: {kv_cache_manager_ctx}", flush=True)
        print("Sending ctx request", flush=True)
        # send ctx request
        kv_cache_transceiver_ctx.respond_and_send_async(ctx_request)

        print("Init gen request", flush=True)

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
        
        print(f"kv_cache_manager_gen: {kv_cache_manager_gen}", flush=True)
        print("Sending gen request", flush=True)
        # send gen request
        kv_cache_transceiver_gen.request_and_receive_async(gen_request)
        
        print("Checking context transfer status", flush=True)
        kv_cache_transceiver_ctx.check_context_transfer_status(1)
        
        print("Checking gen transfer status", flush=True)
        kv_cache_transceiver_gen.check_gen_transfer_status(1)
        
        print("Done", flush=True)

        ctx_buf = kv_cache_manager_ctx.get_buffers(0).to(torch.float32)
        gen_buf = kv_cache_manager_gen.get_buffers(0).to(torch.float32)
        
        print("================================================= FINAL VALUES OF THE CACHES =================================================")
        print(f"ctx_buf (from {ctx_kv_cache_dtype}): shape={ctx_buf.shape}, dtype={ctx_buf.dtype}")
        print(f"gen_buf (from {gen_kv_cache_dtype}): shape={gen_buf.shape}, dtype={gen_buf.dtype}")
        print(f"Transmission format used: {transmission_dtype}")
        print(f"ctx_buf sample: {ctx_buf.flatten()[:10]}")
        print(f"gen_buf sample: {gen_buf.flatten()[:10]}")
        
        # Adjust tolerance based on the precision path
        # FP8 has ~1e-2 relative precision, FP16 has ~1e-3, FP32 has ~1e-7
        # For FP8 conversions, we need higher tolerance
        if ctx_kv_cache_dtype == DataType.FP8 or gen_kv_cache_dtype == DataType.FP8:
            rtol = 5e-2  # 5% relative tolerance for FP8
            atol = 1e-2  # Higher absolute tolerance
        else:
            rtol = 5e-3  # 0.5% for FP16/BF16
            atol = 1e-3
        
        print(f"Using rtol={rtol}, atol={atol} for comparison", flush=True)

        torch.testing.assert_close(
            gen_buf,
            ctx_buf,
            rtol=rtol,
            atol=atol,
            msg=f"Different KV-cache values after transfer (ctx={ctx_kv_cache_dtype}, "
                f"gen={gen_kv_cache_dtype}, transmission={transmission_dtype})",
        )
        
        print(f"✓ Test passed: KV cache successfully transferred from {ctx_kv_cache_dtype} "
              f"to {gen_kv_cache_dtype} via {transmission_dtype}", flush=True)
        
        # torch.testing.assert_close(
        #     gen_buf.flatten().argsort(),
        #     ctx_buf.flatten().argsort(),
        #     rtol=5e-3,
        #     atol=1e-3,
        #     msg="Different KV-cache values after transfer",
        # )
    
    finally:
        # Cleanup resources to prevent hanging between tests
        print("Cleaning up resources...", flush=True)
        try:
            del kv_cache_transceiver_ctx
            del kv_cache_transceiver_gen
            del kv_cache_manager_ctx
            del kv_cache_manager_gen
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            print("Cleanup complete", flush=True)
        except Exception as e:
            print(f"Cleanup error: {e}", flush=True)
