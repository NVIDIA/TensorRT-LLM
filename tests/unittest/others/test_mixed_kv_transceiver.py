import os
import pytest
import torch

os.environ['TRTLLM_ENABLE_KVCACHE_PRECISION_CONVERSION'] = '1'

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
def ctx_gen_kv_cache_dtype(request):
    print(f"FIXTURE CALLED with param: {request.param}", flush=True)
    if request.param == "ctx_fp8_gen_fp8": 
        result = DataType.FP8, DataType.FP8
    elif request.param == "ctx_bf16_gen_fp8": 
        result = DataType.BF16, DataType.FP8
    elif request.param == "ctx_fp16_gen_fp8":
        result = DataType.HALF, DataType.FP8
    else:
        raise ValueError(f"Invalid config: {request.param}")
    print(f"FIXTURE RETURNING: {result}", flush=True)
    return result


# TODO: Test MLA TODO: Test MLA TODO: Test MLA 

@pytest.mark.parametrize(
    "ctx_gen_kv_cache_dtype",
    ["ctx_fp8_gen_fp8", "ctx_fp16_gen_fp8"],  # BF16->FP8 not supported yet
    ids=["ctx_fp8_gen_fp8", "ctx_fp16_gen_fp8"],
    indirect=True)
@pytest.mark.parametrize("attention_type",
                         [AttentionTypeCpp.DEFAULT],
                         ids=["mha"])
def test_kv_cache_transceiver_single_process(ctx_gen_kv_cache_dtype,
                                             attention_type):
    print(f"TEST FUNCTION ENTERED with ctx_gen={ctx_gen_kv_cache_dtype}, attention={attention_type}", flush=True)
    # Init kv_cache manager and cache transceiver
    print("A", flush=True)
    
    mapping = Mapping(world_size=1, rank=0)
    ctx_kv_cache_dtype, gen_kv_cache_dtype = ctx_gen_kv_cache_dtype
    kv_cache_manager_ctx = create_kv_cache_manager(mapping, ctx_kv_cache_dtype)
    kv_cache_manager_gen = create_kv_cache_manager(mapping, gen_kv_cache_dtype)
    
    print("B", flush=True)
    
    cache_transceiver_config = trtllm.CacheTransceiverConfig(
        backend=trtllm.CacheTransceiverBackendType.DEFAULT,
        max_tokens_in_buffer=256)

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
        print("ctx_buf: ", ctx_buf)
        print("gen_buf: ", gen_buf)
        
        # print("ctx_buf: ", ctx_buf.flatten().argsort())
        # print("gen_buf: ", gen_buf.flatten().argsort())

        torch.testing.assert_close(
            gen_buf,
            ctx_buf,
            rtol=5e-3,
            atol=1e-3,
            msg="Different KV-cache values after transfer",
        )
        
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