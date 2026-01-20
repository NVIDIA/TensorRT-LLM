import gc

import pytest
import torch
from utils.util import get_current_process_gpu_memory

import tensorrt_llm
from tensorrt_llm._torch import virtual_memory
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
from tensorrt_llm.bindings.executor import KvCacheConfig
from tensorrt_llm.bindings.internal.batch_manager import CacheType
from tensorrt_llm.mapping import Mapping


@pytest.fixture(scope="function", autouse=True)
def cuda_sync_fixture():
    """
    Synchronizes CUDA to catch device errors.
    """

    torch.cuda.synchronize()
    yield
    torch.cuda.synchronize()


@pytest.fixture(scope="function", autouse=True)
def clean_cache():
    gc.collect()
    torch.cuda.empty_cache()
    yield
    gc.collect()
    torch.cuda.empty_cache()


def test_basic(process_gpu_memory_info_available):
    memory_usage_begin = get_current_process_gpu_memory()

    alloc_size = 256 * 1024 * 1024
    tag = "test_tag"

    with virtual_memory.scope(tag) as pool:
        tensor = torch.full([alloc_size], 42, dtype=torch.int8, device='cuda')
        memory_usage_materialized = get_current_process_gpu_memory()
        if process_gpu_memory_info_available:
            assert memory_usage_begin + alloc_size == memory_usage_materialized

    assert tensor[0].item() == 42

    torch.cuda.synchronize()
    virtual_memory.release_with_tag(tag)

    memory_usage_released = get_current_process_gpu_memory()
    if process_gpu_memory_info_available:
        assert memory_usage_begin == memory_usage_released

    torch.cuda.synchronize()
    virtual_memory.materialize_with_tag(tag)

    memory_usage_rematerialized = get_current_process_gpu_memory()
    if process_gpu_memory_info_available:
        assert memory_usage_begin + alloc_size == memory_usage_rematerialized

    torch.fill_(tensor, 24)
    assert tensor[0].item() == 24

    del tensor
    del pool

    memory_usage_end = get_current_process_gpu_memory()
    if process_gpu_memory_info_available:
        assert memory_usage_begin == memory_usage_end


def test_restore():
    alloc_size = 1024 * 1024
    tag = "test_tag"

    with virtual_memory.scope(tag, virtual_memory.RestoreMode.PINNED) as pool:
        tensor = torch.full([alloc_size], 42, dtype=torch.int8, device='cuda')

    assert tensor[0].item() == 42

    torch.cuda.synchronize()
    virtual_memory.release_with_tag(tag)

    torch.cuda.synchronize()

    virtual_memory.materialize_with_tag(tag)
    torch.cuda.synchronize()

    assert tensor[0].item() == 42

    del tensor
    del pool


def test_kv_cache_manager(process_gpu_memory_info_available):
    kv_cache_params = {
        "kv_cache_config": KvCacheConfig(max_tokens=1024),
        "kv_cache_type": CacheType.SELF,
        "num_layers": 8,
        "num_kv_heads": 256,
        "head_dim": 64,
        "tokens_per_block": 64,
        "max_seq_len": 1024,
        "max_batch_size": 1,
        "mapping": Mapping(world_size=1, tp_size=1, rank=0),
        "dtype": tensorrt_llm.bindings.DataType.FP8,
    }

    mgr = KVCacheManager(**kv_cache_params)
    mgr.shutdown()
    del mgr

    memory_usage_begin = get_current_process_gpu_memory()

    tag = "test_tag"
    cache_size = torch.empty(
        [
            2,  # KV
            8,  # Layers
            256,  # Heads
            1024,  # Tokens
            64,  # Head dim
        ],
        dtype=torch.float8_e4m3fn,
        device='meta')

    alloc_size = cache_size.nelement()

    with virtual_memory.scope(tag) as pool:
        mgr = KVCacheManager(**kv_cache_params)
        memory_usage_materialized = get_current_process_gpu_memory()
        if process_gpu_memory_info_available:
            assert memory_usage_begin + alloc_size == memory_usage_materialized

    torch.cuda.synchronize()
    virtual_memory.release_with_tag(tag)

    memory_usage_released = get_current_process_gpu_memory()
    if process_gpu_memory_info_available:
        assert memory_usage_begin == memory_usage_released

    torch.cuda.synchronize()
    virtual_memory.materialize_with_tag(tag)

    memory_usage_rematerialized = get_current_process_gpu_memory()
    if process_gpu_memory_info_available:
        assert memory_usage_begin + alloc_size == memory_usage_rematerialized

    mgr.shutdown()
    del mgr
    del pool

    memory_usage_end = get_current_process_gpu_memory()
    if process_gpu_memory_info_available:
        assert memory_usage_begin == memory_usage_end


@pytest.mark.skip("https://nvbugspro.nvidia.com/bug/5458911")
def test_cuda_graph(process_gpu_memory_info_available):

    def work(input: torch.Tensor) -> torch.Tensor:
        intermediate = input + input
        output = input + intermediate
        return output

    g = torch.cuda.CUDAGraph()
    tag = "cuda_graph"

    with virtual_memory.scope(tag) as pool:
        static_input = torch.ones(1024, dtype=torch.float32, device='cuda')
        static_output = torch.zeros(1024, dtype=torch.float32, device='cuda')

        with torch.cuda.graph(g):
            static_output.copy_(work(static_input))

    torch.fill_(static_input, 1.0)
    g.replay()

    torch.cuda.synchronize()
    assert static_output[0].item() == 3.0

    memory_usage_before = get_current_process_gpu_memory()

    torch.cuda.synchronize()
    virtual_memory.release_with_tag(tag)

    memory_usage_released = get_current_process_gpu_memory()
    if process_gpu_memory_info_available:
        assert memory_usage_released < memory_usage_before

    torch.cuda.synchronize()
    virtual_memory.materialize_with_tag(tag)

    torch.fill_(static_input, 1.0)
    torch.fill_(static_output, 0.0)
    g.replay()

    torch.cuda.synchronize()
    assert static_output[0].item() == 3.0

    del static_input, static_output, g, pool
