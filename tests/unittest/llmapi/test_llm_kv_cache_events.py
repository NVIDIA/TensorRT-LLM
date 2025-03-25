import asyncio
import time

import pytest
from test_llm import get_model_path

import tensorrt_llm
from tensorrt_llm._torch.pyexecutor.config import PyTorchConfig
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
from tensorrt_llm._utils import KVCacheEventSerializer
from tensorrt_llm.llmapi import LLM, KvCacheConfig
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.sampling_params import SamplingParams

default_model_name = "llama-models-v2/TinyLlama-1.1B-Chat-v1.0"
llama_model_path = get_model_path(default_model_name)

global_kvcache_config = KvCacheConfig(free_gpu_memory_fraction=0.4,
                                      event_buffer_max_size=1024,
                                      enable_block_reuse=True,
                                      onboard_blocks=True,
                                      max_tokens=256)


def create_kv_cache_manager():
    num_layers = 2
    num_kv_heads = 2
    head_dim = 128
    tokens_per_block = 64
    max_seq_len = 1024
    max_batch_size = 1
    mapping = Mapping()
    return KVCacheManager(
        kv_cache_config=global_kvcache_config,
        kv_cache_type=tensorrt_llm.bindings.internal.batch_manager.CacheType.
        SELF,
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        tokens_per_block=tokens_per_block,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
        mapping=mapping,
    )


def create_llm(tensor_parallel_size=1):
    return LLM(model=llama_model_path,
               tensor_parallel_size=tensor_parallel_size,
               kv_cache_config=global_kvcache_config,
               pytorch_backend_config=PyTorchConfig(autotuner_enabled=False),
               backend="pytorch")


def create_llm_request(id, input_tokens, new_tokens=1):
    sampling_params = SamplingParams()
    req = LlmRequest(request_id=id,
                     max_new_tokens=new_tokens,
                     input_tokens=input_tokens,
                     sampling_config=tensorrt_llm.bindings.SamplingConfig(
                         sampling_params._get_sampling_config()),
                     is_streaming=False)
    return req


def flush_events(kv_cache_manager):
    kv_cache_manager.flush_iteration_events()
    time.sleep(0.001)


def test_kv_cache_event_data_serialization():
    kv_cache_manager = create_kv_cache_manager()
    flush_events(kv_cache_manager)
    events = kv_cache_manager.get_latest_events(10)
    serialized_event = KVCacheEventSerializer.serialize(events)
    assert len(serialized_event) == 1 and serialized_event[0]["event_id"] == 0
    assert serialized_event[0]["data"]["type"] == "created"
    assert len(serialized_event[0]["data"]["num_blocks_per_cache_level"]) == 2

    req = create_llm_request(0, [1, 2, 3, 4, 5])
    kv_cache_manager.impl.add_sequence(req.py_request_id, req.prompt_len, 1,
                                       req)
    kv_cache_manager.free_resources(req)

    flush_events(kv_cache_manager)
    events = kv_cache_manager.get_latest_events(10)
    serialized_event = KVCacheEventSerializer.serialize(events)

    assert serialized_event[0]["data"]["type"] == "stored"
    assert serialized_event[0]["data"]["parent_hash"] is None
    assert len(serialized_event[0]["data"]["blocks"]) == 1
    assert len(serialized_event[0]["data"]["blocks"][0]["tokens"]) == 4

    req2 = create_llm_request(1, [1, 2, 3, 4, 5])
    kv_cache_manager.impl.add_sequence(req2.py_request_id, req2.prompt_len, 1,
                                       req2)
    kv_cache_manager.free_resources(req2)

    flush_events(kv_cache_manager)
    events = kv_cache_manager.get_latest_events(10)
    serialized_event = KVCacheEventSerializer.serialize(events)


def test_expected_kv_cache_events():
    llm = create_llm()
    sampling_params = SamplingParams(max_tokens=6, temperature=0.01)
    prompt = "Hello, my name is"

    _ = llm.generate(prompt, sampling_params=sampling_params)

    events = llm.get_kv_cache_events(5)
    # created + stored events
    assert events and len(events) >= 2
    for event in events:
        if event:
            if event[0]["event_id"] == 0:
                assert event[0]["data"]["type"] == "created"
            elif event[0]["event_id"] == 1:
                assert event[0]["data"]["type"] == "stored"


@pytest.mark.skip("https://nvbugs/5150466: flaky fail")
def test_kv_cache_event_async_api():
    llm = create_llm()
    sampling_params = SamplingParams(max_tokens=6, temperature=0.01)
    prompt = "Hello, my name is"

    async def task0():
        async for output in llm.generate_async(prompt,
                                               streaming=True,
                                               sampling_params=sampling_params):
            pass

    events = []

    async def task1():
        async for event in llm.get_kv_cache_events_async():
            events.append(event)

        assert events

    async def main():
        await asyncio.gather(task0(), task1())
        for i in range(2):
            await asyncio.gather(task0(), task1())

    asyncio.run(main())


def test_llm_kv_events_api():
    llm = create_llm()
    sampling_params = SamplingParams(max_tokens=6, temperature=0.01)

    requests = []
    for i in range(3):
        input_tokens = list(range(127 + i))[i:]
        requests.append(input_tokens)

    _ = llm.generate(requests[0], sampling_params=sampling_params)
    events1 = llm.get_kv_cache_events(5)

    # Should have 1 stored event and 1 created event
    event = events1.pop(0)  # created event
    while events1:
        event = events1.pop(0)
        if event:
            assert event[0]["event_id"] == 1
            assert event[0]["data"]["type"] == "stored"
            assert len(event[0]["data"]["blocks"]) == 5

    _ = llm.generate(requests[1], sampling_params=sampling_params)
    events2 = llm.get_kv_cache_events(5)

    while events2:
        event = events2.pop(0)
        if event:
            if event[0]["event_id"] == 2:
                # 2 removed events needed
                # should be a removed event to make space for context block
                assert event[0]["data"]["type"] == "removed"
                assert event[0]["data"]["block_hashes"]
            elif event[0]["event_id"] == 3:
                assert event[0]["data"]["type"] == "removed"
                assert event[0]["data"]["block_hashes"]
            # stored event for 2nd request
            elif event[0]["event_id"] == 4:
                assert event[0]["data"]["type"] == "stored"
                assert len(event[0]["data"]["blocks"]) == 5

    _ = llm.generate(requests[2], sampling_params=sampling_params)
    events3 = llm.get_kv_cache_events(5)

    while events3:
        event = events3.pop(0)
        if event:
            if event[0]["event_id"] == 5:
                assert event[0]["data"]["type"] == "removed"
                assert event[0]["data"]["block_hashes"]
            elif event[0]["event_id"] == 6:
                assert event[0]["data"]["type"] == "removed"
                assert event[0]["data"]["block_hashes"]
            elif event[0]["event_id"] == 7:
                assert event[0]["data"]["type"] == "stored"
                assert len(event[0]["data"]["blocks"]) == 5

    # no more events after request is finished
    assert not llm.get_kv_cache_events(5)
