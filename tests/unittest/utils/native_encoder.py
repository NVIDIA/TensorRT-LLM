# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native encoder test fixtures shared by native storage and serving adapters."""

import ctypes
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.disaggregation import encoder_nixl as transport
from tensorrt_llm._torch.disaggregation import encoder_output as oe
from tensorrt_llm._torch.shared_tensor import SharedTensorContainer
from tensorrt_llm.disaggregated_params import DisaggregatedParams


def request():
    return oe.EncoderInput(
        prompt="Describe <image_1>",
        token_ids=None,
        media=(oe.EncoderMedia("image", "data_uri", "data:image/png;base64,aW1hZ2U="),),
    )


def model():
    values = {"model_type": "qwen3_vl"}
    processor = SimpleNamespace(
        config=SimpleNamespace(model_type="qwen3_vl", to_dict=lambda: values),
        processor=SimpleNamespace(image_processor=None),
        get_preferred_media_io_kwargs=lambda: {},
        support_mm_disagg=True,
    )
    seen = {}

    def preprocess(inputs, sampling, params):
        seen["prompt"] = inputs["prompt"]
        seen["metadata"] = params.multimodal_embedding_handles[0]["metadata"]
        seen["hashes"] = params.multimodal_hashes
        handles = [
            *params.multimodal_embedding_handles,
            params.mrope_position_ids_handle,
            params.mrope_position_deltas_handle,
        ]
        seen["tensors"] = [
            SharedTensorContainer.from_dict(h).get_local_view().clone() for h in handles
        ]
        return SimpleNamespace(multimodal_params=None)

    return SimpleNamespace(input_processor=processor, preprocess=preprocess, tokenizer=None), seen


def native_result():
    tensors = [
        torch.arange(8, dtype=torch.float32).reshape(2, 4),
        torch.arange(6, dtype=torch.int64).reshape(3, 1, 2),
        torch.tensor([[1]]),
    ]
    handles = [SharedTensorContainer.from_tensor(tensor).dump_to_dict() for tensor in tensors]
    handles[0]["metadata"] = {"image_grid_thw": [1, 2, 2]}
    params = DisaggregatedParams(
        multimodal_embedding_handles=[handles[0]],
        multimodal_hashes=[[2147483647] * 8],
        mrope_position_ids_handle=handles[1],
        mrope_position_deltas_handle=handles[2],
    )
    return SimpleNamespace(disaggregated_params=params), tensors


def _check_handoff_hash_initialization(producer, consumer, supplied):
    llm, seen = model()
    llm.input_processor.multimodal_hashing_supported = supplied
    identities = []
    for offset in (1, 2):
        result, tensors = native_result()
        tensors[0].add_(offset)
        expected = [[offset] * 8] if supplied else None
        result.disaggregated_params.multimodal_hashes = expected
        ref = producer.publish(result, request(), llm)
        incoming = DisaggregatedParams(
            request_type="context_only",
            disagg_request_id=37,
            ctx_dp_rank=0,
            ctx_info_endpoint="context.example:1234",
            ctx_usage={"prompt_tokens": 88},
        )
        _, params = consumer.prepare(ref, request(), llm, None, incoming)
        assert params.request_type == incoming.request_type
        assert params.disagg_request_id == 37 and params.ctx_dp_rank == 0
        assert params.ctx_info_endpoint == incoming.ctx_info_endpoint
        assert params.ctx_usage == incoming.ctx_usage
        assert incoming.multimodal_embedding_handles is None
        assert torch.equal(seen["tensors"][0], tensors[0])
        hashes = seen["hashes"]
        assert isinstance(hashes, list) and len(hashes) == 1
        assert len(hashes[0]) == 8 and all(isinstance(value, int) for value in hashes[0])
        if supplied:
            assert hashes == expected
        identities.append(hashes)
    assert identities[0] != identities[1]


def _check_malformed_handoff_hashes(producer, consumer, hashes, monkeypatch):
    llm, seen = model()
    result, tensors = native_result()
    result.disaggregated_params.multimodal_hashes = hashes
    ref = producer.publish(result, request(), llm)
    released = []
    release = SharedTensorContainer.release_dict

    def record(handle):
        released.append(handle)
        release(handle)

    monkeypatch.setattr(SharedTensorContainer, "release_dict", record)
    with pytest.raises(ValueError, match="invalid encoder handoff parameters"):
        consumer.prepare(ref, request(), llm, None)
    assert "hashes" not in seen
    assert len(released) == len(tensors)


class FakeAgent:
    agents = {}

    def __init__(self, name):
        self.name = name
        self.registration = None
        self.registrations = {}
        self.loaded = set()
        self.writes = 0
        self.complete = True
        self.stopped = False
        self.agents[name] = self

    def register_memory(self, desc):
        self.registrations[id(desc)] = desc
        self.registration = desc

    def deregister_memory(self, desc):
        assert self.registrations.pop(id(desc)) is desc
        self.registration = next(iter(self.registrations.values()), None)

    def get_local_agent_desc(self):
        return self.name.encode()

    def load_remote_agent(self, name, desc):
        assert desc == name.encode()
        assert name not in self.loaded, "peer metadata replaced before WRITE completed"
        self.loaded.add(name)

    def invalidate_remote_agent(self, name):
        self.loaded.discard(name)

    def check_remote_descs(self, name, descs):
        peer = self.agents[name]
        targets = [
            desc if isinstance(desc, tuple) else (desc.addr, desc.len, desc.device_id)
            for desc in descs.descs
        ]
        return all(
            any(
                source[0] <= target[0]
                and target[0] + target[1] <= source[0] + source[1]
                and source[2] == target[2]
                for reg in peer.registrations.values()
                for source in reg.descs
            )
            for target in targets
        )

    def submit_transfer_requests(self, req):
        self.writes += 1
        peer = self.agents[req.remote_name]
        sources = req.src_descs.descs
        targets = req.dst_descs.descs
        assert len(sources) == len(targets) > 0
        for source, target in zip(sources, targets):
            source = source if isinstance(source, tuple) else (source.addr, source.len)
            target = target if isinstance(target, tuple) else (target.addr, target.len)
            assert any(
                source[:2] == desc[:2] for reg in self.registrations.values() for desc in reg.descs
            )
            assert any(
                target[:2] == desc[:2] for reg in peer.registrations.values() for desc in reg.descs
            )
            ctypes.memmove(target[0], source[0], source[1])
        return SimpleNamespace(wait=lambda timeout: self.complete)

    def shutdown(self):
        self.stopped = True
        self.registrations.clear()
        self.registration = None
        self.loaded.clear()


def _cache(endpoint="", request_transfer=None):
    agents = []

    def factory(name):
        agent = FakeAgent(name)
        agents.append(agent)
        return agent

    cache = transport.NixlEncoderCache(
        endpoint,
        "a" * 40,
        device="cpu",
        agent_factory=factory,
        request_transfer=request_transfer,
    )
    agents.clear()
    cache.test_agents = agents
    return cache


def _receiver(cache):
    return cache.test_agents[-1]


def _sender(cache):
    return cache.test_agents[-1]


def _writes(cache):
    return sum(agent.writes for agent in cache.test_agents)


def _receive(cache, reference):
    with cache._operation():
        return cache._receive(reference, cache._reference(reference))


@pytest.fixture
def native_pair():
    producer = _cache("encoder.example:30001")
    consumer = _cache(
        request_transfer=lambda endpoint, ref, destination, timeout: producer.transfer(
            ref, destination
        )
    )
    try:
        yield producer, consumer
    finally:
        for cache in (consumer, producer):
            if not cache._closed:
                cache.close()
