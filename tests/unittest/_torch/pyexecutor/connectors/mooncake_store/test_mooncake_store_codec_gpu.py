# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The mooncake-store worker over a real KV cache and its cold-page codec.

The store is replaced by an in-process one that keeps the bytes it is handed,
so a page saved by the worker and loaded back goes through exactly the encode,
store and decode a real pool would see, on one GPU and without a Mooncake
master. The reference for every comparison is the cache manager's own codec,
called through its native bindings with page pairs built here rather than
through the connectors' codec handle: the worker must reproduce it bit for bit.
"""

import contextlib
import ctypes
import json
import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tensorrt_llm._torch.pyexecutor._util import create_kv_cache_compression_manager
from tensorrt_llm._torch.pyexecutor.connectors.kv_cache_layout import build_kv_cache_layout_v2
from tensorrt_llm._torch.pyexecutor.connectors.mooncake_store import worker as worker_module
from tensorrt_llm._torch.pyexecutor.connectors.mooncake_store.metadata import (
    PageTransfer,
    RequestTransfers,
)
from tensorrt_llm._torch.pyexecutor.connectors.mooncake_store.worker import (
    MooncakeStoreConnectorWorker,
)
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm._torch.pyexecutor.resource_manager import DataType
from tensorrt_llm._utils import is_sm_100f
from tensorrt_llm.bindings.internal.batch_manager import CacheType
from tensorrt_llm.llmapi.llm_args import ColdPageQuantizationCompressionConfig, KvCacheConfig
from tensorrt_llm.mapping import Mapping

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU"),
]

needs_nvfp4 = pytest.mark.skipif(
    not torch.cuda.is_available() or not is_sm_100f(),
    reason="NVFP4 cold-page kernels need an SM100-family GPU",
)

TOKENS_PER_BLOCK = 32
NUM_LAYERS = 3


class MemoryStore:
    """A store that keeps what it is given, read from and written to host memory.

    The worker only ever hands it its registered pinned arenas, so every
    address here is host memory ctypes can reach.
    """

    def __init__(self):
        self.objects = {}
        self.workers = []

    def register_buffer(self, address, size):
        return 0

    def unregister_buffer(self, address):
        return 0

    def batch_is_exist(self, keys):
        return [1 if key in self.objects else 0 for key in keys]

    def batch_put_from_multi_buffers(self, keys, addresses, sizes, *_args, **_kwargs):
        results = []
        for key, page_addresses, page_sizes in zip(keys, addresses, sizes, strict=True):
            data = b"".join(
                ctypes.string_at(address, size)
                for address, size in zip(page_addresses, page_sizes, strict=True)
            )
            self.objects[key] = data
            results.append(len(data))
        return results

    def batch_get_into_multi_buffers(self, keys, addresses, sizes):
        results = []
        for key, page_addresses, page_sizes in zip(keys, addresses, sizes, strict=True):
            data = self.objects.get(key)
            if data is None or len(data) != sum(page_sizes):
                results.append(-1)
                continue
            offset = 0
            for address, size in zip(page_addresses, page_sizes, strict=True):
                ctypes.memmove(address, data[offset : offset + size], size)
                offset += size
            results.append(len(data))
        return results

    def close(self):
        pass


@pytest.fixture(autouse=True)
def on_a_created_stream():
    """Codecs may refuse the legacy default stream, so each test runs on one of its own."""
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        yield
    stream.synchronize()


@pytest.fixture
def memory_store(tmp_path, monkeypatch):
    path = tmp_path / "mooncake.json"
    path.write_text(
        json.dumps(
            {
                "metadata_server": "http://127.0.0.1:8080/metadata",
                "master_server_address": "127.0.0.1:50051",
                "protocol": "tcp",
                "device_name": "",
                "global_segment_size": "1GiB",
                "local_buffer_size": "256MiB",
                "model_key": "codec-test",
            }
        )
    )
    monkeypatch.setenv("MOONCAKE_CONFIG_PATH", str(path))
    store = MemoryStore()
    monkeypatch.setattr(worker_module, "_open_store", lambda _config: (store, "10.0.0.1"))
    monkeypatch.setattr(worker_module, "validate_node_budget", lambda _config, **_kwargs: None)
    yield store
    for worker in store.workers:
        worker.shutdown()
    worker_module._LOCAL_WORKER = None
    worker_module._LOCAL_WORKER_READY.clear()


def make_llm_args():
    return SimpleNamespace(
        model="/models/codec-test",
        kv_cache_config=SimpleNamespace(tokens_per_block=TOKENS_PER_BLOCK),
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        context_parallel_size=1,
        sparse_attention_config=None,
        kv_connector_config=SimpleNamespace(
            connector="mooncake-store", mooncake_store=SimpleNamespace(run_dir=None)
        ),
    )


def model_engine():
    return SimpleNamespace(
        mapping=SimpleNamespace(has_cp_helix=lambda: False),
        cuda_graph_batch_sizes=[],
        cuda_graph_padding_enabled=False,
        spec_config=None,
        model=SimpleNamespace(
            # The codec reads the model type to pick a K/V layout; a plain decoder is the default one.
            model_config=SimpleNamespace(
                pretrained_config=SimpleNamespace(model_type="llama"), quant_config=None
            )
        ),
    )


@contextlib.contextmanager
def make_cache(dtype: DataType, *, compression=True, windows=None):
    """A V2 cache as the connector runs it: GPU tier only, with or without NVFP4.

    `windows` gives layers sliding windows, which puts them in layer groups of
    their own.
    """
    kv_cache_config = KvCacheConfig(
        max_tokens=4096,
        enable_block_reuse=True,
        host_cache_size=0,
        disk_cache_size=0,
        max_attention_window=windows,
    )
    provider = None
    if compression:
        provider = create_kv_cache_compression_manager(
            ColdPageQuantizationCompressionConfig(),
            model_engine=model_engine(),
            kv_cache_config=kv_cache_config,
        )
    manager = KVCacheManagerV2(
        kv_cache_config=kv_cache_config,
        kv_cache_type=CacheType.SELF,
        num_layers=NUM_LAYERS,
        num_kv_heads=4,
        head_dim=128,
        tokens_per_block=TOKENS_PER_BLOCK,
        max_seq_len=1024,
        max_batch_size=4,
        mapping=Mapping(world_size=1, tp_size=1, rank=0),
        dtype=dtype,
        vocab_size=32000,
        cold_page_codec_provider=provider,
    )
    try:
        yield manager
    finally:
        manager.shutdown()


def fill_pages(layout, pages, dtype: DataType, seed=0):
    """Random KV values in every region of `pages`, as the cache's dtype stores them."""
    generator = torch.Generator(device="cuda").manual_seed(seed)
    for group in layout.groups:
        for region in group.regions:
            for page in pages:
                view = region.slot_tensor(page)
                values = torch.randn(
                    view.numel() // (1 if dtype == DataType.FP8 else 2),
                    generator=generator,
                    device="cuda",
                )
                if dtype == DataType.FP8:
                    values = values.to(torch.float8_e4m3fn)
                else:
                    values = values.to(torch.bfloat16)
                view.copy_(values.view(torch.uint8))
    torch.cuda.synchronize()


def page_bytes(layout, group, page):
    return torch.cat([region.slot_tensor(page).clone() for region in group.regions])


def codec_round_trip(layout, sources, targets):
    """Through the connectors' codec handle: encode `sources`, decode into `targets`."""
    codec = layout.page_codec
    stream = torch.cuda.current_stream().cuda_stream
    for group in layout.groups:
        size = codec.page_bytes(group.layer_group_id)
        buffer = torch.empty(size * len(sources), dtype=torch.uint8, device="cuda")
        slots = list(range(len(sources)))
        codec.encode(group.layer_group_id, buffer.data_ptr(), slots, sources, stream)
        codec.decode(group.layer_group_id, buffer.data_ptr(), targets, slots, stream)
    torch.cuda.synchronize()


def reference_round_trip(manager, layout, sources, targets):
    """The cache's codec through its native bindings: encode `sources`, decode into `targets`.

    The (dst, src) page pairs are built here, so a mistake in how the codec
    handle the worker uses builds them cannot cancel out against the reference.
    """
    impl = manager.impl
    stream = torch.cuda.current_stream().cuda_stream
    for group in layout.groups:
        layer_group_id = group.layer_group_id
        size = impl.cold_page_bytes(layer_group_id)
        buffer = torch.empty(size * len(sources), dtype=torch.uint8, device="cuda")
        encode_pairs = torch.tensor(
            [[slot, page] for slot, page in enumerate(sources)], dtype=torch.int32
        )
        decode_pairs = torch.tensor(
            [[page, slot] for slot, page in enumerate(targets)], dtype=torch.int32
        )
        if impl.cold_page_index_location(layer_group_id) == 1:
            encode_pairs, decode_pairs = encode_pairs.cuda(), decode_pairs.cuda()
        assert impl.encode_cold_pages(
            layer_group_id, buffer.data_ptr(), encode_pairs.data_ptr(), len(sources), stream
        )
        assert impl.decode_cold_pages(
            layer_group_id, buffer.data_ptr(), decode_pairs.data_ptr(), len(targets), stream
        )
        torch.cuda.synchronize()


@contextlib.contextmanager
def make_worker(store, layout):
    worker = MooncakeStoreConnectorWorker(make_llm_args())
    store.workers.append(worker)
    worker.register_kv_cache_layout(layout)
    try:
        yield worker
    finally:
        worker.shutdown()


@needs_nvfp4
@pytest.mark.parametrize("dtype", [DataType.BF16, DataType.FP8], ids=["bf16", "fp8"])
def test_pages_through_the_pool_match_the_caches_own_codec(memory_store, dtype):
    # Hot pages that are not their arena slots, so a slot and a page mixed up shows.
    sources, targets, references = [3, 1, 2], [5, 6, 7], [9, 10, 11]
    hashes = [bytes([index]) * 16 for index in range(len(sources))]
    with make_cache(dtype) as manager:
        layout = build_kv_cache_layout_v2(manager)
        assert layout.page_codec is not None, "NVFP4 gives the cache a cold-page codec"
        fill_pages(layout, sources, dtype)
        reference_round_trip(manager, layout, sources, references)

        with make_worker(memory_store, layout) as worker:
            worker._put(
                [
                    RequestTransfers(
                        1,
                        [
                            PageTransfer(block_hash, group.layer_group_id, page)
                            for block_hash, page in zip(hashes, sources)
                            for group in layout.groups
                        ],
                    )
                ]
            )
            worker.bind_connector_meta(
                SimpleNamespace(
                    loads=[
                        RequestTransfers(
                            2,
                            [
                                PageTransfer(block_hash, group.layer_group_id, page)
                                for block_hash, page in zip(hashes, targets)
                                for group in layout.groups
                            ],
                        )
                    ],
                    saves=[],
                )
            )
            worker.bind_allocation_fence(lambda _request_ids: None)
            worker.start_load_kv(torch.cuda.current_stream())
            torch.cuda.synchronize()

        for group in layout.groups:
            encoded = layout.page_codec.page_bytes(group.layer_group_id)
            assert encoded < group.bytes_per_page, "the pool must hold the smaller encoding"
            for target, reference in zip(targets, references):
                assert torch.equal(
                    page_bytes(layout, group, target), page_bytes(layout, group, reference)
                ), "a page through the pool must decode exactly as the cache's codec does"
        stored_sizes = {len(data) for data in memory_store.objects.values()}
        assert stored_sizes == {
            layout.page_codec.page_bytes(group.layer_group_id) for group in layout.groups
        }


@needs_nvfp4
def test_decoded_pages_stay_close_to_the_originals():
    """NVFP4 is lossy; this bounds the loss so a mangled layout cannot pass."""
    sources, targets = [0, 1], [4, 5]
    with make_cache(DataType.BF16) as manager:
        layout = build_kv_cache_layout_v2(manager)
        fill_pages(layout, sources, DataType.BF16, seed=7)
        codec_round_trip(layout, sources, targets)
        for group in layout.groups:
            for source, target in zip(sources, targets):
                original = page_bytes(layout, group, source).view(torch.bfloat16).float()
                decoded = page_bytes(layout, group, target).view(torch.bfloat16).float()
                cosine = torch.nn.functional.cosine_similarity(original, decoded, dim=0)
                assert cosine > 0.98, f"cosine {cosine.item():.4f}"


@needs_nvfp4
def test_every_layer_group_of_a_windowed_cache_round_trips():
    """The page codec covers each layer group a cache has, at its own encoded size.

    `mooncake-store` refuses sliding windows, but the codec handle is the
    connectors' common one, so a cache with several layer groups is checked
    through it directly.
    """
    sources, targets = [0, 1], [4, 5]
    with make_cache(DataType.BF16, windows=[64, 1024]) as manager:
        layout = build_kv_cache_layout_v2(manager)
        assert len(layout.groups) > 1, "sliding windows give the cache several layer groups"
        assert layout.page_codec.layer_group_ids == tuple(
            group.layer_group_id for group in layout.groups
        )
        fill_pages(layout, sources, DataType.BF16, seed=11)
        codec_round_trip(layout, sources, targets)
        for group in layout.groups:
            assert layout.page_codec.page_bytes(group.layer_group_id) < group.bytes_per_page
            for source, target in zip(sources, targets):
                original = page_bytes(layout, group, source).view(torch.bfloat16).float()
                decoded = page_bytes(layout, group, target).view(torch.bfloat16).float()
                cosine = torch.nn.functional.cosine_similarity(original, decoded, dim=0)
                assert cosine > 0.98, (
                    f"layer group {group.layer_group_id}: cosine {cosine.item():.4f}"
                )


@needs_nvfp4
def test_the_codec_identity_reaches_the_keys(memory_store):
    with make_cache(DataType.BF16) as manager:
        layout = build_kv_cache_layout_v2(manager)
        identity = manager.cold_page_codec_identity
        assert identity.startswith("quantization_for_cold_page-")
        with make_worker(memory_store, layout) as worker:
            key = worker._namespaces[0].key(b"\x00" * 16)
        assert f"/e{identity}.BF16.r{layout.groups[0].bytes_per_page}/" in key


@needs_nvfp4
def test_the_forward_stream_waits_for_the_decodes_on_the_device(memory_store):
    """Decodes run on the worker's load stream; the pass reading the pages waits for them there.

    The load stream is held back past the point where the host returns, so a
    read on the forward stream that is not ordered after the decodes sees the
    pages as they were before the load.
    """
    sources, targets, references = [3, 1], [5, 6], [9, 10]
    hashes = [bytes([index + 40]) * 16 for index in range(len(sources))]
    with make_cache(DataType.BF16) as manager:
        layout = build_kv_cache_layout_v2(manager)
        fill_pages(layout, sources, DataType.BF16, seed=5)
        reference_round_trip(manager, layout, sources, references)
        with make_worker(memory_store, layout) as worker:
            worker._put(
                [
                    RequestTransfers(
                        1,
                        [
                            PageTransfer(block_hash, group.layer_group_id, page)
                            for block_hash, page in zip(hashes, sources)
                            for group in layout.groups
                        ],
                    )
                ]
            )
            for group in layout.groups:
                for region in group.regions:
                    for target in targets:
                        region.slot_tensor(target).zero_()
            torch.cuda.synchronize()

            fenced = []

            def fence(request_ids):
                fenced.append(set(request_ids))
                with torch.cuda.stream(worker._load_stream):
                    torch.cuda._sleep(200_000_000)

            worker.bind_allocation_fence(fence)
            worker.bind_connector_meta(
                SimpleNamespace(
                    loads=[
                        RequestTransfers(
                            2,
                            [
                                PageTransfer(block_hash, group.layer_group_id, page)
                                for block_hash, page in zip(hashes, targets)
                                for group in layout.groups
                            ],
                        )
                    ],
                    saves=[],
                )
            )
            forward = torch.cuda.current_stream()
            worker.start_load_kv(forward)
            # Copied on the forward stream with no host wait in between.
            seen = {
                (group.layer_group_id, target): page_bytes(layout, group, target)
                for group in layout.groups
                for target in targets
            }
            torch.cuda.synchronize()

        assert fenced == [{2}]
        for group in layout.groups:
            for target, reference in zip(targets, references):
                assert torch.equal(
                    seen[(group.layer_group_id, target)], page_bytes(layout, group, reference)
                ), "the forward stream read a page before its decode finished"


@needs_nvfp4
def test_one_page_batches_alternate_the_arena_halves_in_both_directions(memory_store, monkeypatch):
    """Every page is its own batch, so the arena halves take turns on save and on load.

    Each encode and decode is held back on its stream. A put that read a half
    before its encode finished would store a stale page, and a get that
    refilled a half the decode of two batches back still reads would decode
    the wrong one; either shows as a page differing from the reference.
    """
    config_path = os.environ["MOONCAKE_CONFIG_PATH"]
    with open(config_path) as handle:
        raw = json.load(handle)
    raw["transfer_batch_size"] = 1
    with open(config_path, "w") as handle:
        json.dump(raw, handle)

    store_calls = {"put": 0, "get": 0}
    put, get = memory_store.batch_put_from_multi_buffers, memory_store.batch_get_into_multi_buffers

    def counted_put(*args, **kwargs):
        store_calls["put"] += 1
        return put(*args, **kwargs)

    def counted_get(*args, **kwargs):
        store_calls["get"] += 1
        return get(*args, **kwargs)

    monkeypatch.setattr(memory_store, "batch_put_from_multi_buffers", counted_put)
    monkeypatch.setattr(memory_store, "batch_get_into_multi_buffers", counted_get)

    sources, targets, references = [3, 1, 2, 7], [20, 21, 22, 23], [40, 41, 42, 43]
    hashes = [bytes([index + 80]) * 16 for index in range(len(sources))]
    with make_cache(DataType.BF16) as manager:
        layout = build_kv_cache_layout_v2(manager)
        fill_pages(layout, sources, DataType.BF16, seed=13)
        reference_round_trip(manager, layout, sources, references)

        codec = layout.page_codec

        def held_back(call):
            def run(layer_group_id, base, first, second, stream):
                with torch.cuda.stream(torch.cuda.ExternalStream(stream)):
                    torch.cuda._sleep(50_000_000)
                return call(layer_group_id, base, first, second, stream)

            return run

        monkeypatch.setattr(codec, "encode", held_back(codec.encode))
        monkeypatch.setattr(codec, "decode", held_back(codec.decode))

        with make_worker(memory_store, layout) as worker:
            worker._put(
                [
                    RequestTransfers(
                        1,
                        [
                            PageTransfer(block_hash, group.layer_group_id, page)
                            for block_hash, page in zip(hashes, sources)
                            for group in layout.groups
                        ],
                    )
                ]
            )
            assert store_calls["put"] == len(sources) * len(layout.groups)
            worker.bind_connector_meta(
                SimpleNamespace(
                    loads=[
                        RequestTransfers(
                            2,
                            [
                                PageTransfer(block_hash, group.layer_group_id, page)
                                for block_hash, page in zip(hashes, targets)
                                for group in layout.groups
                            ],
                        )
                    ],
                    saves=[],
                )
            )
            worker.bind_allocation_fence(lambda _request_ids: None)
            worker.start_load_kv(torch.cuda.current_stream())
            torch.cuda.synchronize()
            assert store_calls["get"] == len(targets) * len(layout.groups)

        for group in layout.groups:
            for target, reference in zip(targets, references):
                assert torch.equal(
                    page_bytes(layout, group, target), page_bytes(layout, group, reference)
                ), f"page {target} decoded differently from the reference"


def test_a_cache_without_compression_offers_no_codec():
    with make_cache(DataType.BF16, compression=False) as manager:
        assert manager.cold_page_codec_identity is None
        assert build_kv_cache_layout_v2(manager).page_codec is None


def test_the_lossless_default_codec_round_trips_bit_exactly():
    """The native entry points over the default codec: the binding contract on its own."""
    with make_cache(DataType.BF16, compression=False) as manager:
        layout = build_kv_cache_layout_v2(manager)
        fill_pages(layout, [1, 2], DataType.BF16, seed=3)
        impl = manager.impl
        stream = torch.cuda.current_stream().cuda_stream
        for group in layout.groups:
            layer_group_id = group.layer_group_id
            size = impl.cold_page_bytes(layer_group_id)
            assert size >= group.bytes_per_page
            assert impl.cold_page_index_location(layer_group_id) in (0, 1)
            buffer = torch.zeros(2 * size, dtype=torch.uint8, device="cuda")
            on_device = impl.cold_page_index_location(layer_group_id) == 1
            encode_pairs = torch.tensor([[0, 1], [1, 2]], dtype=torch.int32)
            decode_pairs = torch.tensor([[6, 0], [7, 1]], dtype=torch.int32)
            if on_device:
                encode_pairs, decode_pairs = encode_pairs.cuda(), decode_pairs.cuda()
            assert impl.encode_cold_pages(
                layer_group_id, buffer.data_ptr(), encode_pairs.data_ptr(), 2, stream
            )
            assert impl.decode_cold_pages(
                layer_group_id, buffer.data_ptr(), decode_pairs.data_ptr(), 2, stream
            )
            torch.cuda.synchronize()
            for source, target in ((1, 6), (2, 7)):
                assert torch.equal(
                    page_bytes(layout, group, source), page_bytes(layout, group, target)
                )
        assert impl.cold_page_bytes(99) == 0
        assert impl.cold_page_index_location(99) == -1
        assert not impl.encode_cold_pages(99, 0, 0, 0, stream)


def allowed_numa_nodes():
    """The NUMA nodes this process's cpuset lets it allocate on, or None when unknown."""
    try:
        with open("/proc/self/status") as handle:
            for line in handle:
                if line.startswith("Mems_allowed_list:"):
                    nodes = set()
                    for part in line.split(":", 1)[1].strip().split(","):
                        low, _, high = part.partition("-")
                        nodes.update(range(int(low), int(high or low) + 1))
                    return nodes
    except (OSError, ValueError):
        return None
    return None


def test_staging_memory_sits_on_the_gpus_numa_node():
    """Staging through the far socket costs every encode and decode the socket link."""
    from tensorrt_llm._torch.pyexecutor.connectors.mooncake_store.codec_staging import (
        PinnedHostBuffer,
        gpu_host_numa_node,
        resident_numa_nodes,
    )

    expected = gpu_host_numa_node(torch.cuda.current_device())
    if expected is None:
        pytest.skip("the driver reports no host NUMA node for this GPU")
    allowed = allowed_numa_nodes()
    if allowed is not None and expected not in allowed:
        pytest.skip(f"this process may not place memory on node {expected} (allowed: {allowed})")
    buffer = PinnedHostBuffer(8 << 20)
    try:
        placed = resident_numa_nodes(buffer.address, buffer.nbytes)
        if placed is None:
            pytest.skip("the kernel does not say where pages live (libnuma or move_pages refused)")
        assert placed == {expected}, f"pages landed on nodes {placed}, not node {expected}"
        assert buffer.numa_node == expected
        # The GPU reaches it at its host address, as the codec does.
        values = torch.arange(1024, dtype=torch.int32, device="cuda")
        host = torch.from_numpy(
            np.ctypeslib.as_array((ctypes.c_int32 * 1024).from_address(buffer.address))
        )
        host.zero_()
        torch.cuda.synchronize()
        from cuda.bindings import runtime as cudart

        cudart.cudaMemcpy(
            buffer.address, values.data_ptr(), 4096, cudart.cudaMemcpyKind.cudaMemcpyDefault
        )
        assert torch.equal(host, values.cpu())
    finally:
        buffer.close()


def test_staging_falls_back_to_pinned_memory_when_registration_fails(monkeypatch):
    """A refused cudaHostRegister leaves usable memory and no CUDA error behind."""
    from cuda.bindings import runtime as cudart

    from tensorrt_llm._torch.pyexecutor.connectors.mooncake_store import codec_staging

    monkeypatch.setattr(
        codec_staging.cudart,
        "cudaHostRegister",
        lambda *_args: (cudart.cudaError_t.cudaErrorMemoryAllocation,),
    )
    buffer = codec_staging.PinnedHostBuffer(4 << 20)
    try:
        assert buffer.numa_node is None
        assert buffer._map is None, "the mapping is released with the failed registration"
        values = torch.arange(1024, dtype=torch.int32, device="cuda")
        cudart.cudaMemcpy(
            buffer.address, values.data_ptr(), 4096, cudart.cudaMemcpyKind.cudaMemcpyDefault
        )
        host = torch.from_numpy(
            np.ctypeslib.as_array((ctypes.c_int32 * 1024).from_address(buffer.address))
        )
        assert torch.equal(host, values.cpu())
        assert cudart.cudaGetLastError()[0] == cudart.cudaError_t.cudaSuccess
    finally:
        buffer.close()
