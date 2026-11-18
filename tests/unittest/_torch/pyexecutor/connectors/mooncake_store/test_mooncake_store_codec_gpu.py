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
called directly: the worker must reproduce it bit for bit.
"""

import contextlib
import ctypes
import json
from types import SimpleNamespace

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
    monkeypatch.setattr(worker_module, "validate_node_budget", lambda _config: None)
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
            model_config=SimpleNamespace(pretrained_config=None, quant_config=None)
        ),
    )


@contextlib.contextmanager
def make_cache(dtype: DataType, *, compression=True):
    """A V2 cache as the connector runs it: GPU tier only, with or without NVFP4."""
    kv_cache_config = KvCacheConfig(
        max_tokens=4096, enable_block_reuse=True, host_cache_size=0, disk_cache_size=0
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


def reference_round_trip(layout, sources, targets):
    """The cache's codec, called directly: encode `sources`, decode into `targets`."""
    codec = layout.page_codec
    stream = torch.cuda.current_stream().cuda_stream
    for group in layout.groups:
        size = codec.page_bytes(group.layer_group_id)
        buffer = torch.empty(size * len(sources), dtype=torch.uint8, device="cuda")
        slots = list(range(len(sources)))
        codec.encode(group.layer_group_id, buffer.data_ptr(), slots, sources, stream)
        codec.decode(group.layer_group_id, buffer.data_ptr(), targets, slots, stream)
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
    sources, targets, references = [0, 1, 2], [5, 6, 7], [9, 10, 11]
    hashes = [bytes([index]) * 16 for index in range(len(sources))]
    with make_cache(dtype) as manager:
        layout = build_kv_cache_layout_v2(manager)
        assert layout.page_codec is not None, "NVFP4 gives the cache a cold-page codec"
        fill_pages(layout, sources, dtype)
        reference_round_trip(layout, sources, references)

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
def test_decoded_pages_stay_close_to_the_originals(memory_store):
    """NVFP4 is lossy; this bounds the loss so a mangled layout cannot pass."""
    sources, targets = [0, 1], [4, 5]
    with make_cache(DataType.BF16) as manager:
        layout = build_kv_cache_layout_v2(manager)
        fill_pages(layout, sources, DataType.BF16, seed=7)
        reference_round_trip(layout, sources, targets)
        for group in layout.groups:
            for source, target in zip(sources, targets):
                original = page_bytes(layout, group, source).view(torch.bfloat16).float()
                decoded = page_bytes(layout, group, target).view(torch.bfloat16).float()
                cosine = torch.nn.functional.cosine_similarity(original, decoded, dim=0)
                assert cosine > 0.98, f"cosine {cosine.item():.4f}"


@needs_nvfp4
def test_the_codec_identity_reaches_the_keys(memory_store):
    with make_cache(DataType.BF16) as manager:
        layout = build_kv_cache_layout_v2(manager)
        identity = manager.cold_page_codec_identity
        assert identity.startswith("quantization_for_cold_page-")
        with make_worker(memory_store, layout) as worker:
            key = worker._namespaces[0].key(b"\x00" * 16)
        assert f"/e{identity}.r{layout.groups[0].bytes_per_page}/" in key


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
