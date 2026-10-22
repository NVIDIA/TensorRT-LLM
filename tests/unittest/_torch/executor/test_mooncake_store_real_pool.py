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
"""KV page round trips through a real Mooncake pool, checked byte for byte.

The fake-store tests in `test_mooncake_store_connector.py` pin down which keys
and which addresses the connector asks for. They cannot say whether those
addresses name the bytes it meant, which is what this file checks: a page saved
and reloaded through a live master either comes back bit-identical or the
connector is serving wrong KV.

Expected byte offsets are computed here rather than read back from
`PageAddressing`, so an error in the page arithmetic cannot cancel out between
the save and the load.

Most tests run over TCP. The RDMA test registers a real `KVCacheManagerV2`
and needs an InfiniBand device and a GPUDirect path.
"""

from __future__ import annotations

import contextlib
import hashlib
import os
import time
from collections.abc import Iterator
from types import SimpleNamespace

import pytest
import torch
from test_common.mooncake_utils import running_master_on_free_ports

import tensorrt_llm.bindings
from tensorrt_llm._torch.pyexecutor.connectors.kv_cache_layout import (
    KvCacheBufferRef,
    KvCacheLayerGroupLayout,
    KvCacheLayout,
    KvCacheRegion,
    build_kv_cache_layout_v2,
)
from tensorrt_llm._torch.pyexecutor.connectors.mooncake_store import gpudirect
from tensorrt_llm._torch.pyexecutor.connectors.mooncake_store import worker as worker_module
from tensorrt_llm._torch.pyexecutor.connectors.mooncake_store.addressing import PageAddressing
from tensorrt_llm._torch.pyexecutor.connectors.mooncake_store.config import (
    CONFIG_PATH_ENV,
    MooncakeStoreConnectorConfig,
    StoreRole,
)
from tensorrt_llm._torch.pyexecutor.connectors.mooncake_store.master import (
    POOL_MANIFEST_NAME,
    _highest_rate_ib_devices,
    provision_pool,
    resolve_device_name,
)
from tensorrt_llm._torch.pyexecutor.connectors.mooncake_store.metadata import (
    MooncakeStoreMetadata,
    PageTransfer,
    RequestTransfers,
)
from tensorrt_llm._torch.pyexecutor.connectors.mooncake_store.worker import (
    MooncakeStoreConnectorWorker,
    _open_store,
)
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm.llmapi.llm_args import KvCacheConfig, MooncakeStoreConfig
from tensorrt_llm.mapping import Mapping

# Mooncake reads this once per process, and pytest imports this module before
# any test opens a client, so a TCP test cannot decide the path for the RDMA one.
if gpudirect.dmabuf_needed():
    os.environ.setdefault(gpudirect.PEERMEM_ENV, "0")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="The connector transfers KV out of device memory.",
)

TOKENS_PER_BLOCK = 32
#: Two layer groups of a key and a value region each, the smallest shape that
#: still exercises per-group addressing.
NUM_GROUPS = 2
REGION_ROLES = ("key", "value")
NUM_SLOTS = 8
REGION_BYTES = 4096
BUFFER_BYTES = NUM_GROUPS * len(REGION_ROLES) * NUM_SLOTS * REGION_BYTES

MODEL_KEY = "mooncake-real-pool-test"
#: The pool's whole capacity, held by the keeper the fixture leaves running.
#: Small enough to pass the node budget check on any machine that can run this.
KEEPER_SEGMENT_SIZE = 1 << 30
#: The workers lend nothing, so the keeper's segment is the only place a page
#: can land and no worker takes part of a block down with it at shutdown.
WORKER_SEGMENT_SIZE = 0
#: Stands in for a `BlockHashChain` digest, which the connector treats as
#: opaque content identity. Fixed so separate workers agree on it.
BLOCK_HASH = hashlib.sha256(b"mooncake-real-pool-test/block-0").digest()

SAVE_TIMEOUT_SECONDS = 60.0


def _region_offset(group_id: int, region_id: int) -> int:
    """Byte offset of a region within the test's KV buffer."""
    return (group_id * len(REGION_ROLES) + region_id) * NUM_SLOTS * REGION_BYTES


def _page_slice(buffer: torch.Tensor, group_id: int, region_id: int, page_index: int):
    """The bytes one page occupies in one region."""
    start = _region_offset(group_id, region_id) + page_index * REGION_BYTES
    return buffer[start : start + REGION_BYTES]


def _every_region() -> Iterator[tuple[int, int]]:
    for group_id in range(NUM_GROUPS):
        for region_id in range(len(REGION_ROLES)):
            yield group_id, region_id


def _device_buffer(*, filled: bool) -> torch.Tensor:
    """A KV buffer on the rank's device.

    Filled with noise rather than a pattern, because repeated bytes survive an
    addressing error that lands on the wrong offset within the same region.
    """
    if filled:
        return torch.randint(0, 256, (BUFFER_BYTES,), dtype=torch.uint8, device="cuda")
    return torch.zeros(BUFFER_BYTES, dtype=torch.uint8, device="cuda")


def _snapshot(buffer: torch.Tensor, page_index: int) -> dict[tuple[int, int], torch.Tensor]:
    """A copy of one page's bytes in every region, to compare against later."""
    return {region: _page_slice(buffer, *region, page_index).clone() for region in _every_region()}


def assert_page_matches(buffer, page_index, expected, *, context: str) -> None:
    for region in _every_region():
        group_id, region_id = region
        got = _page_slice(buffer, group_id, region_id, page_index)
        assert torch.equal(got, expected[region]), (
            f"{context}: layer group {group_id}'s {REGION_ROLES[region_id]} region "
            "came back changed. These KV slots were reported as computed, so a "
            "forward pass would have read whatever landed here."
        )


def _layout(buffer: torch.Tensor) -> KvCacheLayout:
    """Describe `buffer` the way a paged KV cache describes its pools."""
    groups = []
    for group_id in range(NUM_GROUPS):
        regions = tuple(
            KvCacheRegion(
                base=buffer.data_ptr() + _region_offset(group_id, region_id),
                size=REGION_BYTES,
                stride=REGION_BYTES,
                num_slots=NUM_SLOTS,
                buffers=(KvCacheBufferRef(layer_id=group_id, role=role),),
            )
            for region_id, role in enumerate(REGION_ROLES)
        )
        groups.append(
            KvCacheLayerGroupLayout(
                layer_group_id=group_id,
                layer_ids=(group_id,),
                window_size=None,
                regions=regions,
            )
        )
    return KvCacheLayout(tokens_per_block=TOKENS_PER_BLOCK, groups=tuple(groups))


def _llm_args(store_config: MooncakeStoreConfig) -> SimpleNamespace:
    """The fields the worker reads off `TorchLlmArgs`, and nothing else."""
    return SimpleNamespace(
        model=f"/models/{store_config.model_key}",
        kv_cache_config=SimpleNamespace(tokens_per_block=TOKENS_PER_BLOCK),
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        context_parallel_size=1,
        sparse_attention_config=None,
        kv_connector_config=SimpleNamespace(
            connector="mooncake-store",
            mooncake_store=store_config,
        ),
    )


@contextlib.contextmanager
def _running_pool(tmp_path, monkeypatch, *, protocol: str) -> Iterator[SimpleNamespace]:
    """A live master and the capacity behind it, over `protocol`.

    The master stores nothing: a page lives in the memory of the participant
    it was allocated in, so the keeper is what lets the pool outlive a worker.

    The master publishes into its own directory and clients claim another, so
    neither claims a run directory the other owns.
    """
    # `provision_pool` is a no-op when a client config is already named, which
    # would point this test at somebody else's pool.
    monkeypatch.delenv(CONFIG_PATH_ENV, raising=False)

    master_dir = tmp_path / "master"
    client_dir = tmp_path / "client"
    master_dir.mkdir()
    client_dir.mkdir()

    with running_master_on_free_ports(str(master_dir), protocol=protocol) as master:
        keeper, _ = _open_store(
            MooncakeStoreConnectorConfig(
                master_server_address=master.address,
                protocol=protocol,
                device_name=resolve_device_name(protocol, ""),
                global_segment_size=KEEPER_SEGMENT_SIZE,
                role=StoreRole.CAPACITY,
            )
        )
        try:
            yield SimpleNamespace(
                run_dir=str(client_dir),
                pool=f"file://{master_dir / POOL_MANIFEST_NAME}",
            )
        finally:
            keeper.close()


@pytest.fixture
def mooncake_pool(tmp_path, monkeypatch):
    """A pool over TCP, shut down with the test.

    TCP rather than RDMA, since a CI node need not have a usable NIC and the
    transport is not what most of these tests are about.
    """
    with _running_pool(tmp_path, monkeypatch, protocol="tcp") as pool:
        yield pool


def _gpudirect_unavailable() -> str | None:
    """Why this host cannot register device memory for RDMA, or None if it can."""
    if not _highest_rate_ib_devices():
        return "no active InfiniBand device"
    if not gpudirect.dmabuf_selected() and not gpudirect.peermem_loaded():
        return (
            "nvidia_peermem is not loaded and WITH_NVIDIA_PEERMEM is not 0, so "
            "Mooncake has no way to register GPU memory (set WITH_NVIDIA_PEERMEM=0 "
            "to use dma-buf)"
        )
    return None


@pytest.fixture
def rdma_pool(tmp_path, monkeypatch):
    """A pool over RDMA, or a skip on a host without GPUDirect RDMA."""
    reason = _gpudirect_unavailable()
    if reason is not None:
        pytest.skip(f"GPUDirect RDMA unavailable: {reason}.")
    with _running_pool(tmp_path, monkeypatch, protocol="rdma") as pool:
        yield pool


@contextlib.contextmanager
def open_worker(
    pool,
    *,
    buffer: torch.Tensor | None = None,
    layout: KvCacheLayout | None = None,
    stage_through_host: bool,
    model_key: str = MODEL_KEY,
) -> Iterator[MooncakeStoreConnectorWorker]:
    """A worker holding a real store handle, over `buffer` or `layout` as its KV cache.

    Shut down inside the test call phase, because registering a layout starts
    the background save thread and pytest-threadleak snapshots threads around
    the call phase only.
    """
    store_config = MooncakeStoreConfig(
        pool=pool.pool,
        model_key=model_key,
        segment_size=WORKER_SEGMENT_SIZE,
        stage_through_host=stage_through_host,
        run_dir=pool.run_dir,
    )
    with provision_pool(store_config, run_dir=pool.run_dir):
        worker = MooncakeStoreConnectorWorker(_llm_args(store_config))
        try:
            worker.register_kv_cache_layout(layout if layout is not None else _layout(buffer))
            yield worker
        finally:
            worker.shutdown()
            # Shutdown clears these only for the worker that installed itself,
            # so a failure before that would leave a dead handle behind.
            worker_module._LOCAL_WORKER = None
            worker_module._LOCAL_WORKER_READY.clear()


def _page_transfers(
    request_id: int, page_index: int, group_ids: tuple[int, ...]
) -> list[RequestTransfers]:
    """One page of every layer group, which is how a block is published."""
    return [
        RequestTransfers(
            request_id,
            [PageTransfer(BLOCK_HASH, group_id, page_index) for group_id in group_ids],
        )
    ]


def save_page(
    worker: MooncakeStoreConnectorWorker,
    request_id: int,
    page_index: int,
    group_ids: tuple[int, ...] = tuple(range(NUM_GROUPS)),
) -> None:
    """Publish a page and wait for the background save thread to drain it."""
    worker.bind_connector_meta(
        MooncakeStoreMetadata(saves=_page_transfers(request_id, page_index, group_ids))
    )
    worker.wait_for_save(torch.cuda.current_stream())

    deadline = time.monotonic() + SAVE_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        finished_saving, _ = worker.get_finished([request_id], [])
        if request_id in finished_saving:
            return
        time.sleep(0.05)
    raise AssertionError(
        f"The save for request {request_id} did not land within "
        f"{SAVE_TIMEOUT_SECONDS}s, so the load below would race it."
    )


def load_page(
    worker: MooncakeStoreConnectorWorker,
    request_id: int,
    page_index: int,
    group_ids: tuple[int, ...] = tuple(range(NUM_GROUPS)),
) -> None:
    """Pull a page into `page_index`, as the executor does before a forward pass."""
    # Zero copy lands bytes by RDMA, outside this stream's ordering, so work
    # the test queued against the KV buffer has to have retired first.
    torch.cuda.current_stream().synchronize()
    worker.bind_allocation_fence(lambda _request_ids: None)
    worker.bind_connector_meta(
        MooncakeStoreMetadata(loads=_page_transfers(request_id, page_index, group_ids))
    )
    worker.start_load_kv(torch.cuda.current_stream())
    torch.cuda.current_stream().synchronize()


@pytest.mark.parametrize("stage_through_host", [False, True], ids=["zero_copy", "host_staged"])
def test_a_saved_page_comes_back_byte_identical(mooncake_pool, stage_through_host):
    """Save a page, lose it, load it back, and compare the bytes.

    The buffer is zeroed between the two halves, so a load that transfers
    nothing fails instead of passing on leftover memory. Host staging adds a
    gather and a scatter through pinned buffers, and its whole point is to be
    indistinguishable from the zero-copy path.
    """
    buffer = _device_buffer(filled=True)
    page_index = 3
    expected = _snapshot(buffer, page_index)

    with open_worker(mooncake_pool, buffer=buffer, stage_through_host=stage_through_host) as worker:
        save_page(worker, request_id=1, page_index=page_index)
        buffer.zero_()
        load_page(worker, request_id=2, page_index=page_index)

    assert_page_matches(buffer, page_index, expected, context="Round trip through the pool")

    # A transfer that overran its page would restore the asked-for bytes and
    # corrupt a neighbour, which the comparison above cannot see.
    for group_id, region_id in _every_region():
        neighbour = _page_slice(buffer, group_id, region_id, page_index + 1)
        assert not neighbour.any(), (
            f"The load wrote into page {page_index + 1} of layer group {group_id} "
            f"({REGION_ROLES[region_id]}), which it was not asked for."
        )


def test_a_page_written_by_one_worker_is_read_back_by_another(mooncake_pool):
    """The pool outlives the worker that filled it.

    This is the claim the connector exists to make: a second engine over the
    same pool replays a prefix it never computed. The reader loads into a
    different page slot than the writer saved from, because the key names the
    content rather than the location.
    """
    writer_page, reader_page = 2, 5
    writer_buffer = _device_buffer(filled=True)
    expected = _snapshot(writer_buffer, writer_page)

    with open_worker(mooncake_pool, buffer=writer_buffer, stage_through_host=True) as writer:
        save_page(writer, request_id=1, page_index=writer_page)

    reader_buffer = _device_buffer(filled=False)
    with open_worker(mooncake_pool, buffer=reader_buffer, stage_through_host=True) as reader:
        assert reader.count_prefix_hit([BLOCK_HASH]) == 1, (
            "The second worker did not find the block the first one published, so "
            "nothing written by a previous engine would ever be reused."
        )
        load_page(reader, request_id=2, page_index=reader_page)

    assert_page_matches(reader_buffer, reader_page, expected, context="Replay by a second worker")


def test_a_prefix_counts_only_once_every_rank_has_its_shard(mooncake_pool, monkeypatch):
    """A prefix missing one rank's shard is a miss; with all of them, a hit.

    `count_prefix_hit` asks about every rank's namespace, because a prefix is
    replayed as a whole. A block that counted on one rank's pages alone would
    have the leader tell the others to load pages that are not in the pool, and
    `start_load_kv` fails the request rather than let a forward pass read
    uninitialized KV.

    Both ranks of a world of two are simulated in this process, one after the
    other, so the miss and the hit differ in exactly one thing: whether rank
    1's shard is there. Without that second half the miss would also be
    reported by a store that answered nothing at all, since a failed probe is
    treated as a miss by design.
    """
    monkeypatch.setattr(worker_module, "mpi_world_size", lambda: 2)

    buffer = _device_buffer(filled=True)
    with open_worker(mooncake_pool, buffer=buffer, stage_through_host=True) as rank_zero:
        save_page(rank_zero, request_id=1, page_index=0)
        assert rank_zero.count_prefix_hit([BLOCK_HASH]) == 0, (
            "A block with only rank 0's shard in the pool was offered for reuse. "
            "Rank 1 would be asked to load a page it never saved."
        )

    monkeypatch.setattr(worker_module, "mpi_rank", lambda: 1)
    with open_worker(mooncake_pool, buffer=buffer, stage_through_host=True) as rank_one:
        save_page(rank_one, request_id=2, page_index=0)
        assert rank_one.count_prefix_hit([BLOCK_HASH]) == 1, (
            "Both ranks' shards are in the pool and the block was still not "
            "offered, so nothing saved above one rank is ever reused."
        )


def test_a_different_model_key_shares_nothing(mooncake_pool):
    """Two models' pages stay apart even when their tokens agree.

    The block hash covers tokens, not weights, so model identity has to come
    from the key prefix. A collision would serve one model's KV to another,
    which produces plausible-looking nonsense rather than an error.

    The block is asked for under both keys, so the miss is the model key
    rather than a page the pool lost: the `model-a` hit would fail first.
    """
    buffer = _device_buffer(filled=True)
    with open_worker(
        mooncake_pool, buffer=buffer, stage_through_host=True, model_key="model-a"
    ) as writer:
        save_page(writer, request_id=1, page_index=0)

    other_buffer = _device_buffer(filled=False)
    with open_worker(
        mooncake_pool, buffer=other_buffer, stage_through_host=True, model_key="model-b"
    ) as reader:
        assert reader.count_prefix_hit([BLOCK_HASH]) == 0, (
            "A block saved under one model key was offered to a different model."
        )

    with open_worker(
        mooncake_pool, buffer=other_buffer, stage_through_host=True, model_key="model-a"
    ) as same_model:
        assert same_model.count_prefix_hit([BLOCK_HASH]) == 1, (
            "The block is gone from the pool under the key it was saved with, "
            "so the miss asserted above says nothing about model keys."
        )


# ---- RDMA over a real KVCacheManagerV2 ----


@contextlib.contextmanager
def _v2_cache_manager() -> Iterator[KVCacheManagerV2]:
    """A cache whose pools span several `cuMemCreate` allocations.

    About 16 MiB of KV, mapped in 2 MiB allocations at this quota.
    """
    manager = KVCacheManagerV2(
        kv_cache_config=KvCacheConfig(max_tokens=2048, enable_block_reuse=False),
        kv_cache_type=tensorrt_llm.bindings.internal.batch_manager.CacheType.SELF,
        num_layers=4,
        num_kv_heads=4,
        head_dim=128,
        tokens_per_block=TOKENS_PER_BLOCK,
        max_seq_len=256,
        max_batch_size=4,
        mapping=Mapping(world_size=1, tp_size=1, rank=0),
        dtype=tensorrt_llm.bindings.DataType.HALF,
        vocab_size=32000,
    )
    try:
        yield manager
    finally:
        manager.shutdown()


def _page_crossing_an_allocation(layout: KvCacheLayout, addressing: PageAddressing) -> int:
    """A page whose bytes the connector must split, or page 0 if none needs it."""
    num_slots = min(region.num_slots for group in layout.groups for region in group.regions)
    for page_index in range(num_slots):
        for group in layout.groups:
            addresses, _ = addressing.buffers(group.layer_group_id, page_index)
            if len(addresses) > len(group.regions):
                return page_index
    return 0


def test_a_real_v2_cache_registers_per_allocation_and_round_trips_over_rdma(rdma_pool):
    """The zero-copy path over the VMM memory a V2 cache really allocates.

    A plain torch tensor is one allocation, so it cannot show a registration
    crossing allocation boundaries.
    """
    with _v2_cache_manager() as manager:
        layout = build_kv_cache_layout_v2(manager)
        if not gpudirect.dmabuf_selected():
            first_region = layout.groups[0].regions[0]
            facts = gpudirect.describe_range(first_region.base, first_region.size)
            if facts.gdr_capable is False:
                pytest.skip(
                    "The V2 allocator fell back to memory without "
                    "gpuDirectRDMACapable, which nvidia_peermem cannot register; "
                    "set WITH_NVIDIA_PEERMEM=0 to test the dma-buf path."
                )

        group_ids = tuple(group.layer_group_id for group in layout.groups)
        regions = [region for group in layout.groups for region in group.regions]
        with open_worker(
            rdma_pool, layout=layout, stage_through_host=False, model_key="v2-rdma"
        ) as worker:
            registered = list(worker._registered_ranges)
            assert len(registered) > 1, (
                "The cache's pools were registered as a single range, so the "
                "allocation walk found no boundary to cut at."
            )
            for start, end in registered:
                allocation = gpudirect.allocation_range(start)
                assert allocation is not None
                base, size = allocation
                assert end <= base + size, (
                    f"[{start:#x}, {end:#x}) runs past its cuMemCreate allocation "
                    f"[{base:#x}, {base + size:#x}), which GPUDirect cannot register."
                )

            page_index = _page_crossing_an_allocation(layout, worker._addressing)
            for region in regions:
                page = region.slot_tensor(page_index)
                page.copy_(torch.randint(0, 256, page.shape, dtype=torch.uint8, device="cuda"))
            expected = [region.slot_tensor(page_index).clone() for region in regions]

            save_page(worker, request_id=1, page_index=page_index, group_ids=group_ids)
            for region in regions:
                region.slot_tensor(page_index).zero_()
            load_page(worker, request_id=2, page_index=page_index, group_ids=group_ids)

            for region, want in zip(regions, expected):
                assert torch.equal(region.slot_tensor(page_index), want), (
                    f"Page {page_index} of the region at {region.base:#x} came back "
                    "changed after an RDMA round trip through the pool."
                )
        assert not worker._registered_ranges, (
            "Shutdown left KV pool registrations behind over memory the cache "
            "manager is about to free."
        )
