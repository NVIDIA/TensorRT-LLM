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
"""

from __future__ import annotations

import contextlib
import hashlib
import time
from collections.abc import Iterator
from types import SimpleNamespace

import pytest
import torch
from test_common.mooncake_utils import running_master_on_free_ports

from tensorrt_llm._torch.pyexecutor.connectors.kv_cache_layout import (
    KvCacheBufferRef,
    KvCacheLayerGroupLayout,
    KvCacheLayout,
    KvCacheRegion,
)
from tensorrt_llm._torch.pyexecutor.connectors.mooncake_store import worker as worker_module
from tensorrt_llm._torch.pyexecutor.connectors.mooncake_store.config import (
    CONFIG_PATH_ENV,
    MooncakeStoreConnectorConfig,
    StoreRole,
)
from tensorrt_llm._torch.pyexecutor.connectors.mooncake_store.master import (
    POOL_MANIFEST_NAME,
    provision_pool,
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
from tensorrt_llm.llmapi.llm_args import MooncakeStoreConfig

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


@pytest.fixture
def mooncake_pool(tmp_path, monkeypatch):
    """A live master and the capacity behind it, shut down with the test.

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

    # TCP rather than RDMA, since a CI node need not have a usable NIC and the
    # transport is not what these tests are about.
    with running_master_on_free_ports(str(master_dir), protocol="tcp") as master:
        keeper, _ = _open_store(
            MooncakeStoreConnectorConfig(
                master_server_address=master.address,
                protocol="tcp",
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


@contextlib.contextmanager
def open_worker(
    pool,
    *,
    buffer: torch.Tensor,
    model_key: str = MODEL_KEY,
) -> Iterator[MooncakeStoreConnectorWorker]:
    """A worker holding a real store handle, with `buffer` as its KV cache.

    Shut down inside the test call phase, because registering a layout starts
    the background save thread and pytest-threadleak snapshots threads around
    the call phase only.
    """
    store_config = MooncakeStoreConfig(
        pool=pool.pool,
        model_key=model_key,
        segment_size=WORKER_SEGMENT_SIZE,
        run_dir=pool.run_dir,
    )
    with provision_pool(store_config, run_dir=pool.run_dir):
        worker = MooncakeStoreConnectorWorker(_llm_args(store_config))
        try:
            worker.register_kv_cache_layout(_layout(buffer))
            yield worker
        finally:
            worker.shutdown()
            # Shutdown clears these only for the worker that installed itself,
            # so a failure before that would leave a dead handle behind.
            worker_module._LOCAL_WORKER = None
            worker_module._LOCAL_WORKER_READY.clear()


def _page_transfers(request_id: int, page_index: int) -> list[RequestTransfers]:
    """One page of every layer group, which is how a block is published."""
    return [
        RequestTransfers(
            request_id,
            [PageTransfer(BLOCK_HASH, group_id, page_index) for group_id in range(NUM_GROUPS)],
        )
    ]


def save_page(worker: MooncakeStoreConnectorWorker, request_id: int, page_index: int) -> None:
    """Publish a page and wait for the background save thread to drain it."""
    worker.bind_connector_meta(MooncakeStoreMetadata(saves=_page_transfers(request_id, page_index)))
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


def load_page(worker: MooncakeStoreConnectorWorker, request_id: int, page_index: int) -> None:
    """Pull a page into `page_index`, as the executor does before a forward pass."""
    # The load lands bytes outside this stream's ordering, so work the test
    # queued against the KV buffer has to have retired first.
    torch.cuda.current_stream().synchronize()
    worker.bind_connector_meta(MooncakeStoreMetadata(loads=_page_transfers(request_id, page_index)))
    worker.start_load_kv(torch.cuda.current_stream())
    torch.cuda.current_stream().synchronize()


def test_a_saved_page_comes_back_byte_identical(mooncake_pool):
    """Save a page, lose it, load it back, and compare the bytes.

    The buffer is zeroed between the two halves, so a load that transfers
    nothing fails instead of passing on leftover memory. The gather and
    scatter through pinned host slots have to leave the page as it was.
    """
    buffer = _device_buffer(filled=True)
    page_index = 3
    expected = _snapshot(buffer, page_index)

    with open_worker(mooncake_pool, buffer=buffer) as worker:
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

    with open_worker(mooncake_pool, buffer=writer_buffer) as writer:
        save_page(writer, request_id=1, page_index=writer_page)

    reader_buffer = _device_buffer(filled=False)
    with open_worker(mooncake_pool, buffer=reader_buffer) as reader:
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
    with open_worker(mooncake_pool, buffer=buffer) as rank_zero:
        save_page(rank_zero, request_id=1, page_index=0)
        assert rank_zero.count_prefix_hit([BLOCK_HASH]) == 0, (
            "A block with only rank 0's shard in the pool was offered for reuse. "
            "Rank 1 would be asked to load a page it never saved."
        )

    monkeypatch.setattr(worker_module, "mpi_rank", lambda: 1)
    with open_worker(mooncake_pool, buffer=buffer) as rank_one:
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
    with open_worker(mooncake_pool, buffer=buffer, model_key="model-a") as writer:
        save_page(writer, request_id=1, page_index=0)

    other_buffer = _device_buffer(filled=False)
    with open_worker(mooncake_pool, buffer=other_buffer, model_key="model-b") as reader:
        assert reader.count_prefix_hit([BLOCK_HASH]) == 0, (
            "A block saved under one model key was offered to a different model."
        )

    with open_worker(mooncake_pool, buffer=other_buffer, model_key="model-a") as same_model:
        assert same_model.count_prefix_hit([BLOCK_HASH]) == 1, (
            "The block is gone from the pool under the key it was saved with, "
            "so the miss asserted above says nothing about model keys."
        )
