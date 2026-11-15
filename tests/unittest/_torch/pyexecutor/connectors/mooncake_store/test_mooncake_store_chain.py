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
"""Unit tests for mooncake-store chain mode: the pool behind the host tier.

`HostChain` is driven directly, against a fake store and a fake KV cache
manager that stand in for the Mooncake client and the KVCacheManagerV2
host-drop hook / ring. Host pages are plain ctypes buffers, so the bytes a put
reads can be checked against the page they came from. Runs without a Mooncake
installation and without a GPU.
"""

import ctypes
import hashlib
import threading
import time
from array import array
from types import SimpleNamespace

import pytest

from tensorrt_llm._torch.pyexecutor.connectors.kv_cache_layout import (
    KvCacheBufferRef,
    KvCacheLayerGroupLayout,
    KvCacheLayout,
    KvCacheRegion,
)
from tensorrt_llm._torch.pyexecutor.connectors.mooncake_store import chain as chain_module
from tensorrt_llm._torch.pyexecutor.connectors.mooncake_store.chain import (
    HostChain,
    _cold_segments,
    chain_enabled,
)
from tensorrt_llm.runtime.kv_cache_manager_v2 import (
    ReuseScope,
    gen_multimodal_cache_key_tokens,
    sequence_to_blockchain_keys,
)

pytestmark = pytest.mark.cpu_only

TOKENS_PER_BLOCK = 4
#: Two hot pools of one layer group: the cold page is pool 0's slot, then pool 1's.
POOL0_BASE, POOL0_SLOT = 0x100000, 256
POOL1_BASE, POOL1_SLOT = 0x200000, 128
COLD_PAGE = POOL0_SLOT + POOL1_SLOT
#: The connector payload: 128 B at offset 64 of pool 0's slot, then pool 1's whole slot.
PAYLOAD_RANGES = [(64, 128), (POOL0_SLOT, 128)]

CHAIN_SWITCHES = (
    "TRTLLM_MOONCAKE_CHAIN",
    "TRTLLM_MOONCAKE_CHAIN_ASYNC",
    "TRTLLM_MOONCAKE_CHAIN_COPY",
    "TRTLLM_MOONCAKE_CHAIN_RING_GIB",
    "TRTLLM_MOONCAKE_CHAIN_STAGING_GIB",
    "TRTLLM_MOONCAKE_CHAIN_MAX_KEYS",
    "TRTLLM_MOONCAKE_PUT_THREADS",
    "TRTLLM_MOONCAKE_EXCLUSIVE",
    "TRTLLM_MOONCAKE_EXCL_DELETE_DELAY_S",
    "TRTLLM_MOONCAKE_EXCL_DELETE_RETRY_S",
    "TRTLLM_MOONCAKE_EXCL_DELETE_TRIES",
)


# ---- fakes ----


class FakeStore:
    """Copies what a put points at, so the bytes written to the pool can be checked."""

    def __init__(self):
        self.lock = threading.Lock()
        self.objects = {}
        self.events = []
        self.registered = []
        self.unregistered = []
        self.exist_calls = 0
        #: key -> codes the next removes of that key return, in order (then 0).
        self.remove_codes = {}
        #: When set, a remove waits for it, so a put can race it.
        self.remove_gate = None
        self.remove_started = threading.Event()

    def register_buffer(self, address, size):
        self.registered.append((address, size))
        return 0

    def unregister_buffer(self, address):
        self.unregistered.append(address)
        return 0

    def batch_is_exist(self, keys):
        with self.lock:
            self.exist_calls += 1
            return [1 if key in self.objects else 0 for key in keys]

    def batch_put_from_multi_buffers(self, keys, addresses, sizes, *options):
        results = []
        for key, addrs, lens in zip(keys, addresses, sizes):
            payload = b"".join(ctypes.string_at(a, n) for a, n in zip(addrs, lens))
            with self.lock:
                self.objects[key] = payload
                self.events.append(("put", key))
            results.append(sum(lens))
        return results

    def batch_remove(self, keys):
        with self.lock:
            self.events.append(("remove_start", tuple(keys)))
        self.remove_started.set()
        if self.remove_gate is not None:
            self.remove_gate.wait(5)
        codes = []
        with self.lock:
            for key in keys:
                pending = self.remove_codes.get(key)
                code = pending.pop(0) if pending else 0
                codes.append(code)
                if code == 0:
                    self.objects.pop(key, None)
            self.events.append(("remove_end", tuple(keys)))
        return codes


class FakeNamespace:
    def key(self, block_hash):
        return "pool/" + block_hash.hex()

    def group(self, block_hash):
        return block_hash


class FakeManagerImpl:
    """The KVCacheManagerV2 surface the chain uses: pool layout, hook, ring."""

    def __init__(self):
        pools = [
            SimpleNamespace(pool_index=0, base_address=POOL0_BASE, slot_bytes=POOL0_SLOT),
            SimpleNamespace(pool_index=1, base_address=POOL1_BASE, slot_bytes=POOL1_SLOT),
        ]
        self.pool_group_descs = [
            SimpleNamespace(
                pools=pools,
                slot_desc=SimpleNamespace(variants=[SimpleNamespace(layer_group_id=0)]),
            )
        ]
        self.hooks = []
        self.ring_calls = []
        self.ring = None
        self.slot_bytes = 512
        self.lock = threading.Lock()
        self.pending = []
        self.released = []
        self.release_event = threading.Event()

    def set_host_drop_hook(self, callback):
        self.hooks.append(callback)

    def host_drop_ring(self, ring_bytes, prefer_kernel):
        self.ring_calls.append((ring_bytes, prefer_kernel))
        if self.ring is None:
            self.ring = (ctypes.c_uint8 * (self.slot_bytes * 8))()
        return ctypes.addressof(self.ring), self.slot_bytes, 8, prefer_kernel

    def ring_slot_address(self, slot):
        return ctypes.addressof(self.ring) + slot * self.slot_bytes

    def push(self, records):
        with self.lock:
            self.pending.extend(records)

    def drain_host_drops(self, max_records):
        with self.lock:
            out, self.pending = self.pending[:max_records], self.pending[max_records:]
        return out

    def release_host_drop_slots(self, slots):
        with self.lock:
            self.released.extend(slots)
        self.release_event.set()

    def host_drop_stats(self):
        return {"handed": 0, "ring_full": 0, "errors": 0}


def make_layout():
    def region(base, size, slot_bytes):
        return KvCacheRegion(
            base=base,
            size=size,
            stride=slot_bytes,
            num_slots=4,
            buffers=(KvCacheBufferRef(layer_id=0, role="key"),),
        )

    return KvCacheLayout(
        tokens_per_block=TOKENS_PER_BLOCK,
        groups=(
            KvCacheLayerGroupLayout(
                layer_group_id=0,
                layer_ids=(0,),
                window_size=None,
                regions=(
                    region(POOL0_BASE + 64, 128, POOL0_SLOT),
                    region(POOL1_BASE, 128, POOL1_SLOT),
                ),
            ),
        ),
    )


def make_worker(store):
    return SimpleNamespace(
        _store=store,
        _rank=0,
        _config=SimpleNamespace(transfer_batch_size=8),
        _stats_interval=None,
        _namespaces={0: FakeNamespace()},
        _group_blocks=False,
        _put_config=None,
        _soft_pin_ttl_ms=0,
    )


def host_page(fill):
    """A cold page in host memory: every byte of pool 0's slot `fill`, pool 1's `fill + 1`."""
    page = (ctypes.c_uint8 * COLD_PAGE)()
    ctypes.memset(page, fill, POOL0_SLOT)
    ctypes.memset(ctypes.addressof(page) + POOL0_SLOT, fill + 1, POOL1_SLOT)
    return page


def expected_payload(page_bytes):
    return b"".join(page_bytes[offset : offset + size] for offset, size in PAYLOAD_RANGES)


def wait_until(predicate, timeout=10.0):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            return False
        time.sleep(0.01)
    return True


@pytest.fixture(autouse=True)
def clean_switches(monkeypatch):
    for name in CHAIN_SWITCHES + ("TRTLLM_MOONCAKE_KEYS_PY",):
        monkeypatch.delenv(name, raising=False)
    # A small staging buffer: the default budget is GiBs.
    monkeypatch.setenv("TRTLLM_MOONCAKE_CHAIN_STAGING_GIB", "0.0001")


@pytest.fixture
def make_chain():
    """Build chains and stop every thread they started before the test's call phase ends."""
    chains = []

    def build(store=None, impl=None):
        store = store or FakeStore()
        impl = impl or FakeManagerImpl()
        manager = SimpleNamespace(impl=impl, kv_cache_map={})
        chain = HostChain(make_worker(store), manager, make_layout())
        chains.append(chain)
        return chain, store, impl, manager

    def stop_all():
        for chain in chains:
            chain.shutdown()
            for name in ("_deleter", "_poller"):
                thread = getattr(chain, name, None)
                if thread is not None:
                    thread.join(timeout=10)
        chains.clear()

    build.stop_all = stop_all
    yield build
    stop_all()


# ---- switches ----


def test_chain_switch_is_off_by_default(monkeypatch):
    assert chain_enabled() is False
    monkeypatch.setenv("TRTLLM_MOONCAKE_CHAIN", "1")
    assert chain_enabled() is True
    monkeypatch.setenv("TRTLLM_MOONCAKE_CHAIN", "0")
    assert chain_enabled() is False


def test_defaults_are_synchronous_shared_pool(make_chain):
    chain, store, impl, _ = make_chain()
    assert chain.exclusive is False
    assert chain._async is False
    assert chain._num_put_threads == 1
    assert impl.hooks == [chain.on_host_drop], "synchronous hand-off installs the hook"
    assert impl.ring_calls == []
    assert chain._deleter is None
    make_chain.stop_all()
    # Shutdown removes the hook and hands the staging registration back.
    assert impl.hooks[-1] is None
    assert store.unregistered == [store.registered[0][0]]


def test_switches_configure_the_chain(monkeypatch, make_chain):
    monkeypatch.setenv("TRTLLM_MOONCAKE_CHAIN_ASYNC", "1")
    monkeypatch.setenv("TRTLLM_MOONCAKE_CHAIN_COPY", "hostfn")
    monkeypatch.setenv("TRTLLM_MOONCAKE_CHAIN_RING_GIB", "0.5")
    monkeypatch.setenv("TRTLLM_MOONCAKE_PUT_THREADS", "3")
    monkeypatch.setenv("TRTLLM_MOONCAKE_EXCLUSIVE", "1")
    monkeypatch.setenv("TRTLLM_MOONCAKE_EXCL_DELETE_DELAY_S", "7.5")
    monkeypatch.setenv("TRTLLM_MOONCAKE_EXCL_DELETE_RETRY_S", "0.25")
    monkeypatch.setenv("TRTLLM_MOONCAKE_EXCL_DELETE_TRIES", "9")
    monkeypatch.setenv("TRTLLM_MOONCAKE_CHAIN_MAX_KEYS", "123")
    chain, store, impl, _ = make_chain()
    assert chain._async is True
    assert impl.ring_calls == [(1 << 29, False)]
    assert impl.hooks == [], "the asynchronous hand-off does not install the hook"
    # The ring is registered with the store as one buffer.
    assert store.registered == [(ctypes.addressof(impl.ring), impl.slot_bytes * 8)]
    assert len(chain._threads) == 3
    assert chain.exclusive is True
    assert (chain._delete_delay_s, chain._delete_retry_s, chain._delete_max_tries) == (7.5, 0.25, 9)
    assert chain._max_keys == 123
    make_chain.stop_all()


@pytest.mark.parametrize("copy, kernel", [(None, True), ("kernel", True), ("hostfn", False)])
def test_ring_copy_switch(monkeypatch, make_chain, copy, kernel):
    monkeypatch.setenv("TRTLLM_MOONCAKE_CHAIN_ASYNC", "1")
    if copy is not None:
        monkeypatch.setenv("TRTLLM_MOONCAKE_CHAIN_COPY", copy)
    _, _, impl, _ = make_chain()
    assert impl.ring_calls == [(8 << 30, kernel)]
    make_chain.stop_all()


# ---- cold-page layout ----


def test_cold_segments_place_each_region_inside_the_cold_page():
    segments = _cold_segments(SimpleNamespace(impl=FakeManagerImpl()), make_layout())
    assert segments == {0: (PAYLOAD_RANGES, COLD_PAGE)}


def test_cold_segments_reject_a_region_outside_the_pools():
    layout = make_layout()
    bad = KvCacheLayout(
        tokens_per_block=TOKENS_PER_BLOCK,
        groups=(
            KvCacheLayerGroupLayout(
                layer_group_id=0,
                layer_ids=(0,),
                window_size=None,
                regions=(
                    layout.groups[0].regions[0],
                    KvCacheRegion(
                        base=0x900000,
                        size=128,
                        stride=128,
                        num_slots=4,
                        buffers=(KvCacheBufferRef(layer_id=0, role="value"),),
                    ),
                ),
            ),
        ),
    )
    with pytest.raises(ValueError, match="lies in none"):
        _cold_segments(SimpleNamespace(impl=FakeManagerImpl()), bad)


def test_cold_segments_reject_a_region_past_its_slot():
    bad = KvCacheLayout(
        tokens_per_block=TOKENS_PER_BLOCK,
        groups=(
            KvCacheLayerGroupLayout(
                layer_group_id=0,
                layer_ids=(0,),
                window_size=None,
                regions=(
                    KvCacheRegion(
                        base=POOL0_BASE + 192,
                        size=128,
                        stride=POOL0_SLOT,
                        num_slots=4,
                        buffers=(KvCacheBufferRef(layer_id=0, role="key"),),
                    ),
                ),
            ),
        ),
    )
    with pytest.raises(ValueError, match="runs past its pool slot"):
        _cold_segments(SimpleNamespace(impl=FakeManagerImpl()), bad)


# ---- synchronous hand-off ----


def test_dropped_page_is_put_with_its_payload_bytes(make_chain):
    chain, store, _, _ = make_chain()
    chain.apply_map([(b"radix-a", b"\xaa" * 8), (b"radix-b", b"\xbb" * 8)])
    page_a, page_b = host_page(0x10), host_page(0x40)
    chain.on_host_drop(
        [
            (b"radix-a", 0, TOKENS_PER_BLOCK, ctypes.addressof(page_a), COLD_PAGE),
            (b"radix-b", 0, TOKENS_PER_BLOCK, ctypes.addressof(page_b), COLD_PAGE),
        ]
    )
    # The hook copied the pages out: their host slots may be reused right away.
    ctypes.memset(page_a, 0, COLD_PAGE)
    ctypes.memset(page_b, 0, COLD_PAGE)
    make_chain.stop_all()
    assert store.objects == {
        "pool/" + (b"\xaa" * 8).hex(): expected_payload(bytes(host_page(0x10))),
        "pool/" + (b"\xbb" * 8).hex(): expected_payload(bytes(host_page(0x40))),
    }
    stats = chain._stats
    assert (stats["drops"], stats["handed"], stats["put_new"]) == (2, 2, 2)
    # Every staging slot is free again.
    assert len(chain._free) == chain._staging._num_slots


def test_drops_that_cannot_be_named_are_not_written(make_chain):
    chain, store, _, _ = make_chain()
    chain.apply_map([(b"radix-a", b"\xaa" * 8)])
    page = host_page(0x10)
    address = ctypes.addressof(page)
    chain.on_host_drop(
        [
            (b"radix-a", 0, TOKENS_PER_BLOCK - 1, address, COLD_PAGE),  # partial block
            (b"radix-unmapped", 0, TOKENS_PER_BLOCK, address, COLD_PAGE),  # never computed here
            (b"radix-a", 5, TOKENS_PER_BLOCK, address, COLD_PAGE),  # unknown layer group
            (b"radix-a", 0, TOKENS_PER_BLOCK, address, COLD_PAGE - 1),  # smaller than a cold page
        ]
    )
    make_chain.stop_all()
    assert store.objects == {}
    stats = chain._stats
    assert (stats["drops"], stats["handed"]) == (4, 0)
    assert (stats["partial"], stats["unmapped"], stats["nogroup"]) == (1, 1, 2)


def test_shared_pool_skips_a_key_that_is_present(make_chain):
    chain, store, _, _ = make_chain()
    key = "pool/" + (b"\xaa" * 8).hex()
    store.objects[key] = b"already there"
    chain.apply_map([(b"radix-a", b"\xaa" * 8)])
    page = host_page(0x10)
    chain.on_host_drop([(b"radix-a", 0, TOKENS_PER_BLOCK, ctypes.addressof(page), COLD_PAGE)])
    make_chain.stop_all()
    assert store.objects[key] == b"already there"
    assert store.exist_calls == 1
    assert (chain._stats["put_present"], chain._stats["put_new"]) == (1, 0)


def test_hook_never_raises_into_eviction(make_chain):
    chain, store, _, _ = make_chain()
    chain.on_host_drop([("not", "a", "record")])
    chain.on_host_drop(None)
    make_chain.stop_all()
    assert chain._stats["hook_errors"] == 2
    assert store.objects == {}


def test_staging_exhaustion_counts_ring_full(make_chain):
    chain, store, _, _ = make_chain()
    chain.apply_map([(b"radix-a", b"\xaa" * 8)])
    with chain._free_lock:
        held, chain._free = chain._free, []
    page = host_page(0x10)
    chain.on_host_drop([(b"radix-a", 0, TOKENS_PER_BLOCK, ctypes.addressof(page), COLD_PAGE)])
    with chain._free_lock:
        chain._free = held
    make_chain.stop_all()
    assert chain._stats["ringfull"] == 1
    assert store.objects == {}


# ---- asynchronous hand-off (ring) ----


def test_ring_records_are_put_from_the_ring_and_their_slots_released(monkeypatch, make_chain):
    monkeypatch.setenv("TRTLLM_MOONCAKE_CHAIN_ASYNC", "1")
    chain, store, impl, _ = make_chain()
    chain.apply_map([(b"radix-a", b"\xaa" * 8)])
    page = host_page(0x21)
    ctypes.memmove(impl.ring_slot_address(3), page, COLD_PAGE)
    impl.push(
        [
            (b"radix-a", 0, TOKENS_PER_BLOCK, 3, COLD_PAGE),
            (b"radix-unmapped", 0, TOKENS_PER_BLOCK, 5, COLD_PAGE),
            (b"radix-a", 0, 1, 6, COLD_PAGE),
        ]
    )
    assert wait_until(lambda: sorted(impl.released) == [3, 5, 6])
    make_chain.stop_all()
    assert store.objects == {"pool/" + (b"\xaa" * 8).hex(): expected_payload(bytes(page))}
    stats = chain._stats
    assert (stats["drops"], stats["handed"], stats["unmapped"], stats["partial"]) == (3, 1, 1, 1)


def test_ring_slot_is_released_after_a_failed_put(monkeypatch, make_chain):
    monkeypatch.setenv("TRTLLM_MOONCAKE_CHAIN_ASYNC", "1")
    store = FakeStore()

    def failing_put(*args, **kwargs):
        raise RuntimeError("transfer failed")

    store.batch_put_from_multi_buffers = failing_put
    chain, _, impl, _ = make_chain(store=store)
    chain.apply_map([(b"radix-a", b"\xaa" * 8)])
    impl.push([(b"radix-a", 0, TOKENS_PER_BLOCK, 2, COLD_PAGE)])
    assert wait_until(lambda: impl.released == [2])
    make_chain.stop_all()
    assert chain._stats["put_failed"] == 1


# ---- exclusive pool ----


@pytest.fixture
def exclusive(monkeypatch):
    monkeypatch.setenv("TRTLLM_MOONCAKE_EXCLUSIVE", "1")
    monkeypatch.setenv("TRTLLM_MOONCAKE_EXCL_DELETE_DELAY_S", "0.05")
    monkeypatch.setenv("TRTLLM_MOONCAKE_EXCL_DELETE_RETRY_S", "0.05")
    monkeypatch.setenv("TRTLLM_MOONCAKE_EXCL_DELETE_TRIES", "3")


def removes(store):
    return [event[1] for event in store.events if event[0] == "remove_start"]


def test_shared_pool_keeps_loaded_blocks(make_chain):
    chain, store, _, _ = make_chain()
    chain.on_loaded(["pool/k1"])
    make_chain.stop_all()
    assert removes(store) == []
    assert chain._stats["excl_loaded"] == 0


def test_exclusive_pool_removes_a_loaded_block_after_the_delay(exclusive, make_chain):
    chain, store, _, _ = make_chain()
    store.objects["pool/k1"] = b"x"
    loaded_at = time.monotonic()
    chain.on_loaded(["pool/k1"])
    assert wait_until(lambda: chain._stats["excl_deleted"] == 1)
    assert time.monotonic() - loaded_at >= 0.05
    make_chain.stop_all()
    assert removes(store) == [("pool/k1",)]
    assert "pool/k1" not in store.objects


def test_exclusive_pool_retries_a_failed_remove(exclusive, make_chain):
    chain, store, _, _ = make_chain()
    store.remove_codes["pool/k1"] = [-706, -706]
    chain.on_loaded(["pool/k1"])
    assert wait_until(lambda: chain._stats["excl_deleted"] == 1)
    make_chain.stop_all()
    assert len(removes(store)) == 3
    assert chain._stats["excl_delete_retry"] == 2
    assert chain._stats["excl_delete_failed"] == 0


def test_exclusive_pool_gives_up_after_the_last_try(exclusive, make_chain):
    chain, store, _, _ = make_chain()
    store.remove_codes["pool/k1"] = [-706] * 10
    chain.on_loaded(["pool/k1"])
    assert wait_until(lambda: chain._stats["excl_delete_failed"] == 1)
    make_chain.stop_all()
    assert len(removes(store)) == 3
    assert dict(chain._delete_codes) == {-706: 1}


def test_exclusive_pool_reeviction_cancels_the_pending_remove(monkeypatch, make_chain):
    monkeypatch.setenv("TRTLLM_MOONCAKE_EXCLUSIVE", "1")
    monkeypatch.setenv("TRTLLM_MOONCAKE_EXCL_DELETE_DELAY_S", "0.3")
    chain, store, _, _ = make_chain()
    key = "pool/" + (b"\xaa" * 8).hex()
    chain.apply_map([(b"radix-a", b"\xaa" * 8)])
    chain.on_loaded([key])
    # Evicted again from the host tier before the remove is due: the fresh copy stays.
    page = host_page(0x10)
    chain.on_host_drop([(b"radix-a", 0, TOKENS_PER_BLOCK, ctypes.addressof(page), COLD_PAGE)])
    assert wait_until(lambda: chain._stats["excl_delete_canceled"] == 1)
    make_chain.stop_all()
    assert removes(store) == []
    assert key in store.objects


def test_exclusive_pool_puts_without_an_existence_check(exclusive, make_chain):
    chain, store, _, _ = make_chain()
    chain.apply_map([(b"radix-a", b"\xaa" * 8)])
    page = host_page(0x10)
    chain.on_host_drop([(b"radix-a", 0, TOKENS_PER_BLOCK, ctypes.addressof(page), COLD_PAGE)])
    make_chain.stop_all()
    assert store.exist_calls == 0
    assert chain._stats["put_new"] == 1


def test_exclusive_put_waits_for_an_in_flight_remove_of_the_same_key(exclusive, make_chain):
    chain, store, _, _ = make_chain()
    key = "pool/" + (b"\xaa" * 8).hex()
    store.objects[key] = b"old copy"
    store.remove_gate = threading.Event()
    chain.on_loaded([key])
    assert store.remove_started.wait(5)
    # The remove is in flight; the same block is evicted again and put.
    chain.apply_map([(b"radix-a", b"\xaa" * 8)])
    page = host_page(0x10)
    chain.on_host_drop([(b"radix-a", 0, TOKENS_PER_BLOCK, ctypes.addressof(page), COLD_PAGE)])
    time.sleep(0.2)
    assert ("put", key) not in store.events, "the put must wait for the remove"
    store.remove_gate.set()
    assert wait_until(lambda: ("put", key) in store.events)
    make_chain.stop_all()
    order = [event for event in store.events if event[0] in ("remove_end", "put")]
    assert order == [("remove_end", (key,)), ("put", key)]
    # The page is in the pool after the race: the remove did not take the fresh copy.
    assert store.objects[key] == expected_payload(bytes(host_page(0x10)))


# ---- radix key mapping ----


def reference_keys(tokens, reuse_scope):
    keys = sequence_to_blockchain_keys(TOKENS_PER_BLOCK, reuse_scope, list(tokens))
    next(keys)  # the root
    # The generator also keys a trailing partial block; only full blocks are mapped.
    return [bytes(key) for _, key in keys][: len(tokens) // TOKENS_PER_BLOCK]


def python_keys(tokens, root):
    """SHA-256 of the parent digest, then the block's tokens as little-endian int32."""
    out, prev = [], root
    for start in range(0, len(tokens) - TOKENS_PER_BLOCK + 1, TOKENS_PER_BLOCK):
        digest = hashlib.sha256(prev)
        digest.update(array("i", tokens[start : start + TOKENS_PER_BLOCK]).tobytes())
        prev = digest.digest()
        out.append(prev)
    return out


@pytest.mark.parametrize("keys_py", [False, True])
def test_map_blocks_matches_the_cpp_key_generator(monkeypatch, make_chain, keys_py):
    monkeypatch.setattr(chain_module, "_KEYS_PY", keys_py)
    chain, _, _, manager = make_chain()
    scope = ReuseScope()
    manager.kv_cache_map[7] = SimpleNamespace(reuse_scope=scope)
    tokens = [(i * 7919) % 200003 for i in range(10 * TOKENS_PER_BLOCK + 3)]
    hashes = [bytes([b]) * 8 for b in range(10)]
    expected = reference_keys(tokens, scope)
    assert len(expected) == 10

    # Incremental: each call returns only the blocks computed since the last one.
    first = chain.map_blocks(7, tokens, hashes, 4)
    second = chain.map_blocks(7, tokens, hashes, 4)
    rest = chain.map_blocks(7, tokens, hashes, 100)
    assert second == []
    assert first + rest == list(zip(expected, hashes))
    if keys_py:
        root = bytes(next(sequence_to_blockchain_keys(TOKENS_PER_BLOCK, scope, []))[1])
        assert python_keys(tokens, root) == expected

    # A second request over the same blocks maps to the same radix keys.
    manager.kv_cache_map[8] = SimpleNamespace(reuse_scope=scope)
    assert chain.map_blocks(8, tokens, hashes, 6) == list(zip(expected, hashes))[:6]
    if keys_py:
        assert chain._stats["map_rev_blocks"] == 6

    # After forget, a request maps again from block 0.
    chain.forget(7)
    assert chain.map_blocks(7, tokens, hashes, 2) == list(zip(expected, hashes))[:2]
    make_chain.stop_all()


def test_python_keys_fall_back_for_tokens_that_are_not_int32(monkeypatch, make_chain):
    monkeypatch.setattr(chain_module, "_KEYS_PY", True)
    chain, _, _, manager = make_chain()
    scope = ReuseScope()
    manager.kv_cache_map[1] = SimpleNamespace(reuse_scope=scope)
    # Multimodal digest tokens are not plain int32 ids: the C++ generator keys them.
    digest = bytes(range(32))
    tokens = [1, 2, 3, 4] + list(gen_multimodal_cache_key_tokens(1000, digest, TOKENS_PER_BLOCK))
    hashes = [b"\x01" * 8, b"\x02" * 8]
    pairs = chain.map_blocks(1, tokens, hashes, 2)
    assert chain._stats["map_fallback"] == 1
    assert pairs == list(zip(reference_keys(tokens, scope), hashes))
    make_chain.stop_all()


def test_map_blocks_skips_a_request_without_a_kv_cache(make_chain):
    chain, _, _, _ = make_chain()
    assert chain.map_blocks(99, [1, 2, 3, 4], [b"\x01" * 8], 1) == []
    assert chain._stats["unmapped_requests"] == 1
    make_chain.stop_all()


def test_apply_map_keeps_the_newest_keys(monkeypatch, make_chain):
    monkeypatch.setenv("TRTLLM_MOONCAKE_CHAIN_MAX_KEYS", "2")
    chain, _, _, _ = make_chain()
    chain.apply_map([(b"k1", b"h1"), (b"k2", b"h2")])
    chain.apply_map([(b"k1", b"h1"), (b"k3", b"h3")])  # k1 refreshed, k2 is the oldest
    assert list(chain._keys) == [b"k1", b"k3"]
    make_chain.stop_all()
