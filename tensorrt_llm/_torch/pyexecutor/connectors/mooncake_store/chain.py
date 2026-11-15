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
"""Chain mode: the pool as the victim tier of the KV cache manager's host tier.

Switched on by `TRTLLM_MOONCAKE_CHAIN=1`. The cache runs GPU -> host tier ->
pool: GPU evictees fall to the host tier as without a connector, and a page the
host tier drops is copied out of host memory and written to the pool, instead of
the pool being written on compute. Lookups are unchanged (GPU + host as one local
match, then the pool past it), and a pool hit loads into GPU pages as before.

Synchronous hand-off (default): the KV cache manager calls `on_host_drop` from
inside its eviction, before the dropped slots are freed, with one record per
committed page: `(radix key, layer group, tokens in page, host address, host
bytes)`. The host copy is the cold-page form, every hot pool's slot back to back
in pool order, so a page's connector payload (its regions in region order) is a
fixed list of byte ranges inside it. The page is copied whole into a registered
staging slot on the executor thread and written to the pool from that slot by a
background thread.

Asynchronous hand-off (`TRTLLM_MOONCAKE_CHAIN_ASYNC=1`): no hook runs on the
executor thread. The KV cache manager copies each dropped page into a pinned ring
it owns, on its own CUDA stream (the batched page-copy kernel, or a host-callback
memcpy), makes the page's ready event the copy's completion so the slot's next
owner waits for it in-stream, and queues the record. A poll thread here takes
records whose copies have completed, puts them from the ring, and gives the ring
slots back.

The radix key is the manager's SHA-256 block key, while the pool is keyed by the
connector's own hash chain. The scheduler maps one to the other for every block
a request computes (`map_blocks`), so a dropped page whose key was never mapped
here, or was forgotten, is not written.

Exclusive pool (`TRTLLM_MOONCAKE_EXCLUSIVE=1`): a block lives in one tier. A pool
hit that landed in GPU pages has its pool copy removed after a delay (past the
master's KV lease), every host-tier eviction is put without an existence check,
and a re-eviction inside the delay cancels the pending removal.

Only the default cold-page layout is supported.
"""

import collections
import ctypes
import hashlib
import os
import queue
import sys
import threading
import time
import traceback
from array import array
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence, Tuple

from tensorrt_llm.logger import logger

from .staging import HostStagingPool

__all__ = ["HostChain", "chain_enabled"]

#: Switches chain mode on.
CHAIN_ENV = "TRTLLM_MOONCAKE_CHAIN"


def _env_flag(name: str, default: bool = False) -> bool:
    return os.environ.get(name, "1" if default else "0") == "1"


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, "") or default)


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, "") or default)


# Map blocks by hashing their tokens in Python, bit-identical to the C++ key generator (SHA-256 of the parent
# digest, then the block's tokens as little-endian int32), and reuse the radix key of a block an earlier request
# already mapped. A request whose tokens are not plain non-negative int32 ids falls back to the C++ generator.
_KEYS_PY = (
    _env_flag("TRTLLM_MOONCAKE_KEYS_PY") and sys.byteorder == "little" and array("i").itemsize == 4
)


def chain_enabled() -> bool:
    """Whether chain mode is switched on for this process."""
    return _env_flag(CHAIN_ENV)


def _cold_segments(manager, layout) -> Dict[int, Tuple[List[Tuple[int, int]], int]]:
    """Per layer group: the payload's `(offset, size)` ranges inside a cold page, and the page's size.

    The default cold-page codec lays every pool of the group's hot pool group
    back to back at the offset of the pools before it. A region of the layout
    is a byte range of one of those pools' slots, so it lands at that pool's
    offset plus its own offset within the slot.
    """
    pools_by_group: Dict[int, List[Tuple[int, int]]] = {}
    for pool_group in manager.impl.pool_group_descs:
        pools = sorted(
            ((int(p.pool_index), int(p.base_address), int(p.slot_bytes)) for p in pool_group.pools)
        )
        for variant in pool_group.slot_desc.variants:
            pools_by_group[int(variant.layer_group_id)] = [(base, size) for _, base, size in pools]

    segments: Dict[int, Tuple[List[Tuple[int, int]], int]] = {}
    for group in layout.groups:
        pools = pools_by_group[group.layer_group_id]
        offsets = []
        cold = 0
        for _, slot_bytes in pools:
            offsets.append(cold)
            cold += slot_bytes
        ranges: List[Tuple[int, int]] = []
        for region in group.regions:
            for (base, slot_bytes), cold_offset in zip(pools, offsets):
                if base <= region.base < base + slot_bytes:
                    within = region.base - base
                    if within + region.size > slot_bytes:
                        raise ValueError(
                            f"chain: region at {region.base:#x} of layer group "
                            f"{group.layer_group_id} runs past its pool slot"
                        )
                    ranges.append((cold_offset + within, int(region.size)))
                    break
            else:
                raise ValueError(
                    f"chain: region at {region.base:#x} of layer group {group.layer_group_id} "
                    "lies in none of its pool group's pools"
                )
        if sum(size for _, size in ranges) != group.bytes_per_page:
            raise ValueError(
                f"chain: layer group {group.layer_group_id} maps "
                f"{sum(size for _, size in ranges)} B of a {group.bytes_per_page} B page"
            )
        segments[group.layer_group_id] = (ranges, cold)
    return segments


class HostChain:
    """Writes the pages the host tier drops to the pool."""

    _hook_errors = 0

    def __init__(self, worker, manager, layout):
        self._worker = worker
        self._manager = manager
        self._store = worker._store
        self._rank = worker._rank
        self._tokens_per_block = int(layout.tokens_per_block)
        self._segments = _cold_segments(manager, layout)
        slot_bytes = max(cold for _, cold in self._segments.values())
        self._async = _env_flag("TRTLLM_MOONCAKE_CHAIN_ASYNC")
        self._stop = False
        self._ring = None
        self._staging = None
        num_slots = 0
        if not self._async:
            budget = int(_env_float("TRTLLM_MOONCAKE_CHAIN_STAGING_GIB", 4) * (1 << 30))
            num_slots = max(64, budget // slot_bytes)
            self._staging = HostStagingPool(
                slot_bytes=slot_bytes, num_slots=num_slots, store=self._store, label="chain"
            )
        self._free: List[int] = list(range(num_slots))
        self._free_lock = threading.Lock()
        self._batch_size = int(worker._config.transfer_batch_size)

        # Radix key -> connector block hash, oldest first; owned by the executor thread.
        self._keys: "OrderedDict[bytes, bytes]" = OrderedDict()
        self._max_keys = _env_int("TRTLLM_MOONCAKE_CHAIN_MAX_KEYS", 4000000)
        # Request id -> blocks already mapped.
        self._mapped: Dict[int, int] = {}
        # One lazy radix-key iterator per request, advanced only over the blocks computed since the last call, so a
        # request's tokens are hashed once rather than from token 0 on every step.
        # request id -> [iterator, tokens it covers, next block ordinal it yields]
        self._key_iters: Dict[int, list] = {}
        # Scope root + connector hash -> radix key, for every block mapped here (bounded like _keys).
        self._rev: "OrderedDict[bytes, bytes]" = OrderedDict()

        self._queue: "queue.Queue" = queue.Queue()
        self._stats_lock = threading.Lock()
        self._stats = self._zero_stats()
        self._interval = worker._stats_interval
        self._last_log = time.monotonic()
        self._log_lock = threading.Lock()

        from tensorrt_llm.runtime.kv_cache_manager_v2 import sequence_to_blockchain_keys

        self._blockchain_keys = sequence_to_blockchain_keys
        # Several put threads drain the shared queue, so pool puts keep up with evictions and the ring does not fill.
        self._num_put_threads = max(1, _env_int("TRTLLM_MOONCAKE_PUT_THREADS", 1))
        self._threads = [
            threading.Thread(
                target=self._drain, name=f"mooncake-store-chain-{self._rank}-{i}", daemon=True
            )
            for i in range(self._num_put_threads)
        ]
        for thread in self._threads:
            thread.start()
        if self._async:
            ring_gib = _env_float("TRTLLM_MOONCAKE_CHAIN_RING_GIB", 8)
            prefer_kernel = os.environ.get("TRTLLM_MOONCAKE_CHAIN_COPY", "kernel") != "hostfn"
            addr, ring_slot, ring_n, kernel = manager.impl.host_drop_ring(
                int(ring_gib * (1 << 30)), prefer_kernel
            )
            status = self._store.register_buffer(addr, ring_slot * ring_n)
            if status != 0:
                raise RuntimeError(
                    f"chain: register_buffer of the host-drop ring failed with status {status}"
                )
            self._ring = (int(addr), int(ring_slot), int(ring_n))
            self._poller = threading.Thread(
                target=self._ring_poll, name=f"mooncake-store-chain-poll-{self._rank}", daemon=True
            )
            self._poller.start()
            hand_off = (
                f"asynchronous hand-off, ring {ring_n} x {ring_slot} B, "
                f"copy={'kernel' if kernel else 'hostfn'}"
            )
        else:
            manager.impl.set_host_drop_hook(self.on_host_drop)
            hand_off = f"synchronous hand-off, {num_slots} staging slots x {slot_bytes} B"

        self._exclusive = _env_flag("TRTLLM_MOONCAKE_EXCLUSIVE")
        self._delete_delay_s = _env_float("TRTLLM_MOONCAKE_EXCL_DELETE_DELAY_S", 3.0)
        self._delete_retry_s = _env_float("TRTLLM_MOONCAKE_EXCL_DELETE_RETRY_S", 1.5)
        self._delete_max_tries = _env_int("TRTLLM_MOONCAKE_EXCL_DELETE_TRIES", 4)
        self._pending_deletes: "collections.deque" = (
            collections.deque()
        )  # (due, loaded_at, key, tries)
        self._delete_lock = threading.Lock()
        self._delete_cv = threading.Condition(self._delete_lock)
        self._inflight_del: set = set()  # keys whose remove is in flight (under _delete_lock)
        self._reput: "OrderedDict[str, float]" = OrderedDict()  # key -> time last queued for a put
        self._put_codes: "collections.Counter" = collections.Counter()
        self._delete_codes: "collections.Counter" = collections.Counter()
        self._deleter = None
        if self._exclusive:
            self._deleter = threading.Thread(
                target=self._delete_loop,
                name=f"mooncake-store-excl-delete-{self._rank}",
                daemon=True,
            )
            self._deleter.start()
        logger.info(
            f"mooncake-store chain rank {self._rank}: pool written on host-tier eviction, not on compute; "
            f"{hand_off}, {self._num_put_threads} put threads"
            f"{f', exclusive (remove {self._delete_delay_s:g} s after a load)' if self._exclusive else ''}"
        )

    @property
    def exclusive(self) -> bool:
        """Whether a pool hit removes the block's pool copy (see the module docstring)."""
        return self._exclusive

    @staticmethod
    def _zero_stats() -> Dict[str, int]:
        return dict.fromkeys(
            (
                "drops",
                "handed",
                "partial",
                "unmapped",
                "nogroup",
                "ringfull",
                "put_new",
                "put_present",
                "put_failed",
                "mapped_blocks",
                "unmapped_requests",
                "hook_errors",
                "ring_errors",
                "excl_loaded",
                "excl_deleted",
                "excl_delete_retry",
                "excl_delete_failed",
                "excl_delete_canceled",
                "map_rev_blocks",
                "map_hashed_blocks",
                "map_fallback",
                "carry_hits",
                "carry_blocks",
                "carry_edited",
                "carry_miss",
                "carry_nocid",
            ),
            0,
        )

    def count(self, **deltas: int) -> None:
        """Add to this window's counters (thread-safe)."""
        with self._stats_lock:
            for name, value in deltas.items():
                self._stats[name] += value

    # ---- scheduler side (executor thread) ----

    def map_blocks(
        self, request_id: int, tokens: Sequence[int], hashes: Sequence[bytes], upto: int
    ) -> List[Tuple[bytes, bytes]]:
        """(radix key, connector hash) of every block below `upto` the request computed since the last call.

        Runs where the scheduler adapter lives (rank 0 under tensor parallelism, every owner under attention DP). The
        pairs travel in the iteration's connector metadata, so every rank, tensor-parallel followers included, applies
        them in `apply_map` and can name its own dropped shard in the pool.
        """
        start = self._mapped.get(request_id, 0)
        upto = min(upto, len(hashes), len(tokens) // self._tokens_per_block)
        if upto <= start:
            return []
        kv = self._manager.kv_cache_map.get(request_id)
        if kv is None:
            self.count(unmapped_requests=1)
            return []
        if _KEYS_PY:
            pairs = self._map_blocks_py(request_id, kv, tokens, hashes, start, upto)
            if pairs is not None:
                self._mapped[request_id] = upto
                return pairs
            self.count(map_fallback=1)
        tpb = self._tokens_per_block
        it = self._key_iters.get(request_id)
        if it is None or upto * tpb > it[1] or it[2] > start:
            # New request, or its tokens grew past the iterator's copy: one iterator over every full block known now
            # (a context request's whole prompt), positioned at block 0.
            full = len(tokens) // tpb
            keys = self._blockchain_keys(tpb, kv.reuse_scope, list(tokens[: full * tpb]))
            next(keys, None)  # the root labels no block
            it = [keys, full * tpb, 0]
            self._key_iters[request_id] = it
        pairs: List[Tuple[bytes, bytes]] = []
        keys = it[0]
        while it[2] < upto:
            _, key = next(keys)
            if it[2] >= start:
                pairs.append((bytes(key), hashes[it[2]]))
            it[2] += 1
        self._mapped[request_id] = upto
        return pairs

    def _map_blocks_py(
        self,
        request_id: int,
        kv,
        tokens: Sequence[int],
        hashes: Sequence[bytes],
        start: int,
        upto: int,
    ) -> Optional[List[Tuple[bytes, bytes]]]:
        """The pairs of blocks [start, upto), or None to use the C++ generator for this request."""
        tpb = self._tokens_per_block
        rev = self._rev
        pairs: List[Tuple[bytes, bytes]] = []
        it = self._key_iters.get(request_id)
        if it is not None and it[0] != "py":
            return None  # this request already runs on the C++ generator
        if it is None or it[2] > start:
            root = bytes(next(self._blockchain_keys(tpb, kv.reuse_scope, []))[1])
            it = ["py", root, 0, root]
            self._key_iters[request_id] = it
        root, nb, prev = it[1], it[2], it[3]
        # Known blocks: an earlier request mapped them (same scope and connector hash, so the same tokens).
        known = nb
        while nb < upto:
            key = rev.get(root + hashes[nb])
            if key is None:
                break
            if nb >= start:
                pairs.append((key, hashes[nb]))
            prev = key
            nb += 1
        if nb > known:
            self.count(map_rev_blocks=nb - known)
            it[2], it[3] = nb, prev
        if nb < upto:
            try:
                arr = array("i", tokens[nb * tpb : upto * tpb])
            except (TypeError, OverflowError, ValueError):
                arr = None
            if arr is None or len(arr) != (upto - nb) * tpb or min(arr) < 0:
                # Not plain int32 token ids (a negative id is a digest handle in C++): C++ generator.
                self._key_iters.pop(request_id, None)
                return None
            raw = memoryview(arr).cast("B")
            step = 4 * tpb
            sha256 = hashlib.sha256
            off = 0
            for j in range(nb, upto):
                h = sha256(prev)
                h.update(raw[off : off + step])
                off += step
                prev = h.digest()
                rev[root + hashes[j]] = prev
                if j >= start:
                    pairs.append((prev, hashes[j]))
            while len(rev) > self._max_keys:
                rev.popitem(last=False)
            self.count(map_hashed_blocks=upto - nb)
            it[2], it[3] = upto, prev
        return pairs

    def apply_map(self, pairs: Sequence[Tuple[bytes, bytes]]) -> None:
        """Remember radix key -> connector hash for blocks computed this iteration (every rank)."""
        for key, block_hash in pairs:
            self._keys[key] = block_hash
            self._keys.move_to_end(key)
        while len(self._keys) > self._max_keys:
            self._keys.popitem(last=False)
        self.count(mapped_blocks=len(pairs))
        # Also from here, so the counters show up even while nothing drops.
        self._maybe_log()

    def forget(self, request_id: int) -> None:
        """Drop a finished or reset request's mapping state."""
        self._mapped.pop(request_id, None)
        self._key_iters.pop(request_id, None)

    def _note_put(self, key: str) -> None:
        if not self._exclusive:
            return
        with self._delete_lock:
            self._reput[key] = time.monotonic()
            self._reput.move_to_end(key)

    # ---- KV cache manager side (executor thread, inside eviction) ----

    def on_host_drop(self, records) -> None:
        """Copy dropped host pages into staging and queue them for the pool.

        Never raises: an error escaping into the C++ eviction path is logged and counted instead.
        """
        try:
            self._on_host_drop(records)
        except BaseException as exc:
            HostChain._hook_errors += 1
            n = HostChain._hook_errors
            self.count(hook_errors=1)
            if n <= 20 or n % 10000 == 0:
                logger.error(
                    f"mooncake-store chain rank {self._rank}: on_host_drop failed (#{n}): "
                    f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
                )

    def _on_host_drop(self, records) -> None:
        handed = partial = unmapped = nogroup = ringfull = 0
        for key, layer_group_id, num_tokens, address, size in records:
            if num_tokens != self._tokens_per_block:
                partial += 1
                continue
            block_hash = self._keys.get(key)
            if block_hash is None:
                unmapped += 1
                continue
            entry = self._segments.get(layer_group_id)
            namespace = self._worker._namespaces.get(layer_group_id)
            if entry is None or namespace is None:
                nogroup += 1
                continue
            ranges, cold = entry
            if cold > size:
                nogroup += 1
                continue
            with self._free_lock:
                slot = self._free.pop() if self._free else None
            if slot is None:
                ringfull += 1
                continue
            base = self._staging.slot_address(slot)
            ctypes.memmove(base, address, cold)
            self._note_put(namespace.key(block_hash))
            self._queue.put(
                (
                    namespace.key(block_hash),
                    namespace.group(block_hash),
                    [base + offset for offset, _ in ranges],
                    [length for _, length in ranges],
                    slot,
                )
            )
            handed += 1
        self.count(
            drops=len(records),
            handed=handed,
            partial=partial,
            unmapped=unmapped,
            nogroup=nogroup,
            ringfull=ringfull,
        )

    # ---- put threads ----

    def _drain(self) -> None:
        from .worker import _replicate_config

        worker = self._worker
        while True:
            items = [self._queue.get()]
            if items[0] is None:
                return
            while len(items) < self._batch_size:
                try:
                    item = self._queue.get_nowait()
                except queue.Empty:
                    break
                if item is None:
                    self._queue.put(None)
                    break
                items.append(item)
            try:
                if self._exclusive:
                    # Every eviction is put; a key still present comes back as an error code and costs no space.
                    present = [0] * len(items)
                else:
                    present = self._store.batch_is_exist([item[0] for item in items])
                pending = [i for i, status in enumerate(present) if status != 1]
                new = failed = 0
                if pending:
                    if self._exclusive:
                        # A put of a key whose remove is in flight would be refused as present and then lose the
                        # page to the remove: wait (bounded) until those removes finish.
                        pending_keys = {items[i][0] for i in pending}
                        deadline = time.monotonic() + 5.0
                        with self._delete_cv:
                            while self._inflight_del & pending_keys and time.monotonic() < deadline:
                                self._delete_cv.wait(0.05)
                    args = (
                        [items[i][0] for i in pending],
                        [items[i][2] for i in pending],
                        [items[i][3] for i in pending],
                    )
                    if worker._group_blocks:
                        args += (
                            _replicate_config(
                                worker._soft_pin_ttl_ms, [items[i][1] for i in pending]
                            ),
                        )
                    elif worker._put_config is not None:
                        args += (worker._put_config,)
                    results = list(self._store.batch_put_from_multi_buffers(*args))
                    bad = [r for r in results if not isinstance(r, int) or r < 0]
                    failed = len(bad)
                    new = len(pending) - failed
                    if self._exclusive and failed:
                        with self._stats_lock:
                            self._put_codes.update(bad)
                self.count(put_new=new, put_failed=failed, put_present=len(items) - len(pending))
            except Exception as exc:
                logger.error(
                    f"mooncake-store chain rank {self._rank}: dropped a put batch: "
                    f"{type(exc).__name__}: {exc}"
                )
                self.count(put_failed=len(items))
            finally:
                ring_slots = [item[4][1] for item in items if isinstance(item[4], tuple)]
                if ring_slots:
                    self._release_ring(ring_slots)
                with self._free_lock:
                    self._free.extend(item[4] for item in items if not isinstance(item[4], tuple))
            self._maybe_log()

    # ---- exclusive pool: remove on load (load threads -> deleter thread) ----

    def on_loaded(self, keys: Sequence[str]) -> None:
        """Pages that landed in GPU pages from the pool: schedule their pool copies for removal."""
        if not self._exclusive or not keys:
            return
        now = time.monotonic()
        due = now + self._delete_delay_s
        with self._delete_lock:
            for key in keys:
                self._pending_deletes.append((due, now, key, 0))
        self.count(excl_loaded=len(keys))

    def _remove(self, keys: List[str]) -> List[int]:
        batch = getattr(self._store, "batch_remove", None)
        if batch is not None:
            try:
                out = batch(keys)
                if isinstance(out, (list, tuple)) and len(out) == len(keys):
                    return [int(c) if isinstance(c, int) else -1 for c in out]
            except Exception as exc:
                logger.error(
                    f"mooncake-store chain rank {self._rank}: batch_remove failed: "
                    f"{type(exc).__name__}: {exc}"
                )
        codes = []
        for key in keys:
            try:
                code = self._store.remove(key)
                codes.append(int(code) if isinstance(code, int) else -1)
            except Exception:
                codes.append(-1)
        return codes

    def _delete_loop(self) -> None:
        horizon = self._delete_delay_s + self._delete_max_tries * self._delete_retry_s + 5.0
        while not self._stop:
            now = time.monotonic()
            due_items = []
            with self._delete_lock:
                while (
                    self._pending_deletes
                    and self._pending_deletes[0][0] <= now
                    and len(due_items) < 4096
                ):
                    due_items.append(self._pending_deletes.popleft())
                while self._reput:
                    key, t = next(iter(self._reput.items()))
                    if now - t <= horizon:
                        break
                    self._reput.popitem(last=False)
                live = []
                canceled = 0
                for item in due_items:
                    reput_at = self._reput.get(item[2])
                    if reput_at is not None and reput_at >= item[1]:
                        canceled += (
                            1  # evicted again after the load: the put brings a fresh copy, keep it
                        )
                    else:
                        live.append(item)
                self._inflight_del.update(item[2] for item in live)
            if canceled:
                self.count(excl_delete_canceled=canceled)
            if not live:
                time.sleep(0.05)
                self._maybe_log()
                continue
            codes = self._remove([item[2] for item in live])
            with self._delete_lock:
                for item in live:
                    self._inflight_del.discard(item[2])
                self._delete_cv.notify_all()
            deleted = retry = failed = 0
            again = []
            for item, code in zip(live, codes):
                if code == 0:
                    deleted += 1
                    continue
                if item[3] + 1 < self._delete_max_tries:
                    retry += 1
                    again.append((now + self._delete_retry_s, item[1], item[2], item[3] + 1))
                else:
                    failed += 1
                    with self._stats_lock:
                        self._delete_codes[code] += 1
            if again:
                with self._delete_lock:
                    self._pending_deletes.extend(again)
            self.count(excl_deleted=deleted, excl_delete_retry=retry, excl_delete_failed=failed)
            self._maybe_log()

    # ---- asynchronous hand-off: ring poll thread ----

    def _release_ring(self, slots) -> None:
        try:
            self._manager.impl.release_host_drop_slots(list(slots))
        except Exception as exc:
            self.count(ring_errors=1)
            logger.error(
                f"mooncake-store chain rank {self._rank}: release of {len(slots)} ring slots failed: {exc}"
            )

    def _ring_poll(self) -> None:
        addr, ring_slot, _ = self._ring
        impl = self._manager.impl
        errors = 0
        while not self._stop:
            try:
                records = impl.drain_host_drops(4096)
            except Exception as exc:
                errors += 1
                self.count(ring_errors=1)
                if errors <= 20 or errors % 10000 == 0:
                    logger.error(
                        f"mooncake-store chain rank {self._rank}: ring drain failed (#{errors}): "
                        f"{type(exc).__name__}: {exc}"
                    )
                time.sleep(0.05)
                continue
            if not records:
                time.sleep(0.002)
                self._maybe_log()
                continue
            handed = partial = unmapped = nogroup = 0
            release = []
            try:
                for key, layer_group_id, num_tokens, slot, size in records:
                    if num_tokens != self._tokens_per_block:
                        partial += 1
                        release.append(slot)
                        continue
                    block_hash = self._keys.get(key)
                    if block_hash is None:
                        unmapped += 1
                        release.append(slot)
                        continue
                    entry = self._segments.get(layer_group_id)
                    namespace = self._worker._namespaces.get(layer_group_id)
                    if entry is None or namespace is None or entry[1] > size:
                        nogroup += 1
                        release.append(slot)
                        continue
                    ranges, _ = entry
                    base = addr + slot * ring_slot
                    self._note_put(namespace.key(block_hash))
                    self._queue.put(
                        (
                            namespace.key(block_hash),
                            namespace.group(block_hash),
                            [base + offset for offset, _ in ranges],
                            [length for _, length in ranges],
                            ("ring", slot),
                        )
                    )
                    handed += 1
            except Exception as exc:
                errors += 1
                self.count(ring_errors=1)
                if errors <= 20 or errors % 10000 == 0:
                    logger.error(
                        f"mooncake-store chain rank {self._rank}: ring record handling failed (#{errors}): "
                        f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
                    )
            if release:
                self._release_ring(release)
            self.count(
                drops=len(records),
                handed=handed,
                partial=partial,
                unmapped=unmapped,
                nogroup=nogroup,
            )
            self._maybe_log()

    # ---- stats ----

    def _maybe_log(self, force: bool = False) -> None:
        if self._interval is None:
            return
        now = time.monotonic()
        if not force and now - self._last_log < self._interval:
            return
        with self._log_lock:
            if not force and now - self._last_log < self._interval:
                return
            window = now - self._last_log
            self._last_log = now
            self._log_window(window)

    def _log_window(self, window: float) -> None:
        with self._stats_lock:
            stats, self._stats = self._stats, self._zero_stats()
            put_codes, delete_codes = dict(self._put_codes), dict(self._delete_codes)
        with self._free_lock:
            free = len(self._free)
        try:
            manager_stats = self._manager.impl.host_drop_stats()
            manager_text = " manager(cum): " + " ".join(
                f"{k}={v}" for k, v in manager_stats.items()
            )
        except Exception:
            manager_text = ""
        excl_text = ""
        if self._exclusive:
            with self._delete_lock:
                pending = len(self._pending_deletes)
            excl_text = (
                f" pending_removes={pending} put_fail_codes(cum)={put_codes}"
                f" remove_fail_codes(cum)={delete_codes}"
            )
        logger.info(
            f"mooncake-store chain stats: rank {self._rank} window_s={window:.0f} "
            + " ".join(f"{name}={value}" for name, value in stats.items())
            + f" keys={len(self._keys)} queued={self._queue.qsize()} free_slots={free}"
            + excl_text
            + manager_text
        )

    def shutdown(self) -> None:
        self._stop = True
        try:
            self._manager.impl.set_host_drop_hook(None)
        except Exception:
            pass
        for _ in self._threads:
            self._queue.put(None)
        for thread in self._threads:
            thread.join(timeout=30)
        self._maybe_log(force=True)
        if self._staging is not None:
            self._staging.close()
