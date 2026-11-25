# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A context server's cold-page codec for the qwen4_exp attention pages (`TRTLLM_QWEN4_EXP_CTX_COLDIDX`).

On a context server with a host (and disk) tier, every attention page that leaves the GPU carries
the packed QSA side planes: one 32 x 256 B index plane per sparse layer and the 32 x 12 B position
page. Under `INDEX_KEY_FP8` a completed group needs only its 128-byte e4m3 anchor; the raw keys of
the groups before the request's last two and the coordinates of a group's later tokens are never
read again. This provider stores those pages without them (`ops/attention/qsa_cold_index_codec`):
on the Qwen3.8 NVFP4 context worker a page of 532,864 B goes to 462,848 B (-13.1%), so the same
host tier holds about 15% more attention pages and a turn's restore reads less from disk.

It sits on the KV cache manager's Python cold-page bridge (`create_python_cold_page_codec`): the
manager calls `configure` once with the provider-owned lifecycles (the attention one: roles `key`,
`value`, `qsa_side_planes`), then `encode_cold_pages` for every GPU -> host migration batch and
`decode_cold_pages` for the way back; host <-> disk moves the cold blob as is. A lifecycle whose
layout is not the one checked here is stored as the plain concatenation (one warning).

Which groups of a page are pooled is marked in the plane by the forward (`qsa_cold_index_mark`
behind every anchor writer, `core.cold_index_marks`). Every byte a kernel reads comes back bitwise;
measured on the overlay this ports: ops micros on engine-size pools, a real KVCacheManagerV2 disk
round trip, context-engine resumes (48 / 48) and 192 / 192 probe records against the switch off.
"""

from __future__ import annotations

import ctypes
import threading
import time
from typing import Any, Optional, Sequence

import torch

from tensorrt_llm._torch.pyexecutor.resource_manager import DataType, KVCacheCompressionManager
from tensorrt_llm._utils import maybe_pin_memory
from tensorrt_llm.logger import logger

from .ops.attention.qsa_cold_index_codec import (
    ANCHOR_BYTES,
    GROUPS_PER_PAGE,
    PLANE_BYTES,
    POSITION_PAGE_BYTES,
    ColdIndexTables,
)

#: The page geometry the codec's plane arithmetic is written for.
TOKENS_PER_BLOCK = 32
#: Codec calls between two counter lines.
LOG_EVERY_CALLS = 200


class QsaColdIndexConfig:
    """The fields the engine reads off a compression manager's config."""

    algorithm = "qsa_cold_index"
    changes_physical_kv_length = False

    def supports_speculative_decoding(self) -> bool:
        return True

    def supports_block_reuse(self) -> bool:
        return True


class _CodecState:
    """State owned by one native codec (one KV cache manager build)."""

    def __init__(self, device: int, num_planes: int) -> None:
        self.device = device
        self.num_planes = num_planes
        self.tables: list[ColdIndexTables] = []


class QsaColdIndexPages(KVCacheCompressionManager):
    """Cold-page codec of the qwen4_exp attention lifecycle: drops the index-plane bytes nothing
    reads."""

    uses_iteration_lifecycle = False
    provides_cold_page_codec = True

    def __init__(self) -> None:
        super().__init__(QsaColdIndexConfig())
        self._lock = threading.Lock()
        self._engaged: set = set()
        self._calls = 0
        self._stats = dict(enc_pages=0, dec_pages=0, enc_s=0.0, dec_s=0.0)
        # Pinned when the first engaged lifecycle is configured (a GPU process).
        self._counter_host: Optional[torch.Tensor] = None
        self._counter_tables: Optional[ColdIndexTables] = None

    def create_cold_page_codec(
        self,
        cache_config: object,
        *,
        runtime_dtype: DataType,
        pp_layers: Sequence[int],
        num_kv_heads_per_layer: Sequence[int],
        head_dim_per_layer: Sequence[int],
        is_draft: bool = False,
    ) -> Optional[object]:
        """The native codec over this manager's layout, or None (the default concatenation) where
        it is not the packed side-plane layout this codec is written for."""
        from tensorrt_llm.runtime.kv_cache_manager_v2 import (
            AttentionLayerConfig,
            HostCacheTierConfig,
        )

        problems = []
        if is_draft:
            problems.append("a separate draft KV cache manager")
        if not any(isinstance(t, HostCacheTierConfig) for t in cache_config.cache_tiers):
            problems.append("no host tier")
        if int(cache_config.tokens_per_block) != TOKENS_PER_BLOCK:
            problems.append(f"tokens_per_block {cache_config.tokens_per_block}")
        side_sizes = []
        for layer in cache_config.layers:
            if not isinstance(layer, AttentionLayerConfig):
                continue
            roles = {str(b.role): int(b.size) for b in layer.buffers}
            if getattr(layer, "sparse_cache", None) is not None:
                problems.append(f"layer {layer.layer_id} has a sparse cache")
            extra = set(roles) - {"key", "value", "qsa_side_planes"}
            if extra:
                problems.append(f"layer {layer.layer_id} roles {sorted(extra)}")
            if "qsa_side_planes" in roles:
                side_sizes.append(roles["qsa_side_planes"])
        planes = _side_planes(side_sizes[0]) if len(side_sizes) == 1 else None
        if planes is None:
            problems.append(f"side-plane buffers {side_sizes}")
        if problems:
            logger.warning(f"QSA cold index codec not applied: {'; '.join(problems[:4])}")
            return None
        from tensorrt_llm.bindings.internal import kv_cache_compression as native

        return native.create_python_cold_page_codec(
            self, _CodecState(torch.cuda.current_device(), planes)
        )

    def configure(self, codec_state: _CodecState, lifecycles: Sequence[object]) -> Sequence[object]:
        """Resolve the provider-owned lifecycles into launch tables and their cold page sizes."""
        from tensorrt_llm.bindings.internal import kv_cache_compression as native

        properties, tables = [], []
        device = torch.device("cuda", codec_state.device)
        for lifecycle in lifecycles:
            verbatim, side, problems = [], None, []
            for layer_id in sorted(int(k) for k in lifecycle.layers.keys()):
                buffers = lifecycle.layers[layer_id]
                order = {"key": 0, "value": 1}
                for role in sorted(buffers.keys(), key=lambda r: order.get(str(r), 2)):
                    hb = buffers[role]
                    base, slot, nbytes = int(hb.raw_base), int(hb.raw_slot_bytes), int(hb.raw_bytes)
                    if base % 16 or slot % 16 or nbytes % 16 or nbytes > slot:
                        problems.append(f"layer {layer_id} {role} not 16-byte aligned")
                    if str(role) == "qsa_side_planes":
                        if _side_planes(nbytes) != codec_state.num_planes:
                            problems.append(f"side planes of {nbytes} B")
                        side = (base, slot, nbytes)
                    else:
                        verbatim.append((base, slot, nbytes))
            if side is None:
                problems.append("no qsa_side_planes buffer")
            with torch.cuda.device(device):
                if problems:
                    logger.warning(
                        f"QSA cold index codec not applied to lifecycle {lifecycle.life_cycle_id}: "
                        f"{'; '.join(problems[:4])} (plain concatenation instead)"
                    )
                    every = verbatim + ([side] if side is not None else [])
                    table = ColdIndexTables(every, None, 0, device)
                else:
                    table = ColdIndexTables(verbatim, side[:2], codec_state.num_planes, device)
                    self._counter_tables = table
                    if self._counter_host is None:
                        self._counter_host = maybe_pin_memory(torch.zeros(4, dtype=torch.int64))
                    _warm(table, device, codec_state.num_planes)
            tables.append(table)
            lifecycle_properties = native.ColdPageLifecycleProperties()
            lifecycle_properties.cold_page_bytes = table.cold_page_bytes
            lifecycle_properties.page_index_location = native.ColdPageIndexLocation.HOST
            properties.append(lifecycle_properties)
            key = (lifecycle.life_cycle_id, table.cold_page_bytes)
            if table.num_planes and key not in self._engaged:
                self._engaged.add(key)
                raw = sum(b for _, _, b in verbatim) + codec_state.num_planes * PLANE_BYTES
                logger.info(
                    f"QSA cold index codec: lifecycle {lifecycle.life_cycle_id}, "
                    f"{codec_state.num_planes} index planes, cold page {table.cold_page_bytes} B "
                    f"(raw {raw + POSITION_PAGE_BYTES}), anchors {GROUPS_PER_PAGE} a plane, "
                    f"marker bytes {ANCHOR_BYTES}-{ANCHOR_BYTES + 15}"
                )
        codec_state.tables = tables
        return properties

    def encode_cold_pages(
        self,
        codec_state: _CodecState,
        lifecycle_index: int,
        cold_base: int,
        page_indices: int,
        num_pages: int,
        stream: int,
    ) -> None:
        self._run(True, codec_state, lifecycle_index, cold_base, page_indices, num_pages, stream)

    def decode_cold_pages(
        self,
        codec_state: _CodecState,
        lifecycle_index: int,
        cold_base: int,
        page_indices: int,
        num_pages: int,
        stream: int,
    ) -> None:
        self._run(False, codec_state, lifecycle_index, cold_base, page_indices, num_pages, stream)

    def _run(
        self,
        encode: bool,
        codec_state: _CodecState,
        lifecycle_index: int,
        cold_base: int,
        page_indices: int,
        num_pages: int,
        stream: int,
    ) -> None:
        """One migration batch on the manager's stream: the host PageIndexPair array (dst, src
        int32) is copied to the device on that stream, then one launch."""
        start = time.perf_counter()
        table = codec_state.tables[lifecycle_index]
        device = torch.device("cuda", codec_state.device)
        with (
            torch.cuda.device(device),
            torch.cuda.stream(torch.cuda.ExternalStream(stream, device=device)),
        ):
            pairs = _pairs(page_indices, num_pages, device)
            if encode:
                table.encode(pairs, num_pages, cold_base)
            else:
                table.decode(pairs, num_pages, cold_base)
            with self._lock:
                kind = "enc" if encode else "dec"
                self._stats[kind + "_pages"] += num_pages
                self._stats[kind + "_s"] += time.perf_counter() - start
                self._calls += 1
                due = self._calls % LOG_EVERY_CALLS == 0
            if due:
                self._log()

    def _log(self) -> None:
        """One counter line; the device counters it prints were read back by the previous one (no
        synchronization here)."""
        with self._lock:
            stats = dict(self._stats)
            counters = self._counter_host
            escaped, mismatched = (
                (0, 0) if counters is None else (int(counters[0]), int(counters[1]))
            )
        if self._counter_tables is not None and counters is not None:
            counters.copy_(self._counter_tables.counters, non_blocking=True)
        logger.info(
            f"QSA cold index codec: encoded {stats['enc_pages']} pages ({stats['enc_s']:.2f} s "
            f"host), decoded {stats['dec_pages']} ({stats['dec_s']:.2f} s), pages with a partial "
            f"group {escaped}, planes off plane 0's partial group {mismatched}"
        )


def _side_planes(nbytes: int) -> Optional[int]:
    """Index planes of a packed side-plane buffer of `nbytes`, or None where it is not exactly
    planes and the position page (a grown buffer) or holds more than the cold layout's 16."""
    planes, rest = divmod(nbytes - POSITION_PAGE_BYTES, PLANE_BYTES)
    return planes if rest == 0 and 0 < planes <= 16 else None


def _pairs(page_indices: int, num_pages: int, device: torch.device) -> torch.Tensor:
    """The manager's host PageIndexPair array (dst, src int32), copied to the device on the current
    stream. A pageable source: the copy has consumed it when the call returns."""
    array = (ctypes.c_int32 * (2 * num_pages)).from_address(page_indices)
    host = torch.frombuffer(array, dtype=torch.int32)
    return host.to(device, non_blocking=True)


def _warm(table: ColdIndexTables, device: torch.device, num_planes: int) -> None:
    """Compile both kernels at configure, not under the first migration: one page of scratch."""
    sizes = table.vb_bytes.tolist()
    kv = sum(sizes)
    hot = torch.zeros(
        kv + num_planes * PLANE_BYTES + POSITION_PAGE_BYTES + 256, dtype=torch.uint8, device=device
    )
    cold = torch.zeros(table.cold_page_bytes + 256, dtype=torch.uint8, device=device)
    hot_base = (hot.data_ptr() + 15) // 16 * 16
    cold_base = (cold.data_ptr() + 15) // 16 * 16
    offsets = [sum(sizes[:i]) for i in range(len(sizes))]
    # The same layout (and so the same compiled kernels), every page id on the scratch page.
    scratch = ColdIndexTables(
        [(hot_base + o, 0, n) for o, n in zip(offsets, sizes)],
        (hot_base + kv, 0),
        num_planes,
        device,
    )
    pairs = torch.zeros(1, 2, dtype=torch.int32, device=device)
    scratch.encode(pairs, 1, cold_base)
    scratch.decode(pairs, 1, cold_base)
    torch.cuda.synchronize(device)


def qsa_cold_pages_provider(
    kv_cache_config: Any, *, index_key_fp8: bool, max_draft_len: Optional[int]
) -> Optional[QsaColdIndexPages]:
    """The provider a context server under `TRTLLM_QWEN4_EXP_CTX_COLDIDX` uses (`RuntimeHooks.
    cold_page_codec`), or None with one warning where its exactness was not shown: it needs the
    e4m3 anchors (`INDEX_KEY_FP8`), block reuse, and -- what the two escaped groups are sized for --
    the `L - 2` in-forward snapshot (`additional_snapshot_offsets_from_end: [2]`) and MTP-3."""
    problems = []
    if not index_key_fp8:
        problems.append("TRTLLM_QWEN4_EXP_INDEX_KEY_FP8 is off")
    if not getattr(kv_cache_config, "enable_block_reuse", False):
        problems.append("block reuse is off")
    state = getattr(kv_cache_config, "mamba_state_config", None)
    offsets = list(getattr(state, "additional_snapshot_offsets_from_end", None) or [])
    if offsets != [2]:
        problems.append(f"snapshot offsets from end {offsets} (needs [2])")
    if max_draft_len != 3:
        problems.append(f"draft length {max_draft_len} (needs MTP-3)")
    if problems:
        logger.warning(f"QSA cold index codec not applied: {'; '.join(problems)}")
        return None
    return QsaColdIndexPages()
