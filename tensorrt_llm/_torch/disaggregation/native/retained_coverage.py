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
"""Retained-only, provider-bound native mapper coverage certificates.

Inputs are snapshots from configured trusted engine services, NEVER public
request metadata. This module neither authenticates those services nor creates
memory/transport authority. The provider negotiates VERSION before publication,
owns allocations/registrations, binds both generations/model/claim/nonce, votes
all-rank errors, and calls certify_destination before any REQUEST_DATA or DMA.

Preview uses the actual native Sender/PeerRegistrar/extractor/mapper. Frozen
plans stay process-local; control payloads contain compact buffer templates and
page tables, plus digests/counts. Source submission compares actual fragments
again; receive completion checks each authorized writer's frozen completed byte
count. Default paired transfer messages are unchanged.
"""

from __future__ import annotations

import base64
import hashlib
import json
from bisect import bisect_right
from dataclasses import asdict, dataclass
from functools import cached_property
from types import MappingProxyType, SimpleNamespace

import numpy as np

from tensorrt_llm._torch.disaggregation.base import CacheKind, TokenRange
from tensorrt_llm._torch.disaggregation.resource.kv_extractor import KVRegionExtractorV1
from tensorrt_llm._torch.disaggregation.resource.page import MapperKind
from tensorrt_llm._torch.disaggregation.resource.utils import get_physical_pool

from .extent import Chunk
from .peer import PeerRegistrar
from .rank_info import RankInfo

VERSION = "retained-native-role-coverage-v1"
MAX_CONTROL_BYTES = 1024 * 1024
MAX_FRAGMENTS = 262144
MAX_REGIONS = 524288
MAX_PLAN_BYTES_PER_CLAIM = 16 * 1024 * 1024
MAX_ADDRESS = (1 << 63) - 1


def _json(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


@dataclass(frozen=True)
class BufferTemplate:
    group: int
    view: int
    global_layer: int
    role: str
    base: int
    stride: int
    slots: int
    offset: int
    size: int
    layout: str
    heads: int
    total_heads: int
    head_start: int
    tokens_per_block: int


@dataclass(frozen=True)
class CacheSnapshot:
    generation: str
    model_identity: str
    request_id: int
    token_end: int
    rank_info_bytes: bytes
    pages: tuple[tuple[int, ...], ...]
    buffers: tuple[BufferTemplate, ...]

    def wire(self) -> dict:
        result = dict(
            version=VERSION,
            generation=self.generation,
            model_identity=self.model_identity,
            request_id=self.request_id,
            token_end=self.token_end,
            rank_info=base64.b64encode(self.rank_info_bytes).decode("ascii"),
            pages=self.pages,
            buffers=[asdict(buffer) for buffer in self.buffers],
        )
        if len(_json(result)) > MAX_CONTROL_BYTES:
            raise ValueError("retained coverage snapshot exceeds bounded control capacity")
        return result

    @cached_property
    def digest(self) -> str:
        return hashlib.sha256(_json(self.wire())).hexdigest()

    @cached_property
    def regions(self):
        # Cache only immutable integer/template metadata on the snapshot, not
        # mutable RankInfo or any allocation/transport object. Snapshot owner
        # lifetime bounds this cache; there is no global metadata registry.
        return _expand_regions(self)

    @property
    def rank_info(self) -> RankInfo:
        return RankInfo.from_bytes(self.rank_info_bytes)

    def chunk(self) -> Chunk:
        info = self.rank_info
        return Chunk(
            [np.array(pages, dtype=np.int64) for pages in self.pages],
            [group.kind for group in info.page_table.layer_groups],
            TokenRange(start=0, end=self.token_end),
            True,
        )


def snapshot_from_trusted_wire(value: dict) -> CacheSnapshot:
    """Parse ONLY a provider-authenticated prepared snapshot, never a client cap."""
    required = {
        "version",
        "generation",
        "model_identity",
        "request_id",
        "token_end",
        "rank_info",
        "pages",
        "buffers",
    }
    if type(value) is not dict or set(value) != required or value["version"] != VERSION:
        raise ValueError("retained coverage protocol/version mismatch")
    if len(_json(value)) > MAX_CONTROL_BYTES:
        raise ValueError("retained coverage snapshot exceeds bounded control capacity")
    snapshot = CacheSnapshot(
        value["generation"],
        value["model_identity"],
        value["request_id"],
        value["token_end"],
        base64.b64decode(value["rank_info"], validate=True),
        tuple(tuple(pages) for pages in value["pages"]),
        tuple(BufferTemplate(**buffer) for buffer in value["buffers"]),
    )
    _regions(snapshot)  # validates counts, bounds, geometry and descriptor consistency
    return snapshot


def _heads(global_heads: int, local_heads: int, tp: int, rank: int) -> int:
    if min(global_heads, local_heads, tp) <= 0 or not 0 <= rank < tp:
        raise ValueError("retained coverage requires positive authoritative head geometry")
    if global_heads >= tp:
        if global_heads % tp or local_heads != global_heads // tp:
            raise ValueError("retained coverage rejects padded/nondivisible head shards")
        return rank * local_heads
    if tp % global_heads or local_heads != 1:
        raise ValueError("retained coverage rejects incompatible replicated head geometry")
    return rank // (tp // global_heads)


def snapshot_cache(
    transceiver, request, chunk: Chunk, *, engine_generation: str, model_identity: str
) -> CacheSnapshot:
    """Freeze metadata from actual managed buffers after provider allocation freeze.

    BufferId.role is recovered from the actual V2 pool descriptors: PoolView's
    role set alone loses individual K/V and scale-buffer labels. No address or
    semantic role supplied by a client participates in this construction.
    """
    manager = transceiver._kv_cache_manager
    mapping = manager.mapping
    table = transceiver._page_table
    manager_type = (type(manager).__module__, type(manager).__name__)
    supported_managers = {
        ("tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2", "KVCacheManagerV2"),
        (
            "tensorrt_llm._torch.attention.backends.sparse.minimax_m3.cache_manager",
            "MiniMaxM3KVCacheManagerV2",
        ),
    }
    if (
        manager_type not in supported_managers
        or not manager.is_disagg
        or mapping.pp_size != 1
        or mapping.cp_size != 1
        or mapping.dp_size != 1
        or manager.is_draft
        or manager.layer_specs is not None
        or manager.mixed_precision_low_dtype is not None
        or manager.get_disagg_ignored_roles()
    ):
        raise ValueError("retained coverage does not qualify this manager/topology/state layout")
    if not engine_generation or not model_identity:
        raise ValueError("retained coverage requires verified generation/model identity")
    cache = manager.kv_cache_map[request.py_request_id]
    if not cache.is_active or table is None or chunk.excluded_pool_views:
        raise ValueError("retained coverage requires active complete managed cache pages")
    tpb = table.tokens_per_block
    if chunk.token_range.start != 0 or chunk.token_range.end <= 0 or chunk.token_range.end % tpb:
        raise ValueError("retained coverage initially requires a complete block-aligned prefix")
    count = chunk.token_range.end // tpb
    if len(chunk.block_ids_per_layer_groups) != len(table.layer_groups):
        raise ValueError("retained coverage requires every native layer group")
    roles = {}
    for desc in manager.impl.pool_group_descs:
        for variant in desc.slot_desc.variants:
            for pool_id, coalesced in enumerate(variant.coalesced_buffers):
                size = int(coalesced.single_buffer_size)
                for index, buffer in enumerate(coalesced.buffer_ids):
                    key = (
                        int(variant.layer_group_id),
                        pool_id,
                        int(buffer.layer_id),
                        index * size,
                        size,
                    )
                    if key in roles:
                        raise ValueError("duplicate authoritative storage buffer identity")
                    roles[key] = str(buffer.role)
    templates, seen = [], set()
    pages = []
    for group_id, group in enumerate(table.layer_groups):
        if group.kind != CacheKind.PAGED:
            raise ValueError("retained coverage does not qualify recurrent checkpoint semantics")
        page_ids = tuple(int(value) for value in chunk.block_ids_per_layer_groups[group_id])
        actual = transceiver._reuse_adapter.get_block_ordinals(request, group_id, group)
        if (
            len(page_ids) != count
            or any(page < 0 for page in page_ids)
            or tuple(int(value) for value in actual[:count]) != page_ids
        ):
            raise ValueError("retained coverage pages differ from actual complete cache ordinals")
        pages.append(page_ids)
        global_layers = {
            layer.local_layer_id: layer.global_layer_id for layer in group.local_layers
        }
        for view_id, view in enumerate(group.pool_views):
            if view.ring_slot_bytes is not None or view.mapper_kind not in (
                MapperKind.INDEXED,
                MapperKind.NHD,
                MapperKind.REPLICATED,
            ):
                raise ValueError("retained coverage rejects unqualified mapper/ring state")
            pool = get_physical_pool(table, group_id, view.pool_idx)
            view_roles = set()
            for entry in view.buffer_entries:
                local = int(entry["local_layer_id"])
                offset, size = int(entry["offset"]), int(entry["size"])
                key = (group_id, view.pool_idx, local, offset, size)
                role = roles[key]
                seen.add(key)
                view_roles.add(role)
                global_layer = int(global_layers[local])
                if view.mapper_kind == MapperKind.REPLICATED:
                    layout, heads, total, start = "REPLICATED", 0, 0, 0
                else:
                    layout = "NHD" if view.mapper_kind == MapperKind.NHD else "HND"
                    heads = int(view.num_kv_heads)
                    original_heads = manager.num_kv_heads
                    total = (
                        original_heads
                        if type(original_heads) is int
                        else original_heads[global_layer]
                    )
                    start = _heads(int(total), heads, mapping.tp_size, mapping.tp_rank)
                templates.append(
                    BufferTemplate(
                        group_id,
                        view_id,
                        global_layer,
                        role,
                        int(pool.base_address),
                        int(pool.slot_stride_bytes),
                        int(pool.num_slots),
                        offset,
                        size,
                        layout,
                        heads,
                        int(total),
                        start,
                        tpb,
                    )
                )
            if view_roles != set(view.pool_role):
                raise ValueError(
                    "native view role set differs from authoritative storage BufferIds"
                )
    if seen != set(roles):
        raise ValueError("native transfer page table omits a managed storage role/buffer")
    result = CacheSnapshot(
        engine_generation,
        model_identity,
        request.py_request_id,
        chunk.token_range.end,
        transceiver._transfer_worker._rank_info.to_bytes(),
        tuple(pages),
        tuple(templates),
    )
    _regions(result)
    result.wire()  # enforce bounded control capacity before provider publishes anything
    return result


@dataclass(frozen=True)
class _Region:
    start: int
    end: int
    buffer: BufferTemplate
    ordinal: int

    def semantic(self, address: int) -> tuple[tuple, int, int]:
        offset = address - self.start
        b = self.buffer
        key = (b.global_layer, b.role, self.ordinal, b.layout, b.total_heads, b.tokens_per_block)
        if b.layout == "REPLICATED":
            return key + (b.size,), offset, self.end - address
        unit = b.size // (b.heads * b.tokens_per_block)
        key += (unit,)
        if b.layout == "HND":
            return key, b.head_start * b.tokens_per_block * unit + offset, self.end - address
        row = b.heads * unit
        token, head_byte = divmod(offset, row)
        return key, token * b.total_heads * unit + b.head_start * unit + head_byte, row - head_byte


def _regions(snapshot: CacheSnapshot) -> tuple[_Region, ...]:
    return snapshot.regions


def logical_layout_identity(snapshot: CacheSnapshot) -> str:
    """Semantic cache identity shared by compatible TP partitions.

    Physical snapshots still bind every address/stride/page/rank separately.
    This hash excludes those coordinates and prefix length, retaining actual
    global layer/individual role/mapper/global heads/token byte geometry.
    """
    _regions(snapshot)
    buffers = []
    for b in snapshot.buffers:
        buffers.append(
            (
                b.global_layer,
                b.role,
                "PAGED",
                b.layout,
                b.total_heads,
                b.tokens_per_block,
                b.size if b.layout == "REPLICATED" else b.size // (b.heads * b.tokens_per_block),
            )
        )
    return hashlib.sha256(
        _json(dict(version=VERSION, model=snapshot.model_identity, buffers=sorted(buffers)))
    ).hexdigest()


def _expand_regions(snapshot: CacheSnapshot) -> tuple[_Region, ...]:
    info = snapshot.rank_info
    if (
        not snapshot.generation
        or not snapshot.model_identity
        or type(snapshot.token_end) is not int
        or type(snapshot.request_id) is not int
        or snapshot.request_id < 0
        or info.pp_size != 1
        or info.cp_size != 1
        or info.dp_size != 1
        or info.page_table is None
        or len(snapshot.pages) != len(info.page_table.layer_groups)
    ):
        raise ValueError("invalid retained authoritative cache snapshot")
    regions, semantic_ids = [], set()
    for b in snapshot.buffers:
        numeric = (
            b.group,
            b.view,
            b.global_layer,
            b.base,
            b.stride,
            b.slots,
            b.offset,
            b.size,
            b.heads,
            b.total_heads,
            b.head_start,
            b.tokens_per_block,
        )
        if (
            not all(type(value) is int for value in numeric)
            or min(b.base, b.stride, b.slots, b.size, b.tokens_per_block) <= 0
        ):
            raise ValueError("invalid retained buffer numeric geometry")
        if (
            b.group < 0
            or b.group >= len(snapshot.pages)
            or b.global_layer < 0
            or b.offset < 0
            or b.offset + b.size > b.stride
            or type(b.role) is not str
            or not b.role
        ):
            raise ValueError("retained buffer lies outside native storage slot")
        group = info.page_table.layer_groups[b.group]
        if b.view < 0 or b.view >= len(group.pool_views) or group.kind != CacheKind.PAGED:
            raise ValueError("retained buffer references an unknown native view")
        view = group.pool_views[b.view]
        pool = get_physical_pool(info.page_table, b.group, view.pool_idx)
        if b.offset + b.size > int(pool.slot_bytes):
            raise ValueError("retained role exceeds its registered physical slot span")
        if (int(pool.base_address), int(pool.slot_stride_bytes), int(pool.num_slots)) != (
            b.base,
            b.stride,
            b.slots,
        ):
            raise ValueError(
                "retained buffer pointer differs from authoritative native rank metadata"
            )
        expected_mapper = (
            "REPLICATED"
            if view.mapper_kind == MapperKind.REPLICATED
            else "NHD"
            if view.mapper_kind == MapperKind.NHD
            else "HND"
        )
        if (
            view.mapper_kind not in (MapperKind.INDEXED, MapperKind.NHD, MapperKind.REPLICATED)
            or view.ring_slot_bytes is not None
            or b.layout != expected_mapper
            or b.role not in view.pool_role
            or b.tokens_per_block != info.page_table.tokens_per_block
        ):
            raise ValueError("retained buffer semantic identity differs from native view")
        local = next(
            (
                layer.local_layer_id
                for layer in group.local_layers
                if layer.global_layer_id == b.global_layer
            ),
            None,
        )
        if not any(
            int(entry["local_layer_id"]) == local
            and int(entry["offset"]) == b.offset
            and int(entry["size"]) == b.size
            for entry in view.buffer_entries
        ):
            raise ValueError("retained buffer offset/size differs from native view")
        if b.layout != "REPLICATED":
            if (
                b.heads != view.num_kv_heads
                or _heads(b.total_heads, b.heads, info.tp_size, info.tp_rank) != b.head_start
                or b.size % (b.heads * b.tokens_per_block)
            ):
                raise ValueError("retained buffer has inconsistent head/token byte geometry")
        elif (b.heads, b.total_heads, b.head_start) != (0, 0, 0):
            raise ValueError("replicated retained role unexpectedly carries sharded head geometry")
        if (
            snapshot.token_end <= 0
            or snapshot.token_end % b.tokens_per_block
            or len(snapshot.pages[b.group]) != snapshot.token_end // b.tokens_per_block
        ):
            raise ValueError("retained buffer does not cover the complete prefix")
        for ordinal, page in enumerate(snapshot.pages[b.group]):
            if type(page) is not int or not 0 <= page < b.slots:
                raise ValueError("retained page lies outside registered physical pool")
            start = b.base + page * b.stride + b.offset
            if start + b.size > MAX_ADDRESS:
                raise ValueError("retained pointer arithmetic overflow")
            semantic_id = (b.global_layer, b.role, ordinal)
            if semantic_id in semantic_ids:
                raise ValueError("retained snapshot duplicates a semantic role/layer/page")
            semantic_ids.add(semantic_id)
            regions.append(_Region(start, start + b.size, b, ordinal))
            if len(regions) > MAX_REGIONS:
                raise ValueError("retained coverage region budget exceeded")
    regions.sort(key=lambda region: region.start)
    if not regions or any(left.end > right.start for left, right in zip(regions, regions[1:])):
        raise ValueError("retained registered role/page regions overlap or are empty")
    return tuple(regions)


@dataclass(frozen=True)
class ClaimBinding:
    claim_id: int
    nonce: str
    model_identity: str
    source_generation: str
    source_instance: str
    source_rank: int
    destination_generation: str
    destination_instance: str
    destination_rank: int
    token_end: int


@dataclass(frozen=True)
class FrozenWritePlan:
    binding: ClaimBinding
    source_digest: str
    destination_digest: str
    src_ptrs: bytes
    dst_ptrs: bytes
    sizes: bytes
    total_bytes: int
    source_rank_info_bytes: bytes
    destination_rank_info_bytes: bytes

    @cached_property
    def digest(self) -> str:
        prefix = _json(
            dict(
                version=VERSION,
                binding=asdict(self.binding),
                source=self.source_digest,
                destination=self.destination_digest,
            )
        )
        return hashlib.sha256(prefix + self.src_ptrs + self.dst_ptrs + self.sizes).hexdigest()

    def arrays(self):
        return tuple(
            np.frombuffer(raw, dtype="<i8") for raw in (self.src_ptrs, self.dst_ptrs, self.sizes)
        )

    def receipt(self) -> dict:
        """Bounded control receipt; actual fragment arrays never cross this API."""
        result = dict(
            version=VERSION,
            binding=asdict(self.binding),
            source=self.source_digest,
            destination=self.destination_digest,
            digest=self.digest,
            completed_bytes=self.total_bytes,
            fragments=len(self.sizes) // 8,
        )
        if len(_json(result)) > MAX_CONTROL_BYTES:
            raise ValueError("retained coverage receipt exceeds bounded control capacity")
        return result


def _check_binding(binding, source, destination):
    a, b = source.rank_info, destination.rank_info
    if (
        type(binding.claim_id) is not int
        or not 0 <= binding.claim_id < 1 << 63
        or type(binding.nonce) is not str
        or not 0 < len(binding.nonce) <= 128
        or binding.model_identity != source.model_identity
        or binding.model_identity != destination.model_identity
        or (binding.source_generation, binding.source_instance, binding.source_rank)
        != (source.generation, a.instance_name, a.instance_rank)
        or (binding.destination_generation, binding.destination_instance, binding.destination_rank)
        != (destination.generation, b.instance_name, b.instance_rank)
        or binding.token_end != source.token_end
        or binding.token_end != destination.token_end
    ):
        raise ValueError("retained coverage claim/generation/model/peer binding mismatch")


def _array_bytes(values) -> bytes:
    array = np.asarray(values)
    if array.ndim != 1 or not np.issubdtype(array.dtype, np.integer) or len(array) > MAX_FRAGMENTS:
        raise ValueError("retained native fragment shape/type/budget is invalid")
    return np.asarray(array, dtype="<i8").tobytes()


def _pieces(source, destination, arrays):
    src_regions, dst_regions = _regions(source), _regions(destination)
    src_starts = [region.start for region in src_regions]
    dst_starts = [region.start for region in dst_regions]
    srcs, dsts, sizes = arrays
    if len(srcs) != len(dsts) or len(srcs) != len(sizes) or len(sizes) > MAX_FRAGMENTS:
        raise ValueError("retained native fragment counts differ or exceed budget")
    pieces = []
    for src, dst, size in zip(srcs, dsts, sizes):
        src, dst, remaining = int(src), int(dst), int(size)
        if remaining <= 0:
            raise ValueError("retained native fragment has a non-positive size")
        while remaining:
            si, di = bisect_right(src_starts, src) - 1, bisect_right(dst_starts, dst) - 1
            if si < 0 or di < 0 or src >= src_regions[si].end or dst >= dst_regions[di].end:
                raise ValueError("retained native fragment escapes the frozen cache allocation")
            sr, dr = src_regions[si], dst_regions[di]
            sk, so, sl = sr.semantic(src)
            dk, do, dl = dr.semantic(dst)
            amount = min(remaining, sl, dl, sr.end - src, dr.end - dst)
            if sk != dk or so != do:
                raise ValueError(
                    "retained native fragment swaps role/layer/head/token/offset semantics"
                )
            pieces.append((di, dst, dst + amount))
            src += amount
            dst += amount
            remaining -= amount
            if len(pieces) > MAX_REGIONS:
                raise ValueError("retained coverage split-fragment budget exceeded")
    return dst_regions, pieces


def freeze_write_plan(
    binding: ClaimBinding, write_meta, source: CacheSnapshot, destination: CacheSnapshot
) -> FrozenWritePlan:
    _check_binding(binding, source, destination)
    destination_info = destination.rank_info
    if (
        write_meta.unique_rid != binding.claim_id
        or write_meta.peer_rank != binding.destination_rank
        or write_meta.peer_name != binding.destination_instance + str(binding.destination_rank)
        or write_meta.peer_endpoint != destination_info.self_endpoint
        or write_meta.dst_device_id != destination_info.device_id
        or write_meta.slice_id != 0
        or write_meta.receiver_slice_id != 0
        or not write_meta.is_last_slice
    ):
        raise ValueError("retained native operation is not the bound monolithic claim")
    raw = tuple(
        _array_bytes(values)
        for values in (write_meta.src_ptrs, write_meta.dst_ptrs, write_meta.sizes)
    )
    arrays = tuple(np.frombuffer(value, dtype="<i8") for value in raw)
    _pieces(source, destination, arrays)
    return FrozenWritePlan(
        binding,
        source.digest,
        destination.digest,
        *raw,
        sum(int(value) for value in arrays[2]),
        source.rank_info_bytes,
        destination.rank_info_bytes,
    )


def certify_destination(destination: CacheSnapshot, plans, sources, authorized_writers) -> str:
    """Require exact role/layer/page span partition from the whole writer cohort."""
    plans = tuple(plans)
    if (
        sum(len(plan.src_ptrs) + len(plan.dst_ptrs) + len(plan.sizes) for plan in plans)
        > MAX_PLAN_BYTES_PER_CLAIM
    ):
        raise ValueError("retained coverage immutable plan memory budget exceeded")
    writers = {(plan.binding.source_instance, plan.binding.source_rank) for plan in plans}
    if not plans or len(writers) != len(plans) or writers != set(authorized_writers):
        raise ValueError("retained coverage plans differ from the authorized full writer cohort")
    cohorts = {
        tuple(value for key, value in asdict(plan.binding).items() if key != "source_rank")
        for plan in plans
    }
    if len(cohorts) != 1:
        raise ValueError("retained coverage mixes claims/nonces/engine generations")
    expected = _regions(destination)
    covered = [[] for _ in expected]
    for plan in plans:
        writer = (plan.binding.source_instance, plan.binding.source_rank)
        source = sources[writer]
        _check_binding(plan.binding, source, destination)
        if plan.source_digest != source.digest or plan.destination_digest != destination.digest:
            raise ValueError("retained coverage plan does not bind the frozen cache snapshots")
        if (
            plan.source_rank_info_bytes != source.rank_info_bytes
            or plan.destination_rank_info_bytes != destination.rank_info_bytes
        ):
            raise ValueError(
                "retained coverage plan does not bind authoritative native peer metadata"
            )
        if plan.total_bytes != sum(int(value) for value in plan.arrays()[2]):
            raise ValueError("retained coverage plan byte count differs from its native fragments")
        _, pieces = _pieces(source, destination, plan.arrays())
        for index, start, end in pieces:
            covered[index].append((start, end))
    for region, spans in zip(expected, covered):
        cursor = region.start
        for start, end in sorted(spans):
            if start != cursor:
                raise ValueError("retained destination role/page coverage has a hole or overlap")
            cursor = end
        if cursor != region.end:
            raise ValueError("retained destination role/page coverage is incomplete")
    return hashlib.sha256(
        _json(
            dict(
                version=VERSION,
                destination=destination.digest,
                plans=sorted(plan.digest for plan in plans),
            )
        )
    ).hexdigest()


def preview_write_plan(
    binding: ClaimBinding, source: CacheSnapshot, destination: CacheSnapshot
) -> FrozenWritePlan:
    """Run exact native pure descriptor builders; no transport or allocation."""
    from .transfer import RecvReqInfo, Sender

    _check_binding(binding, source, destination)
    src_info, dst_info = source.rank_info, destination.rank_info
    registrar = PeerRegistrar(src_info, KVRegionExtractorV1(src_info.page_table))
    registrar.register(dst_info.instance_name, dst_info.instance_rank, dst_info)
    task = SimpleNamespace(
        _chunk=source.chunk(),
        _prompt_len=binding.token_end,
        _perf_timer=None,
        _unique_rid=binding.claim_id,
        slice_id=0,
    )
    request = RecvReqInfo(
        sender_req_id=binding.claim_id,
        instance_name=dst_info.instance_name,
        instance_rank=dst_info.instance_rank,
        block_ids_per_layer_groups=destination.chunk().block_ids_per_layer_groups,
        unique_rid=binding.claim_id,
        slice_id=0,
        context_first=True,
    )
    meta = Sender._build_kv_write_meta(SimpleNamespace(_registrar=registrar), task, request)
    return freeze_write_plan(binding, meta, source, destination)


def check_native_operation(
    plan: FrozenWritePlan, meta, *, source_info=None, destination_info=None, source_device_id=None
) -> None:
    """Source hook immediately before real native descriptor construction."""
    from .transfer import _TransferGroup

    binding = plan.binding
    destination = RankInfo.from_bytes(plan.destination_rank_info_bytes)
    source = RankInfo.from_bytes(plan.source_rank_info_bytes)
    if (
        meta.unique_rid != binding.claim_id
        or meta.peer_rank != binding.destination_rank
        or meta.peer_name != binding.destination_instance + str(binding.destination_rank)
        or meta.peer_endpoint != destination.self_endpoint
        or meta.dst_device_id != destination.device_id
        or meta.slice_id != 0
        or meta.receiver_slice_id != 0
        or not meta.is_last_slice
        or tuple(_array_bytes(values) for values in (meta.src_ptrs, meta.dst_ptrs, meta.sizes))
        != (plan.src_ptrs, plan.dst_ptrs, plan.sizes)
    ):
        raise ValueError("actual native operation differs from the certified retained frozen plan")
    groups = meta.transfer_groups
    if type(groups) is not tuple or len(groups) > 1:
        raise ValueError("retained native transfer groups require one complete GPU span")
    if groups:
        group = groups[0]
        if (
            type(group) is not _TransferGroup
            or group.src_memory_type != "VRAM"
            or group.dst_memory_type != "VRAM"
            or type(group.start) is not int
            or type(group.stop) is not int
            or group.start != 0
            or group.stop != meta.src_ptrs.size
        ):
            raise ValueError("retained native transfer groups require one complete GPU span")
    if (
        (source_info is not None and source_info.to_bytes() != plan.source_rank_info_bytes)
        or (
            destination_info is not None
            and destination_info.to_bytes() != plan.destination_rank_info_bytes
        )
        or (source_device_id is not None and source_device_id != source.device_id)
    ):
        raise ValueError(
            "actual native peer registration/device changed after retained coverage certification"
        )


def check_completed_writer(
    plans_by_rank, claim_id, peer_rank, slice_id, is_last_slice, completed_bytes
) -> None:
    """Validate successful native DONE report before ordinary task completion."""
    plan = plans_by_rank.get(peer_rank)
    if (
        plan is None
        or plan.binding.claim_id != claim_id
        or slice_id != 0
        or not is_last_slice
        or type(completed_bytes) is not int
        or completed_bytes != plan.total_bytes
    ):
        raise ValueError(
            "retained completed writer does not match its authorized frozen coverage plan"
        )


def bind_source_session(session, source: CacheSnapshot, plans) -> None:
    """Bind negotiated/certified plans before a fresh source session sends.

    Provider must receive the destination's whole-cohort certificate agreement
    before calling this or granting publication. This function validates local
    native identity; it never grants remote authority from a digest alone.
    """
    plans = tuple(plans)
    if (
        sum(len(plan.src_ptrs) + len(plan.dst_ptrs) + len(plan.sizes) for plan in plans)
        > MAX_PLAN_BYTES_PER_CLAIM
    ):
        raise ValueError("retained source immutable plan memory budget exceeded")
    local = session._sender._registrar.self_rank_info
    if (
        session.kv_tasks
        or getattr(session, "_retained_coverage_by_rank", None) is not None
        or not session._enforce_physical_ownership
    ):
        raise ValueError("retained coverage requires a fresh physically owned source session")
    for plan in plans:
        b = plan.binding
        if (
            b.claim_id != session.disagg_request_id
            or plan.source_digest != source.digest
            or source.request_id != session.request_id
            or b.source_generation != source.generation
            or (b.source_instance, b.source_rank) != (local.instance_name, local.instance_rank)
            or b.token_end != session._base_args.prompt_len
        ):
            raise ValueError("source session differs from retained coverage identity")
    by_rank = {plan.binding.destination_rank: plan for plan in plans}
    if not plans or len(by_rank) != len(plans):
        raise ValueError("retained source plans must name distinct authorized destination ranks")
    session._retained_coverage_by_rank = MappingProxyType(by_rank)


def bind_destination_session(
    session, destination: CacheSnapshot, plans, sources, authorized_writers
) -> str:
    """Certify all actual native preview fragments before destination exposure."""
    plans = tuple(plans)
    local = session._receiver._registrar.self_rank_info
    if (
        session._kv_tasks
        or getattr(session, "_retained_coverage_expected_writers", None) is not None
        or not session._enforce_physical_ownership
    ):
        raise ValueError("retained coverage requires a fresh physically owned destination session")
    if destination.request_id != session.request_id:
        raise ValueError("retained coverage is for a different allocated destination request")
    for plan in plans:
        b = plan.binding
        if (
            b.claim_id != session.disagg_request_id
            or (b.destination_instance, b.destination_rank)
            != (local.instance_name, local.instance_rank)
            or b.token_end != session._base_args.prompt_len
        ):
            raise ValueError("destination session differs from retained coverage identity")
    certificate = certify_destination(destination, plans, sources, authorized_writers)
    by_rank = {plan.binding.source_rank: plan for plan in plans}
    if len(by_rank) != len(plans):
        raise ValueError("retained native writer ranks must belong to one source instance")
    session._retained_coverage_expected_writers = MappingProxyType(by_rank)
    session._retained_destination_snapshot = destination
    return certificate


def check_destination_publication(session, chunk, expected_write_bytes) -> None:
    snapshot = session._retained_destination_snapshot
    if (
        not chunk.is_last
        or chunk.excluded_pool_views
        or chunk.token_range.start != 0
        or chunk.token_range.end != snapshot.token_end
        or tuple(chunk.kind_per_layer_group)
        != tuple(group.kind for group in snapshot.rank_info.page_table.layer_groups)
        or len(chunk.block_ids_per_layer_groups) != len(snapshot.pages)
        or any(
            tuple(int(value) for value in actual) != expected
            for actual, expected in zip(chunk.block_ids_per_layer_groups, snapshot.pages)
        )
        or expected_write_bytes
        != sum(plan.total_bytes for plan in session._retained_coverage_expected_writers.values())
    ):
        raise ValueError(
            "native destination publication differs from its frozen retained certificate"
        )
