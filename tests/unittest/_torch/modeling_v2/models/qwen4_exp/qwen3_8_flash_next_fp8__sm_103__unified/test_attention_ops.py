# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU tests for the FP8 unified target's attention entries other than `thop_attention`: every
certified cell against the entry's `reference` over the engine's paged state, plus the TP index
split, captured stores, kernels without cells, the call sites and the FP8 KV range probe."""

import math
import types
import zlib
from types import SimpleNamespace
from unittest.mock import Mock

import cuda.bindings.driver as cuda_drv
import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.ops.attention.qsa_topk_expand_decode import (  # noqa: E501
    qsa_topk_expand_decode as family_op,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_fp8__sm_103__unified import (  # noqa: E501
    modeling as M,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_fp8__sm_103__unified import (  # noqa: E501
    pool as P,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_fp8__sm_103__unified.ops.attention import (  # noqa: E501
    qsa_index_scores_cute__prefill as _isc,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_fp8__sm_103__unified.ops.attention import (  # noqa: E501
    qsa_index_scores_groups_cute__decode as _igc,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_fp8__sm_103__unified.ops.attention import (  # noqa: E501
    qsa_sparse_gqa_rowtile__prefill as _module,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_fp8__sm_103__unified.ops.attention import (  # noqa: E501
    qsa_sparse_gqa_splitk_cute__decode as mod,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_fp8__sm_103__unified.ops.attention import (  # noqa: E501
    qsa_triton_kernels__stable_jit as _stable,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_fp8__sm_103__unified.ops.attention.draft_continue_rows__decode import (  # noqa: E501
    draft_continue_select,
    draft_rows_gather,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_fp8__sm_103__unified.ops.attention.qsa_fp8_pages_to_bf16__prefill import (  # noqa: E501
    qsa_fp8_pages_to_bf16 as _f2b_op,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_fp8__sm_103__unified.ops.attention.qsa_index_split_plan__decode import (  # noqa: E501
    index_split_plan,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_fp8__sm_103__unified.ops.attention.qsa_paged_index_store import (  # noqa: E501
    qsa_paged_index_store,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_fp8__sm_103__unified.ops.attention.qsa_paged_kv_store import (  # noqa: E501
    qsa_paged_kv_store,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_fp8__sm_103__unified.ops.attention.qsa_paged_sparse_gqa import (  # noqa: E501
    qsa_paged_sparse_gqa,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_fp8__sm_103__unified.ops.attention.qsa_sparse_gqa_grouped_cute__decode import (  # noqa: E501
    qsa_sparse_gqa_grouped_cute as _sgg_op,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_fp8__sm_103__unified.ops.attention.qsa_sparse_gqa_rowtile__prefill import (  # noqa: E501
    qsa_sparse_gqa_rowtile as _rtl_op,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_fp8__sm_103__unified.ops.attention.qsa_sparse_gqa_rowtile__prefill import (  # noqa: E501
    rows_per_group,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_fp8__sm_103__unified.ops.attention.qsa_sparse_gqa_splitk_cute__decode import (  # noqa: E501
    qsa_sparse_gqa_splitk_cute as _sgc_op,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_fp8__sm_103__unified.ops.attention.qsa_spec_step_metadata__decode import (  # noqa: E501
    qsa_spec_step_metadata,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_fp8__sm_103__unified.ops.attention.qsa_topk_expand_cluster__decode import (  # noqa: E501
    qsa_topk_expand_cluster as _tkc_op,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_fp8__sm_103__unified.ops.attention.qsa_topk_expand_cluster__decode import (  # noqa: E501
    supports,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_fp8__sm_103__unified.ops.attention.qsa_topk_rows_cute__prefill import (  # noqa: E501
    qsa_topk_rows as _tkr_op,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_fp8__sm_103__unified.ops.attention.qsa_topk_rows_cute__prefill import (  # noqa: E501
    supports as _tkr_supports,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_fp8__sm_103__unified.ops.attention.qwen4_exp_attn_decode_prologue__decode import (  # noqa: E501
    qwen4_exp_attn_decode_prologue as _pro_op,
)
from tensorrt_llm._torch.attention.backends.sparse.qsa.kernels import (
    triton_qsa_generation_token_mapping,
)

__extra_import_path__ = ["../../.."]
from _validating import (  # noqa: E402,F401 — needs the path declared above; fixture by name
    assert_within_ulp,
    certifying,
)

assert torch.cuda.is_available(), "the attention entries require a CUDA device"

DEV = "cuda"
#: The element offset a straddling cell's pages lie on both sides of: a 32-bit offset breaks.
_OFFSET_EDGE = 2**31


def _seed(text: str) -> int:
    """A seed that is the same in every process; `hash()` of a str is not."""
    return zlib.crc32(text.encode())


def _bits(tensor: torch.Tensor) -> torch.Tensor:
    as_int = {1: torch.uint8, 2: torch.int16, 4: torch.int32}[tensor.element_size()]
    return tensor.contiguous().view(as_int)


def _hole(spec, table, allocated_cols):
    """One row that resolves to no slot: its (request, position)."""
    assert spec["hole"] == "unallocated_page", spec["hole"]
    assert int(table[0, allocated_cols[0]]) == -1
    return 0, allocated_cols[0] * spec["tokens_per_block"]


def _index_store_compare(out, ref) -> None:
    """Bitwise over both whole planes, as raw bits: a store is a move with one conversion that
    `reference` performs too, and comparing every slot is the stray-write check."""
    for name, got, want in zip(("index_cache", "position_cache"), out, ref):
        assert got.dtype == want.dtype, (name, got.dtype, want.dtype)
        assert got.shape == want.shape, (name, got.shape, want.shape)
        torch.testing.assert_close(
            _bits(got), _bits(want), rtol=0, atol=0, msg=lambda m, n=name: f"{n}: {m}"
        )


KEY_FILL = -7.0
POS_FILL = -7
ENGINE_MAX_NUM_TOKENS = 16384


def _key_fill_bits() -> int:
    return int(_bits(torch.tensor([KEY_FILL], dtype=torch.bfloat16))[0])


def _shared_page_planes(spec, extras: dict):
    """The NVFP4-KV pool's side planes: a page holds the `layers` index-K pages then the position
    page, so both caches are strided views of one allocation."""
    layers, tpb, hkv, head_dim, axes, pages = (
        spec[k] for k in ("layers", "tokens_per_block", "hkv", "head_dim", "axes", "pages")
    )
    index_bytes = tpb * hkv * head_dim * 2
    page = torch.empty(pages, layers * index_bytes + tpb * axes * 4, dtype=torch.uint8, device=DEV)
    page[:, : layers * index_bytes].view(torch.bfloat16).fill_(KEY_FILL)
    page[:, layers * index_bytes :].view(torch.int32).fill_(POS_FILL)
    index_cache = page[:, (layers - 1) * index_bytes : layers * index_bytes].view(torch.bfloat16)
    index_cache = index_cache.view(pages, tpb, hkv, head_dim)
    position_cache = page[:, layers * index_bytes :].view(torch.int32).view(pages, tpb, axes)
    assert index_cache.stride() == (spec["page_stride"], hkv * head_dim, head_dim, 1)
    assert position_cache.stride() == spec["position_strides"]
    extras["roles"] = page[:, : (layers - 1) * index_bytes]
    return index_cache, position_cache


def _coalesced_planes(spec, extras: dict):
    """The default target's bf16 index pool: the sparse layers' INDEX_KEY buffers coalesced into
    one allocation, the cell's plane a strided view; positions in their own pool."""
    layers, layer, tpb, hkv, head_dim, axes, pages = (
        spec[k] for k in ("layers", "layer", "tokens_per_block", "hkv", "head_dim", "axes", "pages")
    )
    pool = torch.full(
        (pages, layers, tpb, hkv, head_dim), KEY_FILL, dtype=torch.bfloat16, device=DEV
    )
    index_cache = pool[:, layer]
    assert index_cache.stride() == (spec["page_stride"], hkv * head_dim, head_dim, 1)
    position_cache = torch.full((pages, tpb, axes), POS_FILL, dtype=torch.int32, device=DEV)
    extras["coalesced"] = (pool, layer)
    return index_cache, position_cache


def _other_layers_untouched(pool: torch.Tensor, layer: int) -> None:
    fill, bits = _key_fill_bits(), pool.view(torch.int16)
    for start in range(0, pool.shape[0], 16384):
        band = bits[start : start + 16384]
        for other in range(pool.shape[1]):
            if other != layer:
                assert bool((band[:, other] == fill).all()), (
                    f"layer {other}'s index plane moved in pages {start}..{start + band.shape[0]}"
                )


def _index_store_build(spec, seed: int, planes=None):
    """The op's kwargs and extras for one cell, storing into `planes` when given."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    tpb, hkv, head_dim, axes = (spec[k] for k in ("tokens_per_block", "hkv", "head_dim", "axes"))
    slots, width, pages = spec["slots"], spec["width"], spec["pages"]
    segments = spec["segments"]
    requests = [r for r, (_, n) in enumerate(segments) for _ in range(n)]
    positions = [p for start, n in segments for p in range(start, start + n)]
    rows = len(positions)

    table = torch.randint(0, pages, (slots, width), generator=gen, dtype=torch.int32)
    order = torch.randperm(pages, generator=gen).to(torch.int32)
    used, allocated_cols = 0, []
    for r, (start, n) in enumerate(segments):
        cols = (start + n - 1) // tpb + 1
        table[r] = -1
        table[r, :cols] = order[used : used + cols]
        used += cols
        allocated_cols.append(cols)
    assert used <= pages and len(segments) <= slots, "pool or table too small for the batch"
    if spec.get("straddle"):
        # The last row stores into the pool's last page: rows land on both sides of 2^31.
        last_start, last_rows = segments[-1]
        column = (last_start + last_rows - 1) // tpb
        live = table[: len(segments)]
        holder = (live == pages - 1).nonzero()
        if len(holder):
            live[holder[0, 0], holder[0, 1]] = live[len(segments) - 1, column]
        live[len(segments) - 1, column] = pages - 1
    if spec["hole"] is not None:
        request, position = _hole(spec, table, allocated_cols)
        requests.append(request)
        positions.append(position)
        rows += 1
    table = table.to(DEV)

    projection = torch.randn(
        rows, spec["index_heads"] + hkv, head_dim, generator=gen, dtype=torch.float32
    ).to(device=DEV, dtype=spec["key_dtype"])
    token_k = projection[:, spec["index_heads"] :]
    if spec["key_layout"] == "fused_row":
        # The fused projection row in heads of 128; the raw index key is the second to last.
        heads = spec.get("row_width", 14080) // head_dim
        row = torch.randn(rows, heads, head_dim, generator=gen, dtype=torch.float32)
        token_k = row.to(device=DEV, dtype=spec["key_dtype"])[:, heads - 2 : heads - 2 + hkv]
        assert token_k.stride(0) == heads * head_dim

    request_indices = torch.tensor(requests, dtype=spec["request_dtype"], device=DEV)
    logical_positions = torch.tensor(positions, dtype=spec["position_dtype"], device=DEV)
    # Three distinct axes (an image span) in the engine's `[3, max_num_tokens]` buffer.
    coords = logical_positions.to(torch.int32)[:, None] + 7 * torch.arange(
        3, dtype=torch.int32, device=DEV
    )
    buffer = torch.full((3, ENGINE_MAX_NUM_TOKENS), -1, dtype=torch.int32, device=DEV)
    buffer[:, :rows] = coords.t()
    coords = buffer[:, :rows].t()

    extras = {}
    position_cache = None
    if planes is not None:
        index_cache, position_cache = planes
    elif spec["cache_layout"] == "shared_page":
        index_cache, position_cache = _shared_page_planes(spec, extras)
    elif spec["cache_layout"] == "coalesced":
        index_cache, position_cache = _coalesced_planes(spec, extras)
    else:
        index_cache = torch.full((pages, tpb, hkv, head_dim), KEY_FILL, device=DEV)
        index_cache = index_cache.to(torch.bfloat16)
    if position_cache is None:
        position_cache = torch.full((pages, tpb, axes), POS_FILL, dtype=torch.int32, device=DEV)
    args = dict(
        token_k=token_k, position_coordinates=coords, index_cache=index_cache,
        position_cache=position_cache, request_indices=request_indices,
        logical_positions=logical_positions, block_table=table, tokens_per_block=tpb,
    )  # fmt: skip
    return args, extras


def _check_index_store(cell, monkeypatch) -> None:
    args, extras = _index_store_build(cell.spec, seed=_seed(cell.why))
    if cell.spec.get("straddle"):
        assert (cell.spec["pages"] - 1) * args["index_cache"].stride(0) >= _OFFSET_EDGE
    ref = qsa_paged_index_store.reference(**args)
    out = qsa_paged_index_store(**args)
    assert out[0] is args["index_cache"] and out[1] is args["position_cache"]
    _index_store_compare(out, ref)
    if "roles" in extras:
        assert bool((extras["roles"].view(torch.int16) == _key_fill_bits()).all()), (
            "another index-K role of the shared page moved"
        )
    if "coalesced" in extras:
        _other_layers_untouched(*extras["coalesced"])


def _kv_store_compare(out, ref) -> None:
    """Bitwise over both role planes, which together are the whole pool: the stray-write check."""
    for name, got, want in zip(("k_cache", "v_cache"), out, ref):
        assert got.dtype == want.dtype, (name, got.dtype, want.dtype)
        assert got.shape == want.shape, (name, got.shape, want.shape)
        torch.testing.assert_close(
            _bits(got), _bits(want), rtol=0, atol=0, msg=lambda m, n=name: f"{n}: {m}"
        )


FILL = -2.0


def _check(result):
    err, *value = result
    assert err == cuda_drv.CUresult.CUDA_SUCCESS, err
    return value[0] if len(value) == 1 else value


class _ArrayInterface:
    def __init__(self, ptr: int, nbytes: int):
        self.__cuda_array_interface__ = dict(
            shape=(nbytes,), strides=None, typestr="|u1", data=(ptr, False), version=2
        )


class _KvEnginePool:
    """Layer `layer_slot`'s K and V views of the stock manager's QSA pool at `pages` pages. The
    range is reserved whole; memory is mapped only under the K and V planes of `touched` pages
    and page 0 (the tensor's base pointer)."""

    def __init__(self, spec, touched):
        dtype = spec["cache_dtype"]
        elem = torch.empty((), dtype=dtype).element_size()
        hkv, tpb, dim = spec["hkv"], spec["tokens_per_block"], spec["head_dim"]
        self.pages, self.touched = spec["pages"], sorted(set(touched))
        self.role = hkv * tpb * dim
        self.stride = spec["role_pages"] * self.role
        self.own = spec["layer_slot"] * 2 * self.role

        def planes(page):
            """This layer's K and V planes of `page`, in bytes of the range."""
            lo = (page * self.stride + self.own) * elem
            return lo, lo + 2 * self.role * elem

        self.written = [planes(p) for p in self.touched]
        dev = torch.cuda.current_device()
        prop = cuda_drv.CUmemAllocationProp()
        prop.type = cuda_drv.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
        prop.location.type = cuda_drv.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
        prop.location.id = dev
        gran = _check(
            cuda_drv.cuMemGetAllocationGranularity(
                prop,
                cuda_drv.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_RECOMMENDED,
            )
        )
        self.size = -(-self.pages * self.stride * elem // gran) * gran
        self.va = int(_check(cuda_drv.cuMemAddressReserve(self.size, gran, 0, 0)))
        merged = []
        for lo, hi in sorted({planes(0), *self.written}):
            lo, hi = lo // gran * gran, min(-(-hi // gran) * gran, self.size)
            if merged and lo <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], hi)
            else:
                merged.append([lo, hi])
        access = cuda_drv.CUmemAccessDesc()
        access.location.type = cuda_drv.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
        access.location.id = dev
        access.flags = cuda_drv.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
        self.maps, self.spans = [], merged
        for lo, hi in merged:
            handle = _check(cuda_drv.cuMemCreate(hi - lo, prop, 0))
            ptr = cuda_drv.CUdeviceptr(self.va + lo)
            _check(cuda_drv.cuMemMap(ptr, hi - lo, 0, handle, 0))
            self.maps.append((ptr, hi - lo, handle))
            _check(cuda_drv.cuMemSetAccess(ptr, hi - lo, [access], 1))

        self.raw = torch.as_tensor(_ArrayInterface(self.va, self.size), device=DEV)
        pool = self.raw.view(dtype).as_strided(
            (self.pages, 2, hkv, tpb, dim),
            (self.stride, self.role, tpb * dim, dim, 1),
            self.own,
        )
        self.k, self.v = pool[:, 0], pool[:, 1]
        self.index = torch.tensor(self.touched, dtype=torch.long, device=DEV)
        self.fill()

    def fill(self) -> None:
        pattern = _bits(torch.full((2,), FILL, dtype=torch.bfloat16).to(self.k.dtype))
        for lo, hi in self.spans:
            self.raw[lo:hi].view(pattern.dtype).fill_(int(pattern[0]))
        torch.cuda.synchronize()
        self.before = [self.raw[lo:hi].clone() for lo, hi in self.spans]

    def dense(self, table: torch.Tensor):
        """`(k, v, table)`: copies of the touched pages and `table` renumbered into them."""
        remap = torch.full((self.pages,), -1, dtype=torch.int32, device=DEV)
        remap[self.index] = torch.arange(len(self.touched), dtype=torch.int32, device=DEV)
        renamed = torch.where(table >= 0, remap[table.clamp_min(0).long()], -1).to(torch.int32)
        return self.k[self.index].clone(), self.v[self.index].clone(), renamed.contiguous()

    def gathered(self):
        return self.k[self.index], self.v[self.index]

    def assert_rest_unchanged(self) -> None:
        torch.cuda.synchronize()
        for (lo, hi), before in zip(self.spans, self.before):
            keep = torch.ones(hi - lo, dtype=torch.bool, device=DEV)
            for p_lo, p_hi in self.written:
                if p_hi > lo and p_lo < hi:
                    keep[max(p_lo, lo) - lo : min(p_hi, hi) - lo] = False
            assert torch.equal(self.raw[lo:hi][keep], before[keep]), (
                f"bytes [{lo}, {hi}) of the pool changed outside the touched pages' planes"
            )

    def close(self) -> None:
        torch.cuda.synchronize()
        del self.k, self.v, self.raw
        for ptr, size, handle in self.maps:
            _check(cuda_drv.cuMemUnmap(ptr, size))
            _check(cuda_drv.cuMemRelease(handle))
        _check(cuda_drv.cuMemAddressFree(cuda_drv.CUdeviceptr(self.va), self.size))


def _kv_store_build(spec, seed: int):
    """`(op kwargs, pool)`: a contiguous pool, or a `_KvEnginePool` for a `pool="engine"` cell."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    tpb, hkv, head_dim = spec["tokens_per_block"], spec["hkv"], spec["head_dim"]
    slots, width, pages = spec["slots"], spec["width"], spec["pages"]
    engine = spec.get("pool") == "engine"
    segments = spec["segments"]
    requests = [r for r, (_, n) in enumerate(segments) for _ in range(n)]
    positions = [p for start, n in segments for p in range(start, start + n)]

    table = torch.randint(0, pages, (slots, width), generator=gen, dtype=torch.int32)
    order = torch.randperm(pages, generator=gen).to(torch.int32)
    if engine:
        # The first row's page is the pool's last one, whose planes are its highest bytes.
        first = segments[0][0] // tpb
        last = int((order == pages - 1).nonzero()[0])
        order[[first, last]] = order[[last, first]]
    used, allocated_cols = 0, []
    for r, (start, n) in enumerate(segments):
        cols = (start + n - 1) // tpb + 1
        assert cols <= width, f"request {r} ends past the table's {width} columns"
        table[r] = -1
        table[r, :cols] = order[used : used + cols]
        used += cols
        allocated_cols.append(cols)
    assert used <= pages and len(segments) <= slots, "pool or table too small for the batch"
    if spec["hole"] is not None:
        request, position = _hole(spec, table, allocated_cols)
        requests.append(request)
        positions.append(position)

    rows = len(positions)
    assert rows == spec.get("rows", rows), (rows, spec["rows"])
    heads = spec["heads_q"] + 2 * hkv + spec.get("row_pad_heads", 0)
    qkv = torch.randn(rows, heads, head_dim, generator=gen, dtype=torch.float32)
    qkv = qkv.to(device=DEV, dtype=spec["row_dtype"])
    assert qkv.stride(0) == spec.get("row_stride", qkv.stride(0)), qkv.stride(0)
    k = qkv[:, spec["heads_q"] : spec["heads_q"] + hkv]
    v = qkv[:, spec["heads_q"] + hkv : spec["heads_q"] + 2 * hkv]
    touched = [int(table[r, p // tpb]) for r, p in zip(requests, positions)]
    table = table.to(DEV)

    if engine:
        pool = _KvEnginePool(spec, [t for t in touched if t >= 0])
        k_cache, v_cache = pool.k, pool.v
    else:
        pool = torch.full(
            (pages, 2, hkv, tpb, head_dim), FILL, dtype=torch.bfloat16, device=DEV
        ).to(spec["cache_dtype"])
        k_cache, v_cache = pool[:, 0], pool[:, 1]
    args = dict(
        k=k, v=v, k_cache=k_cache, v_cache=v_cache,
        request_indices=torch.tensor(requests, dtype=spec["request_dtype"], device=DEV),
        logical_positions=torch.tensor(positions, dtype=spec["position_dtype"], device=DEV),
        block_table=table, tokens_per_block=tpb,
    )  # fmt: skip
    return args, pool


def _kv_engine_reference(args, pool: _KvEnginePool):
    """The entry's reference over a dense copy of the touched pages."""
    dense_k, dense_v, renamed = pool.dense(args["block_table"])
    return qsa_paged_kv_store.reference(
        **{**args, "k_cache": dense_k, "v_cache": dense_v, "block_table": renamed}
    )


def _check_kv_store(cell, monkeypatch) -> None:
    args, pool = _kv_store_build(cell.spec, seed=_seed(cell.why))
    if not isinstance(pool, _KvEnginePool):
        before = pool.clone()
        ref = qsa_paged_kv_store.reference(**args)
        out = qsa_paged_kv_store(**args)
        assert out[0] is args["k_cache"] and out[1] is args["v_cache"]
        _kv_store_compare(out, ref)
        wrote = not torch.equal(_bits(pool), _bits(before))
        assert wrote == bool(cell.spec["segments"]), "a cell with rows writes, an empty one not"
        return
    try:
        top = max(pool.touched)
        assert top == pool.pages - 1, (top, pool.pages)
        assert top * pool.k.stride(0) >= 2**32, top * pool.k.stride(0)
        before = tuple(t.clone() for t in pool.gathered())
        ref = _kv_engine_reference(args, pool)
        out = qsa_paged_kv_store(**args)
        assert out[0] is args["k_cache"] and out[1] is args["v_cache"]
        got = pool.gathered()
        _kv_store_compare(got, ref)
        pool.assert_rest_unchanged()
        assert any(not torch.equal(_bits(g), _bits(b)) for g, b in zip(got, before)), "no write"
    finally:
        pool.close()


def _gqa_compare(out: torch.Tensor, ref: torch.Tensor) -> None:
    """4 ulp of each (row, head)'s magnitude and 2 ulp of relative RMS in q's dtype, the band of
    every sparse GQA entry here. The kernels round softmax probabilities to the cache dtype
    before P.V, so the error tracks the averaged values rather than the output, and a gated
    output is rounded twice; a page-shifted table or a dropped scale or token falls outside it."""
    assert out.dtype == ref.dtype, (out.dtype, ref.dtype)
    assert out.shape == ref.shape, (out.shape, ref.shape)
    if out.numel() == 0:
        return
    assert_within_ulp(out, ref, element_ulp=4.0, rms_ulp=2.0)


#: NaN's bit pattern per pool dtype, filled through an integer view (not every fill takes e4m3).
_NAN_BITS = {torch.bfloat16: (torch.int16, 0x7FC0), torch.float8_e4m3fn: (torch.uint8, 0x7F)}


def _selected_tokens(positions, seed, compress_ratio, token_topk, shared=False, chunk=None):
    """QSA's expansion layout: a random `min(available, block_topk)`-subset of the complete
    visible groups expanded to their tokens, the incomplete trailing group, then `-1`, front-packed.
    `shared`: every row ranks the groups by one score vector, as real selections overlap."""
    block_topk = token_topk // compress_ratio
    final_topk = token_topk + compress_ratio - 1
    gen = torch.Generator(device=DEV).manual_seed(seed)
    rows = positions.numel()
    visible = positions.long() + 1
    available = visible // compress_ratio
    taken = available.clamp(max=block_topk)
    width = max(block_topk, int(available.max().item()) if rows else 0)
    out = torch.empty(rows, final_topk, dtype=torch.int32, device=DEV)
    column = torch.arange(final_topk, device=DEV)[None, :]
    step = chunk or max(rows, 1)
    for start in range(0, rows, step):
        stop = min(start + step, rows)
        avail, take, vis = available[start:stop], taken[start:stop], visible[start:stop]
        if shared:
            scores = torch.rand(1, width, generator=gen, device=DEV).expand(stop - start, -1)
            gen.manual_seed(seed)
        else:
            scores = torch.rand(stop - start, width, generator=gen, device=DEV)
        group_ids = torch.arange(width, device=DEV)[None, :]
        scores = torch.where(group_ids < avail[:, None], scores, torch.inf)
        chosen = scores.argsort(dim=1)[:, :block_topk]
        group_of_column = (column // compress_ratio).clamp(max=block_topk - 1)
        body = torch.gather(chosen, 1, group_of_column.expand(stop - start, -1))
        body = body * compress_ratio + column % compress_ratio
        in_body = column < (take * compress_ratio)[:, None]
        tail_column = column - (take * compress_ratio)[:, None]
        tail = (avail * compress_ratio)[:, None] + tail_column
        in_tail = ~in_body & (tail_column >= 0) & (tail < vis[:, None])
        out[start:stop] = torch.where(in_body, body, torch.where(in_tail, tail, -1)).to(torch.int32)
    return out


def _straddling_pages(spec, needed: int, edge: int = _OFFSET_EDGE) -> int:
    """Pool size whose last `needed` pages lie on both sides of `edge` elements of the K plane."""
    role_page = spec["kv_heads"] * spec["tokens_per_block"] * spec["head_dim"]
    edge_page = math.ceil((edge - 2 * spec["layer"] * role_page) / (spec["slot_pages"] * role_page))
    return max(edge_page + needed // 2, needed)


def _paged_kv_state(spec, seed, dtype, pool_pages_of, slot_pages=2, layer=0, chunk=None,
                    scratch=False) -> SimpleNamespace:  # fmt: skip
    """The engine's paged K/V: a `[slots, pages]` table over seeded-permuted slots, `-1` past each
    sequence; K and V layer `layer`'s role planes of a NaN `[pool_pages, slot_pages, kv, tpb, D]`
    pool, or with `scratch` two contiguous pools (what `qsa_fp8_pages_to_bf16` writes)."""
    gen = torch.Generator(device=DEV).manual_seed(seed)
    tpb, kv, d = spec["tokens_per_block"], spec["kv_heads"], spec["head_dim"]
    segments = list(spec["segments"])
    positions = torch.cat([torch.arange(s, s + n, device=DEV) for s, n in segments])
    req = torch.cat([torch.full((n,), i, device=DEV) for i, (_, n) in enumerate(segments)])
    slots = spec["slots"]
    assert len(segments) <= slots
    slot_of = torch.randperm(slots, generator=torch.Generator().manual_seed(seed))[: len(segments)]
    mapped = [math.ceil((s + n) / tpb) for s, n in segments]
    assert max(mapped) <= spec["pages"]
    needed = sum(mapped) + 7
    pool_pages = pool_pages_of(needed)
    assert pool_pages >= needed
    first = pool_pages - needed
    page_ids = first + torch.randperm(needed, generator=gen, device=DEV).to(torch.int32)
    table = torch.full((slots, spec["pages"]), -1, dtype=torch.int32, device=DEV)
    cursor = 0
    for slot, count in zip(slot_of.tolist(), mapped):
        table[slot, :count] = page_ids[cursor : cursor + count]
        cursor += count
    if scratch:
        k, v = (torch.randn((pool_pages, kv, tpb, d), generator=gen, device=DEV).to(dtype)
                for _ in range(2))  # fmt: skip
    else:
        bits, nan = _NAN_BITS[dtype]
        pool = torch.full((pool_pages, slot_pages, kv, tpb, d), nan, dtype=bits, device=DEV)
        pool = pool.view(dtype)
        for p0 in range(first, pool_pages, chunk or needed):
            p1 = min(p0 + (chunk or needed), pool_pages)
            fresh = torch.randn(p1 - p0, 2, kv, tpb, d, generator=gen, device=DEV)
            pool[p0:p1, 2 * layer : 2 * layer + 2] = fresh.to(dtype)
        k, v = pool[:, 2 * layer], pool[:, 2 * layer + 1]
    rows = slot_of.to(DEV)[req]
    return SimpleNamespace(gen=gen, positions=positions, rows=rows, table=table, k=k, v=v)


def _assert_straddles(args, edge: int = _OFFSET_EDGE) -> None:
    """The mapped pages put both planes' element offsets on both sides of `edge`."""
    table = args["block_table"]
    pages = table[(table >= 0) & (table < args["k_cache"].shape[0])].long()
    for name in ("k_cache", "v_cache"):
        plane = args[name]
        ends = plane.storage_offset() + pages * plane.stride(0) + plane[0].numel() - 1
        assert bool((ends >= edge).any()), f"{name}: no mapped page lies past {edge}"
        assert bool((ends < edge).any()), f"{name}: every mapped page lies past {edge}"


def _assert_prefill_selection(selected, positions, spec) -> None:
    """The precondition the prefill kernels cannot check: front-packed, and exactly as many
    valid entries as the per-row scan bound the positions give."""
    valid = selected >= 0
    columns = torch.arange(selected.shape[1], device=selected.device)[None, :]
    assert bool((valid == (columns < valid.sum(1)[:, None])).all())
    cr, visible = spec["compress_ratio"], positions.long() + 1
    bound = (visible // cr).clamp(max=spec["token_topk"] // cr) * cr + visible % cr
    assert torch.equal(valid.sum(1), bound)


def _refill(args, seed, selection) -> None:
    """The engine's next step in the call's own buffers: new q, gate, selection and mapped K/V."""
    gen = torch.Generator(device=DEV).manual_seed(seed)
    for name in ("q", "output_gate"):
        if args.get(name) is not None:
            view = args[name]
            view.copy_(torch.randn(view.shape, generator=gen, device=DEV).to(view.dtype))
    args["selected_tokens"].copy_(selection(args))
    pages = args["block_table"][args["block_table"] >= 0].long()
    for name in ("k_cache", "v_cache"):
        plane = args[name]
        bits = _NAN_BITS[plane.dtype][0]
        fresh = torch.randn((pages.numel(), *plane.shape[1:]), generator=gen, device=DEV)
        plane.view(bits)[pages] = fresh.to(plane.dtype).view(bits)


def _captured(fn, refill):
    """`fn()` captured in a CUDA graph, `refill()` run, the graph replayed: the captured output."""
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = fn()
    refill()
    graph.replay()
    torch.cuda.synchronize()
    return out


def _replay_in_a_graph(entry, args, refill, first) -> None:
    """Captured once, inputs refilled in place, replayed: within the band of the new inputs'
    reference and bitwise an eager call on them."""
    replayed = _captured(lambda: entry(**args), refill).clone()
    assert not torch.equal(replayed, first), "the replay did not read the refilled inputs"
    _gqa_compare(replayed, entry.reference(**args))
    assert torch.equal(replayed, entry(**args)), "the replay differs from an eager call"


class _LaunchRecorder:
    """Records a kernel's grid and PDL flag, then delegates to the kernel."""

    def __init__(self, inner, name, log):
        self._inner, self._name, self._log = inner, name, log

    def __getitem__(self, grid):
        entry = [self._name, tuple(grid), False]
        self._log.append(entry)
        launch = self._inner[grid]

        def run(*args, **kwargs):
            entry[2] = bool(kwargs.get("launch_pdl", False))
            return launch(*args, **kwargs)

        return run

    def __call__(self, selected_tokens, *args, **kwargs):
        # the row-tile token plan: one CTA a group (its 10th argument); the block plan: its 9th
        grid = (args[8],) if self._name == "_group_plan" else (args[7],)
        self._log.append([self._name, grid, False])
        return self._inner(selected_tokens, *args, **kwargs)

    def warmup(self, *args, **kwargs):
        """A compile-only call (`jit_variants.compile_only`): no launch to record."""
        return self._inner.warmup(*args, **kwargs)


def _record_launches(monkeypatch, module, names) -> list:
    """Which of `module`'s kernels the next calls launch, in order: [name, grid, PDL]."""
    log: list = []
    for name in names:
        monkeypatch.setattr(module, name, _LaunchRecorder(getattr(module, name), name, log))
    return log


_KERNELS = {
    "_qsa_prefill_pair_stable_kernel": "prefill",
    "_qsa_splitk_stable_kernel": "splitk",
    "_qsa_merge_stable_kernel": "merge",
}


def _row_gate(q, spec):
    """The output gate's columns of the fused row `q` is read off: after q, k and v."""
    rows, d = q.shape[0], spec["head_dim"]
    row = q.as_strided((rows, q.stride(0)), (q.stride(0), 1))
    start = (spec["q_heads"] + 2 * spec["kv_heads"]) * d
    return row[:, start : start + spec["q_heads"] * d]


def _sparse_gqa_selection(spec, seed):
    positions = torch.cat([torch.arange(s, s + n, device=DEV) for s, n in spec["segments"]])
    return _selected_tokens(positions, seed, spec["compress_ratio"], spec["token_topk"])


def _sparse_gqa_build(spec, seed):
    """One cell's args: q off the packed or the fused projection row (`row_heads` heads)."""
    st = _paged_kv_state(
        spec, seed, spec["cache_dtype"], lambda needed: spec.get("pool_pages", needed),
        slot_pages=spec.get("slot_pages", 2), layer=spec.get("layer", 0),
    )  # fmt: skip
    rows, d, q_heads = st.positions.numel(), spec["head_dim"], spec["q_heads"]
    fused = spec["q_layout"] == "fused_row"
    row_heads = spec.get("row_heads", 55 if fused else q_heads + 2 * spec["kv_heads"])
    row = torch.randn(rows, row_heads * d, generator=st.gen, device=DEV).to(spec["q_dtype"])
    q = row.view(rows, row_heads, d)[:, :q_heads]
    args = dict(
        q=q, k_cache=st.k, v_cache=st.v, block_table=st.table,
        selected_tokens=_sparse_gqa_selection(spec, seed + 3),
        request_indices=st.rows.to(spec["request_dtype"]),
        tokens_per_block=spec["tokens_per_block"], softmax_scale=d**-0.5,
    )  # fmt: skip
    if spec["prefill_pair"]:
        args.update(query_positions=st.positions, compress_ratio=spec["compress_ratio"])
    if spec.get("gate") == "fused_row":
        assert fused, "the gate is read off the fused row"
        if spec.get("gate_fused", True):
            args.update(output_gate=_row_gate(q, spec))
    return args


def _check_sparse_gqa(cell, monkeypatch) -> None:
    """The cell's branch (its kernels, grid and PDL, recorded), then the band; a launch with no
    merge refuses the gate (the call site applies it after); a `capture` cell is replayed."""
    launches = _record_launches(monkeypatch, _stable, _KERNELS)
    spec, seed = cell.spec, sum(map(ord, cell.why))
    args = _sparse_gqa_build(spec, seed)
    q, rows = args["q"], args["q"].shape[0]
    assert rows == spec.get("rows", rows)
    if "row_heads" in spec:
        assert q.stride(0) == spec["row_heads"] * spec["head_dim"]
        if "output_gate" in args:
            gate = args["output_gate"]
            assert gate.stride(0) == q.stride(0)
            assert gate.untyped_storage().data_ptr() == q.untyped_storage().data_ptr()
    if "rows_per_request" in spec:
        assert {n for _, n in spec["segments"]} == {spec["rows_per_request"]}
    if "offset_edge" in spec:
        _assert_straddles(args, spec["offset_edge"])
    if spec["prefill_pair"]:
        _assert_prefill_selection(args["selected_tokens"], args["query_positions"], spec)
    out = qsa_paged_sparse_gqa(**args)
    assert [_KERNELS[name] for name, _, _ in launches] == list(spec["launches"])
    assert launches[0][1][:2] == (rows, spec["kv_heads"])
    if "splits" in spec:
        assert launches[0][1][2] == spec["splits"], launches
    assert all(pdl == spec.get("pdl", False) for _, _, pdl in launches), launches
    if "output_gate" in args:
        gated = (q, args["k_cache"], args["selected_tokens"], args["output_gate"])
        assert qsa_paged_sparse_gqa.fuses_output_gate(*gated)
    assert out.shape == (rows, spec["q_heads"], spec["head_dim"])
    assert out.dtype == q.dtype and out.is_contiguous()
    assert not bool(out.isnan().any())
    _gqa_compare(out, qsa_paged_sparse_gqa.reference(**args))
    if spec.get("gate") == "fused_row" and not spec.get("gate_fused", True):
        gate = _row_gate(q, spec)
        assert not qsa_paged_sparse_gqa.fuses_output_gate(
            q, args["k_cache"], args["selected_tokens"], gate
        )
        with pytest.raises(ValueError, match="output-gate fusion requires"):
            qsa_paged_sparse_gqa(**args, output_gate=gate)
    if spec.get("capture"):
        eager = [tuple(entry) for entry in launches]
        del launches[:]

        def refill():
            assert [tuple(entry) for entry in launches] == eager, "the capture's launches"
            _refill(args, seed + 1, lambda a: _sparse_gqa_selection(spec, seed + 4))

        _replay_in_a_graph(qsa_paged_sparse_gqa, args, refill, out)


PRO_D, PRO_IH, PRO_ID, PRO_TPB, PRO_CR, PRO_ROT, PRO_WIDTH, PRO_SLOTS = (
    256, 4, 128, 32, 4, 64, 8196, 8
)  # fmt: skip
#: (q heads, KV heads) per rank; fused row `[q | k | v | gate | index q 4 x 128, k | v slot]`.
PRO_GEOMS = {"tp4": (6, 1), "tp2": (12, 1), "dep4": (24, 2)}
PRO_ROLE_PAGES, PRO_LAYER_SLOT, PRO_INDEX_LAYERS = 24, 11, 12
#: One bf16 step in 2^-8 units of the row scale; the relative RMS bound is empirical.
PRO_ELEMENT_ULP, PRO_RMS_ULP = 2.0, 0.05


def _pro_geometry(geom):
    hq, hk = PRO_GEOMS[geom]
    qkv = (hq + 2 * hk) * PRO_D
    iq_col = qkv + hq * PRO_D
    return hq, hk, qkv, iq_col, iq_col + (PRO_IH + 2) * PRO_ID


def _pro_edge_rows(proj, hq, hk) -> None:
    """Zero, -0.0 and outlier blocks in the k head past the rotary prefix and in the v head."""
    k0, v0 = hq * PRO_D, (hq + hk) * PRO_D
    edges = (
        (0, slice(k0 + 128, k0 + 144), 0.0),
        (1, slice(k0 + 160, k0 + 176), -0.0),
        (2, slice(k0 + 200, k0 + 201), 900.0),
        (0, slice(v0 + 16, v0 + 32), 0.0),
        (1, slice(v0 + 48, v0 + 64), torch.tensor([-0.0] * 8 + [0.0] * 8)),
        (3, slice(v0 + 40, v0 + 41), -900.0),
        (2, slice(v0 + 64, v0 + 80), 1e-30),
    )
    for r, cols, value in edges:
        if r < proj.shape[0]:
            proj[r, cols] = value


def _pro_build(spec, seed):
    g = torch.Generator(device=DEV).manual_seed(seed)
    hq, hk, qkv_width, iq_col, row_width = _pro_geometry(spec["geom"])
    per = spec.get("rows_per_request", 1)
    layout = [(r, p + i) for r, p in enumerate(spec["pos"]) for i in range(per)]
    pos = [p for _, p in layout]
    rows = len(pos)
    last = {}
    for req, p in layout:
        last[req] = max(last.get(req, 0), p)
    slots = max(PRO_SLOTS, len(last) + 3)
    engine = spec.get("engine", False)
    if engine:
        # Enough pages that the last ones start past 2^31 elements; rows go on those.
        page_elems = PRO_ROLE_PAGES * hk * PRO_TPB * PRO_D
        pages = _OFFSET_EDGE // page_elems + 2 + sum(p // PRO_TPB + 1 for p in last.values())
        perm = torch.arange(pages - 1, -1, -1, device=DEV, dtype=torch.int32)
    else:
        pages = sum(p // PRO_TPB + 1 for p in last.values()) + 64
        perm = torch.randperm(pages, generator=g, device=DEV).to(torch.int32)
    block_table = torch.full((slots, PRO_WIDTH), -1, dtype=torch.int32, device=DEV)
    used = 0
    for req, p in last.items():
        n = p // PRO_TPB + 1
        block_table[req, :n] = perm[used : used + n]
        used += n
    request = torch.tensor([req for req, _ in layout], dtype=torch.int32, device=DEV)
    if spec.get("unmapped") is not None:
        request[spec["unmapped"]] = slots + 5
    logical = torch.tensor(pos, dtype=torch.int64, device=DEV)
    ids = torch.stack([logical, logical + 3, logical + 7]).to(torch.int32).contiguous()
    buf = torch.empty(3, 1, rows + 5, dtype=torch.int32, device=DEV)
    buf[:, 0, :rows] = ids
    coords = torch.transpose(buf[:, 0, :rows], 0, 1)
    table_rows = max(pos) + 8
    angles = torch.rand(table_rows, PRO_ROT // 2, generator=g, device=DEV) * 6.28
    cos_sin = torch.cat([torch.cos(angles), torch.sin(angles)], dim=1).contiguous()
    proj = (torch.randn(rows, row_width, generator=g, device=DEV) * 2.0).to(torch.bfloat16)
    if spec.get("edge"):
        _pro_edge_rows(proj, hq, hk)
    out = torch.randn(rows, qkv_width + 64, generator=g, device=DEV).to(torch.bfloat16)
    index_scratch = torch.randn(rows, PRO_IH + 2, PRO_ID, generator=g, device=DEV)
    index_scratch = index_scratch.to(torch.bfloat16)
    weights = [
        (0.3 * torch.randn(n, generator=g, device=DEV)).to(torch.bfloat16)
        for n in (PRO_D, PRO_D, PRO_ID, PRO_ID)
    ]
    kv_dtype = torch.float8_e4m3fn if spec.get("fp8") else torch.bfloat16
    if engine:
        roles = torch.zeros(pages, PRO_ROLE_PAGES, hk, PRO_TPB, PRO_D, dtype=kv_dtype, device=DEV)
        pool = roles[:, 2 * PRO_LAYER_SLOT : 2 * PRO_LAYER_SLOT + 2]
        planes = torch.randn(pages, PRO_INDEX_LAYERS, PRO_TPB, 1, PRO_ID, generator=g, device=DEV)
        index_cache = planes.to(torch.bfloat16)[:, PRO_LAYER_SLOT]
    else:
        pool = torch.randn(pages, 2, hk, PRO_TPB, PRO_D, generator=g, device=DEV).to(kv_dtype)
        index_cache = torch.randn(pages, PRO_TPB, 1, PRO_ID, generator=g, device=DEV)
        index_cache = index_cache.to(torch.bfloat16)
    position_cache = torch.randint(
        0, table_rows, (pages, PRO_TPB, 3), generator=g, device=DEV, dtype=torch.int32
    )
    return dict(
        proj=proj, out=out[:, :qkv_width], index_q=index_scratch[:, :PRO_IH], position_ids=ids,
        coords=coords, cos_sin=cos_sin, q_weight=weights[0], k_weight=weights[1],
        index_q_weight=weights[2], index_k_weight=weights[3], k_cache=pool[:, 0],
        v_cache=pool[:, 1], index_cache=index_cache, position_cache=position_cache,
        request_indices=request, logical_positions=logical, block_table=block_table,
        tokens_per_block=PRO_TPB, compress_ratio=PRO_CR, num_heads_q=hq, num_heads_k=hk, head_dim=PRO_D,
        index_heads=PRO_IH, index_head_dim=PRO_ID, index_q_col=iq_col, rotary_dim=PRO_ROT, eps=1e-6,
        mrope_section1=11, mrope_section2=10, rows_per_request=per,
    )  # fmt: skip


def _pro_clone(args):
    """The arguments over fresh copies of every allocation they view, and the copies keyed by
    the first argument viewing each."""
    out, copies, keys = dict(args), {}, {}
    for name, t in args.items():
        if not isinstance(t, torch.Tensor):
            continue
        key = keys.setdefault(t.untyped_storage().data_ptr(), name)
        if key not in copies:
            storage = torch.empty(0, dtype=torch.uint8, device=t.device).set_(t.untyped_storage())
            copies[key] = storage.clone()
        view = torch.empty(0, dtype=t.dtype, device=t.device)
        view.set_(copies[key].untyped_storage(), t.storage_offset(), t.shape, t.stride())
        out[name] = view
    return out, copies


def _pro_written_slots(args):
    """(row, page, slot) of every row whose page lookup holds, from the host's view."""
    table = args["block_table"].cpu()
    pages = args["k_cache"].shape[0]
    found = []
    for t, (req, p) in enumerate(
        zip(args["request_indices"].tolist(), args["logical_positions"].tolist())
    ):
        column = p // PRO_TPB
        if 0 <= req < table.shape[0] and column < table.shape[1]:
            page = int(table[req, column])
            if 0 <= page < pages:
                found.append((t, page, p % PRO_TPB))
    return found


def _pro_check(got, want) -> None:
    """Band the rotated values (only the rms's reduction order differs), overwrite them in `got`,
    then require every byte bitwise."""
    g, w = got[0], want[0]
    hq, hk = g["num_heads_q"], g["num_heads_k"]
    width = (hq + hk) * PRO_D

    def banded(a, b):
        a, b = a.float().reshape(-1, a.shape[-1]), b.float().reshape(-1, b.shape[-1])
        assert_within_ulp(a.to(torch.bfloat16), b.to(torch.bfloat16), PRO_ELEMENT_ULP, PRO_RMS_ULP)

    banded(g["out"][:, :width], w["out"][:, :width])
    banded(g["index_q"], w["index_q"])
    slots = _pro_written_slots(g)
    k_got = g["out"][:, hq * PRO_D : width].reshape(-1, hk, PRO_D)
    for t, page, slot in slots:
        own = k_got[t].to(g["k_cache"].dtype)
        assert torch.equal(g["k_cache"][page, :, slot].view(torch.uint8), own.view(torch.uint8)), (
            f"row {t}: the K pool's row is not the packed k converted"
        )
        g["k_cache"][page, :, slot] = w["k_cache"][page, :, slot]
    ends = [(t, page, slot) for t, page, slot in slots if (slot + 1) % PRO_CR == 0]
    if ends:
        a = torch.stack([g["index_cache"][page, slot, 0] for _, page, slot in ends])
        b = torch.stack([w["index_cache"][page, slot, 0] for _, page, slot in ends])
        banded(a, b)
        for _, page, slot in ends:
            g["index_cache"][page, slot, 0] = w["index_cache"][page, slot, 0]
    g["out"][:, :width] = w["out"][:, :width]
    g["index_q"].copy_(w["index_q"])
    assert got[1].keys() == want[1].keys()
    for key, copy in got[1].items():
        differ = (copy != want[1][key]).nonzero()
        assert differ.numel() == 0, f"{key}: {differ.shape[0]} bytes differ"


def _check_decode_prologue(cell, monkeypatch) -> None:
    args = _pro_build(cell.spec, seed=_seed(cell.why) % (2**31))
    want = _pro_clone(args)
    _pro_op.reference(**want[0])
    got = _pro_clone(args)
    _pro_op(**got[0])
    torch.cuda.synchronize()
    _pro_check(got, want)


def _plan_operands(spec, g):
    n, p = spec["requests"], spec["rows_per_request"]
    kind = spec["kind"]
    if kind == "equal":
        windows = torch.full((n,), 25000, dtype=torch.int64)
    else:
        windows = torch.randint(500, 62000, (n,), generator=g)
        if kind == "padded":
            windows[-3:] = 1
    rows = n * p
    visible = windows.repeat_interleave(p) + torch.arange(p).repeat(n) // 2
    return (
        visible.to(torch.int32).to(DEV),
        torch.arange(n, dtype=torch.int32).repeat_interleave(p).to(DEV),
        (visible * 4 + torch.randint(0, 4, (rows,), generator=g)).to(DEV),
        (visible * 4 + 7).to(torch.int32).to(DEV),
        torch.zeros(rows, dtype=torch.int32, device=DEV),
    )


def _check_index_split_plan(cell, monkeypatch) -> None:
    """Bitwise the reference, eager and replayed on new windows; the ranks' summed windows
    balanced and the padding requests spread over them."""
    sp = cell.spec
    n, p, world, rank = sp["requests"], sp["rows_per_request"], sp["world"], sp["rank"]
    g = torch.Generator().manual_seed(_seed(cell.why))

    def check(got, vectors):
        want = index_split_plan.reference(*vectors, n, p, world, rank)
        for a, b in zip(got, want):
            assert a.dtype == b.dtype and torch.equal(a, b)
        perm = got.perm_rows.long()
        share = n // world * p
        win = vectors[0].long()
        sums = [int(win[perm[r * share : (r + 1) * share]].sum()) for r in range(world)]
        assert max(sums) - min(sums) <= int(win.max()) * p, sums
        if sp["kind"] == "padded":
            pads = set(range((n - 3) * p, n * p))
            owners = {int(torch.nonzero(perm == r_).item()) // share for r_ in pads}
            assert len(owners) > 1, "the padding requests land on more than one rank"

    vectors = _plan_operands(sp, g)
    check(index_split_plan(*vectors, n, p, world, rank), vectors)
    s_vec, fresh = tuple(v.clone() for v in vectors), _plan_operands(sp, g)
    out = _captured(
        lambda: index_split_plan(*s_vec, n, p, world, rank),
        lambda: [a.copy_(b) for a, b in zip(s_vec, fresh)],
    )
    check(out, fresh)


def _check_draft_continue_select(cell, monkeypatch) -> None:
    """Bitwise `_draft_continue_rows` and the row metadata's `index_select`s, eager and replayed."""
    seqs, p = cell.spec["seqs"], cell.spec["rows_per_request"]
    g = torch.Generator(device=DEV).manual_seed(_seed(cell.why))
    rows = seqs * p + 5

    def operands():
        accepted = torch.randint(1, p + 1, (seqs + 3,), device=DEV, generator=g).to(torch.int32)
        vectors = (
            torch.randint(0, 1 << 20, (rows,), device=DEV, generator=g).to(torch.int32),
            torch.randint(0, 1 << 20, (rows,), device=DEV, generator=g).to(torch.int32),
            torch.randint(0, 1 << 40, (rows,), device=DEV, generator=g),
            torch.randint(0, seqs, (rows,), device=DEV, generator=g).to(torch.int32),
        )
        return accepted, vectors

    def check(got, accepted, vectors):
        want = draft_continue_select.reference(accepted, seqs, p, vectors)
        assert torch.equal(got[0], want[0])
        for a, b in zip(got[1], want[1]):
            assert a.dtype == b.dtype and torch.equal(a, b)
        core = types.SimpleNamespace(dp_size=1)
        md = types.SimpleNamespace(qsa_mtp_num_accepted=accepted)
        assert torch.equal(got[0], M._draft_continue_rows(core, md, seqs, p, False))

    accepted, vectors = operands()
    check(draft_continue_select(accepted, seqs, p, vectors), accepted, vectors)
    s_acc, s_vec = accepted.clone(), tuple(v.clone() for v in vectors)
    accepted, vectors = operands()
    out = _captured(
        lambda: draft_continue_select(s_acc, seqs, p, s_vec),
        lambda: [a.copy_(b) for a, b in zip((s_acc, *s_vec), (accepted, *vectors))],
    )
    check(out, accepted, vectors)


def _check_draft_rows_gather(cell, monkeypatch) -> None:
    """Bitwise `index_select` of strided and one-column row operands, eager and replayed."""
    rows, widths = cell.spec["rows"], cell.spec["widths"]
    g = torch.Generator(device=DEV).manual_seed(_seed(cell.why))
    src_rows = 3 * rows + 2
    index = torch.randperm(src_rows, device=DEV, generator=g)[:rows].to(torch.int64)
    operands = []
    for k, w in enumerate(widths):
        pad = 64 * (k % 2)  # every other operand a column slice of a wider row
        buf = torch.randn(src_rows, w + pad, device=DEV, generator=g).to(torch.bfloat16)
        operands.append(buf[:, pad : pad + w])
    got = draft_rows_gather(index, tuple(operands))
    want = draft_rows_gather.reference(index, tuple(operands))
    for a, b, t in zip(got, want, operands):
        assert a.is_contiguous() and torch.equal(a, b)
        assert torch.equal(a, torch.index_select(t, 0, index))
    s_idx = index.clone()
    index2 = torch.randperm(src_rows, device=DEV, generator=g)[:rows].to(torch.int64)
    out = _captured(lambda: draft_rows_gather(s_idx, tuple(operands)), lambda: s_idx.copy_(index2))
    for a, t in zip(out, operands):
        assert torch.equal(a, torch.index_select(t, 0, index2))


def _sgc_build(spec, seed):
    st = _paged_kv_state(
        spec, seed, spec["kv"], lambda needed: _straddling_pages(spec, needed),
        slot_pages=spec["slot_pages"], layer=spec["layer"], chunk=4096,
    )  # fmt: skip
    rows, d, hq, kv = st.positions.numel(), spec["head_dim"], spec["q_heads"], spec["kv_heads"]
    qkv = torch.randn(rows, hq + 2 * kv, d, generator=st.gen, device=DEV).to(torch.bfloat16)
    q = qkv[:, :hq]
    cr, topk = spec["compress_ratio"], spec["token_topk"]
    selected = _selected_tokens(st.positions, seed + 3, cr, topk)
    if spec.get("holes"):
        selected[:, 1::3] = -1
    offset = spec.get("row_offset", 0)
    if offset:
        # The generation rows of a mixed step: a row slice of a wider selection buffer.
        wide = torch.full((offset + rows, selected.shape[1]), -1, dtype=torch.int32, device=DEV)
        wide[offset:] = selected
        selected = wide[offset:]
    request_indices = st.rows.to(torch.int32)
    if spec.get("foreign"):
        request_indices[0] = -1
    args = dict(
        q=q, k_cache=st.k, v_cache=st.v, block_table=st.table, selected_tokens=selected,
        request_indices=request_indices, tokens_per_block=spec["tokens_per_block"],
        softmax_scale=d**-0.5,
    )  # fmt: skip
    gate = spec.get("gate")
    if gate == "row":
        # The trunk's fused input projection row `[q | k | v | gate | index q, k]`.
        row = torch.randn(rows, spec["row_heads"] * d, generator=st.gen, device=DEV).to(q.dtype)
        start = (hq + 2 * kv) * d
        args["output_gate"] = row[:, start : start + hq * d]
    elif gate == "interleaved":
        # The draft layer's q projection `[T, Hq, 2 D]`: q then the gate, per head.
        qg = torch.randn(rows, hq, 2 * d, generator=st.gen, device=DEV).to(q.dtype)
        args["output_gate"] = qg[:, :, d:]
    elif gate == "contiguous":
        # A gate gathered out of the projection row (a draft-extend step's continuing rows).
        args["output_gate"] = torch.randn(rows, hq * d, generator=st.gen, device=DEV).to(q.dtype)
    return args


def _rolled_selection(args):
    """The same selection rolled by one column: other tokens, the layout kept."""
    sel = args["selected_tokens"]
    return torch.where(sel >= 0, torch.roll(sel, 1, dims=1).clamp_min(0), sel)


def _check_sparse_gqa_splitk(cell, monkeypatch) -> None:
    spec, seed = cell.spec, sum(map(ord, cell.why))
    args = _sgc_build(spec, seed)
    _assert_straddles(args)
    q, k = args["q"], args["k_cache"]
    rows = q.shape[0]
    assert _sgc_op.serves(q, k, args["selected_tokens"]), "the target sends this call here"
    if "splits" in spec:
        width = args["selected_tokens"].shape[1]
        elem = 1 if spec["kv"] == torch.float8_e4m3fn else 2
        assert mod.plan_splits(rows, spec["kv_heads"], width, elem)[0] == spec["splits"]
    out = _sgc_op(**args)
    assert out.shape == (rows, spec["q_heads"], spec["head_dim"])
    assert out.dtype == torch.bfloat16 and out.is_contiguous()
    assert not bool(out.isnan().any())
    _gqa_compare(out, _sgc_op.reference(**args))
    if spec.get("foreign"):
        assert bool((out[0] == 0).all()), "a row of no request attends nothing"
    if spec.get("capture"):
        _replay_in_a_graph(_sgc_op, args, lambda: _refill(args, seed + 1, _rolled_selection), out)


def _sgg_selection(spec, seed):
    """Rows of a request select from its block scores plus `overlap` x noise (None: independent)."""
    cr, token_topk, spread = spec["compress_ratio"], spec["token_topk"], spec.get("overlap")
    block_topk, final_topk = token_topk // cr, token_topk + cr - 1
    segments = list(spec["segments"])
    positions = torch.cat([torch.arange(s, s + n, device=DEV) for s, n in segments])
    requests = torch.cat([torch.full((n,), i, device=DEV) for i, (_, n) in enumerate(segments)])
    gen = torch.Generator(device=DEV).manual_seed(seed)
    rows = positions.numel()
    visible = positions.long() + 1
    available = visible // cr
    taken = available.clamp(max=block_topk)
    width = max(block_topk, int(available.max().item()))
    if spread is None:
        scores = torch.rand(rows, width, generator=gen, device=DEV)
    else:
        base = torch.rand(int(requests.max()) + 1, width, generator=gen, device=DEV)
        scores = base[requests] + spread * torch.rand(rows, width, generator=gen, device=DEV)
    group_ids = torch.arange(width, device=DEV)[None, :]
    scores = torch.where(group_ids < available[:, None], scores, -torch.inf)
    chosen = scores.topk(block_topk, dim=1).indices.sort(dim=1).values
    column = torch.arange(final_topk, device=DEV)[None, :]
    group_of_column = (column // cr).clamp(max=block_topk - 1).expand(rows, -1)
    body = torch.gather(chosen, 1, group_of_column) * cr + column % cr
    in_body = column < (taken * cr)[:, None]
    tail_column = column - (taken * cr)[:, None]
    tail = (available * cr)[:, None] + tail_column
    in_tail = ~in_body & (tail_column >= 0) & (tail < visible[:, None])
    selected = torch.where(in_body, body, torch.where(in_tail, tail, torch.full_like(body, -1)))
    return selected.to(torch.int32)


def _check_sparse_gqa_grouped(cell, monkeypatch) -> None:
    """Within the band of the reference and of the per-row entry, and deterministic."""
    spec, seed = cell.spec, sum(map(ord, cell.why))
    args = _sgc_build({k: v for k, v in spec.items() if k != "holes"}, seed)
    args["selected_tokens"].copy_(_sgg_selection(spec, seed + 5))
    args["rows_per_request"] = spec["rows_per_request"]
    _assert_straddles(args)
    q, k = args["q"], args["k_cache"]
    rows = q.shape[0]
    assert _sgg_op.serves(q, k, args["selected_tokens"], args["rows_per_request"])
    out = _sgg_op(**args)
    assert out.shape == (rows, spec["q_heads"], spec["head_dim"])
    assert out.dtype == torch.bfloat16 and out.is_contiguous()
    assert not bool(out.isnan().any())
    _gqa_compare(out, _sgg_op.reference(**args))
    _gqa_compare(out, _sgc_op(**{k_: v for k_, v in args.items() if k_ != "rows_per_request"}))
    assert torch.equal(out, _sgg_op(**args)), "two calls on the same inputs differ"
    if spec.get("foreign"):
        assert bool((out[0] == 0).all()), "a row of no request attends nothing"
    if spec.get("capture"):
        refill = lambda: _refill(args, seed + 1, lambda a: _sgg_selection(spec, seed + 8))  # noqa: E731
        _replay_in_a_graph(_sgg_op, args, refill, out)


TKC_K, TKC_CR, TKC_TOKEN_TOPK = 512, 4, 2048


def _tkc_build(spec, seed):
    """One cell's arguments, with garbage past each row's visible blocks that must not be read."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    W = spec["width"]
    if "requests" in spec:
        # A verify step's 1 + draft rows: every row's KV length is the request's last row + 1.
        pos = torch.cat(
            [torch.arange(s, s + c, dtype=torch.int64, device=DEV) for s, c in spec["requests"]]
        )
        seqlen = torch.cat(
            [torch.full((c,), s + c, dtype=torch.int32, device=DEV) for s, c in spec["requests"]]
        )
    else:
        pos = torch.tensor(spec["pos"], dtype=torch.int64, device=DEV)
        seqlen = (pos + 1).to(torch.int32)
    rows = pos.numel()
    assert rows == spec["rows"], (rows, spec["rows"])
    scores = torch.randn(rows, W, generator=g, device=DEV).abs() * 3.0
    if spec.get("relu_sparse"):
        # Fewer positive scores than the budget, so the zero class straddles it.
        scores = torch.relu(torch.randn(rows, W, generator=g, device=DEV) - 2.2)
        scores[:, ::7] = -0.0  # negative zero ties with zero
    if spec.get("relu_sum"):
        # The indexer's form: four heads' ReLUs summed, times a scale (zero in 1 of 16).
        scores = torch.zeros(rows, W, device=DEV)
        for _ in range(4):
            scores += torch.relu(torch.randn(rows, W, generator=g, device=DEV))
        scores *= 0.0884
    if spec.get("equal"):
        # Every score tied: the budget is the highest-index blocks.
        scores = torch.full((rows, W), 0.5, device=DEV)
    if spec.get("lognormal"):
        # bf16-like resolution, so equal scores tie; a zero in every other row.
        scores = torch.exp(1.74 + 0.2 * torch.randn(rows, W, generator=g, device=DEV))
        scores = scores.to(torch.bfloat16).float()
        scores[::2, 5] = 0.0
    ends = ((pos + 1) // TKC_CR).to(torch.int32)
    assert int(ends.max()) <= W, "a window wider than the score table"
    cols = torch.arange(W, device=DEV)
    scores = torch.where(cols[None, :] >= ends[:, None], 1.0e30, scores)
    scores = torch.where(cols[None, :] == ends[:, None] + 3, float("nan"), scores)
    topk = torch.full((rows, TKC_K), -7, dtype=torch.int32, device=DEV)
    return [scores, ends, pos, seqlen, topk, TKC_CR, TKC_TOKEN_TOPK]


def _check_topk_expand_cluster(cell, monkeypatch) -> None:
    args = _tkc_build(cell.spec, seed=_seed(cell.why) % 2**31)
    assert supports(args[0]), "a cell outside the target's routing to this entry"
    want_topk, want_sel = _tkc_op.reference(*args)
    sel = _tkc_op(*args)
    assert sel.dtype == torch.int32 and sel.shape == want_sel.shape
    assert torch.equal(args[4], want_topk), "block list"
    assert torch.equal(sel, want_sel), "token selection"


# Scratch and table are pre-filled with sentinels so a missing or a stray write shows.

F2B_SCRATCH_FILL = 0x7FC1  # a bf16 NaN
F2B_TABLE_FILL = -7


def _f2b_build(spec, seed):
    gen = torch.Generator(device=DEV).manual_seed(seed)
    tpb, kv, d = spec["tokens_per_block"], spec["kv_heads"], spec["head_dim"]
    counts = [math.ceil(n / tpb) for n in spec["kv_lens"]]
    needed = sum(counts) + 7
    pool_pages = (_straddling_pages(spec, needed) if spec["straddle"] else needed) + 5
    role_pages, layer = spec["slot_pages"], spec["layer"]
    # Random e4m3 codes (NaN codes 0x7F / 0xFF excluded), in the layer's two planes only.
    data = torch.zeros((pool_pages, role_pages, kv, tpb, d), dtype=torch.uint8, device=DEV)
    first = pool_pages - needed
    for p0 in range(first, pool_pages, 1024):
        p1 = min(p0 + 1024, pool_pages)
        codes = torch.randint(0, 254, (p1 - p0, 2, kv, tpb, d), generator=gen, device=DEV)
        codes = torch.where(codes >= 0x7F, codes + 1, codes)
        data[p0:p1, 2 * layer : 2 * layer + 2] = codes.to(torch.uint8)
    pool = data.view(torch.float8_e4m3fn)
    width, slots_total = spec["width"], spec["slots"]
    table = torch.full((slots_total, width), -1, dtype=torch.int32, device=DEV)
    slots = torch.randperm(slots_total, generator=torch.Generator().manual_seed(seed))
    slots = slots[: len(counts)]
    page_ids = first + torch.randperm(needed, generator=gen, device=DEV).to(torch.int32)
    cursor = 0
    for slot, count in zip(slots.tolist(), counts):
        table[slot, :count] = page_ids[cursor : cursor + count]
        cursor += count
    if spec["holes"]:
        head = slots[0].item()
        table[head, 1] = -1  # unmapped inside the sequence
        table[head, 2] = pool_pages + 3  # past the pool
    counts_t = torch.tensor(counts, dtype=torch.int64, device=DEV)
    out_shape = (int(counts_t.sum()), kv, tpb, d)

    def scratch():
        fill = torch.full(out_shape, F2B_SCRATCH_FILL, dtype=torch.int16, device=DEV)
        return fill.view(torch.bfloat16)

    return dict(
        k_cache=pool[:, 2 * layer], v_cache=pool[:, 2 * layer + 1], block_table=table,
        slots=slots.to(DEV, torch.int32), counts=counts_t,
        offsets=torch.cumsum(counts_t, 0) - counts_t, table=torch.full_like(table, F2B_TABLE_FILL),
        k_out=scratch(), v_out=scratch(),
    )  # fmt: skip


def _check_fp8_pages_to_bf16(cell, monkeypatch) -> None:
    args = _f2b_build(cell.spec, _seed(cell.why))
    if cell.spec["straddle"]:
        _assert_straddles(args)
    ref = _f2b_op.reference(**{k: v.clone() if torch.is_tensor(v) else v for k, v in args.items()})
    out = _f2b_op(**args)
    for name, got, want in zip(("k_out", "v_out", "table"), out, ref):
        assert got.shape == want.shape and torch.equal(_bits(got), _bits(want)), name
    # Every listed page was written, and only the listed requests' table entries.
    assert not bool(out[0].isnan().any()) and not bool(out[1].isnan().any())
    listed_rows = torch.zeros_like(args["block_table"], dtype=torch.bool)
    for slot, count in zip(args["slots"].tolist(), args["counts"].tolist()):
        listed_rows[slot, :count] = True
    assert bool((out[2][~listed_rows] == F2B_TABLE_FILL).all())


#: The entry's launches with the token plan and with the block plan (where it serves).
_RTL_FUSED = ("_group_plan", "_rowtile_lone_kernel", "_rowtile_attention_cute")
_RTL_BLOCK = ("_block_plan", "_rowtile_lone_kernel", "_rowtile_attention_cute")


def _rtl_build(spec, seed):
    trunk = spec["cache_layout"] == "trunk"
    assert trunk or spec["cache_layout"] == "scratch"
    assert spec["cache_dtype"] == torch.bfloat16 and spec["q_layout"] == "qkv_slice"
    st = _paged_kv_state(
        spec, seed, spec["cache_dtype"],
        lambda needed: _straddling_pages(spec, needed, spec["offset_edge"]) if trunk else needed,
        slot_pages=spec.get("slot_pages", 2), layer=spec.get("layer", 0), chunk=1024,
        scratch=not trunk,
    )  # fmt: skip
    rows, kv, d, q_heads = st.positions.numel(), spec["kv_heads"], spec["head_dim"], spec["q_heads"]
    qkv = torch.randn(rows, q_heads + 2 * kv, d, generator=st.gen, device=DEV).to(spec["q_dtype"])
    selected = _selected_tokens(
        st.positions, seed + 3, spec["compress_ratio"], spec["token_topk"],
        shared=spec.get("shared_scores", False), chunk=1024,
    )  # fmt: skip
    return dict(
        q=qkv[:, :q_heads], k_cache=st.k, v_cache=st.v, block_table=st.table,
        selected_tokens=selected, request_indices=st.rows.to(spec["request_dtype"]),
        tokens_per_block=spec["tokens_per_block"], softmax_scale=d**-0.5,
        query_positions=st.positions.to(spec.get("position_dtype", spec["request_dtype"])),
        compress_ratio=spec["compress_ratio"],
    )  # fmt: skip


def _check_sparse_gqa_rowtile(cell, monkeypatch) -> None:
    """The plan, lone-row and row-tile passes with their grids, then the band of the reference."""
    launches = _record_launches(monkeypatch, _module, dict.fromkeys(_RTL_FUSED + _RTL_BLOCK))
    spec = cell.spec
    args = _rtl_build(spec, seed=sum(map(ord, cell.why)))
    if spec["cache_layout"] == "trunk":
        _assert_straddles(args, spec["offset_edge"])
    rows = args["q"].shape[0]
    _assert_prefill_selection(args["selected_tokens"], args["query_positions"], spec)
    out = _rtl_op(**args)
    group_rows = rows_per_group(spec["q_heads"] // spec["kv_heads"])
    assert group_rows == {12: 10, 6: 20}[spec["q_heads"] // spec["kv_heads"]]
    groups = math.ceil(rows / group_rows)
    blocks = _module.block_plan_serves(spec["compress_ratio"], spec["tokens_per_block"], group_rows)
    assert [name for name, _, _ in launches] == list(_RTL_BLOCK if blocks else _RTL_FUSED)
    assert launches[0][1] == (groups,)
    assert launches[1][1] == (min(rows, 256), spec["kv_heads"])
    # The row-tile pass is persistent: one CTA per SM, never more than (group, KV head) items.
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    assert launches[2][1] == (min(groups * spec["kv_heads"], sms),)
    assert out.shape == (rows, spec["q_heads"], spec["head_dim"])
    assert out.dtype == args["q"].dtype and out.is_contiguous()
    assert not bool(out.isnan().any())
    _gqa_compare(out, _rtl_op.reference(**args))


def _check_spec_step_metadata(cell, monkeypatch) -> None:
    """Bitwise the engine's composed launches over a column slice of a wider position buffer,
    eager and replayed on new offsets."""
    n, p = cell.spec["n"], cell.spec["p"]
    rows = n * p
    g = torch.Generator(device=DEV).manual_seed(_seed(cell.why) % 2**31)

    def operands():
        buf = torch.randint(0, 200000, (3, 1, rows + 9), generator=g, device=DEV, dtype=torch.int32)
        pos_off = torch.randint(0, 4, (rows + 5,), generator=g, device=DEV, dtype=torch.int32)
        kv = torch.randint(90000, 100000, (n,), generator=g, device=DEV, dtype=torch.int32)
        kv_off = torch.randint(-3, 1, (n + 2,), generator=g, device=DEV, dtype=torch.int32)
        seq = torch.full((n + 1,), p, device=DEV, dtype=torch.int32)
        mapping = tuple(
            torch.full((rows + 4,), -9, device=DEV, dtype=dt)
            for dt in (torch.int32, torch.int64, torch.int32, torch.int32)
        )
        return buf, buf[:, :, :rows], pos_off, kv, kv_off, seq, mapping

    def copy(ops):
        c = tuple(x.clone() if torch.is_tensor(x) else tuple(y.clone() for y in x) for x in ops)
        return (c[0], c[0][:, :, :rows]) + c[2:]

    def composed(buf, pos, pos_off, kv, kv_off, seq, mapping):
        pos += pos_off[:rows]
        kv += kv_off[:n]
        triton_qsa_generation_token_mapping(
            kv_lens=kv, seq_lens=seq[:n], request_indices=mapping[0], logical_positions=mapping[1],
            sequence_lengths=mapping[2], visible_blocks=mapping[3], num_rows=rows, compress_ratio=4,
        )  # fmt: skip

    a = operands()
    b, c = copy(a), copy(a)
    pos_out = torch.full((3, rows), -9, device=DEV, dtype=torch.int32)
    qsa_spec_step_metadata(a[1], a[2], a[3], a[4], a[5][:n], a[6], 4, pos_out=pos_out)
    composed(*b)
    assert torch.equal(pos_out, b[1].reshape(3, rows)), "the contiguous positions"
    qsa_spec_step_metadata.reference(c[1], c[2], c[3], c[4], c[5][:n], c[6], 4)
    for got, want, ref in zip((a[0], a[3]) + a[6], (b[0], b[3]) + b[6], (c[0], c[3]) + c[6]):
        assert torch.equal(got, want) and torch.equal(got, ref)
    if not cell.spec["capture"]:
        return
    s, fresh = operands(), operands()
    want = copy(fresh)
    _captured(
        lambda: qsa_spec_step_metadata(s[1], s[2], s[3], s[4], s[5][:n], s[6], 4),
        lambda: [dst.copy_(fresh[i]) for i, dst in ((0, s[0]), (2, s[2]), (3, s[3]), (4, s[4]))],
    )
    composed(*want)
    assert torch.equal(s[0], want[0]) and torch.equal(s[3], want[3])
    for got, exp in zip(s[6], want[6]):
        assert torch.equal(got[:rows], exp[:rows])


_ENTRIES = (
    (qsa_paged_index_store, qsa_paged_index_store.CELLS, _check_index_store),
    (qsa_paged_kv_store, qsa_paged_kv_store.CELLS, _check_kv_store),
    (qsa_paged_sparse_gqa, qsa_paged_sparse_gqa.CELLS, _check_sparse_gqa),
    (_pro_op, _pro_op.CELLS, _check_decode_prologue),
    (index_split_plan, index_split_plan.CELLS, _check_index_split_plan),
    (draft_continue_select, draft_continue_select.CELLS, _check_draft_continue_select),
    (draft_rows_gather, draft_rows_gather.CELLS, _check_draft_rows_gather),
    (_sgc_op, _sgc_op.CELLS, _check_sparse_gqa_splitk),
    (_sgg_op, _sgg_op.CELLS, _check_sparse_gqa_grouped),
    (_tkc_op, _tkc_op.CELLS, _check_topk_expand_cluster),
    (_f2b_op, _f2b_op.CELLS, _check_fp8_pages_to_bf16),
    (_rtl_op, _rtl_op.CELLS, _check_sparse_gqa_rowtile),
    (qsa_spec_step_metadata, qsa_spec_step_metadata.CELLS, _check_spec_step_metadata),
)
CASES = [
    pytest.param(
        entry, cell, check,
        id=f"{type(entry).__name__}: {cell.why[:44]}",
        marks=[pytest.mark.high_cuda_memory] if cell.spec.get("large") else [],
    )
    for entry, cells, check in _ENTRIES
    for cell in cells
]  # fmt: skip


@pytest.mark.parametrize("entry, cell, check", CASES)
def test_certified_cells(entry, cell, check, certifying, monkeypatch) -> None:  # noqa: F811
    """Every (entry, cell) pair against the entry's own `reference`, with `certifying` armed."""
    certifying(entry)
    check(cell, monkeypatch)


def _stable_kv_store_is_shared(cell) -> None:
    args, pool = _kv_store_build(cell.spec, seed=_seed(cell.why))
    if not isinstance(pool, _KvEnginePool):
        shared = _stable.qsa_paged_kv_store.reference(**args)
        _stable.qsa_paged_kv_store(**args)
        _kv_store_compare((args["k_cache"], args["v_cache"]), shared)
        return
    try:
        # The shared launcher stores into dense copies of the touched pages; the copy into the
        # engine's pool, whose pages straddle 2^32 elements.
        dense_k, dense_v, renamed = pool.dense(args["block_table"])
        shared = _stable.qsa_paged_kv_store.reference(
            **{**args, "k_cache": dense_k, "v_cache": dense_v, "block_table": renamed}
        )
        _stable.qsa_paged_kv_store(**args)
        _kv_store_compare(pool.gathered(), shared)
    finally:
        pool.close()


def _stable_sparse_gqa_is_shared(cell) -> None:
    args = _sparse_gqa_build(cell.spec, _seed(cell.why))
    got = _stable.qsa_paged_sparse_gqa(**args)
    want = _stable.qsa_paged_sparse_gqa.reference(**args)
    assert got.dtype == want.dtype and got.shape == want.shape, (got.shape, want.shape)
    assert torch.equal(_bits(got), _bits(want)), "the stable-JIT kernels differ from the shared"


_STABLE_CASES = [
    pytest.param(
        check, cell,
        id=f"{name}: {cell.why[:44]}",
        marks=[pytest.mark.high_cuda_memory] if cell.spec.get("large") else [],
    )
    for name, cells, check in (
        ("kv_store", qsa_paged_kv_store.CELLS, _stable_kv_store_is_shared),
        ("sparse_gqa", qsa_paged_sparse_gqa.CELLS, _stable_sparse_gqa_is_shared),
    )
    for cell in cells
]  # fmt: skip


@pytest.mark.parametrize("check, cell", _STABLE_CASES)
def test_stable_kernels_are_the_shared_launchers(check, cell) -> None:
    """`qsa_triton_kernels__stable_jit`'s kernel copies write bitwise what the shared QSA
    launchers they copy write, at every cell of the entries that launch them."""
    check(cell)


# `DecodeTarget._attention_block` at the checkpoint's shapes for one rank of a layout.

PRL_DEV, PRL_BF, PRL_F8 = "cuda", torch.bfloat16, torch.float8_e4m3fn
PRL_D, PRL_ROT, PRL_IH, PRL_ID, PRL_TPB, PRL_CR, PRL_HIDDEN, PRL_LAYERS, PRL_LAYER = (
    256, 64, 4, 128, 32, 4, 2560, 12, 5
)  # fmt: skip
PRL_CFG = dict(theta=1.0e7, eps=1e-6, mrope_section1=11, mrope_section2=10, max_positions=262144)
#: (q heads, KV heads, context lengths after the step, rows a request, pool dtype).
PRL_LAYOUTS = {
    "tep4_verify_bf16": (6, 1, (7003,), 4, PRL_BF),
    "tep2_decode_bf16": (12, 1, (4096, 5003, 6150), 1, PRL_BF),
    "dep4_verify_e4m3": (24, 2, (2500, 4100, 6000, 8000), 4, PRL_F8),
}
#: A tep2 rank, kv_fp8, MTP2 verify (3 rows a request) at 90 requests (270 rows).
IDX_LAYOUT = (12, 1, tuple(2400 + 61 * r for r in range(90)), 3, PRL_F8)


def _prl_core(hq, hk):
    c = types.SimpleNamespace(
        **PRL_CFG, head_dim=PRL_D, rotary_dim=PRL_ROT, heads_q=hq, heads_kv=hk
    )
    c.index_heads, c.index_kv_heads, c.index_head_dim = PRL_IH, 1, PRL_ID
    c.token_topk, c.compress_ratio = 2048, PRL_CR
    c.block_topk = c.token_topk // PRL_CR
    c.index_width = (PRL_IH + 1) * PRL_ID
    c.index_scratch_width = c.index_width + PRL_ID
    c.softmax_scale = PRL_D**-0.5
    c.q_width, c.kv_width = hq * PRL_D, hk * PRL_D
    c.qkv_width = c.q_width + 2 * c.kv_width
    c.attn_in_width = c.qkv_width + c.q_width + c.index_scratch_width
    c.attn_in_tail_heads = c.attn_in_width // PRL_D - hq - hk
    c.index_col = c.qkv_width + c.q_width
    c._cos_sin = M.Qwen4ExpUnifiedModelingV2._index_rope_table(c, c.theta, PRL_DEV)
    c.tp_sum = lambda x: x
    return c


def _prl_build(hq, hk, ctx, p, kv_dtype, seed):
    g = torch.Generator(device=PRL_DEV).manual_seed(seed)
    core = _prl_core(hq, hk)
    need = [-(-n // PRL_TPB) for n in ctx]
    pages = sum(need) + 16
    perm = torch.randperm(pages, generator=g, device=PRL_DEV).to(torch.int32)
    table = torch.full((max(8, len(ctx)), 8196), -1, dtype=torch.int32, device=PRL_DEV)
    used = 0
    for r, n in enumerate(need):
        table[r, :n] = perm[used : used + n]
        used += n
    roles = torch.randn(pages, 2 * PRL_LAYERS, hk, PRL_TPB, PRL_D, generator=g, device=PRL_DEV)
    roles = roles.to(kv_dtype)
    # Index keys 2..3x larger at 512 groups a request win the block top-k by a 2x margin, so
    # paths a sub-ulp apart in the query select the same tokens.
    scale = 0.25 + 0.75 * torch.rand(pages, PRL_TPB, generator=g, device=PRL_DEV)
    for r, n in enumerate(ctx):
        hot = torch.randperm((n - p) // PRL_CR - 1, generator=g, device=PRL_DEV)[:512]
        token = hot * PRL_CR + PRL_CR - 1
        scale[table[r, token // PRL_TPB].long(), token % PRL_TPB] = 2.0 + torch.rand(
            hot.shape[0], generator=g, device=PRL_DEV
        )
    planes = (
        scale[:, None, :, None, None]
        * torch.randn(PRL_LAYERS, 1, 1, PRL_ID, generator=g, device=PRL_DEV)[None]
    ).to(PRL_BF)
    position = torch.randint(
        0, 8192, (pages, PRL_TPB, 3), generator=g, device=PRL_DEV, dtype=torch.int32
    )
    core._bound = types.SimpleNamespace(
        pools=types.SimpleNamespace(
            kv={PRL_LAYER: P.KvPlanes(roles[:, 2 * PRL_LAYER], roles[:, 2 * PRL_LAYER + 1])},
            index_cache={PRL_LAYER: planes[:, PRL_LAYER]},
            position_cache=position,
        )
    )
    n = len(ctx)
    req = torch.arange(n, device=PRL_DEV, dtype=torch.int32).repeat_interleave(p)
    lens = torch.tensor(ctx, device=PRL_DEV, dtype=torch.int64).repeat_interleave(p)
    logical = lens - p + torch.arange(p, device=PRL_DEV, dtype=torch.int64).repeat(n)
    ids = logical.to(torch.int32).expand(3, n * p).contiguous()
    qsa = (
        req, logical, lens.to(torch.int32), ((logical + 1) // PRL_CR).to(torch.int32),
        torch.empty(n * p, core.block_topk, dtype=torch.int32, device=PRL_DEV),
        torch.zeros(n * p, dtype=torch.int32, device=PRL_DEV), table, PRL_TPB, ids.clone(), ids[0],
        torch.transpose(ids, 0, 1), True,
    )  # fmt: skip
    w = lambda *shape: torch.randn(*shape, generator=g, device=PRL_DEV)  # noqa: E731
    L = types.SimpleNamespace(
        attn_in=(w(core.attn_in_width, PRL_HIDDEN) * 0.02).to(PRL_BF).t(),
        o=(w(PRL_HIDDEN, core.q_width) * 0.02).to(PRL_BF).t(),
        q_norm=(w(PRL_D) * 0.3).to(PRL_BF), k_norm=(w(PRL_D) * 0.3).to(PRL_BF),
        idx_qn=(w(PRL_ID) * 0.3).to(PRL_BF), idx_kn=(w(PRL_ID) * 0.3).to(PRL_BF),
    )  # fmt: skip
    x = w(n * p, PRL_HIDDEN).to(PRL_BF)
    return core, L, x, qsa, (roles, planes, position), p


def test_decode_index_selection_split_over_tp_is_the_replicated_selection(monkeypatch) -> None:
    """The decode index selection split over two attention-TP ranks, gathers emulated in one
    process: split by position or by `_index_split_plan`, the gathered selection and the layer
    over it are bitwise the replicated ones."""
    core, L, x, qsa, pools, p = _prl_build(*IDX_LAYOUT, seed=37)
    rows = x.shape[0]
    monkeypatch.setattr(M, "_INDEX_SPLIT_MIN_REQUESTS", 32)
    # bitwise halves need one scoring kernel for 45 and 90 groups alike (the family's)
    monkeypatch.setattr(M, "_INDEX_SCORES_CUTE_MIN_GROUPS", 1 << 30)
    q8 = qsa[:6] + (qsa[6], qsa[7])
    cache = core._bound.pools.index_cache[PRL_LAYER]
    saved = [t.clone() for t in pools]
    assert M._index_split_rows(None, rows, p) is None
    assert M._index_split_rows(([0, 1], None, 1), rows, p) == (135, 270)
    assert M._index_split_rows(([0, 1], None, 1), 31 * p, p) is None  # below the threshold

    def restore():
        for t, sv in zip(pools, saved):
            t.copy_(sv)

    def layer(target, gather):
        qkv = torch.empty(rows, core.qkv_width, dtype=PRL_BF, device=PRL_DEV)
        scratch = torch.empty(rows, core.index_scratch_width, dtype=PRL_BF, device=PRL_DEV)
        with monkeypatch.context() as m:
            m.setattr(M, "allgather", gather)
            out = M.DecodeTarget._attention_block(
                target, x, PRL_LAYER, L, qkv, scratch, None, qsa,
                {"predicted_tokens_per_seq": p}, False,
            )  # fmt: skip
        torch.cuda.synchronize()
        return out.clone(), scratch

    def select(rank, q, vectors, n, rpr, gather=lambda part, sizes, g_, pg: part, **kw):
        with monkeypatch.context() as m:
            m.setattr(M, "allgather", gather)
            split = None if rank is None else ([0, 1], None, rank)
            return M._decode_index_selection(core, split, q, vectors, n, rpr, PRL_DEV, cache, **kw)

    # The selections are taken over the planes the layer left (its rows' index keys stored).
    full_out, scratch = layer(
        types.SimpleNamespace(core=core, _index_split=None, _kv_probe=None), None
    )
    q = M._index_query(core, scratch)
    full = select(None, q, q8, rows, p)
    halves = [select(rank, q, q8, rows, p) for rank in (0, 1)]
    assert [h.shape[0] for h in halves] == [135, 135]
    assert torch.equal(torch.cat(halves), full), "the halves are bitwise the replicated selection"
    # A draft step's shape: one row a request (each request's last verify row here).
    last = torch.arange(p - 1, rows, p, device=PRL_DEV)
    q1, d8 = q.index_select(0, last), tuple(t.index_select(0, last) for t in q8[:6]) + q8[6:]
    halves1 = [select(rank, q1, d8, last.numel(), 1) for rank in (0, 1)]
    assert [h.shape[0] for h in halves1] == [45, 45]
    assert torch.equal(torch.cat(halves1), select(None, q1, d8, last.numel(), 1))

    lam = object()  # `_index_split_plan` needs a Lamport gather; the emulation below is NCCL's
    plans = [M._index_split_plan(([0, 1], None, r, lam), q8, rows, p) for r in (0, 1)]
    assert all(pl is not None for pl in plans)
    assert torch.equal(plans[0].perm_rows, plans[1].perm_rows)
    perm = plans[0].perm_rows.long()
    assert torch.equal(torch.sort(perm).values, torch.arange(rows, device=PRL_DEV))
    assert torch.equal(plans[0].iq_rows.long()[perm], torch.arange(rows, device=PRL_DEV))
    windows = qsa[3].long()
    by_plan = [int(windows[perm[r * 135 : (r + 1) * 135]].sum()) for r in (0, 1)]
    by_position = [int(windows[r * 135 : (r + 1) * 135].sum()) for r in (0, 1)]
    assert max(by_position) / min(by_position) > 1.1
    assert max(by_plan) / min(by_plan) < 1.01, by_plan
    parts = {}
    for permuted in (False, True):
        qs = q.index_select(0, perm) if permuted else q
        for rank in (0, 1):
            kw = dict(plan=plans[rank], q_permuted=permuted)
            keep = lambda part, sizes, g_, pg, key=(permuted, rank): parts.setdefault(  # noqa: E731
                key, part.clone()
            ).repeat(2, 1)
            select(rank, qs, q8, rows, p, gather=keep, **kw)
        gathered = torch.cat([parts[(permuted, 0)], parts[(permuted, 1)]])
        assert torch.equal(gathered, full.index_select(0, perm))
        for rank in (0, 1):
            kw = dict(plan=plans[rank], q_permuted=permuted)
            got = select(rank, qs, q8, rows, p, gather=lambda *a, g=gathered: g, **kw)
            assert torch.equal(got, full), (permuted, rank)
    restore()

    for rank in (0, 1):
        # By position: this rank's half handed over, the gathered selection back.
        handed = []

        def nccl(part, sizes, group, process_group):
            assert sizes is None and group == [0, 1]
            handed.append(part.clone())
            return torch.cat(halves)

        out, _ = layer(
            types.SimpleNamespace(core=core, _index_split=([0, 1], None, rank), _kv_probe=None),
            nccl,
        )
        restore()
        assert len(handed) == 1 and torch.equal(handed[0], halves[rank])
        assert torch.equal(out, full_out), "the layer over the gathered selection"
        # The plan: the prologue stores the index queries in the permuted order.
        target = types.SimpleNamespace(
            core=core, _index_split=([0, 1], None, rank), _index_plan=plans[rank], _kv_probe=None
        )
        out, scr = layer(target, lambda part, sizes, g_, pg: gathered)
        restore()
        assert torch.equal(M._index_query(core, scr), q.index_select(0, perm))
        assert torch.equal(out, full_out), rank


def _replay_index_store(prefix, monkeypatch) -> None:
    """A store captured at the cell's batch, replayed after the next step's metadata is written
    into the captured buffers, stores that step bitwise (the pool is sized to the batch)."""
    cell = next(c.spec for c in qsa_paged_index_store.CELLS if c.why.startswith(prefix))
    spec = dict(cell, pages=24576, straddle=False)
    args, extras = _index_store_build(spec, seed=101)
    index_cache, position_cache = args["index_cache"], args["position_cache"]
    qsa_paged_index_store(**args)  # compiles the launch before the capture
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        qsa_paged_index_store(**args)
    shifted = dict(spec, segments=tuple((max(0, s - 3), n) for s, n in spec["segments"]))
    following, _ = _index_store_build(shifted, seed=202, planes=(index_cache, position_cache))
    before = qsa_paged_index_store.reference(**args)
    names = ("token_k", "position_coordinates", "request_indices", "logical_positions")
    for name in (*names, "block_table"):
        args[name].copy_(following[name])
    index_cache.fill_(KEY_FILL)
    position_cache.fill_(POS_FILL)
    ref = qsa_paged_index_store.reference(**args)
    assert not torch.equal(_bits(ref[0]), _bits(before[0])), "the next step stores nothing new"
    graph.replay()
    torch.cuda.synchronize()
    _index_store_compare((index_cache, position_cache), ref)
    _other_layers_untouched(*extras["coalesced"])


def _replay_kv_store(prefix, monkeypatch) -> None:
    """A step over the engine's pool captured with every position at -1 (nothing read or
    written), replayed after the real positions are written into the same buffer."""
    cell = next(c for c in qsa_paged_kv_store.CELLS if c.why.startswith(prefix))
    args, pool = _kv_store_build(cell.spec, seed=_seed(cell.why))
    try:
        positions = args["logical_positions"]
        saved = positions.clone()
        positions.fill_(-1)
        before = tuple(t.clone() for t in pool.gathered())
        for _ in range(3):  # compile the Triton kernel before capture
            qsa_paged_kv_store(**args)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            qsa_paged_kv_store(**args)
        torch.cuda.synchronize()
        _kv_store_compare(pool.gathered(), before)  # a row at position -1 writes nothing
        positions.copy_(saved)
        ref = _kv_engine_reference(args, pool)
        graph.replay()
        torch.cuda.synchronize()
        _kv_store_compare(pool.gathered(), ref)
        pool.assert_rest_unchanged()
    finally:
        pool.close()


@pytest.mark.parametrize(
    "replay, prefix",
    [
        pytest.param(_replay_index_store, "tp2 decode: 128 requests", id="index tp2-decode"),
        pytest.param(_replay_index_store, "dep4 MTP3 verify", id="index dep4-mtp3"),
        pytest.param(_replay_kv_store, "tp2 decode, 128 rows (max_batch_size)", id="kv tp2-decode"),
        pytest.param(_replay_kv_store, "dep4 MTP3 verify, 128 requests x 4", id="kv dep4-mtp3"),
    ],
)
def test_captured_stores_replay_rewritten_metadata(replay, prefix, monkeypatch) -> None:
    replay(prefix, monkeypatch)


def _index_pool(n_requests, need, seed, pool_pages, slots):
    """A trunk-like index pool's layer-11 plane and a `[slots, 8196]` table of random pages."""
    gen = torch.Generator(device=DEV).manual_seed(seed)
    pages = max(pool_pages, sum(need) + 8)
    pool = torch.randn(pages, 12, 32, 1, 128, generator=gen, device=DEV).to(torch.bfloat16)
    perm = torch.randperm(pages, generator=gen, device=DEV).to(torch.int32)
    table = torch.full((slots, 8196), -1, dtype=torch.int32, device=DEV)
    used = 0
    slot_of = torch.randperm(slots, generator=torch.Generator().manual_seed(seed))[:n_requests]
    for i, k in enumerate(need):
        table[int(slot_of[i]), :k] = perm[used : used + k]
        used += k
    return gen, pool[:, 11], table, slot_of


def _family_index_scores_groups(contexts, monkeypatch) -> None:
    """The CuTe request-group index scores: within fp32 summation-order noise of the family's
    kernel, the same top-512 except at near-ties, deterministic, and replayable."""
    rpr = 3
    gen, cache, table, slot_of = _index_pool(
        len(contexts), [-(-c // 32) for c in contexts], len(contexts) * 7 + 1, 6000, 128
    )
    req = slot_of.to(DEV).to(torch.int32).repeat_interleave(rpr)
    lens = torch.tensor(contexts, device=DEV, dtype=torch.int64).repeat_interleave(rpr)
    pos = lens - rpr + torch.arange(rpr, device=DEV, dtype=torch.int64).repeat(len(contexts))
    q = torch.randn(len(contexts) * rpr, 6, 128, generator=gen, device=DEV)
    q = q.to(torch.bfloat16)[:, :4]
    assert _igc.serves(q, cache, rpr)
    fam = M.qsa_paged_index_scores(
        q, cache, table, pos, req, 32, 4, only_visible_blocks=True, context_rows=False,
        rows_per_request=rpr,
    )  # fmt: skip
    out = _igc.index_scores_groups(q, cache, table, pos, req, 32, 4, False, rpr)
    vis = (pos + 1) // 4
    inside = torch.arange(out.shape[1], device=DEV)[None, :] < vis[:, None]
    torch.testing.assert_close(out[inside], fam[inside], rtol=1e-5, atol=1e-6)
    neg = torch.full_like(out, -1.0)
    k = min(512, int(vis.min().item())) if int(vis.min().item()) > 0 else 0
    if k:
        a = torch.where(inside, out, neg).topk(k, dim=1)
        b = torch.where(inside, fam, neg).topk(k, dim=1)
        differ = (a.indices.sort(dim=1).values != b.indices.sort(dim=1).values).any(dim=1)
        # a row may differ only where its k-th and (k+1)-th scores are within the noise
        next_value = torch.where(inside, fam, neg).topk(k + 1, dim=1).values[:, -1]
        gap = (b.values[:, -1] - next_value).abs()
        assert bool((~differ | (gap <= 1e-5 * b.values[:, -1].abs())).all())
    out2 = _igc.index_scores_groups(q, cache, table, pos, req, 32, 4, False, rpr)
    assert torch.equal(out2[inside], out[inside]), "two calls on the same inputs differ"
    refill = lambda: q.copy_(torch.randn(q.shape, device=DEV).to(q.dtype))  # noqa: E731
    captured = _captured(
        lambda: _igc.index_scores_groups(q, cache, table, pos, req, 32, 4, False, rpr), refill
    )
    eager = _igc.index_scores_groups(q, cache, table, pos, req, 32, 4, False, rpr)
    assert torch.equal(captured[inside], eager[inside]), "the replay differs from an eager call"


def _family_index_scores_prefill(case, monkeypatch) -> None:
    """The tcgen05 context-row index scores, `case` = (((prefix, rows), ...), position dtype):
    bitwise the family's kernel over every row's visible columns, and replayable."""
    requests, pos_dtype = case
    need = [-(-(p + r) // 32) for p, r in requests]
    gen, cache, table, slot_of = _index_pool(
        len(requests), need, sum(r for _, r in requests), 12000, 64
    )
    req = torch.cat(
        [torch.full((r,), int(slot_of[i]), dtype=torch.int32) for i, (_, r) in enumerate(requests)]
    ).to(DEV)
    pos = torch.cat([torch.arange(p, p + r, dtype=pos_dtype) for p, r in requests]).to(DEV)
    q = torch.randn(pos.numel(), 6, 128, generator=gen, device=DEV).to(torch.bfloat16)[:, :4]
    assert _isc.supports(q, cache, 32, 4)
    assert not _isc.supports(q, cache, 64, 4)
    assert _isc.supports(q, cache, 16, 2)
    fam = M.qsa_paged_index_scores(
        q, cache, table, pos, req, 32, 4, only_visible_blocks=True, context_rows=True
    )
    out = _isc.qsa_index_scores_prefill(q, cache, table, pos, req, 32, 4)
    vis = ((pos + 1) // 4).long()
    inside = torch.arange(out.shape[1], device=DEV)[None, :] < vis[:, None]
    assert torch.equal(out[inside], fam[inside]), "visible scores differ from the family's"
    refill = lambda: q.copy_(torch.randn(q.shape, device=DEV).to(q.dtype))  # noqa: E731
    captured = _captured(
        lambda: _isc.qsa_index_scores_prefill(q, cache, table, pos, req, 32, 4), refill
    )
    eager = _isc.qsa_index_scores_prefill(q, cache, table, pos, req, 32, 4)
    assert torch.equal(captured[inside], eager[inside]), "the replay differs from an eager call"


def _family_topk_rows(case, monkeypatch) -> None:
    """The prefill row top-k: bitwise the family's `qsa_topk_expand_decode` on a wide table."""
    name, spec = case
    args = _tkc_build(dict(spec, width=65568), seed=_seed(name) % 2**31)
    assert _tkr_supports(args[0], args[4]), "a table outside the row kernel's range"
    family_args = [a.clone() if torch.is_tensor(a) else a for a in args]
    sel = _tkr_op(*args)
    family_sel = family_op(*family_args)
    assert torch.equal(args[4], family_args[4]), "block list"
    assert torch.equal(sel, family_sel), "token selection"


_FAMILY_CASES = [
    *(
        pytest.param(_family_index_scores_groups, contexts, id=f"index scores groups {name}")
        for name, contexts in (
            ("32x50k", tuple([50000] * 32)),
            ("64x20k", tuple([20000] * 64)),
            ("32_spread", tuple(4400 + 6007 * i for i in range(32))),
            ("edges", (3, 4, 5, 515, 516, 517, 2051, 66000, 131071, 262143, 9000, 31)),
            ("8x100k", tuple([100000] * 8)),
            ("20_spread", tuple(70000 + 1500 * i for i in range(20))),
        )
    ),
    *(
        pytest.param(_family_index_scores_prefill, (requests, pos_dtype),
                     id=f"index scores prefill {name}")
        for name, requests, pos_dtype in (
            ("88k_one", ((88064, 1024),), torch.int32),
            ("three_requests", ((88064, 700), (40000, 333), (3, 300)), torch.int32),
            ("windows_from_0", ((0, 1500),), torch.int32),
            ("131k_250k", ((131072, 320), (250000, 64)), torch.int32),
            ("three_requests_int64", ((88064, 700), (40000, 333), (3, 300)), torch.int64),
        )
    ),
    *(
        pytest.param(_family_topk_rows, (name, spec), id=f"topk rows {name}")
        for name, spec in (
            ("chunk_88k_relu_sum", dict(rows=1024, requests=((88064, 1024),), relu_sum=True)),
            ("chunk_131k_relu_sum", dict(rows=512, requests=((131072, 512),), relu_sum=True)),
            ("requests_12_relu_sum", dict(rows=768, requests=tuple((88064 + 97 * j, 64)
                                          for j in range(12)), relu_sum=True)),
            ("chunk_250k_relu_sum", dict(rows=256, requests=((250000, 256),), relu_sum=True)),
            ("zero_class_straddles", dict(rows=256, requests=((70000, 256),), relu_sparse=True)),
            ("every_score_tied", dict(rows=256, requests=((88064, 256),), equal=True)),
            ("lognormal_bf16_ties", dict(rows=512, requests=((60000, 512),), lognormal=True)),
            ("windows_0_to_past_k", dict(rows=2600, requests=((0, 2600),), relu_sum=True)),
        )
    ),
]  # fmt: skip


@pytest.mark.parametrize("check, case", _FAMILY_CASES)
def test_kernels_without_cells_match_the_family(check, case, monkeypatch) -> None:
    check(case, monkeypatch)


def _prl_site(monkeypatch, built, prologue, site):
    """One `trunk` (`DecodeTarget._attention_block`) or `draft` (`_mtp_attention_planes`) run
    over the built pools, restored after, with the fused prologue on or off."""
    core, L, x, qsa, pools, p = built
    saved = [t.clone() for t in pools]
    calls, entry = [], M.qwen4_exp_attn_decode_prologue

    def counted(*args, **kwargs):
        calls.append(kwargs["rows_per_request"])
        return entry(*args, **kwargs)

    rows = x.shape[0]
    with monkeypatch.context() as m:
        m.setattr(M, "qwen4_exp_attn_decode_prologue", counted)
        if not prologue:
            m.setattr(M, "_decode_prologue", lambda kv, sparse: False)
        if site == "trunk":
            qkv = torch.empty(rows, core.qkv_width, dtype=PRL_BF, device=PRL_DEV)
            scratch = torch.empty(rows, core.index_scratch_width, dtype=PRL_BF, device=PRL_DEV)
            # The FP8 KV range probe reads the packed K/V on both branches; this is its last layer.
            target = types.SimpleNamespace(core=core, _kv_probe=M._KvRangeProbe(PRL_LAYER))
            out = M.DecodeTarget._attention_block(
                target, x, PRL_LAYER, L, qkv, scratch, None, qsa,
                {"predicted_tokens_per_seq": p}, False,
            )  # fmt: skip
            assert target._kv_probe is None, "the probe reported on its last layer and was dropped"
        else:
            roles, planes, position = pools
            draft = types.SimpleNamespace(
                kv=P.KvPlanes(roles[:, 2 * PRL_LAYER], roles[:, 2 * PRL_LAYER + 1]),
                index_cache=planes[:, PRL_LAYER],
                position_cache=position,
            )
            vectors = (qsa[0], qsa[1], qsa[6], qsa[7], qsa[8], qsa[9], qsa[10])
            qkv, gate, iq = M._mtp_attention_planes(
                core, L, x, draft, vectors, p, True, True, M._decode_projection
            )
    torch.cuda.synchronize()
    if site == "trunk":
        got = dict(out=out, qkv=qkv, iq=M._index_query(core, scratch).clone())
    else:
        got = dict(qkv=qkv.clone(), gate=gate.clone(), iq=iq.clone())
    got.update(calls=calls, planes=[t.clone() for t in pools])
    for t, s in zip(pools, saved):
        t.copy_(s)
    return got


def _site_prologue(case, monkeypatch) -> None:
    """The one-launch prologue against the composition it replaces: v, index keys, coordinates
    and the draft's gate bitwise; rotated values within rounding, since the entry reads the fp32
    table and `fused_qk_norm_rope` takes `__sincosf`."""
    site, layout = case
    hq, hk, ctx, p, kv_dtype = PRL_LAYOUTS[layout]
    built = _prl_build(hq, hk, ctx, p, kv_dtype, seed=len(layout) + (7 if site == "draft" else 0))
    if site == "draft":
        core = built[0]
        core.mtp_theta = 5.0e6
        core._mtp_cos_sin = M.Qwen4ExpUnifiedModelingV2._index_rope_table(core, 5.0e6, PRL_DEV)
    fused = _prl_site(monkeypatch, built, True, site)
    composed = _prl_site(monkeypatch, built, False, site)
    assert fused["calls"] == [p] and composed["calls"] == []
    width = (hq + hk) * PRL_D
    assert_within_ulp(fused["qkv"][:, :width], composed["qkv"][:, :width], 2.0, 0.25)
    assert torch.equal(fused["qkv"][:, width:], composed["qkv"][:, width:]), "v"
    if site == "draft":
        assert torch.equal(fused["gate"], composed["gate"]), "the gate columns"
    assert_within_ulp(fused["iq"], composed["iq"], 2.0, 0.25)
    (roles_f, index_f, pos_f), (roles_c, index_c, pos_c) = fused["planes"], composed["planes"]
    assert torch.equal(index_f, index_c), "raw and compressed index keys"
    assert torch.equal(pos_f, pos_c), "coordinates"
    v_f, v_c = roles_f[:, 2 * PRL_LAYER + 1], roles_c[:, 2 * PRL_LAYER + 1]
    assert torch.equal(v_f.view(torch.uint8), v_c.view(torch.uint8)), "the V pool"
    others = [s for s in range(2 * PRL_LAYERS) if s != 2 * PRL_LAYER]
    assert torch.equal(roles_f[:, others].view(torch.uint8), roles_c[:, others].view(torch.uint8))
    k_f, k_c = roles_f[:, 2 * PRL_LAYER], roles_c[:, 2 * PRL_LAYER]
    if kv_dtype == PRL_BF:
        assert_within_ulp(k_f, k_c, 2.0, 0.25)
    else:
        a, b = k_f.view(torch.uint8).int(), k_c.view(torch.uint8).int()
        moved = a != b
        assert moved.float().mean().item() < 1e-3
        if site == "trunk":
            # One bf16 step of k moves an e4m3 rounding by at most one code: the codes'
            # magnitudes are monotonic, a sign may differ only next to zero.
            mag = ((a & 0x7F) - (b & 0x7F)).abs()
            near_zero = ((a & 0x7F) <= 1) & ((b & 0x7F) <= 1)
            assert ((mag <= 1) & (((a ^ b) & 0x80 == 0) | near_zero))[moved].all()
    if site == "trunk":
        # The layer output follows within the selection both paths make from those keys.
        o_f, o_c = fused["out"].float(), composed["out"].float()
        assert ((o_f - o_c).norm() / o_c.norm()).item() < 1e-2


def _site_rowtile_q_in_place(segments, monkeypatch) -> None:
    """q in place on the attention projection row gives the compact row's bits, and the
    lone-row kernel (q's row stride a runtime argument) compiles once for both strides."""
    spec = {**_rtl_op.CELLS[0].spec, "segments": tuple(segments), "slots": 16}
    assert (spec["q_heads"], spec["kv_heads"], spec["head_dim"]) == (6, 1, 256)
    args = _rtl_build(spec, seed=7900 + len(segments))
    q = args["q"]
    rows, heads, d = q.shape
    # A projection row: q | k | v | gate | index planes, wider than the compact row.
    width = (2 * heads + 5) * d
    proj = torch.randn(rows, width, device=DEV).to(q.dtype)
    proj[:, : heads * d].copy_(q.reshape(rows, heads * d))
    in_place = torch.reshape(proj, [rows, width // d, d])[:, :heads]
    assert in_place.stride(0) != q.stride(0) and torch.equal(in_place, q)
    dev = torch.cuda.current_device()
    compact = _rtl_op(**args)
    variants = len(_module._rowtile_lone_kernel.device_caches[dev][0])
    got = _rtl_op(**{**args, "q": in_place})
    assert len(_module._rowtile_lone_kernel.device_caches[dev][0]) == variants, "recompiled"
    assert torch.equal(_bits(got), _bits(compact)), "q in place vs compact"
    _gqa_compare(got, _rtl_op.reference(**args))


def _site_rowtile_scratch_groups(_, monkeypatch) -> None:
    """`_rowtile_scratch_groups`: one launch (None) while the pages fit the scratch, else the
    fewest launches of consecutive requests that fit, each request in exactly one."""
    tpb = 32
    page_bytes = 32 * 256 * 4  # one fp8 K/V head of 256 at 32 tokens a page, widened to bf16
    cap_pages = M._FP8_ROWTILE_MAX_SCRATCH // page_bytes

    def md(q_lens, kv_lens):
        return types.SimpleNamespace(
            num_contexts=len(q_lens),
            tokens_per_block=tpb,
            seq_lens=torch.tensor(q_lens + [1, 1], dtype=torch.int32),
            kv_lens_runtime=torch.tensor(kv_lens + [9, 9], dtype=torch.int32),
        )

    assert M._rowtile_scratch_groups(md([1024] * 4, [68608] * 4), page_bytes) is None
    # Sixteen 1024-token chunks at a 67k prefix: 2144 pages each, 34304 in all, past the cap.
    q_lens, kv_lens = [1024] * 16, [67584 + 1024] * 16
    groups = M._rowtile_scratch_groups(md(q_lens, kv_lens), page_bytes)
    pages = [-(-k // tpb) for k in kv_lens]
    assert sum(pages) > cap_pages and groups is not None and len(groups) > 1
    starts = [0]
    for q in q_lens:
        starts.append(starts[-1] + q)
    first = 0
    for a, b, n in groups:
        assert n <= cap_pages
        last = starts.index(b)
        assert a == starts[first] and n == sum(pages[first:last])
        first = last
    assert first == len(q_lens) and groups[-1][1] == sum(q_lens)
    assert len(groups) == -(-sum(pages) // cap_pages) == 2
    assert [b - a for a, b, _ in groups] == [8192, 8192]
    # Uneven requests: still launches that fit, every page in one.
    q_lens = [512, 4096, 256, 2048, 1024, 6000, 300, 2000]
    kv_lens = [200000, 150000, 180000, 90000, 210000, 120000, 160000, 70000]
    groups = M._rowtile_scratch_groups(md(q_lens, kv_lens), page_bytes)
    assert groups[0][0] == 0 and groups[-1][1] == sum(q_lens)
    assert all(n <= cap_pages for _, _, n in groups)
    assert sum(n for _, _, n in groups) == sum(-(-k // tpb) for k in kv_lens)
    assert all(prev[1] == nxt[0] for prev, nxt in zip(groups, groups[1:]))


def _site_draft_share_chunks(case, monkeypatch) -> None:
    """The IndexShare draft-extend step's context selection (`_score_and_select`, as
    `mtp_forward` calls it) in equal chunks of a lowered score-table bound, `case` =
    (((prefix, rows), ...), bound, chunk rows): bitwise the one-call scores and `_selection`
    the step ran unchunked, block lists and token selection alike."""
    requests, bound, want = case
    tpb, cr, token_topk = 32, 4, TKC_TOKEN_TOPK
    need = [-(-(p + r) // tpb) for p, r in requests]
    rows = sum(r for _, r in requests)
    gen, cache, table, slot_of = _index_pool(len(requests), need, rows, 12000, 64)
    req = torch.cat(
        [torch.full((r,), int(slot_of[i]), dtype=torch.int32) for i, (_, r) in enumerate(requests)]
    ).to(DEV)
    logical = torch.cat([torch.arange(p, p + r, dtype=torch.int64) for p, r in requests]).to(DEV)
    seq_lens = torch.cat([torch.full((r,), p + r, dtype=torch.int32) for p, r in requests]).to(DEV)
    visible = ((logical + 1) // cr).to(torch.int32)
    row_starts = torch.zeros(rows, dtype=torch.int32, device=DEV)
    q = torch.randn(rows, 6, 128, generator=gen, device=DEV).to(torch.bfloat16)[:, :4]
    core = SimpleNamespace(compress_ratio=cr, token_topk=token_topk)
    columns = table.shape[1] * tpb // cr
    # At the engine's bound, `max_seq_len` 262144 (8196 pages of 32) scores 32704 rows a call.
    assert M._score_chunk_rows(core, table, tpb) == 32704

    def topk():
        return torch.full((rows, token_topk // cr), -7, dtype=torch.int32, device=DEV)

    def selected():
        return torch.full((rows, token_topk + cr - 1), -7, dtype=torch.int32, device=DEV)

    one_topk, one = topk(), selected()
    scores = M._context_index_scores(q, cache, table, logical, req, tpb, cr, True)
    M._selection(core, scores, visible, logical, seq_lens, one_topk, row_starts, one)
    del scores

    monkeypatch.setattr(M, "_SCORE_TABLE_ELEMENTS", bound * columns + 1)
    assert M._score_chunk_rows(core, table, tpb) == bound
    calls = []
    scorer = M._context_index_scores

    def counted(q_rows, *args):
        calls.append(q_rows.shape[0])
        return scorer(q_rows, *args)

    monkeypatch.setattr(M, "_context_index_scores", counted)
    chunked_topk, chunked = topk(), selected()
    M._score_and_select(
        core, cache, q, table, logical, req, tpb, visible, seq_lens, chunked_topk, row_starts,
        0, rows, True, chunked,
    )  # fmt: skip
    assert calls == list(want)
    assert torch.equal(chunked_topk, one_topk), "block lists"
    assert torch.equal(chunked, one), "token selection"


def _site_gen_rows_order(rows, monkeypatch) -> None:
    """A mixed step's generation rows through `_score_and_select` with the step's `_RowOrder`:
    past one wave of rows on the wide (65568-column) table the selection launches in that
    order, bitwise the unordered launch, block lists and token selection alike."""
    tpb, cr, token_topk = 32, 4, TKC_TOKEN_TOPK
    gen, cache, table, slot_of = _index_pool(1, [2900], rows, 3000, rows)
    table[:] = table[int(slot_of[0])]  # every request reads the same pages
    req = torch.arange(rows, dtype=torch.int32, device=DEV)
    logical = torch.randint(1, 2900 * tpb - 1, (rows,), generator=gen, device=DEV)
    seq_lens = (logical + 1).to(torch.int32)
    visible = ((logical + 1) // cr).to(torch.int32)
    row_starts = torch.zeros(rows, dtype=torch.int32, device=DEV)
    q = torch.randn(rows, 6, 128, generator=gen, device=DEV).to(torch.bfloat16)[:, :4]
    core = SimpleNamespace(compress_ratio=cr, token_topk=token_topk)
    seen = []
    op = M.qsa_topk_expand_decode

    def spy(*args, row_order=None, **kwargs):
        seen.append(row_order)
        return op(*args, row_order=row_order, **kwargs)

    monkeypatch.setattr(M, "qsa_topk_expand_decode", spy)
    outs = []
    for order in (None, M._RowOrder()):
        topk = torch.full((rows, token_topk // cr), -7, dtype=torch.int32, device=DEV)
        sel = torch.full((rows, token_topk + cr - 1), -7, dtype=torch.int32, device=DEV)
        M._score_and_select(
            core, cache, q, table, logical, req, tpb, visible, seq_lens, topk, row_starts,
            0, rows, False, sel, order,
        )  # fmt: skip
        outs.append((topk, sel))
    assert seen[0] is None and seen[1] is not None, "only the ordered call passes a row order"
    assert torch.equal(outs[0][0], outs[1][0]), "block lists"
    assert torch.equal(outs[0][1], outs[1][1]), "token selection"


_SITE_CASES = [
    *(
        pytest.param(_site_prologue, (site, layout), id=f"{site} prologue {layout}")
        for site in ("trunk", "draft")
        for layout in PRL_LAYOUTS
    ),
    pytest.param(_site_rowtile_q_in_place, [(5000, 2048)], id="rowtile q in place, one request"),
    pytest.param(
        _site_rowtile_q_in_place, [(3000, 1040), (7000, 1043)], id="rowtile q in place, lone group"
    ),
    pytest.param(_site_rowtile_scratch_groups, None, id="rowtile fp8 scratch groups"),
    pytest.param(
        _site_draft_share_chunks,
        (((88064, 900), (40000, 600), (3, 300)), 512, (512, 512, 512, 264)),
        id="draft index share chunks, three requests",
    ),
    pytest.param(
        _site_draft_share_chunks,
        (((250000, 1100),), 512, (384, 384, 332)),
        id="draft index share chunks, 250k",
    ),
    pytest.param(_site_gen_rows_order, 300, id="generation rows in the step's row order"),
]


@pytest.mark.parametrize("check, case", _SITE_CASES)
def test_call_sites(check, case, monkeypatch) -> None:
    """What the target's call sites hand the entries beyond a cell."""
    check(case, monkeypatch)


# ---- the FP8 KV range probe (`TRTLLM_MODELING_V2_VALIDATE` over an fp8 pool) ----------------
# The pool stores K/V at the unit scale, so a magnitude past the e4m3 maximum saturates. The probe
# reports the largest magnitude of the first eager forward's attention layers, once, and never
# reads a tensor under CUDA graph capture.

PROBE_LAST = 11


def _probe_kv(peak: float = 1.0):
    """K and V of one layer, 5 rows x 2 heads x 8 dims, with `peak` as V's largest value."""
    k = torch.linspace(-1.0, 1.0, 80, device=DEV).reshape(5, 2, 8).to(torch.bfloat16)
    v = torch.linspace(-0.5, 0.5, 80, device=DEV).reshape(5, 2, 8).to(torch.bfloat16)
    v[3, 1, 4] = peak
    return k, v


def _probe_forward(probe, peaks):
    """One forward's attention layers 3, 7 and PROBE_LAST; returns each `observe` result."""
    return [
        probe.observe(layer, *_probe_kv(peak)) for layer, peak in zip((3, 7, PROBE_LAST), peaks)
    ]


def test_a_forward_within_the_e4m3_range_is_silent(monkeypatch) -> None:
    logger = Mock()
    monkeypatch.setattr(M, "logger", logger)
    assert _probe_forward(M._KvRangeProbe(PROBE_LAST), (1.0, 300.0, 448.0)) == [True, True, False]
    logger.warning.assert_not_called()


@pytest.mark.parametrize("peak", [-600.0, float("nan")])
def test_a_magnitude_past_the_e4m3_maximum_is_reported_once(monkeypatch, peak) -> None:
    """A negative value counts by its magnitude; a NaN counts as out of range. The oversized value
    is in a middle layer: the report waits for the forward's last one."""
    logger = Mock()
    monkeypatch.setattr(M, "logger", logger)
    assert _probe_forward(M._KvRangeProbe(PROBE_LAST), (1.0, peak, 1.0)) == [True, True, False]
    logger.warning.assert_called_once()
    assert "e4m3 maximum 448" in logger.warning.call_args.args[0]


def test_the_probe_runs_only_when_asked_over_an_fp8_pool() -> None:
    core = SimpleNamespace(quant_mode=M.QM_FP8_KV_CACHE, attn_layers=[3, 7, PROBE_LAST])
    assert M._KvRangeProbe.for_target(core, True)._last == PROBE_LAST
    assert M._KvRangeProbe.for_target(core, False) is None
    for quant_mode in (0, M.QM_NVFP4_KV_CACHE):
        other = SimpleNamespace(quant_mode=quant_mode, attn_layers=core.attn_layers)
        assert M._KvRangeProbe.for_target(other, True) is None


def test_observing_under_capture_reads_nothing(monkeypatch) -> None:
    logger = Mock()
    monkeypatch.setattr(M, "logger", logger)
    probe = M._KvRangeProbe(PROBE_LAST)
    k, v = _probe_kv(1000.0)
    scratch = torch.zeros(1, device=DEV)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        scratch.add_(1)  # warm the allocator off the capture stream
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    kept = []
    with torch.cuda.graph(graph):
        scratch.add_(1)
        # The last layer, with an out-of-range value: a read here would fail the capture (a host
        # sync) and a report would fire.
        kept.append(probe.observe(PROBE_LAST, k, v))
    assert kept == [True], "a captured forward does not end the probe"
    logger.warning.assert_not_called()
    # The first eager forward after capture is what reports.
    assert _probe_forward(probe, (1.0, 1.0, 1000.0)) == [True, True, False]
    logger.warning.assert_called_once()
