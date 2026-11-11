# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for this target's qsa_sparse_gqa_rowtile variant (`__prefill`).

The paged state is the target's: a trunk cell's K and V are layer 11's planes of a `[pages,
24, kv, tpb, D]` HND pool (24 role pages a slot), sized so the requests' pages straddle 2^31
elements, every other page and plane NaN; a scratch cell's are a contiguous `[pages, kv, tpb,
D]` bf16 pool, as `qsa_fp8_pages_to_bf16` widens a kv_fp8 pool into. The slot table is the
engine's at the tune's max_seq_len (8196 columns, a row per slot, a seeded permutation, -1
past each sequence). Selection rows are laid out the way QSA's top-k + expansion emit them
(complete groups expanded to tokens, the trailing partial group, then -1, packed at the
front). The band and the reference are `qsa_paged_sparse_gqa`'s: this op replaces it on
context launches.
"""

import math

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__default.ops.attention import (  # noqa: E501
    qsa_sparse_gqa_rowtile__prefill as _module,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__default.ops.attention.qsa_sparse_gqa_rowtile__prefill import (  # noqa: E501
    qsa_sparse_gqa_rowtile as op,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__default.ops.attention.qsa_sparse_gqa_rowtile__prefill import (  # noqa: E501
    rows_per_group,
)

__extra_import_path__ = ["../../../.."]
from _validating import assert_within_ulp, certifying, validating  # noqa: E402,F401

assert torch.cuda.is_available(), "qsa_sparse_gqa_rowtile requires a CUDA device"

DEV = "cuda"


def _compare(out: torch.Tensor, ref: torch.Tensor) -> None:
    """`qsa_paged_sparse_gqa`'s band: 4 ulp of each (row, head)'s magnitude, 2 ulp relative
    RMS, in q's dtype, against the fp32 reference rounded once. The row-tile pass rounds P to
    bf16 for the PV MMA and sums it in fp32, as the per-row kernel rounds its probabilities to
    the cache dtype before P.V."""
    assert out.dtype == ref.dtype, (out.dtype, ref.dtype)
    assert out.shape == ref.shape, (out.shape, ref.shape)
    if out.numel() == 0:
        return
    assert_within_ulp(out, ref, element_ulp=4.0, rms_ulp=2.0)


class _LaunchRecorder:
    """Records which of the entry's kernels a call indexed with a grid; the bitmap pass
    (`_row_bits`, a plain call) is recorded with its one-CTA-a-row grid."""

    def __init__(self, inner, name, log):
        self._inner, self._name, self._log = inner, name, log

    def __getitem__(self, grid):
        self._log.append((self._name, tuple(grid)))
        return self._inner[grid]

    def __call__(self, selected_tokens, *args, **kwargs):
        self._log.append((self._name, (selected_tokens.shape[0],)))
        return self._inner(selected_tokens, *args, **kwargs)


_KERNELS = (
    "_row_bits",
    "_rowtile_plan_kernel",
    "_rowtile_lone_kernel",
    "_rowtile_attention_cute",
)


@pytest.fixture
def launches(monkeypatch):
    log: list = []
    for name in _KERNELS:
        inner = getattr(_module, name)
        monkeypatch.setattr(_module, name, _LaunchRecorder(inner, name, log))
    return log


def _selected_tokens(positions, seed, compress_ratio, token_topk, shared_scores=False):
    """QSA's expansion layout: a random `min(available, block_topk)`-subset of the complete
    visible groups, expanded, then the trailing partial group, then `-1`, packed at the
    front. With `shared_scores` every row ranks the groups by one score vector, so adjacent
    rows pick nearly the same set (a sparse union, as on real selections)."""
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
    for start in range(0, rows, 1024):
        stop = min(start + 1024, rows)
        avail, take, vis = available[start:stop], taken[start:stop], visible[start:stop]
        if shared_scores:
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


def _nan_pool(shape, dtype):
    assert dtype == torch.bfloat16
    return torch.full(shape, 0x7FC0, dtype=torch.int16, device=DEV).view(dtype)


def _straddling(spec, needed: int) -> int:
    """The pool page count that puts the requests -- the pool's last `needed` pages -- on
    both sides of `offset_edge` elements of layer `layer`'s K plane."""
    role_page = spec["kv_heads"] * spec["tokens_per_block"] * spec["head_dim"]
    edge, slot_pages, layer = spec["offset_edge"], spec["slot_pages"], spec["layer"]
    edge_page = math.ceil((edge - 2 * layer * role_page) / (slot_pages * role_page))
    return max(edge_page + needed // 2, needed)


def _build(spec, seed):
    """Real paged state for one spec, keyed by `__call__`'s parameter names."""
    gen = torch.Generator(device=DEV).manual_seed(seed)
    tpb, kv, d, q_heads = (
        spec["tokens_per_block"],
        spec["kv_heads"],
        spec["head_dim"],
        spec["q_heads"],
    )
    segments = list(spec["segments"])
    positions = torch.cat([torch.arange(s, s + n, device=DEV) for s, n in segments])
    req = torch.cat([torch.full((n,), i, device=DEV) for i, (_, n) in enumerate(segments)])
    rows, slots = positions.numel(), spec["slots"]
    assert len(segments) <= slots
    slot_of = torch.randperm(slots, generator=torch.Generator().manual_seed(seed))[: len(segments)]
    mapped = [math.ceil((s + n) / tpb) for s, n in segments]
    assert max(mapped) <= spec["pages"]
    needed = sum(mapped) + 7
    trunk = spec["cache_layout"] == "trunk"
    pool_pages = _straddling(spec, needed) if trunk else needed
    first = pool_pages - needed
    page_ids = first + torch.randperm(needed, generator=gen, device=DEV).to(torch.int32)
    block_table = torch.full((slots, spec["pages"]), -1, dtype=torch.int32, device=DEV)
    cursor = 0
    for slot, count in zip(slot_of.tolist(), mapped):
        block_table[slot, :count] = page_ids[cursor : cursor + count]
        cursor += count

    dtype = spec["cache_dtype"]
    if trunk:
        # Layer `layer`'s K and V planes of the trunk; every other page and plane is NaN.
        layer = spec["layer"]
        pool = _nan_pool((pool_pages, spec["slot_pages"], kv, tpb, d), dtype)
        for p0 in range(first, pool_pages, 1024):
            p1 = min(p0 + 1024, pool_pages)
            pool[p0:p1, 2 * layer : 2 * layer + 2] = torch.randn(
                (p1 - p0, 2, kv, tpb, d), generator=gen, device=DEV
            ).to(dtype)
        k_cache, v_cache = pool[:, 2 * layer], pool[:, 2 * layer + 1]
    else:
        # The widened scratch: two contiguous bf16 pools.
        assert spec["cache_layout"] == "scratch"
        k_cache = torch.randn((pool_pages, kv, tpb, d), generator=gen, device=DEV).to(dtype)
        v_cache = torch.randn((pool_pages, kv, tpb, d), generator=gen, device=DEV).to(dtype)
    assert spec["q_layout"] == "qkv_slice"
    qkv = torch.randn(rows, q_heads + 2 * kv, d, generator=gen, device=DEV).to(spec["q_dtype"])
    q = qkv[:, :q_heads]

    selected = _selected_tokens(positions, seed + 3, spec["compress_ratio"], spec["token_topk"],
                                spec.get("shared_scores", False))  # fmt: skip
    request_indices = slot_of.to(DEV)[req].to(spec["request_dtype"])
    return dict(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        block_table=block_table,
        selected_tokens=selected,
        request_indices=request_indices,
        tokens_per_block=tpb,
        softmax_scale=d**-0.5,
        query_positions=positions.to(spec["request_dtype"]),
        compress_ratio=spec["compress_ratio"],
    )


def _bound(positions, spec):
    cr = spec["compress_ratio"]
    visible = positions.long() + 1
    return (visible // cr).clamp(max=spec["token_topk"] // cr) * cr + visible % cr


def _is_prefix_packed(table):
    valid = table >= 0
    columns = torch.arange(table.shape[1], device=table.device)[None, :]
    return (valid == (columns < valid.sum(1)[:, None])).all(1)


def _assert_straddles(args, edge: int) -> None:
    """The mapped pages put the K and V planes' element offsets on both sides of `edge`."""
    table = args["block_table"]
    pages = table[table >= 0].long()
    for name in ("k_cache", "v_cache"):
        plane = args[name]
        ends = plane.storage_offset() + pages * plane.stride(0) + plane[0].numel() - 1
        assert bool((ends >= edge).any()), f"{name}: no mapped page lies past {edge}"
        assert bool((ends < edge).any()), f"{name}: every mapped page lies past {edge}"


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying, launches) -> None:  # noqa: F811
    certifying(op)
    spec = cell.spec
    args = _build(spec, seed=sum(map(ord, cell.why)))
    if spec["cache_layout"] == "trunk":
        _assert_straddles(args, spec["offset_edge"])
    rows = args["q"].shape[0]
    # The per-row kernel's precondition, which this op shares.
    selected = args["selected_tokens"]
    assert bool(_is_prefix_packed(selected).all())
    assert torch.equal((selected >= 0).sum(1), _bound(args["query_positions"], spec))
    out = op(**args)
    assert [name for name, _ in launches] == list(_KERNELS)
    group_rows = rows_per_group(spec["q_heads"] // spec["kv_heads"])
    assert group_rows == {12: 10, 6: 20}[spec["q_heads"] // spec["kv_heads"]]
    groups = math.ceil(rows / group_rows)
    assert launches[0][1] == (rows,) and launches[1][1] == (groups,)
    assert launches[2][1] == (min(rows, 256), spec["kv_heads"])
    # The row-tile pass is persistent: one CTA per SM, never more than (group, KV head) items.
    items = groups * spec["kv_heads"]
    assert launches[3][1] == (
        min(items, torch.cuda.get_device_properties(0).multi_processor_count),
    )
    assert out.shape == (rows, spec["q_heads"], spec["head_dim"])
    assert out.dtype == args["q"].dtype and out.is_contiguous()
    assert not bool(out.isnan().any())
    _compare(out, op.reference(**args))


TP4 = op.CELLS[0].spec
assert (TP4["q_heads"], TP4["kv_heads"], TP4["head_dim"]) == (6, 1, 256)


def _with(segments, **overrides):
    return {**TP4, "segments": tuple(segments), **overrides}


@pytest.mark.parametrize(
    "segments",
    [
        # Fewer items (2 groups x 1 KV head) than SMs: a 2-CTA grid, one item a CTA.
        [(5000, 37)],
        # A lone group (rows of two requests) between row-tile groups of each request.
        [(3000, 50), (7000, 50)],
    ],
    ids=["fewer items than SMs", "a lone group between row-tile groups"],
)
def test_small_launches_match_the_reference(segments) -> None:
    """The persistent pass's edges at 20-row groups: CTAs left without a second item, and
    an item the pass skips (a lone group) between items it serves."""
    args = _build(_with(segments, slots=8), seed=7700 + len(segments))
    _compare(op(**args), op.reference(**args))
