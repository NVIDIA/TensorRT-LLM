# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the unified target's qsa_paged_sparse_gqa_nvfp4 variant (`__r38`).

The paged state is real: one NVFP4 pool as the QSA cache manager serves it --
e2m1 data `[pages, 2, kv, tpb, D/2]` and e4m3 block scales `[pages, 2, kv, tpb,
D/16]` -- sized from the pages its requests map, and the engine's slot table,
a row per request slot, more slots than requests, assigned by a seeded
permutation, `-1` past each sequence. Selection rows are built the way QSA's
top-k + expansion lay them out, packed at the front.

Beside `_compare` against the reference, every cell runs QSA's bf16 sparse GQA
over a bf16 pool that holds the dequantized values and holds the two outputs to
the same band: the NVFP4 kernels are the bf16 ones with a dequantizing load.
"""

import math
import re

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention import (
    qsa_paged_sparse_gqa_nvfp4 as _entry,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention.qsa_paged_sparse_gqa_nvfp4 import (  # noqa: E501
    qsa_paged_sparse_gqa_nvfp4 as op,
)
from tensorrt_llm._torch.attention.backends.sparse.qsa.kernels import triton_qsa_paged_sparse_gqa

__extra_import_path__ = [".."]
from _validating import assert_within_ulp, certifying, validating  # noqa: E402,F401

assert torch.cuda.is_available(), "qsa_paged_sparse_gqa_nvfp4 requires a CUDA device"

DEV = "cuda"
ENGINE_MAX_SEQ_LEN = 8192
#: The e4m3fn NaN bit pattern.
E4M3_NAN = 0x7F

_KERNELS = {
    "_nvfp4_sparse_gqa_prefill_kernel": "prefill",
    "_nvfp4_sparse_gqa_splitk_kernel": "splitk",
    "_nvfp4_merge_splitk_kernel": "merge",
}


class _LaunchRecorder:
    """Records a Triton kernel's grid indexing, then delegates to the kernel."""

    def __init__(self, inner, name, log):
        self._inner, self._name, self._log = inner, name, log

    def __getitem__(self, grid):
        self._log.append((self._name, tuple(grid)))
        return self._inner[grid]


@pytest.fixture
def launches(monkeypatch):
    """Which branch kernels the next call launches, in order, with grids."""
    log: list = []
    for name, tag in _KERNELS.items():
        monkeypatch.setattr(_entry, name, _LaunchRecorder(getattr(_entry, name), tag, log))
    return log


def _compare(out: torch.Tensor, ref: torch.Tensor) -> None:
    """4 ulp of each (row, head)'s magnitude, 2 ulp relative RMS, in q's dtype.

    After the dequantize the kernels are the bf16 sparse GQA kernels, whose band
    this is (they round the online-softmax probabilities to the value dtype
    before P.V); the dequantize itself is exact at unit scale and rounded once,
    as `reference` rounds it, otherwise.
    """
    assert out.dtype == ref.dtype, (out.dtype, ref.dtype)
    assert out.shape == ref.shape, (out.shape, ref.shape)
    if out.numel() == 0:
        return
    assert_within_ulp(out, ref, element_ulp=4.0, rms_ulp=2.0)


def _selected_tokens(positions, seed, compress_ratio, token_topk):
    """QSA's expansion layout: chosen complete groups, the trailing partial group, -1."""
    block_topk = token_topk // compress_ratio
    final_topk = token_topk + compress_ratio - 1
    gen = torch.Generator(device=DEV).manual_seed(seed)
    rows = positions.numel()
    visible = positions.long() + 1
    available = visible // compress_ratio
    taken = available.clamp(max=block_topk)
    width = max(block_topk, int(available.max().item()) if rows else 0)
    scores = torch.rand(rows, width, generator=gen, device=DEV)
    group_ids = torch.arange(width, device=DEV)[None, :]
    scores = torch.where(group_ids < available[:, None], scores, torch.inf)
    chosen = scores.argsort(dim=1)[:, :block_topk]
    column = torch.arange(final_topk, device=DEV)[None, :]
    group_of_column = (column // compress_ratio).clamp(max=block_topk - 1).expand(rows, -1)
    body = torch.gather(chosen, 1, group_of_column) * compress_ratio + column % compress_ratio
    in_body = column < (taken * compress_ratio)[:, None]
    tail_column = column - (taken * compress_ratio)[:, None]
    tail = (available * compress_ratio)[:, None] + tail_column
    in_tail = ~in_body & (tail_column >= 0) & (tail < visible[:, None])
    selected = torch.where(in_body, body, torch.where(in_tail, tail, torch.full_like(body, -1)))
    return selected.to(torch.int32)


def _random_pool(pages, kv, tpb, d, gen):
    """Pool bytes: any e2m1 nibble, and positive e4m3 scales from 2^-6 to 0.5."""
    data = torch.randint(0, 256, (pages, 2, kv, tpb, d // 2), generator=gen, device=DEV)
    codes = torch.randint(0x08, 0x31, (pages, 2, kv, tpb, d // 16), generator=gen, device=DEV)
    return data.to(torch.uint8), codes.to(torch.uint8)


def _build(spec, seed):
    """Real paged state for one spec, keyed by the op's parameter names."""
    gen = torch.Generator(device=DEV).manual_seed(seed)
    tpb, kv, d, q_heads = (spec[k] for k in ("tokens_per_block", "kv_heads", "head_dim", "q_heads"))
    segments = list(spec["segments"])
    positions = torch.cat([torch.arange(s, s + n, device=DEV) for s, n in segments])
    req = torch.cat([torch.full((n,), i, device=DEV) for i, (_, n) in enumerate(segments)])
    rows, slots = positions.numel(), spec["slots"]
    assert len(segments) <= slots
    slot_of = torch.randperm(slots, generator=torch.Generator().manual_seed(seed))[: len(segments)]
    mapped = [math.ceil((s + n) / tpb) for s, n in segments]
    assert not mapped or max(mapped) <= spec["pages"]
    needed = sum(mapped) + 7
    slot_pages, layer = spec["slot_pages"], spec["layer"]
    planes = slice(2 * layer, 2 * layer + 2)
    # The requests sit on the pool's last `needed` pages, and every scale outside them is
    # NaN, so a page address that strayed could not come back finite.
    pool_pages = spec.get("pool_pages", needed)
    assert pool_pages >= needed
    first = pool_pages - needed
    page_ids = first + torch.randperm(needed, generator=gen, device=DEV).to(torch.int32)
    data = torch.zeros(pool_pages, slot_pages, kv, tpb, d // 2, dtype=torch.uint8, device=DEV)
    scales = torch.full(
        (pool_pages, slot_pages, kv, tpb, d // 16), E4M3_NAN, dtype=torch.uint8, device=DEV
    )
    data[first:, planes], scales[first:, planes] = _random_pool(needed, kv, tpb, d, gen)
    block_table = torch.full((slots, spec["pages"]), -1, dtype=torch.int32, device=DEV)
    cursor = 0
    for slot, count in zip(slot_of.tolist(), mapped):
        block_table[slot, :count] = page_ids[cursor : cursor + count]
        cursor += count

    assert spec["q_layout"] == "qkv_slice", spec["q_layout"]
    qkv = torch.randn(rows, q_heads + 2 * kv, d, generator=gen, device=DEV).to(spec["q_dtype"])
    q = qkv[:, :q_heads]
    selected = _selected_tokens(positions, seed + 3, spec["compress_ratio"], spec["token_topk"])
    request_indices = slot_of.to(DEV)[req].to(spec["request_dtype"])

    args = dict(
        q=q,
        k_cache=data[:, 2 * layer],
        v_cache=data[:, 2 * layer + 1],
        k_scale_cache=scales[:, 2 * layer],
        v_scale_cache=scales[:, 2 * layer + 1],
        kv_scale_quant_orig=torch.tensor(
            [1.0, spec["global_k"], spec["global_v"]], dtype=torch.float32, device=DEV
        ),
        block_table=block_table,
        selected_tokens=selected,
        request_indices=request_indices,
        tokens_per_block=tpb,
        softmax_scale=d**-0.5,
    )
    if spec["prefill_pair"]:
        args.update(query_positions=positions, compress_ratio=spec["compress_ratio"])
    return args


def _bf16_twin(args):
    """QSA's bf16 sparse GQA over a bf16 pool holding the dequantized values.

    The twin pool holds only the pages the table maps, renumbered in order, so a cell's
    pool can reach past 2^31 bytes without the twin dequantizing all of it.
    """
    table = args["block_table"]
    mapped = torch.unique(table[table >= 0]).long()
    renumber = torch.full((args["k_cache"].shape[0],), -1, dtype=torch.int32, device=DEV)
    renumber[mapped] = torch.arange(mapped.numel(), dtype=torch.int32, device=DEV)
    twin_table = torch.where(table >= 0, renumber[table.clamp_min(0).long()], table)
    tpb = args["tokens_per_block"]
    pages = mapped[:, None].expand(-1, tpb)
    slots = torch.arange(tpb, device=DEV)[None, :].expand(mapped.numel(), -1)
    planes = []
    for data, scales, index, grouped in (
        (args["k_cache"], args["k_scale_cache"], 1, False),
        (args["v_cache"], args["v_scale_cache"], 2, True),
    ):
        values = _entry._dequantize_tokens(
            data, scales, pages, slots, args["kv_scale_quant_orig"][index], grouped
        )  # [mapped pages, tpb, H, D]
        planes.append(values.to(args["q"].dtype).transpose(1, 2).contiguous())
    twin = {
        name: args[name] for name in ("q", "selected_tokens", "request_indices", "tokens_per_block")
    }
    twin.update(
        softmax_scale=args["softmax_scale"],
        block_table=twin_table,
        k_cache=planes[0],
        v_cache=planes[1],
    )
    if "query_positions" in args:
        twin.update(query_positions=args["query_positions"], compress_ratio=args["compress_ratio"])
    return triton_qsa_paged_sparse_gqa(**twin)


def _bound(positions, spec):
    cr = spec["compress_ratio"]
    visible = positions.long() + 1
    return (visible // cr).clamp(max=spec["token_topk"] // cr) * cr + visible % cr


def _assert_straddles(args, edge: int) -> None:
    """The pages the cell's rows read put the scale plane's byte offsets on both sides of
    `edge` and the data plane's past it: what the cell claims about its pool."""
    table = args["block_table"]
    pages = table[table >= 0].long()
    for name, side in (("k_scale_cache", "both"), ("v_scale_cache", "both"), ("k_cache", "past")):
        plane = args[name]
        base = plane.data_ptr() - plane.untyped_storage().data_ptr()
        ends = base + pages * plane.stride(0) + plane[0].numel() - 1
        assert bool((ends >= edge).any()), f"{name}: no page the rows read lies past {edge}"
        if side == "both":
            assert bool((ends < edge).any()), f"{name}: every page lies past {edge}"


def _is_prefix_packed(table):
    valid = table >= 0
    columns = torch.arange(table.shape[1], device=table.device)[None, :]
    return (valid == (columns < valid.sum(1)[:, None])).all(1)


_OPERANDS = (
    "q",
    "k_cache",
    "v_cache",
    "k_scale_cache",
    "v_scale_cache",
    "block_table",
    "selected_tokens",
    "request_indices",
)


def _capture_sweep(spec, seed, launches):
    """The decode site under a CUDA graph, at each G for each rows-per-request:
    capture, write a second state's operands in place, replay, and require the
    eager call on that second state bitwise."""
    sweep = spec["capture"]
    for per_request in sweep["rows_per_request"]:
        for groups in sweep["groups"]:
            _capture_once(
                _decode_step(spec, groups, per_request),
                seed + 100 * per_request + groups,
                launches,
                where=f"G={groups} x {per_request} rows",
            )


def _start(request: int, rows: int) -> int:
    """A request's first position, across the budget."""
    return min(100 + (499 * request) % 8000, ENGINE_MAX_SEQ_LEN - rows)


def _decode_step(spec, groups, per_request):
    """The decode site's step: `groups` requests x `per_request` rows."""
    segments = tuple((_start(g, per_request), per_request) for g in range(groups))
    return {**spec, "segments": segments}


def _capture_once(shape, seed, launches, where) -> None:
    """One call of `shape` captured, a second state's operands written in place,
    replayed: the replay must be bitwise the eager call on the second state."""
    where = f"{where}, [{shape['slots']}, {shape['pages']}] table"
    args = _build(shape, seed)
    fresh = _build(shape, seed + 50)
    launches.clear()
    op(**args)
    assert [name for name, _ in launches] == list(shape["launches"]), where
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            captured = op(**args)
        for name in _OPERANDS:
            args[name].copy_(fresh[name])
        graph.replay()
        torch.cuda.synchronize()
        eager = op(**fresh)
        assert torch.equal(captured, eager), f"replay differs from eager at {where}"
        _compare(captured, op.reference(**fresh))
    finally:
        del graph
        torch.cuda.synchronize()


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying, launches) -> None:  # noqa: F811
    certifying(op)
    spec = cell.spec
    if "capture" in spec:
        _capture_sweep(spec, seed=sum(map(ord, cell.why)), launches=launches)
        return
    args = _build(spec, seed=sum(map(ord, cell.why)))
    rows = args["q"].shape[0]
    if spec["prefill_pair"]:
        selected = args["selected_tokens"]
        assert bool(_is_prefix_packed(selected).all())
        valid = (selected >= 0).sum(1)
        assert torch.equal(valid, _bound(args["query_positions"], spec))
    if "offset_edge" in spec:
        _assert_straddles(args, spec["offset_edge"])

    out = op(**args)
    assert [name for name, _ in launches] == list(spec["launches"])
    assert launches[0][1][:2] == (rows, spec["kv_heads"])
    assert out.shape == (rows, spec["q_heads"], spec["head_dim"])
    assert out.dtype == args["q"].dtype and out.is_contiguous()
    assert not bool(out.isnan().any())
    _compare(out, op.reference(**args))
    # QSA's bf16 kernels over the same values: inside the same band, not bitwise
    # (the dequantizing load changes the compiled kernel).
    _compare(out, _bf16_twin(args))


CKPT = op.CELLS[0].spec


def _with(**overrides):
    return {**CKPT, **overrides}


def test_calls_are_bitwise_deterministic() -> None:
    args = _build(_with(segments=tuple((4000 + 37 * i, 2) for i in range(4))), seed=8800)
    assert torch.equal(op(**args), op(**args))


def test_rejects_positions_without_a_compress_ratio() -> None:
    """The op's own loud refusal -- the reason the entry carries no `is_valid`: half of
    the prefill pair."""
    args = _build(_with(segments=((600, 1),) * 4), seed=9100)
    args["selected_tokens"] = args["selected_tokens"][:, :64].contiguous()
    args["query_positions"] = torch.zeros(4, dtype=torch.int64, device=DEV)
    message = "pass both query_positions and compress_ratio, or neither"
    with pytest.raises(ValueError, match=re.escape(message)):
        with validating(op):
            op(**args)
