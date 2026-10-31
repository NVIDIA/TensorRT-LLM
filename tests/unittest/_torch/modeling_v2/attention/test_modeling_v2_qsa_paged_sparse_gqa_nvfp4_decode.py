# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the unified target's `qsa_paged_sparse_gqa_nvfp4_decode` variant (`__r39`), copied from
the tp1 target's: against the reference of `qsa_paged_sparse_gqa_nvfp4` (the unified target's `__r38`)
and QSA's bf16 kernel over the dequantized pool, inside the NVFP4 op's band (4 ulp of each
(row, head)'s max, 2 ulp relative RMS), over its cells with and without the output gate; plus
determinism and graph replay against eager.

The dep4 cells run at one rank's pool as the engine allocates it under `kv_nvfp4`: the
requests' pages scattered over all 211797 pages, both planes' byte offsets on both sides of
2^31 and 2^32 (asserted), every scale outside the mapped pages NaN, and each captured and
replayed at its own compile variant. The bf16 twin dequantizes only the mapped pages."""

import math
import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention.qsa_paged_sparse_gqa_nvfp4_decode import (  # noqa: E501
    qsa_paged_sparse_gqa_nvfp4_decode as op,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp_for_conditional_generation.hf_fc694b5_orig__sm_103__unified.ops.attention import (  # noqa: E501
    qsa_paged_sparse_gqa_nvfp4 as _nvfp4_module,
)
from tensorrt_llm._torch.attention.backends.sparse.qsa.kernels import triton_qsa_paged_sparse_gqa

__extra_import_path__ = [".."]
from _validating import certifying  # noqa: E402,F401 — path above; fixture by name

assert torch.cuda.is_available(), "qsa_paged_sparse_gqa_nvfp4_decode requires a CUDA device"

DEV = "cuda"
E4M3_NAN = 0x7F
ULP = 2.0**-7  # bf16


def within_ulp(out, ref, element_ulp=4.0, rms_ulp=2.0):
    o = out.float().reshape(-1, out.shape[-1])
    r = ref.float().reshape(-1, ref.shape[-1])
    scale = r.abs().amax(dim=1, keepdim=True).clamp_min(1e-9)
    el = ((o - r).abs() / scale).max().item() / ULP
    rms = (((o - r) ** 2).sum() / (r**2).sum().clamp_min(1e-30)).sqrt().item() / ULP
    return el, rms, el <= element_ulp and rms <= rms_ulp


def _selected_tokens(positions, seed, compress_ratio, token_topk):
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


def build(
    segments,
    seed,
    slots=32,
    pages=260,
    slot_pages=2,
    layer=0,
    global_k=1.0,
    global_v=1.0,
    request_dtype=torch.int32,
    pool_pages=None,
    spread=False,
):
    gen = torch.Generator(device=DEV).manual_seed(seed)
    tpb, kv, d, q_heads = 32, 2, 256, 24
    positions = torch.cat([torch.arange(s, s + n, device=DEV) for s, n in segments])
    req = torch.cat([torch.full((n,), i, device=DEV) for i, (_, n) in enumerate(segments)])
    rows = positions.numel()
    slots = max(slots, len(segments))
    slot_of = torch.randperm(slots, generator=torch.Generator().manual_seed(seed))[: len(segments)]
    mapped = [math.ceil((s + n) / tpb) for s, n in segments]
    assert max(mapped) <= pages, (mapped, pages)
    needed = sum(mapped) + 7
    planes = slice(2 * layer, 2 * layer + 2)
    pool_pages = pool_pages or needed
    data = torch.zeros(pool_pages, slot_pages, kv, tpb, d // 2, dtype=torch.uint8, device=DEV)
    # Every scale outside the requests' pages is NaN, so a page address that strayed could
    # not come back finite.
    scales = torch.full(
        (pool_pages, slot_pages, kv, tpb, d // 16), E4M3_NAN, dtype=torch.uint8, device=DEV
    )
    if spread:
        # The requests' pages scattered over the whole pool; the pool's last page is the first
        # request's first page, so it is always mapped.
        picked = torch.randperm(pool_pages - 1, generator=gen, device=DEV)[: needed - 1]
        last = torch.full((1,), pool_pages - 1, device=DEV)
        page_ids = torch.cat([last, picked]).to(torch.int32)
        resident = page_ids.long()
    else:
        first = pool_pages - needed
        page_ids = first + torch.randperm(needed, generator=gen, device=DEV).to(torch.int32)
        resident = slice(first, None)
    data[resident, planes] = torch.randint(
        0, 256, (needed, 2, kv, tpb, d // 2), generator=gen, device=DEV
    ).to(torch.uint8)
    scales[resident, planes] = torch.randint(
        0x08, 0x31, (needed, 2, kv, tpb, d // 16), generator=gen, device=DEV
    ).to(torch.uint8)
    block_table = torch.full((slots, pages), -1, dtype=torch.int32, device=DEV)
    cursor = 0
    for slot, count in zip(slot_of.tolist(), mapped):
        block_table[slot, :count] = page_ids[cursor : cursor + count]
        cursor += count
    qkv = torch.randn(rows, q_heads + 2 * kv, d, generator=gen, device=DEV).to(torch.bfloat16)
    selected = _selected_tokens(positions, seed + 3, 4, 2048)
    return dict(
        q=qkv[:, :q_heads],
        k_cache=data[:, 2 * layer],
        v_cache=data[:, 2 * layer + 1],
        k_scale_cache=scales[:, 2 * layer],
        v_scale_cache=scales[:, 2 * layer + 1],
        kv_scale_quant_orig=torch.tensor(
            [1.0, global_k, global_v], dtype=torch.float32, device=DEV
        ),
        block_table=block_table,
        selected_tokens=selected,
        request_indices=slot_of.to(DEV)[req].to(request_dtype),
        tokens_per_block=tpb,
        softmax_scale=d**-0.5,
    )


def bf16_twin(args):
    """QSA's bf16 sparse GQA over a bf16 pool holding the dequantized values of the pages the
    table maps, renumbered in order, so a cell's pool can reach past 2^32 bytes without the
    twin dequantizing all of it."""
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
        values = _nvfp4_module._dequantize_tokens(
            data, scales, pages, slots, args["kv_scale_quant_orig"][index], grouped
        )
        planes.append(values.to(args["q"].dtype).transpose(1, 2).contiguous())
    twin = {n: args[n] for n in ("q", "selected_tokens", "request_indices", "tokens_per_block")}
    twin.update(
        softmax_scale=args["softmax_scale"],
        block_table=twin_table,
        k_cache=planes[0],
        v_cache=planes[1],
    )
    return triton_qsa_paged_sparse_gqa(**twin)


def _assert_straddles(args, edges) -> None:
    """The pages the cell's rows read put both planes' byte offsets on both sides of every
    edge: what the cell claims about its pool."""
    table = args["block_table"]
    pages = table[table >= 0].long()
    for name in ("k_cache", "v_cache", "k_scale_cache", "v_scale_cache"):
        plane = args[name]
        base = plane.data_ptr() - plane.untyped_storage().data_ptr()
        ends = base + pages * plane.stride(0) + plane[0].numel() - 1
        for edge in edges:
            assert bool((ends >= edge).any()), f"{name}: no page the rows read lies past {edge}"
            assert bool((ends < edge).any()), f"{name}: every page the rows read lies past {edge}"
    last = args["k_cache"].shape[0] - 1
    assert bool((pages == last).any()), "the pool's last page is not among the rows' pages"


#: Spec keys that describe the cell rather than the state `build` makes.
_CELL_ONLY = ("segments", "gated", "capture", "large", "offset_edges")


def _args(spec, seed):
    kw = {k: v for k, v in spec.items() if k not in _CELL_ONLY}
    args = build(spec["segments"], seed=seed, **kw)
    if spec.get("gated"):
        rows = args["q"].shape[0]
        g = torch.Generator(device=DEV).manual_seed(seed + 1)
        args["output_gate"] = torch.randn(rows, 24 * 256, generator=g, device=DEV).to(
            torch.bfloat16
        )
    return args


def _replay_in_place(args, seed) -> None:
    """The call captured at the cell's own compile variant, fresh q (and gate) written into
    the captured buffers, replayed: bitwise the eager call on the fresh values."""
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            captured = op(**args)
        g = torch.Generator(device=DEV).manual_seed(seed)
        for name in ("q", "output_gate"):
            if name in args:
                args[name].copy_(torch.randn(args[name].shape, generator=g, device=DEV))
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(captured, op(**args)), "replay differs from eager"
    finally:
        del graph
        torch.cuda.synchronize()


def _marks(cell):
    return (pytest.mark.large, pytest.mark.high_cuda_memory) if cell.spec.get("large") else ()


@pytest.mark.parametrize(
    "cell", [pytest.param(c, marks=_marks(c), id=c.why[:44]) for c in op.CELLS]
)
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture
    certifying(op)
    seed = zlib.crc32(cell.why.encode()) % (2**31)
    args = _args(cell.spec, seed)
    if "offset_edges" in cell.spec:
        _assert_straddles(args, cell.spec["offset_edges"])
    out = op(**args)
    assert out.dtype == torch.bfloat16 and not out.isnan().any()
    el, rms, ok = within_ulp(out, op.reference(**args))
    assert ok, f"vs reference: {el:.2f} ulp element, {rms:.2f} ulp RMS"
    # QSA's bf16 kernel has no output gate, so a gated cell checks its ungated call over the
    # same state against it.
    gate = args.pop("output_gate", None)
    plain = out if gate is None else op(**args)
    el, rms, ok = within_ulp(plain, bf16_twin(args))
    assert ok, f"vs QSA bf16 kernel: {el:.2f} ulp element, {rms:.2f} ulp RMS"
    if gate is not None:
        args["output_gate"] = gate
    if cell.spec.get("capture"):
        _replay_in_place(args, seed + 7)


def test_band_refuses_a_wrong_answer() -> None:
    """The band's negative controls, at the dep4 32-request step over a pool of the pages it
    maps: the reference read through a table one page off, and with one selected token
    dropped a row, both lie outside the band the cells pass."""
    dep4 = next(c.spec for c in op.CELLS if c.spec.get("large") and len(c.spec["segments"]) == 32)
    args = _args({**dep4, "pool_pages": None, "spread": False}, seed=9900)
    out = op(**args)
    assert within_ulp(out, op.reference(**args))[2], "the unaltered reference is refused"
    table = args["block_table"]
    shifted = dict(args, block_table=torch.where(table >= 0, table + 1, table))
    selection = args["selected_tokens"].clone()
    last = (selection >= 0).sum(1) - 1
    selection[torch.arange(selection.shape[0], device=DEV), last] = -1
    dropped = dict(args, selected_tokens=selection)
    for name, wrong in (("a table one page off", shifted), ("one token dropped a row", dropped)):
        el, rms, ok = within_ulp(out, op.reference(**wrong))
        assert not ok, f"the band accepts {name}: {el:.2f} ulp element, {rms:.2f} ulp RMS"


def test_unaligned_row_slice_of_the_selection() -> None:
    """A mixed step hands its generation rows as a row slice of the step's 2051-wide
    selection, 16-byte aligned only at every fourth row: an unaligned slice gives the same
    rows as an aligned copy."""
    args = build(tuple((3000 + 211 * i, 1) for i in range(9)), seed=4242)
    aligned = dict(args)
    for name in ("q", "selected_tokens", "request_indices"):
        args[name] = args[name][1:]
        aligned[name] = args[name].clone()
    assert args["selected_tokens"].data_ptr() % 16
    assert torch.equal(op(**args), op(**aligned))


def test_calls_are_bitwise_deterministic() -> None:
    args = build(tuple((4000 + 37 * i, 2) for i in range(4)), seed=8800)
    assert torch.equal(op(**args), op(**args))


@pytest.mark.parametrize("rows", [1, 64])
def test_replay_matches_eager(rows) -> None:
    segments = (
        ((4000, 1),) if rows == 1 else tuple((100 + (499 * g) % 8000, 1) for g in range(rows))
    )
    args = build(segments, seed=77)
    fresh = build(segments, seed=127)
    op(**args)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = op(**args)
    for name in (
        "q",
        "k_cache",
        "v_cache",
        "k_scale_cache",
        "v_scale_cache",
        "block_table",
        "selected_tokens",
        "request_indices",
    ):
        args[name].copy_(fresh[name])
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(captured, op(**fresh))
