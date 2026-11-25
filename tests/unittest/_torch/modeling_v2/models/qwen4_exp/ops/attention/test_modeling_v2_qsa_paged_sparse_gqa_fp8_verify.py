# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test (SM103): `qsa_paged_sparse_gqa_fp8_verify` over an e4m3 pool against the per-row
op's reference and the per-row kernel, inside the decode op's band (4 ulp of each (row, head)'s
max, 2 ulp relative RMS), over its cells; plus the routed floor (`min_groups`), the two-wave and
tail plans, forced splits, an invalid request row, the selections `is_valid` refuses,
determinism and graph replay against eager."""

import math
import zlib

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.attention import (  # noqa: E501
    qsa_paged_sparse_gqa_fp8_verify as _verify_module,
)
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.attention.qsa_paged_sparse_gqa import (  # noqa: E501
    qsa_paged_sparse_gqa,
)
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.attention.qsa_paged_sparse_gqa_fp8_verify import (  # noqa: E501
    qsa_paged_sparse_gqa_fp8_verify as op,
)

assert torch.cuda.is_available(), "qsa_paged_sparse_gqa_fp8_verify requires a CUDA device"
if torch.cuda.get_device_capability() != (10, 3):
    pytest.skip("the bitmap-union e4m3 verify kernel serves SM103", allow_module_level=True)

DEV = "cuda"
ULP = 2.0**-7  # bf16
COMPRESS_RATIO, TOKEN_TOPK = 4, 2048
WIDTH = TOKEN_TOPK + COMPRESS_RATIO - 1


def within_ulp(out, ref, element_ulp=4.0, rms_ulp=2.0):
    o = out.float().reshape(-1, out.shape[-1])
    r = ref.float().reshape(-1, ref.shape[-1])
    scale = r.abs().amax(dim=1, keepdim=True).clamp_min(1e-9)
    el = ((o - r).abs() / scale).max().item() / ULP
    rms = (((o - r) ** 2).sum() / (r**2).sum().clamp_min(1e-30)).sqrt().item() / ULP
    return el, rms, el <= element_ulp and rms <= rms_ulp


def _selected_tokens(positions, requests, seed, overlap):
    """Each row's top 512 visible four-token blocks (the request's scores plus `overlap` x
    N(0, 1) a row), ascending as `qsa_topk_expand_decode` writes them, expanded to 2051 columns:
    each block's four tokens, then the causal tail, then -1."""
    block_topk = TOKEN_TOPK // COMPRESS_RATIO
    gen = torch.Generator(device=DEV).manual_seed(seed)
    rows = positions.numel()
    visible = positions.long() + 1
    available = visible // COMPRESS_RATIO
    taken = available.clamp(max=block_topk)
    width = max(block_topk, int(available.max().item()))
    block_ids = torch.arange(width, device=DEV)[None, :]
    base = torch.rand(int(requests.max().item()) + 1, width, generator=gen, device=DEV)
    scores = base[requests] + overlap * torch.randn(rows, width, generator=gen, device=DEV)
    scores = torch.where(block_ids < available[:, None], scores, -torch.inf)
    top = scores.topk(block_topk, dim=1).indices
    valid = torch.arange(block_topk, device=DEV)[None, :] < taken[:, None]
    blocks = torch.where(valid, top, torch.full_like(top, 1 << 30)).sort(dim=1).values
    column = torch.arange(WIDTH, device=DEV)[None, :]
    of_column = (column // COMPRESS_RATIO).clamp(max=block_topk - 1).expand(rows, -1)
    body = torch.gather(blocks, 1, of_column) * COMPRESS_RATIO + column % COMPRESS_RATIO
    in_body = column < (taken * COMPRESS_RATIO)[:, None]
    tail_column = column - (taken * COMPRESS_RATIO)[:, None]
    tail = (available * COMPRESS_RATIO)[:, None] + tail_column
    in_tail = ~in_body & (tail_column >= 0) & (tail < visible[:, None])
    sel = torch.where(in_body, body, torch.where(in_tail, tail, torch.full_like(body, -1)))
    return sel.to(torch.int32)


def build(segments, seed, overlap=0.05, pages=516, request_dtype=torch.int32, slot_pages=2):
    """An e4m3 pool `[pool pages, slot pages, 2 KV heads, 32 tokens, 256]` (layer 0's K and V
    planes of each slot), a block table of `pages` pages a request and the call's operands for
    `segments` ((first position, rows) per request, in row order)."""
    gen = torch.Generator(device=DEV).manual_seed(seed)
    tpb, kv, d, heads = 32, 2, 256, 24
    positions = torch.cat([torch.arange(s, s + n, device=DEV) for s, n in segments])
    req = torch.cat([torch.full((n,), i, device=DEV) for i, (_, n) in enumerate(segments)])
    rows = positions.numel()
    slots = max(32, len(segments))
    slot_of = torch.randperm(slots, generator=torch.Generator().manual_seed(seed))[: len(segments)]
    mapped = [math.ceil((s + n) / tpb) for s, n in segments]
    needed = sum(mapped) + 7
    # uint8 codes drawn directly (a 220k-token request is ~7k pages: no int64 staging)
    raw = torch.randint(
        0, 256, (needed, slot_pages, kv, tpb, d), generator=gen, device=DEV, dtype=torch.uint8
    )
    raw = torch.where((raw & 0x7F) == 0x7F, raw - 1, raw)  # no e4m3 NaN
    pool = raw.view(torch.float8_e4m3fn)
    page_ids = torch.randperm(needed, generator=gen, device=DEV).to(torch.int32)
    block_table = torch.full((slots, pages), -1, dtype=torch.int32, device=DEV)
    cursor = 0
    for slot, count in zip(slot_of.tolist(), mapped):
        block_table[slot, :count] = page_ids[cursor : cursor + count]
        cursor += count
    # q rows inside the packed [q | k | v] row, as the target hands them
    qkv = torch.randn(rows, heads + 2 * kv, d, generator=gen, device=DEV).to(torch.bfloat16)
    return dict(
        q=qkv[:, :heads],
        k_cache=pool[:, 0],
        v_cache=pool[:, 1],
        block_table=block_table,
        selected_tokens=_selected_tokens(positions, req.long(), seed + 3, overlap),
        request_indices=slot_of.to(DEV)[req].to(request_dtype),
        tokens_per_block=tpb,
        softmax_scale=d**-0.5,
    )


def _args(spec, seed):
    kw = {k: spec[k] for k in ("overlap", "pages", "request_dtype") if k in spec}
    args = build(spec["segments"], seed=seed, **kw)
    args["rows_per_request"] = spec["rows_per_request"]
    args["min_groups"] = spec.get("min_groups")
    if spec.get("gated"):
        rows = args["q"].shape[0]
        g = torch.Generator(device=DEV).manual_seed(seed + 1)
        args["output_gate"] = torch.randn(rows, 24 * 256, generator=g, device=DEV).to(
            torch.bfloat16
        )
    return args


def _takes_grouped(args):
    return _verify_module.takes_grouped(
        args["q"],
        args["k_cache"],
        args["block_table"],
        args["selected_tokens"],
        args["tokens_per_block"],
        args["rows_per_request"],
        min_groups=args.get("min_groups"),
    )


def _per_row(args):
    out = qsa_paged_sparse_gqa(
        q=args["q"],
        k_cache=args["k_cache"],
        v_cache=args["v_cache"],
        block_table=args["block_table"],
        selected_tokens=args["selected_tokens"],
        request_indices=args["request_indices"],
        tokens_per_block=args["tokens_per_block"],
        softmax_scale=args["softmax_scale"],
        query_positions=None,
        compress_ratio=None,
        output_gate=None,
    )
    gate = args.get("output_gate")
    if gate is None:
        return out
    gated = out.float() * torch.sigmoid(gate.reshape(out.shape).float())
    return gated.to(torch.bfloat16)


def _segments(requests, rows_each=4, first=4000, step=37):
    return tuple((first + step * i, rows_each) for i in range(requests))


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell) -> None:
    args = _args(cell.spec, zlib.crc32(cell.why.encode()) % (2**31))
    op.is_valid(**args)
    assert _takes_grouped(args), "every cell runs on the grouped kernel"
    out = op(**args)
    assert out.dtype == torch.bfloat16 and not out.isnan().any()
    el, rms, ok = within_ulp(out, op.reference(**args))
    assert ok, f"vs reference: {el:.2f} ulp element, {rms:.2f} ulp RMS"
    el, rms, ok = within_ulp(out, _per_row(args))
    assert ok, f"vs the per-row kernel: {el:.2f} ulp element, {rms:.2f} ulp RMS"


@pytest.mark.parametrize(
    "requests, min_groups, grouped",
    [(7, None, False), (8, None, True), (2, 2, True), (1, 2, False), (3, 8, False)],
)
def test_routed_floor(requests, min_groups, grouped) -> None:
    """The routed entry takes the grouped kernel from the caller's floor (`MIN_GROUPS` by
    default) and the per-row kernel with the gate apart below it."""
    args = build(_segments(requests), seed=31 + requests)
    args["rows_per_request"] = 4
    args["min_groups"] = min_groups
    assert _takes_grouped(args) == grouped
    out = op(**args)
    if not grouped:
        assert torch.equal(out, _per_row(args))
    el, rms, ok = within_ulp(out, op.reference(**args))
    assert ok, f"vs reference: {el:.2f} ulp element, {rms:.2f} ulp RMS"


def test_floor_outside_the_op_range_raises() -> None:
    args = build(_segments(8), seed=5)
    for bad in (1, 9):
        with pytest.raises(ValueError):
            _verify_module.takes_grouped(
                args["q"], args["k_cache"], args["block_table"], args["selected_tokens"], 32, 4,
                min_groups=bad,
            )  # fmt: skip


def test_launch_plan_two_waves() -> None:
    plan = _verify_module.plan_launch
    # more than half of the SMs as units: two splits a unit (two waves), not one CTA a unit
    assert plan(45, 2, WIDTH, 152) == (_verify_module.TWO_WAVE_SPLITS, None)
    assert plan(76, 2, WIDTH, 152) == (_verify_module.TWO_WAVE_SPLITS, None)
    # one wave of several splits a unit below half of the SMs
    assert plan(32, 2, WIDTH, 152) == (2, None)
    assert plan(3, 2, WIDTH, 152) == (25, None)
    # past one wave: the last partial wave split, as the NVFP4 op
    assert plan(256, 2, WIDTH, 152) == (2, 456)


@pytest.mark.parametrize("rows_per_request, splits", [(2, None), (4, 1), (4, 3)], ids=str)
def test_other_rows_a_request_and_forced_splits(rows_per_request, splits) -> None:
    args = build(_segments(8, rows_per_request), seed=500 + rows_per_request)
    out = _verify_module.fp8_sparse_gqa_verify_grouped(
        **args, rows_per_request=rows_per_request, splits=splits
    )
    el, rms, ok = within_ulp(out, op.reference(**args, rows_per_request=rows_per_request))
    assert ok, f"vs reference: {el:.2f} ulp element, {rms:.2f} ulp RMS"


def test_wide_table_matches_the_narrow_one() -> None:
    """The same requests over an 8190-page block table (the wide plan: bitmaps in the K/V slots,
    the block table from global memory, a bounded multi-round scan) and over a 516-page one
    (the union scratch in raw slot 0) give the same output: same union, same tiles. The
    requests' contexts (up to 14971 tokens, 468 pages) fit the narrow table."""
    segments = _segments(12, first=4000, step=997)
    splits = _verify_module.plan_splits(
        12, 2, WIDTH, _verify_module._num_sms(torch.cuda.current_device())
    )
    assert not _verify_module._layout(4, 12, WIDTH, 516, 32, splits).wide
    assert _verify_module._layout(4, 12, WIDTH, 8190, 32, splits).wide
    narrow = build(segments, seed=2024, pages=516)
    wide = dict(narrow)
    table = torch.full((narrow["block_table"].shape[0], 8190), -1, dtype=torch.int32, device=DEV)
    table[:, :516] = narrow["block_table"]
    wide["block_table"] = table
    assert torch.equal(op(**wide, rows_per_request=4), op(**narrow, rows_per_request=4))


def test_rows_of_an_invalid_request_are_zero() -> None:
    args = build(_segments(8), seed=99)
    args["request_indices"][5] = -1
    out = op(**args, rows_per_request=4)
    assert not out[5].any()


def test_is_valid_refuses_selections_that_are_not_expanded() -> None:
    args = build(_segments(8), seed=7)
    op.is_valid(**args, rows_per_request=4)
    good = args["selected_tokens"]
    shifted = good.clone()
    shifted[0, 0] += 1
    repeated = good.clone()
    repeated[0, 4:8] = good[0, 0:4]
    for bad in (shifted, repeated):
        args["selected_tokens"] = bad
        with pytest.raises(AssertionError):
            op.is_valid(**args, rows_per_request=4)


def test_calls_are_bitwise_deterministic() -> None:
    args = build(_segments(12), seed=8800)
    assert torch.equal(op(**args, rows_per_request=4), op(**args, rows_per_request=4))


def test_replay_matches_eager() -> None:
    segments = tuple((100 + (499 * g) % 8000, 4) for g in range(8))
    args = build(segments, seed=77)
    fresh = build(segments, seed=127)
    op(**args, rows_per_request=4, min_groups=2)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = op(**args, rows_per_request=4, min_groups=2)
    for name in ("q", "k_cache", "v_cache", "block_table", "selected_tokens", "request_indices"):
        args[name].copy_(fresh[name])
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(captured, op(**fresh, rows_per_request=4, min_groups=2))
