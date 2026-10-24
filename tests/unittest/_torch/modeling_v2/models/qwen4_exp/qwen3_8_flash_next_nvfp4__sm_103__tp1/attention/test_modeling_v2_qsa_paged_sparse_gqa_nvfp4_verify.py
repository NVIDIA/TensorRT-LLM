# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test: `qsa_paged_sparse_gqa_nvfp4_verify` against the reference of
`qsa_paged_sparse_gqa_nvfp4_decode` and against the per-row kernel, inside the NVFP4 op's band
(4 ulp of each (row, head)'s max, 2 ulp relative RMS), over its cells; plus the routing to the
per-row kernel, forced splits and other rows a request, the launch plan (a partial last wave's
units split and merged), invalid request rows, an unaligned row slice of the selection, the
selections `is_valid` refuses, determinism and graph replay against eager."""

import math
import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1.ops.attention import (  # noqa: E501
    qsa_paged_sparse_gqa_nvfp4_verify as _verify_module,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1.ops.attention.qsa_paged_sparse_gqa_nvfp4_decode import (  # noqa: E501
    nvfp4_sparse_gqa_decode as per_row,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1.ops.attention.qsa_paged_sparse_gqa_nvfp4_verify import (  # noqa: E501
    qsa_paged_sparse_gqa_nvfp4_verify as op,
)

assert torch.cuda.is_available(), "qsa_paged_sparse_gqa_nvfp4_verify requires a CUDA device"

DEV = "cuda"
E4M3_NAN = 0x7F
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


def _expand(blocks, taken, available, visible):
    """Block ids `[rows, 512]` (the first `taken` valid) to the 2051-wide token selection:
    each block's four tokens, then the causal tail, then -1."""
    column = torch.arange(WIDTH, device=DEV)[None, :]
    rows = blocks.shape[0]
    block_of_column = (column // COMPRESS_RATIO).clamp(max=blocks.shape[1] - 1).expand(rows, -1)
    body = torch.gather(blocks, 1, block_of_column) * COMPRESS_RATIO + column % COMPRESS_RATIO
    in_body = column < (taken * COMPRESS_RATIO)[:, None]
    tail_column = column - (taken * COMPRESS_RATIO)[:, None]
    tail = (available * COMPRESS_RATIO)[:, None] + tail_column
    in_tail = ~in_body & (tail_column >= 0) & (tail < visible[:, None])
    selected = torch.where(in_body, body, torch.where(in_tail, tail, torch.full_like(body, -1)))
    return selected.to(torch.int32)


def _selected_tokens(positions, requests, seed, overlap):
    """Each row's top 512 visible four-token blocks, expanded. `overlap` None: independent
    random scores a row, in random order (the decode op's test); else the request's scores plus
    `overlap` x N(0, 1) a row, blocks ascending (as `qsa_topk_expand_decode` writes them)."""
    block_topk = TOKEN_TOPK // COMPRESS_RATIO
    gen = torch.Generator(device=DEV).manual_seed(seed)
    rows = positions.numel()
    visible = positions.long() + 1
    available = visible // COMPRESS_RATIO
    taken = available.clamp(max=block_topk)
    width = max(block_topk, int(available.max().item()) if rows else 0)
    block_ids = torch.arange(width, device=DEV)[None, :]
    if overlap is None:
        scores = torch.rand(rows, width, generator=gen, device=DEV)
        scores = torch.where(block_ids < available[:, None], scores, torch.inf)
        return _expand(scores.argsort(dim=1)[:, :block_topk], taken, available, visible)
    base = torch.rand(int(requests.max().item()) + 1, width, generator=gen, device=DEV)
    noise = torch.randn(rows, width, generator=gen, device=DEV)
    scores = base[requests] + overlap * noise
    scores = torch.where(block_ids < available[:, None], scores, -torch.inf)
    top = scores.topk(block_topk, dim=1).indices
    valid = torch.arange(block_topk, device=DEV)[None, :] < taken[:, None]
    blocks = torch.where(valid, top, torch.full_like(top, 1 << 30)).sort(dim=1).values
    return _expand(blocks, taken, available, visible)


def build(
    segments,
    seed,
    overlap=0.05,
    slots=32,
    pages=260,
    slot_pages=2,
    layer=0,
    global_k=1.0,
    global_v=1.0,
    request_dtype=torch.int32,
    pool_pages=None,
):
    """The decode op's test state for `segments` ((first position, rows) per request, in row
    order) with `_selected_tokens(overlap)`."""
    gen = torch.Generator(device=DEV).manual_seed(seed)
    tpb, kv, d, q_heads = 32, 2, 256, 24
    positions = torch.cat([torch.arange(s, s + n, device=DEV) for s, n in segments])
    req = torch.cat([torch.full((n,), i, device=DEV) for i, (_, n) in enumerate(segments)])
    rows = positions.numel()
    slots = max(slots, len(segments))
    slot_of = torch.randperm(slots, generator=torch.Generator().manual_seed(seed))[: len(segments)]
    mapped = [math.ceil((s + n) / tpb) for s, n in segments]
    needed = sum(mapped) + 7
    planes = slice(2 * layer, 2 * layer + 2)
    pool_pages = pool_pages or needed
    first = pool_pages - needed
    page_ids = first + torch.randperm(needed, generator=gen, device=DEV).to(torch.int32)
    data = torch.zeros(pool_pages, slot_pages, kv, tpb, d // 2, dtype=torch.uint8, device=DEV)
    scales = torch.full(
        (pool_pages, slot_pages, kv, tpb, d // 16), E4M3_NAN, dtype=torch.uint8, device=DEV
    )
    data[first:, planes] = torch.randint(
        0, 256, (needed, 2, kv, tpb, d // 2), generator=gen, device=DEV
    ).to(torch.uint8)
    scales[first:, planes] = torch.randint(
        0x08, 0x31, (needed, 2, kv, tpb, d // 16), generator=gen, device=DEV
    ).to(torch.uint8)
    block_table = torch.full((slots, pages), -1, dtype=torch.int32, device=DEV)
    cursor = 0
    for slot, count in zip(slot_of.tolist(), mapped):
        block_table[slot, :count] = page_ids[cursor : cursor + count]
        cursor += count
    qkv = torch.randn(rows, q_heads + 2 * kv, d, generator=gen, device=DEV).to(torch.bfloat16)
    selected = _selected_tokens(positions, req.long(), seed + 3, overlap)
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


def _args(spec, seed):
    """A cell's arguments, `rows_per_request` and the gate included."""
    kw = {k: v for k, v in spec.items() if k not in ("segments", "gated", "rows_per_request")}
    args = build(spec["segments"], seed=seed, **kw)
    args["rows_per_request"] = spec["rows_per_request"]
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
    )


def _per_row(args):
    return per_row(**{k: v for k, v in args.items() if k != "rows_per_request"})


def _segments(requests, rows_each, first=4000, step=37):
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
    "segments, rows_per_request",
    [(_segments(1, 4), 4), (_segments(7, 4), 4), (_segments(128, 1, step=29), 1)],
    ids=["4 rows", "7 groups of 4", "128 rows of 1"],
)
def test_small_batches_and_single_rows_take_the_per_row_kernel(segments, rows_per_request):
    args = build(segments, seed=31)
    args["rows_per_request"] = rows_per_request
    assert not _takes_grouped(args)
    assert torch.equal(op(**args), _per_row(args))


@pytest.mark.parametrize(
    "rows_per_request, splits", [(2, None), (3, None), (4, 1), (4, 2)], ids=str
)
def test_other_rows_a_request_and_forced_splits(rows_per_request, splits) -> None:
    args = build(_segments(8, rows_per_request), seed=500 + rows_per_request)
    out = _verify_module.nvfp4_sparse_gqa_verify_grouped(
        **args, rows_per_request=rows_per_request, splits=splits
    )
    el, rms, ok = within_ulp(out, op.reference(**args, rows_per_request=rows_per_request))
    assert ok, f"vs reference: {el:.2f} ulp element, {rms:.2f} ulp RMS"


def test_launch_plan_splits_only_a_partial_last_wave() -> None:
    plan = _verify_module.plan_launch
    # within one wave every unit (group, KV head) takes the same splits
    assert plan(32, 2, WIDTH, 152) == (2, None)
    # 512 units on 152 SMs: three full waves of one CTA a unit, the last 56 units split in two
    assert plan(256, 2, WIDTH, 152) == (2, 456)
    # a last wave too full to split (104 units): one CTA a unit
    assert plan(128, 2, WIDTH, 152) == (1, None)


@pytest.mark.parametrize("gated", [False, True], ids=["plain", "gated"])
def test_split_last_wave_matches_one_cta_a_unit(gated) -> None:
    """The units of the full waves are bitwise the one-CTA-a-unit launch; the split and
    merged last wave stays inside the band of it and of the reference."""
    args = build(tuple((2100 + (37 * i) % 1900, 4) for i in range(256)), seed=2718)
    rows = args["q"].shape[0]
    if gated:
        g = torch.Generator(device=DEV).manual_seed(2719)
        args["output_gate"] = torch.randn(rows, 24 * 256, generator=g, device=DEV).to(
            torch.bfloat16
        )
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    splits, n_whole = _verify_module.plan_launch(rows // 4, 2, WIDTH, sms)
    if n_whole is None:
        pytest.skip(f"{sms} SMs: no partial last wave at {rows} rows")
    out = op(**args, rows_per_request=4)
    one = _verify_module.nvfp4_sparse_gqa_verify_grouped(**args, rows_per_request=4, splits=1)
    head = (n_whole // 2) * 4
    assert torch.equal(out[:head], one[:head])
    el, rms, ok = within_ulp(out, one)
    assert ok, f"vs one CTA a unit: {el:.2f} ulp element, {rms:.2f} ulp RMS"
    el, rms, ok = within_ulp(out, op.reference(**args, rows_per_request=4))
    assert ok, f"vs reference: {el:.2f} ulp element, {rms:.2f} ulp RMS"


def test_rows_of_an_invalid_request_are_zero() -> None:
    args = build(_segments(8, 4), seed=99)
    args["request_indices"][5] = -1
    out = op(**args, rows_per_request=4)
    assert not out[5].any()
    el, rms, ok = within_ulp(out, op.reference(**args, rows_per_request=4))
    assert ok, f"vs reference: {el:.2f} ulp element, {rms:.2f} ulp RMS"


def test_unaligned_row_slice_of_the_selection() -> None:
    """A mixed step hands its generation rows as a row slice of the step's 2051-wide
    selection, 16-byte aligned only at every fourth row: an unaligned slice gives the same
    rows as an aligned copy."""
    args = build(((3000, 1),) + _segments(8, 4), seed=4242)
    aligned = dict(args)
    for name in ("q", "selected_tokens", "request_indices"):
        args[name] = args[name][1:]
        aligned[name] = args[name].clone()
    assert args["selected_tokens"].data_ptr() % 16
    assert torch.equal(op(**args, rows_per_request=4), op(**aligned, rows_per_request=4))


def test_is_valid_refuses_selections_that_are_not_expanded() -> None:
    args = build(_segments(8, 4), seed=7)
    op.is_valid(**args, rows_per_request=4)
    good = args["selected_tokens"]
    shifted = good.clone()
    shifted[0, 0] += 1  # off its column's offset within the group
    mixed = good.clone()
    mixed[0, 1] = good[0, 5]  # two groups in one quad
    repeated = good.clone()
    repeated[0, 4:8] = good[0, 0:4]  # one group twice in a row
    for bad in (shifted, mixed, repeated):
        args["selected_tokens"] = bad
        with pytest.raises(AssertionError):
            op.is_valid(**args, rows_per_request=4)


def test_calls_are_bitwise_deterministic() -> None:
    args = build(_segments(8, 4), seed=8800)
    assert torch.equal(op(**args, rows_per_request=4), op(**args, rows_per_request=4))


@pytest.mark.parametrize("rows", [32, 128])
def test_replay_matches_eager(rows) -> None:
    segments = tuple((100 + (499 * g) % 8000, 4) for g in range(rows // 4))
    args = build(segments, seed=77)
    fresh = build(segments, seed=127)
    op(**args, rows_per_request=4)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = op(**args, rows_per_request=4)
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
    assert torch.equal(captured, op(**fresh, rows_per_request=4))
