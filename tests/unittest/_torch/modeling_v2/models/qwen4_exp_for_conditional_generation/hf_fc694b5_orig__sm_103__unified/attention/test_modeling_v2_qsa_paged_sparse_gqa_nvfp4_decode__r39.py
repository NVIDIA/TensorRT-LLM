# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for this target's `qsa_paged_sparse_gqa_nvfp4_decode` variant (`__r39`), copied from
the tp1 target's: against the reference of `qsa_paged_sparse_gqa_nvfp4` (this target's `__r38`)
and QSA's bf16 kernel over the dequantized pool, inside the NVFP4 op's band (4 ulp of each
(row, head)'s max, 2 ulp relative RMS), over its cells with and without the output gate; plus
determinism and graph replay against eager."""

import math
import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp_for_conditional_generation.hf_fc694b5_orig__sm_103__unified.ops.attention import (  # noqa: E501
    qsa_paged_sparse_gqa_nvfp4__r38 as _nvfp4_module,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp_for_conditional_generation.hf_fc694b5_orig__sm_103__unified.ops.attention.qsa_paged_sparse_gqa_nvfp4_decode__r39 import (  # noqa: E501
    qsa_paged_sparse_gqa_nvfp4_decode as op,
)
from tensorrt_llm._torch.attention.backends.sparse.qsa.kernels import triton_qsa_paged_sparse_gqa

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
):
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
    pages = args["k_cache"].shape[0]
    all_pages = torch.arange(pages, device=DEV)[:, None].expand(-1, args["tokens_per_block"])
    slots = torch.arange(args["tokens_per_block"], device=DEV)[None, :].expand(pages, -1)
    planes = []
    for data, scales, index, grouped in (
        (args["k_cache"], args["k_scale_cache"], 1, False),
        (args["v_cache"], args["v_scale_cache"], 2, True),
    ):
        values = _nvfp4_module._dequantize_tokens(
            data, scales, all_pages, slots, args["kv_scale_quant_orig"][index], grouped
        )
        planes.append(values.to(args["q"].dtype).transpose(1, 2).contiguous())
    twin = {
        n: args[n]
        for n in ("q", "block_table", "selected_tokens", "request_indices", "tokens_per_block")
    }
    twin.update(softmax_scale=args["softmax_scale"], k_cache=planes[0], v_cache=planes[1])
    return triton_qsa_paged_sparse_gqa(**twin)


def _args(spec, seed):
    kw = {k: v for k, v in spec.items() if k not in ("segments", "gated")}
    args = build(spec["segments"], seed=seed, **kw)
    if spec.get("gated"):
        rows = args["q"].shape[0]
        g = torch.Generator(device=DEV).manual_seed(seed + 1)
        args["output_gate"] = torch.randn(rows, 24 * 256, generator=g, device=DEV).to(
            torch.bfloat16
        )
    return args


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell) -> None:
    args = _args(cell.spec, zlib.crc32(cell.why.encode()) % (2**31))
    out = op(**args)
    assert out.dtype == torch.bfloat16 and not out.isnan().any()
    el, rms, ok = within_ulp(out, op.reference(**args))
    assert ok, f"vs reference: {el:.2f} ulp element, {rms:.2f} ulp RMS"
    if "output_gate" not in args:
        el, rms, ok = within_ulp(out, bf16_twin(args))
        assert ok, f"vs QSA bf16 kernel: {el:.2f} ulp element, {rms:.2f} ulp RMS"


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
