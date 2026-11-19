# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU tests for the inkling2_paged_attention entry."""

import zlib

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.inkling2.kernels.paged_attention import (
    block_tokens,
    tile_plan,
)
from tensorrt_llm._torch.modeling_v2.models.inkling2.ops.attention.inkling2_paged_attention import (
    SM_SCALE,
)
from tensorrt_llm._torch.modeling_v2.models.inkling2.ops.attention.inkling2_paged_attention import (
    inkling2_paged_attention as op,
)

__extra_import_path__ = ["../../../.."]
from _validating import assert_within_ulp, certifying, validating  # noqa: E402,F401

assert torch.cuda.is_available(), "inkling2_paged_attention requires a CUDA device"

PAGE_SIZE = 32
HEAD_DIM = 128
REL_EXTENT = 1024


def _build(spec, seed):
    """A step of `spec["requests"]` ((new, cached) pairs) over a pool of shuffled pages whose
    cached positions already hold keys and values."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    cpu = torch.Generator().manual_seed(seed)
    heads, kv_heads, window = spec["heads"], spec["kv_heads"], spec["window"]
    requests = spec["requests"]
    lengths = [n for n, _ in requests]
    kv_lens = [n + c for n, c in requests]
    widths = [-(-length // PAGE_SIZE) for length in kv_lens]
    num_pages = sum(widths) + 3
    pool = torch.randn(num_pages, 2, kv_heads, PAGE_SIZE, HEAD_DIM, generator=g, device="cuda")
    # Keys after the clip have norm <= 22.6; values after the RMS norm have unit RMS.
    pool[:, 0] *= 1.5
    pool = pool.to(torch.bfloat16)
    order = torch.randperm(num_pages, generator=cpu)
    page_table = torch.full((len(requests), max(widths)), -1, dtype=torch.int32)
    used = 0
    for r, width in enumerate(widths):
        page_table[r, :width] = order[used : used + width].to(torch.int32)
        used += width
    num_tokens = sum(lengths)
    # Clipped queries reach norm 22.6 and the global layers' tau scales them further.
    q = (1.5 * torch.randn(num_tokens, heads, HEAD_DIM, generator=g, device="cuda")).to(
        torch.bfloat16
    )
    k = (1.5 * torch.randn(num_tokens, kv_heads, HEAD_DIM, generator=g, device="cuda")).to(
        torch.bfloat16
    )
    v = torch.randn(num_tokens, kv_heads, HEAD_DIM, generator=g, device="cuda").to(torch.bfloat16)
    rel_logits = 2.0 * torch.randn(num_tokens, heads, REL_EXTENT, generator=g, device="cuda")
    sink = (2.0 * torch.randn(num_tokens, heads, generator=g, device="cuda")).to(torch.bfloat16)
    gate = (2.0 * torch.randn(num_tokens, heads, HEAD_DIM, generator=g, device="cuda")).to(
        torch.bfloat16
    )
    bounds = [0]
    for n in lengths:
        bounds.append(bounds[-1] + n)
    block_t = block_tokens(max(lengths))
    tile_request, tile_start = tile_plan(lengths, block_t, torch.device("cuda"))
    return [
        q,
        k,
        v,
        pool,
        page_table.cuda(),
        torch.tensor(bounds, dtype=torch.int32, device="cuda"),
        torch.tensor(kv_lens, dtype=torch.int32, device="cuda"),
        torch.repeat_interleave(
            torch.arange(len(lengths), dtype=torch.int32), torch.tensor(lengths)
        ).cuda(),
        tile_request,
        tile_start,
        block_t,
        rel_logits,
        sink,
        gate,
        window,
        SM_SCALE,
    ]


def _run_both(args):
    op_args, ref_args = list(args), list(args)
    op_args[3] = args[3].clone()
    ref_args[3] = args[3].clone()
    out = op(*op_args)
    ref = op.reference(*ref_args)
    return out, ref, op_args[3], ref_args[3]


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:48] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811
    certifying(op)
    out, ref, pool, ref_pool = _run_both(_build(cell.spec, zlib.crc32(cell.why.encode()) % 2**31))
    torch.testing.assert_close(pool, ref_pool, rtol=0, atol=0)
    # Flash attention rounds the unnormalized probabilities to bf16 for the value product and
    # sums in another order than the dense fp32 reference: a few bf16 ulps of each row's scale.
    assert_within_ulp(
        out.reshape(-1, out.shape[-1]), ref.reshape(-1, ref.shape[-1]), element_ulp=4.0, rms_ulp=0.5
    )


@pytest.mark.parametrize("what", ["bias_extent", "window", "sink"])
def test_negative_controls_are_visible(what) -> None:
    """Each mechanism the cells certify changes the output when it is broken."""
    spec = dict(
        heads=12,
        kv_heads=2,
        window=1024 if what == "window" else 0,
        requests=((1, 1500), (40, 990)),
    )
    args = _build(spec, 21)
    if what == "sink":
        # Sinks near the logsumexp of 1500 keys, where they take a visible share of the mass.
        args[12] = (args[12].float() + 10.0).to(torch.bfloat16)
    out, *_ = _run_both(args)
    broken = list(args)
    if what == "bias_extent":
        # Bias leaking past the extent: shift it by one distance.
        broken[11] = torch.roll(args[11], shifts=1, dims=2)
    elif what == "window":
        broken[14] = 1023
    else:
        broken[12] = torch.zeros_like(args[12])
    broken[3] = args[3].clone()
    assert not torch.allclose(op(*broken), out, rtol=0.01, atol=0.01)


@pytest.mark.parametrize(
    "case,match",
    [
        ("bf16_rel_logits", None),
        ("short_page_table", "page_table must cover"),
        ("wrong_tiles", "tile plan"),
        ("strided_kv", "contiguous head dims"),
    ],
)
def test_guard_refuses_inputs_the_op_answers_wrongly(case, match) -> None:
    args = _build(dict(heads=12, kv_heads=2, window=1024, requests=((5, 3), (1, 40))), 4)
    if case == "bf16_rel_logits":
        args[11] = args[11].to(torch.bfloat16)
    elif case == "short_page_table":
        args[4] = args[4][:, :1]
    elif case == "strided_kv":
        # The same values every other element of a wider buffer: the kernels read unit stride.
        for i in (1, 2):
            t = args[i]
            wide = torch.zeros(*t.shape[:2], 2 * t.shape[2], dtype=t.dtype, device=t.device)
            wide[..., ::2] = t
            args[i] = wide[..., ::2]
    else:
        args[9] = torch.zeros_like(args[9])
    with pytest.raises(AssertionError, match=match):
        with validating(op):
            op(*args)
