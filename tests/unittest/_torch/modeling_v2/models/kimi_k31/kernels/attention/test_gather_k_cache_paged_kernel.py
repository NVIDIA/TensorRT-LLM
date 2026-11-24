# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the family's indexer K-cache gather (CuTe DSL) against the engine's
indexer_k_cache_gather_op and the op's torch reference."""

import pytest
import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kernels.attention import (
    gather_k_cache_paged as kernel,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.attention.gather_k_cache_paged import (  # noqa: E501
    k31_gather_k_cache_paged as op,
)

assert torch.cuda.is_available(), "the K-cache gather requires a CUDA device"


def _cache(pages: int, tpb: int, head_dim: int, extra_page_bytes: int, seed: int):
    """A ``[pages, tpb, 1, head_dim + 4]`` uint8 view inside a pool with ``extra_page_bytes``
    of other layers' data after each page, and the packed-page byte offsets of each token."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    per_token = head_dim + 4
    page_bytes = tpb * per_token
    pool = torch.randint(
        0,
        256,
        (pages, page_bytes + extra_page_bytes),
        generator=g,
        device="cuda",
        dtype=torch.uint8,
    )
    view = pool[:, :page_bytes].view(pages, tpb, 1, per_token)
    return view, page_bytes, per_token


def _slots(
    num_tokens: int,
    pages: int,
    tpb: int,
    per_token: int,
    head_dim: int,
    seed: int,
    holes: bool = False,
):
    """Token t's key bytes at packed offset page * page_bytes + slot * head_dim, its scale in the
    scale region after the page's keys (the engine's per-page [keys | scales] layout)."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    page_bytes = tpb * per_token
    flat = torch.randperm(pages * tpb, generator=g, device="cuda")[:num_tokens]
    page, slot = flat // tpb, flat % tpb
    fp8 = page * page_bytes + slot * head_dim
    scale = page * page_bytes + tpb * head_dim + slot * 4
    if holes:
        fp8[::7] = -1
    return fp8.to(torch.int64), scale.to(torch.int64)


@pytest.mark.parametrize(
    "tokens,start,head_dim,holes",
    (
        (31, 3, 128, False),
        (64, 0, 64, False),
        (64, 0, 256, False),
        (65536, 7, 128, False),
    ),
)
def test_bytes_equal_engine_op(tokens, start, head_dim, holes) -> None:
    tpb, pages = 64, (start + tokens) // 64 + 4
    cache, _, per_token = _cache(pages, tpb, head_dim, extra_page_bytes=4096, seed=tokens)
    fp8, scale = _slots(
        start + tokens, pages, tpb, per_token, head_dim, seed=tokens + 1, holes=holes
    )
    k, s = kernel.k31_gather_k_cache_paged(cache, fp8, scale, start, tokens, head_dim)
    tk, ts = op.reference(cache, fp8, scale, start, tokens, head_dim)
    ok = fp8[start : start + tokens] >= 0
    assert torch.equal(k.view(torch.uint8)[ok], tk.view(torch.uint8)[ok])
    assert torch.equal(s.view(torch.int32)[ok], ts.view(torch.int32)[ok])
    if head_dim != 128:
        return  # the engine op gathers the 128-byte FP8 layout only; the bytes are checked below
    ek, es = torch.ops.trtllm.indexer_k_cache_gather_op(cache, fp8, scale, start, tokens, head_dim)
    ok = fp8[start : start + tokens] >= 0
    assert torch.equal(k.view(torch.uint8)[ok], ek.view(torch.uint8)[ok])
    assert torch.equal(s.view(torch.int32)[ok], es.view(torch.int32)[ok])


@pytest.mark.parametrize("head_dim", (64, 256))
def test_other_widths_gather_the_right_bytes(head_dim) -> None:
    tpb, pages, tokens = 64, 6, 200
    cache, page_bytes, per_token = _cache(pages, tpb, head_dim, extra_page_bytes=512, seed=9)
    fp8, scale = _slots(tokens, pages, tpb, per_token, head_dim, seed=10)
    k, s = kernel.k31_gather_k_cache_paged(cache, fp8, scale, 0, tokens, head_dim)
    flat = cache.as_strided((pages, page_bytes), (cache.stride(0), 1))
    for t in (0, 77, tokens - 1):
        f, fs = int(fp8[t]), int(scale[t])
        assert torch.equal(
            k.view(torch.uint8)[t], flat[f // page_bytes, f % page_bytes :][:head_dim]
        )
        assert torch.equal(
            s.view(torch.uint8).view(tokens, 4)[t], flat[fs // page_bytes, fs % page_bytes :][:4]
        )


@pytest.mark.parametrize("tokens", (100, 40000))
@pytest.mark.parametrize("use_pdl", (False, True))
@pytest.mark.parametrize("trigger", (False, True))
def test_launch_variants_agree(tokens, use_pdl, trigger) -> None:
    tpb, head_dim = 64, 128
    pages = tokens // tpb + 4
    cache, _, per_token = _cache(pages, tpb, head_dim, extra_page_bytes=256, seed=11)
    fp8, scale = _slots(tokens, pages, tpb, per_token, head_dim, seed=12)
    out_k = torch.empty(tokens, head_dim, dtype=torch.uint8, device="cuda")
    out_s = torch.empty(tokens, 4, dtype=torch.uint8, device="cuda")
    kernel.gather_k_cache_paged_into(
        cache, fp8, scale, 0, out_k, out_s, use_pdl=use_pdl, trigger=trigger
    )
    tk, ts = op.reference(cache, fp8, scale, 0, tokens, head_dim)
    assert torch.equal(out_k, tk.view(torch.uint8))
    assert torch.equal(out_s.view(torch.int32).view(-1), ts.view(torch.int32).view(-1))


def test_empty_chunk() -> None:
    cache = torch.zeros(4, 64, 1, 132, dtype=torch.uint8, device="cuda")
    slots = torch.zeros(8, dtype=torch.int64, device="cuda")
    k, s = kernel.k31_gather_k_cache_paged(cache, slots, slots, 0, 0, 128)
    assert k.shape == (0, 128) and s.shape == (0, 1)


def test_rejects_other_layouts() -> None:
    cache = torch.zeros(4, 64, 2, 132, dtype=torch.uint8, device="cuda")  # two heads
    slots = torch.zeros(8, dtype=torch.int64, device="cuda")
    with pytest.raises(ValueError):
        kernel.k31_gather_k_cache_paged(cache, slots, slots, 0, 8, 128)
    assert not kernel.supports_layout(cache, 128)


@pytest.mark.parametrize("bad", ("int32", "strided"))
def test_rejects_other_slot_mappings(bad) -> None:
    cache = torch.zeros(4, 64, 1, 132, dtype=torch.uint8, device="cuda")
    slots = torch.zeros(16, dtype=torch.int64, device="cuda")
    bad_slots = slots.int() if bad == "int32" else slots[::2]
    with pytest.raises(ValueError):
        kernel.k31_gather_k_cache_paged(cache, bad_slots, slots[:8], 0, 8, 128)
