# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the GLM-5.2 target's indexer_k_cache_gather_op entry."""

import math
import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tep4.ops.attention.indexer_k_cache_gather_op import (  # noqa: E501
    indexer_k_cache_gather_op as op,
)

__extra_import_path__ = ["../../../.."]
from _validating import validating  # noqa: E402 — needs the path declared above

assert torch.cuda.is_available(), "indexer_k_cache_gather_op requires a CUDA device"


def _compare(out: torch.Tensor, ref: torch.Tensor) -> None:
    """Bit-exact: a gather moves bytes and computes nothing. Call once for
    the data and once for the scale."""
    assert out.dtype == ref.dtype, (out.dtype, ref.dtype)
    assert out.shape == ref.shape, (out.shape, ref.shape)
    assert torch.equal(out.contiguous().view(torch.uint8), ref.contiguous().view(torch.uint8)), (
        "gathered bytes differ from the reference"
    )


def _slot_mappings(table, seq_lens, head_dim, tpb):
    """Flat byte offsets of every token's data and scale, in sequence order.

    The same arithmetic the DSA indexer uses: a page is `tpb * (head_dim + 4)`
    bytes, all data first, then all scales.
    """
    page_bytes = tpb * (head_dim + 4)
    fp8, scale = [], []
    for s, n in enumerate(seq_lens):
        pos = torch.arange(n)
        page = table[s, pos // tpb].long()
        fp8.append(page * page_bytes + (pos % tpb) * head_dim)
        scale.append(page * page_bytes + tpb * head_dim + (pos % tpb) * 4)
    return torch.cat(fp8).cuda(), torch.cat(scale).cuda()


def _build(spec, seed):
    """`(k_cache, slot_mapping_fp8, slot_mapping_scale, k_token_start, num_tokens, head_dim)`.

    With `pool_layers > 1` the cache is one layer's view of a pool that
    interleaves layers page by page, so dim 0 is not contiguous -- the layout
    the K cache manager hands out.
    """
    g = torch.Generator().manual_seed(seed)
    hd, tpb = spec["head_dim"], spec["tokens_per_block"]
    pages = [math.ceil(n / tpb) for n in spec["seq_lens"]]
    num_pages = 2 * sum(pages) + 1
    pool = torch.randperm(num_pages, generator=g).to(torch.int32)
    table = torch.zeros(len(pages), max(pages), dtype=torch.int32)
    used = 0
    for s, n in enumerate(pages):
        table[s, :n] = pool[used : used + n]
        used += n
    layers = spec["pool_layers"]
    parent = torch.randint(
        0, 256, (num_pages, layers, tpb * (hd + 4)), generator=g, dtype=torch.uint8
    ).cuda()
    k_cache = parent[:, layers // 2].view(num_pages, tpb, 1, hd + 4)
    assert layers == 1 or not k_cache.is_contiguous()
    fp8, scale = _slot_mappings(table, spec["seq_lens"], hd, tpb)
    return k_cache, fp8, scale, spec["k_token_start"], spec["num_tokens"], hd


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell) -> None:
    args = _build(cell.spec, seed=zlib.crc32(cell.why.encode()) % 2**31)
    cache_before = args[0].clone()
    with validating(op):
        data, scale = op(*args)
    assert torch.equal(args[0], cache_before), "the gather modified the cache"
    assert data.dtype == torch.float8_e4m3fn and scale.dtype == torch.float32
    ref_data, ref_scale = op.reference(*args)
    _compare(data, ref_data)
    _compare(scale, ref_scale)


def test_zero_tokens_returns_empty() -> None:
    k_cache, fp8, scale, _, _, hd = _build(
        dict(
            head_dim=128,
            tokens_per_block=32,
            seq_lens=[8],
            k_token_start=0,
            num_tokens=8,
            pool_layers=1,
        ),
        seed=3,
    )
    data, sc = op(k_cache, fp8, scale, 0, 0, hd)
    assert tuple(data.shape) == (0, hd) and tuple(sc.shape) == (0, 1)


def _ok():
    return _build(
        dict(
            head_dim=128,
            tokens_per_block=32,
            seq_lens=[64],
            k_token_start=0,
            num_tokens=64,
            pool_layers=1,
        ),
        seed=5,
    )


def test_guard_refuses_a_negative_start() -> None:
    """The op checks only the upper end of the window."""
    k_cache, fp8, scale, _, _, hd = _ok()
    with pytest.raises(AssertionError, match="k_token_start must be >= 0"):
        with validating(op):
            op(k_cache, fp8, scale, -8, 8, hd)


def test_guard_refuses_a_negative_slot() -> None:
    """The kernel skips the row and leaves torch.empty's contents in it."""
    k_cache, fp8, scale, _, _, hd = _ok()
    fp8 = fp8.clone()
    fp8[3] = -1
    with pytest.raises(AssertionError, match="must be >= 0"):
        with validating(op):
            op(k_cache, fp8, scale, 0, 64, hd)
