# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the Kimi K3.1 gather_k_cache_paged entry: every cell's gathered bytes exactly, against
the reference."""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Cell
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.attention.gather_k_cache_paged import (  # noqa: E501
    k31_gather_k_cache_paged as op,
)

__extra_import_path__ = ["../../../.."]
from _cells import certified_cells  # noqa: E402 — needs the path declared above
from _validating import validating  # noqa: E402 — needs the path declared above

assert torch.cuda.is_available(), "gather_k_cache_paged requires a CUDA device"

TARGET_CELLS: dict[str, tuple[Cell, ...]] = {}
CELLS = certified_cells(op, TARGET_CELLS)


def _build(spec, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    tokens, start, tpb, head_dim = (
        spec["tokens"],
        spec["start"],
        spec["tokens_per_block"],
        spec["head_dim"],
    )
    per_token = head_dim + 4
    page_bytes = tpb * per_token
    pages = (start + tokens) // tpb + 2
    pool = torch.randint(
        0, 256, (pages, page_bytes + 1024), generator=g, device="cuda", dtype=torch.uint8
    )
    cache = pool[:, :page_bytes].view(pages, tpb, 1, per_token)
    flat = torch.randperm(pages * tpb, generator=g, device="cuda")[: start + tokens]
    page, slot = flat // tpb, flat % tpb
    fp8 = (page * page_bytes + slot * head_dim).to(torch.int64)
    scale = (page * page_bytes + tpb * head_dim + slot * 4).to(torch.int64)
    if spec.get("holes"):
        fp8[start::5] = -1
    if spec.get("misaligned"):
        # Any byte offset inside a page that keeps the key (or the scale) inside it.
        n = fp8.shape[0]
        fp8 = page * page_bytes + torch.randint(
            0, page_bytes - head_dim + 1, (n,), generator=g, device="cuda"
        )
        scale = page * page_bytes + torch.randint(
            0, page_bytes - 4 + 1, (n,), generator=g, device="cuda"
        )
    return cache, fp8, scale, start, tokens, head_dim


@pytest.mark.parametrize("cell", CELLS, ids=[c.why[:44] for c in CELLS])
def test_certified_cells(cell) -> None:
    args = _build(cell.spec, seed=zlib.crc32(cell.why.encode()) % 2**31)
    with validating(op):
        k, s = op(*args)
    ref_k, ref_s = op.reference(*args)
    _, fp8, _, start, tokens, _ = args
    ok = fp8[start : start + tokens] >= 0
    assert torch.equal(k.view(torch.uint8)[ok], ref_k.view(torch.uint8)[ok])
    assert torch.equal(s.view(torch.int32)[ok], ref_s.view(torch.int32)[ok])


def test_guard_refuses_a_range_past_the_mapping() -> None:
    cache, fp8, scale, _, _, head_dim = _build(
        dict(tokens=16, start=0, tokens_per_block=64, head_dim=128), seed=1
    )
    with pytest.raises(AssertionError, match="inside both slot mappings"):
        with validating(op):
            op(cache, fp8, scale, 8, 16, head_dim)
