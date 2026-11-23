# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the HY4-Flash target's convert_req_index_to_global entry."""

import math
import zlib

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.hyv4_flash.hy4_flash_fp8__sm_103__default.ops.attention.convert_req_index_to_global import (  # noqa: E501
    convert_req_index_to_global as op,
)
from tensorrt_llm._utils import get_sm_version

__extra_import_path__ = ["../../../../.."]
from _validating import certifying  # noqa: E402,F401 — path above; fixture by name

assert torch.cuda.is_available(), "convert_req_index_to_global requires a CUDA device"

pytestmark = pytest.mark.skipif(
    get_sm_version() != 103, reason="the hyv4_flash target's entries are certified on sm_103"
)


def _build(spec, seed):
    """`(req_id, block_table, token_indices, block_size, topk, stride_factor, layer_id)`.

    Row `j` of request `s` holds up to `topk` distinct positions from `[0, kv_lens[s] - seq_rows[s] + j + 1)`,
    unordered, padded with -1. Pages are a random permutation; unused table entries are -1.
    """
    g = torch.Generator().manual_seed(seed)
    bs, topk = spec["block_size"], spec["topk"]
    pages_needed = [math.ceil(n / bs) for n in spec["kv_lens"]]
    width = max(pages_needed)
    pad = spec.get("pad_block_table", False)
    if pad:
        # Narrower than the longest request needs, plus one unused in-range page.
        width = max(2, width // 2)
    num_pages = 2 * sum(pages_needed) + 2
    pool = torch.randperm(num_pages, generator=g).to(torch.int32)
    table = torch.full((len(spec["kv_lens"]), width), -1, dtype=torch.int32)
    used = 0
    for s, need in enumerate(pages_needed):
        n = min(need, width)
        table[s, :n] = pool[used : used + n]
        used += n
    if pad:
        table[0, 1] = -1

    req, rows = [], []
    for s, (r_count, kv) in enumerate(zip(spec["seq_rows"], spec["kv_lens"])):
        for j in range(r_count):
            end = kv - r_count + j + 1
            row = torch.full((topk,), -1, dtype=torch.int32)
            pick = torch.randperm(end, generator=g)[:topk].to(torch.int32)
            row[: pick.numel()] = pick
            rows.append(row)
            req.append(s)
    return (
        torch.tensor(req, dtype=torch.int32, device="cuda"),
        table.cuda(),
        torch.stack(rows).cuda(),
        bs,
        topk,
        spec["pool_layers"] * bs,
        spec["layer_id"],
    )


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    certifying(op)
    args = _build(cell.spec, seed=zlib.crc32(cell.why.encode()) % 2**31)
    snap = [a.clone() for a in args[:3]]
    out = op(*args)
    for a, s in zip(args[:3], snap):
        assert torch.equal(a, s), "an input tensor was mutated"
    ref = op.reference(*args)
    # Exact: an off-by-one row is a different token's KV.
    assert out.dtype == ref.dtype == torch.int32 and out.shape == ref.shape
    assert torch.equal(out, ref), f"{int((out != ref).sum())} of {out.numel()} pool rows differ"
    if cell.spec.get("pad_block_table"):
        assert bool((out[args[2] >= 0] == -1).any()), "the cell drove no unresolvable position"
