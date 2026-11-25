# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the GLM-5.2 target's cute_dsl_mha_kv_gather_mxf8 entry.

The pool is synthetic: random bytes (NaN encodings included) in `[blocks * layers *
32, 576]` rows, a shuffled block table and request slots, the new rows the tail of
2624-wide projection rows. Both operands and both scale buffers, padding bytes included,
are compared bit for bit against the torch twin.
"""

import zlib

import pytest
import torch

if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3):
    pytest.skip("the vendored CuTe DSL kernel needs an sm_103 GPU", allow_module_level=True)

from tensorrt_llm._torch.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tep4.ops.attention import (  # noqa: E402, E501
    cute_dsl_mha_kv_gather_mxf8 as entry,
)

__extra_import_path__ = ["../../../.."]
from _validating import validating  # noqa: E402 — needs the path declared above

op = entry.cute_dsl_mha_kv_gather_mxf8
LAT = entry.LATENT_DIM
PAGE, LAYERS, LAYER, ROW = 32, 3, 1, 2624


def _build(spec, seed):
    """The op's arguments for one context step (lat is the tail of 2624-wide rows)."""
    cpu = torch.Generator().manual_seed(seed)
    prefixes, news = spec["prefixes"], spec["news"]
    reqs = len(prefixes)
    per_req = [-(-(c + n) // PAGE) + 1 for c, n in zip(prefixes, news)]
    num_blocks = sum(per_req) + 5
    perm = torch.randperm(num_blocks, generator=cpu)
    table = torch.full((reqs + 2, max(per_req) + 2), -1, dtype=torch.int32)
    slots = torch.randperm(reqs + 2, generator=cpu)[:reqs]
    used = 0
    for r, n in enumerate(per_req):
        table[slots[r], :n] = perm[used : used + n].to(torch.int32)
        used += n
    stride_factor = LAYERS * PAGE
    pool = torch.randint(
        0, 256, (num_blocks * stride_factor, LAT), generator=cpu, dtype=torch.uint8
    )
    req_idx = torch.repeat_interleave(slots.to(torch.int32), torch.tensor(news))
    cu_k = [0]
    for c, n in zip(prefixes, news):
        cu_k.append(cu_k[-1] + c + n)
    dst_c, tok_c, pos_c, dst_n, tok0 = [], [], [], [], 0
    for r, (c, n) in enumerate(zip(prefixes, news)):
        dst_c += [cu_k[r] + p for p in range(c)]
        tok_c += [tok0] * c
        pos_c += list(range(c))
        dst_n += [cu_k[r] + c + i for i in range(n)]
        tok0 += n
    rows = torch.randn(sum(news), ROW, generator=cpu)
    # Latent-like magnitudes: c_kv blocks from tiny (below e4m3's normal range at 2^0) to O(10).
    block_exp = torch.randint(-12, 4, (sum(news), ROW // 32), generator=cpu)
    rows = rows * torch.exp2(block_exp.float()).repeat_interleave(32, dim=1)
    rows = rows.to(torch.bfloat16).cuda()
    a8 = tuple(t.fill_(0xA5) for t in entry.buffers(cu_k[-1], "cuda"))
    return dict(
        pool_rows=pool.cuda(),
        idx=torch.tensor(dst_c + tok_c + pos_c + dst_n, dtype=torch.int64).cuda(),
        num_cached=sum(prefixes),
        req_idx=req_idx.cuda(),
        block_table=table.cuda(),
        latent_new=rows[:, ROW - LAT :],
        a8=a8,
        layer_offset=LAYER * PAGE,
        stride_factor=stride_factor,
        tokens_per_block=PAGE,
        new_dst_from_idx=bool(sum(prefixes)),
    )


@pytest.fixture(scope="module")
def compiled():
    """The handle the target builds at load."""
    return entry.compile(torch.cuda.current_device())


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, compiled) -> None:
    """Exact on every byte: a cached row is a copy, a new row the activation quant's rule
    (power-of-two scales, one e4m3 rounding) in the same fp32 operations, and every scale
    byte the GEMMs read is written, padding included."""
    args = _build(cell.spec, zlib.crc32(cell.why.encode()) % 2**31)
    ref = op.reference(**args, compiled=compiled)
    with validating(op):
        got = op(**args, compiled=compiled)
    torch.cuda.synchronize()
    assert got is args["a8"]
    for name, g, r in zip(("a_k", "a_v", "sfa_k", "sfa_v"), got, ref):
        bad = int((g != r).sum())
        assert bad == 0, f"{name}: {bad} bytes differ from the reference"


def test_is_valid_refuses_an_unallocated_prefix_block(compiled) -> None:
    """A -1 block would read before the pool's start."""
    args = _build(dict(prefixes=[40], news=[8]), seed=3)
    assert args["new_dst_from_idx"]
    table = args["block_table"].clone()
    table[table >= 0] = -1
    args["block_table"] = table
    with pytest.raises(AssertionError, match="unallocated"):
        with validating(op):
            op(**args, compiled=compiled)
