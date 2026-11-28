# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for csa2_sparse_context_o_a_input (the engine's FP8 sparse context kernel).

Against fp32 attention over the dequantized bank with the sink, rotated back, with the scales as `csa2_shared_stage`
makes them (a power-of-two staging scale shared by Q and KV). Q, K and P are E4M3 in the kernel and the output is
rounded to E4M3 under a per-row, per-128-channel scale: gate 7% relative RMS of the dequantized output.
"""

import pytest
import torch
from _v41_util import DEV, gen_for

__extra_import_path__ = ["../../../.."]
from _validating import certifying, validating  # noqa: E402,F401

from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.attention.csa2_rope import (
    csa2_rope_table,
)
from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.attention.csa2_sparse_context import (  # noqa: E501
    csa2_sparse_context_o_a_input,
    o_a_input_dequant,
)
from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.attention.csa2_store_rows import (
    csa2_stock_rope_table,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3),
    reason="SM103 kernel",
)
_LOG2E = 1.4426950408889634


def _rel(a, b):
    return (
        (a.float() - b.float()).square().mean().sqrt() / b.float().square().mean().sqrt()
    ).item()


def _inputs(s, gen):
    n, rows, keys = s["n"], s["rows"], s["keys"]
    q = (torch.randn(n, 64, 512, generator=gen, device=DEV) * 0.5).bfloat16()
    kv_dequant = torch.tensor(
        [2.0**-3], device=DEV
    )  # stock's context scale: Q and KV share one power of two
    bank = ((torch.randn(rows, 512, generator=gen, device=DEV) * 2).clamp(-50, 50) / kv_dequant).to(
        torch.float8_e4m3fn
    )
    bank[0] = 0
    idx = torch.randint(1, rows, (n, keys), generator=gen, device=DEV, dtype=torch.int32)
    lens = torch.randint(1, keys + 1, (n,), generator=gen, device=DEV, dtype=torch.int32)
    idx = torch.where(torch.arange(keys, device=DEV)[None, :] < lens[:, None], idx, -1).int()
    sink = torch.randn(64, generator=gen, device=DEV)
    bmm1 = kv_dequant * kv_dequant * 512**-0.5
    bmm1 = torch.cat([bmm1, bmm1 * _LOG2E])
    return q, bank, idx, lens, 1.0 / kv_dequant, bmm1, kv_dequant.clone(), sink


@pytest.mark.parametrize(
    "cell",
    csa2_sparse_context_o_a_input.CELLS,
    ids=[c.why[:44] for c in csa2_sparse_context_o_a_input.CELLS],
)
def test_o_a_input_cells(cell, certifying):  # noqa: F811
    certifying(csa2_sparse_context_o_a_input)
    gen = gen_for(cell)
    args = _inputs(cell.spec, gen)
    n = cell.spec["n"]
    pos = torch.randint(0, 4000, (n,), generator=gen, device=DEV, dtype=torch.int32)
    cos_sin = csa2_stock_rope_table(csa2_rope_table(64, 4096, 0, 10000.0, 1.0, 32, 1, DEV))
    full = (*args, pos, cos_sin, cell.spec["groups"])
    with validating(csa2_sparse_context_o_a_input):
        data, scale = csa2_sparse_context_o_a_input(*full)
    g = cell.spec["groups"]
    assert data.shape == (g, n, 64 // g * 512) and data.dtype == torch.float8_e4m3fn
    assert scale.shape == (g, 64 // g * 4, (n + 3) // 4 * 4)
    got = o_a_input_dequant(data, scale)
    ref = csa2_sparse_context_o_a_input.reference(*full)
    print(f"csa2_sparse_context_o_a_input rel RMS {cell.why}: {_rel(got, ref):.4f}")
    assert _rel(got, ref) < 0.07, _rel(got, ref)


def test_o_a_input_e4m3_query():
    """An E4M3 query at unit scale (stock's fused context query, `q_scale` None) gives what the kernel computes from the
    bf16 query it quantizes on load at unit scale: the same E4M3 bytes, so the same output."""
    cell = csa2_sparse_context_o_a_input.CELLS[0]
    gen = gen_for(cell)
    q, bank, idx, lens, _, _, bmm2, sink = _inputs(cell.spec, gen)
    n = cell.spec["n"]
    bmm1 = bmm2 * 512**-0.5  # unit query scale: BMM1 carries the KV dequant scale alone
    bmm1 = torch.cat([bmm1, bmm1 * _LOG2E])
    pos = torch.randint(0, 4000, (n,), generator=gen, device=DEV, dtype=torch.int32)
    cos_sin = csa2_stock_rope_table(csa2_rope_table(64, 4096, 0, 10000.0, 1.0, 32, 1, DEV))
    g = cell.spec["groups"]
    unit = torch.ones(1, device=DEV)
    with validating(csa2_sparse_context_o_a_input):
        d8, s8 = csa2_sparse_context_o_a_input(
            q.to(torch.float8_e4m3fn), bank, idx, lens, None, bmm1, bmm2, sink, pos, cos_sin, g
        )
    db, sb = csa2_sparse_context_o_a_input(
        q, bank, idx, lens, unit, bmm1, bmm2, sink, pos, cos_sin, g
    )
    rel = _rel(o_a_input_dequant(d8, s8), o_a_input_dequant(db, sb))
    print(f"e4m3 query vs bf16 query at unit scale: rel RMS {rel:.2e}")
    assert rel < 1e-3, rel
    ref = csa2_sparse_context_o_a_input.reference(
        q.to(torch.float8_e4m3fn), bank, idx, lens, None, bmm1, bmm2, sink, pos, cos_sin, g
    )
    assert _rel(o_a_input_dequant(d8, s8), ref) < 0.07


def _o_a_args(cell, gen):
    """The cell's inputs with positions and the engine's rope table, as `csa2_sparse_context_o_a_input` takes them."""
    n = cell.spec["n"]
    args = _inputs(cell.spec, gen)
    pos = torch.randint(0, 4000, (n,), generator=gen, device=DEV, dtype=torch.int32)
    cos_sin = csa2_stock_rope_table(csa2_rope_table(64, 4096, 0, 10000.0, 1.0, 32, 1, DEV))
    return (*args, pos, cos_sin, cell.spec["groups"])


def test_reference_ignores_spare_metadata_rows():
    """`indices` / `lens` may hold more rows than the live queries; the reference attends with the live ones only."""
    cell = csa2_sparse_context_o_a_input.CELLS[0]
    args = list(_o_a_args(cell, gen_for(cell)))
    idx, lens = args[2], args[3]
    want = csa2_sparse_context_o_a_input.reference(*args)
    args[2] = torch.cat([idx, idx[:1].flip(1).abs()]).contiguous()
    args[3] = torch.cat([lens, lens[:1]])
    got = csa2_sparse_context_o_a_input.reference(*args)
    assert got.shape == want.shape and torch.equal(got, want)


def test_rejections():
    cell = csa2_sparse_context_o_a_input.CELLS[0]
    args = list(_o_a_args(cell, gen_for(cell)))
    csa2_sparse_context_o_a_input.is_valid(*args)
    args[1] = args[1].view(torch.uint8).float()
    with pytest.raises(AssertionError):
        csa2_sparse_context_o_a_input.is_valid(*args)
