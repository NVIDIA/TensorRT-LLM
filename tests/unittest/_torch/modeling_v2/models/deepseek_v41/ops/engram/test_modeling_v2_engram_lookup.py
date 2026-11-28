# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test for the engram_lookup entry: the engine's FP8 gather kernel against the float64 dequantizing reference.

The kernel multiplies an E4M3 value by a power of two (exact in fp32) and rounds once to bf16, as the reference
does, so the two must agree bit for bit. MXFP4 tables (every byte value, so every E2M1 code pair) go through the same
check: the kernel widens E2M1 to E4M3 exactly.
"""

import pytest
import torch
from _v41_util import DEV, gen_for

__extra_import_path__ = ["../../../.."]
from _validating import certifying, validating  # noqa: E402,F401

from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.engram.engram_lookup import (
    engram_lookup,
)

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10),
    reason="the engram lookup is a Blackwell (sm_10x) kernel",
)

DIM, BLOCK = 128, 32


def _inputs(spec, cell):
    gen = gen_for(cell)
    rows, heads, local = spec["rows"], spec["heads"], spec["local_heads"]
    if spec.get("mxfp4"):  # E2M1 pairs: any byte is a valid pair of codes
        weight = torch.randint(
            0, 256, (rows, DIM // 2), generator=gen, device=DEV, dtype=torch.uint8
        )
    else:
        values = torch.randn(rows, DIM, generator=gen, device=DEV) * 4
        weight = values.to(torch.float8_e4m3fn).view(torch.uint8)
    scale = torch.randint(
        118, 134, (rows, DIM // BLOCK), generator=gen, device=DEV, dtype=torch.uint8
    )
    bucket = rows // heads
    offsets_all = torch.arange(heads, device=DEV, dtype=torch.int64) * bucket
    ids = torch.randint(
        0, bucket, (spec["tokens"], heads), generator=gen, device=DEV, dtype=torch.int64
    )
    head_start = spec["head_start"]
    offsets = offsets_all[head_start : head_start + local].clone()
    if spec["masked"]:  # this rank holds the lower half of the rows
        vocab_start, vocab_end = 0, rows // 2
    else:  # a head shard holds its heads' buckets, rebased to its first row
        vocab_start, vocab_end = int(offsets[0]), int(offsets[-1]) + bucket
        offsets = offsets - vocab_start
        weight, scale = (
            weight[vocab_start:vocab_end].contiguous(),
            scale[vocab_start:vocab_end].contiguous(),
        )
    out = torch.empty(spec["tokens"], local, DIM, dtype=torch.bfloat16, device=DEV)
    kwargs = dict(head_start=head_start, block_size=BLOCK, masked=spec["masked"], vocab_start=vocab_start,
                  vocab_end=vocab_end, num_ctas=64, mxfp4=spec.get("mxfp4", False))  # fmt: skip
    return (weight, scale, ids, offsets, out), kwargs


@pytest.mark.parametrize("cell", engram_lookup.CELLS, ids=[c.why[:44] for c in engram_lookup.CELLS])
def test_certified_cells(cell, certifying):  # noqa: F811
    certifying(engram_lookup)
    args, kwargs = _inputs(cell.spec, cell)
    got = engram_lookup(*args, **kwargs)
    ref = engram_lookup.reference(*args, **kwargs)
    assert got.dtype == torch.bfloat16 and got.shape == args[4].shape
    torch.testing.assert_close(got, ref, rtol=0, atol=0)


def test_unowned_rows_read_zero():
    """Both sides of the ownership boundary: with only row 0 owned, a token reading rows 0 and 1 gets row 0's
    (nonzero) value and zero for row 1, and every unowned read is zero."""
    cell = engram_lookup.CELLS[0]
    args, kwargs = _inputs(cell.spec, cell)
    weight, scale, ids, offsets = args[:4]
    assert int(offsets[0]) == 0  # head 0's bucket starts at row 0
    weight[0] = (
        torch.tensor([1.5], device=DEV).to(torch.float8_e4m3fn).view(torch.uint8)
    )  # nonzero, known
    scale[0] = 127  # 2**0
    ids[0, 0], ids[1, 0] = 0, 1  # token 0 reads row 0 (owned), token 1 reads row 1 (not owned)
    kwargs["vocab_end"] = 1  # only row 0 is owned
    got = engram_lookup(*args, **kwargs)
    assert torch.equal(got[0, 0], torch.full((DIM,), 1.5, dtype=torch.bfloat16, device=DEV))
    assert not got[1, 0].any()
    owned = (ids + offsets < 1).unsqueeze(-1)
    assert not got.masked_select(~owned).any()
    torch.testing.assert_close(got, engram_lookup.reference(*args, **kwargs), rtol=0, atol=0)


def test_cuda_graph_replay_is_bitwise_the_eager_result():
    cell = engram_lookup.CELLS[0]
    args, kwargs = _inputs(cell.spec, cell)
    engram_lookup(*args, **kwargs)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        engram_lookup(*args, **kwargs)
    expect = args[4].clone()
    args[4].zero_()
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(args[4], expect, rtol=0, atol=0)


def test_rejections():
    cell = engram_lookup.CELLS[0]
    args, kwargs = _inputs(cell.spec, cell)
    engram_lookup.is_valid(*args, **kwargs)
    weight, scale, ids, offsets, out = args
    with pytest.raises(AssertionError):
        engram_lookup.is_valid(weight.view(torch.float8_e4m3fn), scale, ids, offsets, out, **kwargs)
    with pytest.raises(AssertionError):
        engram_lookup.is_valid(weight, scale, ids.int(), offsets, out, **kwargs)
    with pytest.raises(AssertionError):
        engram_lookup.is_valid(weight, scale, ids, offsets[:-1], out, **kwargs)


def test_e2m1_widening_is_the_code_table():
    """The reference's E2M1 -> E4M3 widening: every code's value (0, 0.5, 1, 1.5, 2, 3, 4, 6 and their negatives)."""
    from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.engram.engram_lookup import (
        e2m1_to_e4m3_bytes,
    )

    codes = torch.arange(16, dtype=torch.uint8, device=DEV)
    packed = codes | (codes.flip(0) << 4)  # element 0 is the low nibble
    wide = e2m1_to_e4m3_bytes(packed).view(torch.float8_e4m3fn).float()
    mags = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], device=DEV)
    values = torch.cat([mags, -mags])
    assert torch.equal(wide[0::2], values) and torch.equal(wide[1::2], values.flip(0))


def test_mxfp4_rejects_a_row_of_e4m3_width():
    cell = next(c for c in engram_lookup.CELLS if c.spec.get("mxfp4"))
    args, kwargs = _inputs(cell.spec, cell)
    engram_lookup.is_valid(*args, **kwargs)
    weight, scale, ids, offsets, out = args
    wide = torch.zeros(weight.shape[0], weight.shape[1] * 2, dtype=torch.uint8, device=DEV)
    with pytest.raises(AssertionError):
        engram_lookup.is_valid(wide, scale, ids, offsets, out, **kwargs)
