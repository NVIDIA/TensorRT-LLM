# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the m3_fused_decode_attn entry (fused norm/RoPE/KV-insert + identity sparse decode + merge)."""

import inspect
import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_103__tp4.ops.attention.m3_fused_decode_attn import (  # noqa: E501
    PAGE,
    mxfp8_rows,
    unswizzle_scales,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_103__tp4.ops.attention.m3_fused_decode_attn import (  # noqa: E501
    m3_fused_decode_attn as op,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_103__tp4.ops.m3_fused_attn import (  # noqa: E501
    default_geometry,
)

__extra_import_path__ = ["../../../..", "."]
from _m3_fused_attn_step import (  # noqa: E402
    ATTN_ELEMENT_ULP,
    ATTN_RMS_ULP,
    MXFP8_ROWS,
    attention_args,
    build_step,
    check_insert,
    decode_over_written_caches,
)
from _validating import assert_within_ulp, certifying, validating  # noqa: E402, F401

assert torch.cuda.is_available(), "m3_fused_decode_attn requires a CUDA device"


def _buffers(s):
    t = s.num_tokens
    out = torch.empty(t, 2048, dtype=torch.bfloat16, device="cuda")
    q8 = torch.zeros(MXFP8_ROWS, 2048, dtype=torch.float8_e4m3fn, device="cuda")
    sf = torch.zeros(MXFP8_ROWS * 64, dtype=torch.uint8, device="cuda")
    return out, q8, sf


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:48] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 -- the fixture from _validating
    """The insert against the reference's recipe, the output against the reference's decode over the caches the
    op wrote (see ATTN_ELEMENT_ULP / CODE_MISMATCH_FRACTION), the MXFP8 hand-off bit-exactly."""
    certifying(op)
    spec = cell.spec
    qg, bpc = spec["geometry"]
    assert default_geometry(spec["qlen"], len(spec["seq_lens"])) == (qg, bpc), (
        "the cell's geometry is the target's"
    )
    s = build_step(spec["seq_lens"], spec["qlen"], seed=zlib.crc32(cell.why.encode()) % 2**31)
    out, q8, sf = _buffers(s)
    args = (s.packed, *attention_args(s), qg, bpc, out, q8, sf, True)
    before = (s.kv_data.clone(), s.kv_scale.clone(), s.idx_k.clone())
    ref = op.reference(*args)
    op(*args)
    check_insert(s, before, ref)
    want = decode_over_written_caches(s, s.packed)
    assert_within_ulp(out.view(-1, 128), want.view(-1, 128), ATTN_ELEMENT_ULP, ATTN_RMS_ULP)
    # The MXFP8 hand-off is the merge's recipe applied to the op's own bf16 rows: bit-exact.
    t = s.num_tokens
    q8_want, ex_want = mxfp8_rows(out)
    assert torch.equal(q8[:t].view(torch.uint8), q8_want), "MXFP8 hand-off values"
    assert torch.equal(unswizzle_scales(sf, t, 64), ex_want), "MXFP8 hand-off scales"


def _cell_inputs(index=1):
    spec = op.CELLS[index].spec
    s = build_step(spec["seq_lens"], spec["qlen"], seed=7)
    out, q8, sf = _buffers(s)
    qg, bpc = spec["geometry"]
    return s, [s.packed, *attention_args(s), qg, bpc, out, q8, sf, True]


def _arg(args, name):
    """Position of a parameter in the op's argument list (``is_valid`` mirrors ``__call__``)."""
    return list(inspect.signature(type(op).is_valid).parameters)[1:].index(name)


def test_guard_refuses_a_request_past_the_identity_window() -> None:
    """16 blocks of 128: a 2049-token request would have its new token placed in block 15 and its tail unread."""
    s, args = _cell_inputs()
    seq_lens = s.seq_lens.clone()
    seq_lens[0] = 16 * PAGE + 1
    args[_arg(args, "seq_lens")] = seq_lens
    with pytest.raises(AssertionError, match="identity selection covers 16 blocks"):
        with validating(op):
            op(*args)


def test_guard_refuses_a_slot_the_block_table_does_not_give() -> None:
    """The attention reads the inserted page from the cache it loaded: a slot elsewhere is never attended."""
    s, args = _cell_inputs()
    loc = s.out_cache_loc.clone()
    loc[0] = (loc[0] + 1) % (s.kv_data.shape[0] * PAGE)
    args[_arg(args, "out_cache_loc")] = loc
    with pytest.raises(AssertionError, match="out_cache_loc must be the slot"):
        with validating(op):
            op(*args)


def test_guard_refuses_a_block_table_narrower_than_the_length() -> None:
    s, args = _cell_inputs()
    args[_arg(args, "block_table")] = s.block_table[:, :1].clone()
    with pytest.raises(AssertionError, match="block_table must hold every attended block"):
        with validating(op):
            op(*args)


def test_guard_refuses_a_strided_kv_pool() -> None:
    """The kernel reads each [128, 64] head block as contiguous bytes; a transposed view is read as if it were."""
    s, args = _cell_inputs()
    args[_arg(args, "kv_data")] = s.kv_data.transpose(3, 4).contiguous().transpose(3, 4)
    with pytest.raises(AssertionError, match="kv_data must be"):
        with validating(op):
            op(*args)


def test_op_rejects_an_uncompiled_geometry() -> None:
    """Only the target's geometries are compiled; anything else is refused by the op itself."""
    s, args = _cell_inputs(index=0)
    args[_arg(args, "tokens_per_cluster")] = 2
    with pytest.raises(RuntimeError, match="is not compiled"):
        op(*args)
