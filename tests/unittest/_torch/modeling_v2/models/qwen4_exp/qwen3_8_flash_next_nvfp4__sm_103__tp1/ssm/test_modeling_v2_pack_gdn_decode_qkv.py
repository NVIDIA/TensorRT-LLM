# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the pack_gdn_decode_qkv entry.

The op is a pure copy, so every comparison is bitwise; the discrimination test
is what shows that gate is not vacuous on these operands.
"""

import zlib

import pytest
import torch
from triton.compiler.errors import CompilationError

from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1.ops.ssm.pack_gdn_decode_qkv import (  # noqa: E501
    CKPT_HEADS,
    HEAD_K_DIM,
    HEAD_V_DIM,
    NUM_K_HEADS,
    NUM_V_HEADS,
    QKV_DIM,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1.ops.ssm.pack_gdn_decode_qkv import (  # noqa: E501
    pack_gdn_decode_qkv as op,
)

__extra_import_path__ = ["../../../.."]
from _validating import certifying, validating  # noqa: E402,F401 — path above; fixture by name

assert torch.cuda.is_available(), "pack_gdn_decode_qkv requires a CUDA device"


def _compare(out: tuple, ref: tuple) -> None:
    """Bitwise, all three: the op moves bits and changes nothing."""
    assert len(out) == len(ref) == 3
    for name, got, want in zip("qkv", out, ref):
        assert got.dtype == want.dtype and got.shape == want.shape, (
            name,
            (got.dtype, got.shape),
            (want.dtype, want.shape),
        )
        assert torch.equal(got, want), f"{name} must be a bit-exact copy"


DEV = "cuda"
Q_WIDTH = NUM_K_HEADS * HEAD_K_DIM  # 2048
V_WIDTH = NUM_V_HEADS * HEAD_V_DIM  # 6144
#: The `z` half of `in_proj_qkvz`, of which the stock mixer's `mixed_qkv` is a column slice.
Z_WIDTH = NUM_V_HEADS * HEAD_V_DIM


def _build(spec, seed):
    """`(mixed_qkv, conv_out or None)`: the input, and for a verify batch the
    `[B, qkv_dim, P]` conv output it was flattened from."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    dtype = spec.get("dtype", torch.bfloat16)
    heads = spec.get("heads", CKPT_HEADS)
    qkv_dim = 2 * heads[0] * heads[1] + heads[2] * heads[3]
    layout = spec.get("layout", "contiguous")

    def randn(*shape):
        return torch.randn(*shape, generator=g, device=DEV).to(dtype)

    if "draft" in spec:
        b, p = spec["num_decodes"], spec["draft"]
        if layout == "conv_dim_major":
            conv_out = randn(b, qkv_dim, p)  # contiguous; the flatten copies unless b == 1
        else:
            # The target: the conv result carries its `[n, C, p]` input view's
            # strides, so transposing back and flattening is a contiguous view.
            conv_out = randn(b, p, qkv_dim).transpose(1, 2)
        mixed = conv_out.transpose(1, 2).reshape(b * p, qkv_dim)
        if layout != "conv_dim_major":
            assert mixed.is_contiguous() and mixed.data_ptr() == conv_out.data_ptr()
        return mixed, conv_out
    n = spec["num_tokens"]
    if layout == "row_slice_of_projection":
        mixed = randn(n + 9, qkv_dim + Z_WIDTH)[9:, :qkv_dim]
        assert mixed.stride() == (qkv_dim + Z_WIDTH, 1)
    elif layout == "column_stride_2":
        mixed = randn(n, 2 * qkv_dim)[:, ::2]
        assert mixed.stride() == (2 * qkv_dim, 2)
    elif layout == "transposed_view":
        mixed = randn(qkv_dim, n).t()
        assert mixed.stride() == (1, n)
    elif layout == "wide":
        mixed = randn(n, qkv_dim + Z_WIDTH)
    else:
        mixed = randn(n, qkv_dim)
    return mixed, None


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    certifying(op)
    spec = cell.spec
    heads = spec.get("heads", CKPT_HEADS)
    mixed, conv_out = _build(spec, seed=zlib.crc32(cell.why.encode()))
    snapshot = mixed.clone()
    out = op(mixed, *heads)
    _compare(out, op.reference(mixed, *heads))

    assert all(t.is_contiguous() for t in out)
    assert torch.equal(mixed, snapshot), "mixed_qkv is read-only"
    if mixed.numel():  # an empty allocation has no storage to tell apart
        storages = {t.untyped_storage().data_ptr() for t in out}
        assert len(storages) == 3 and mixed.untyped_storage().data_ptr() not in storages

    if "draft" in spec:
        # The verify caller regroups each return as [B, P, heads, dim]: that is
        # a view of the same storage, and it lands on (sequence, draft) exactly.
        b, p = spec["num_decodes"], spec["draft"]
        num_k_heads, head_k_dim, num_v_heads, head_v_dim = heads
        grouped = (
            out[0].reshape(b, p, num_k_heads, head_k_dim),
            out[1].reshape(b, p, num_k_heads, head_k_dim),
            out[2].reshape(b, p, num_v_heads, head_v_dim),
        )
        assert all(gt.data_ptr() == t.data_ptr() for gt, t in zip(grouped, out))
        per_token = conv_out.permute(0, 2, 1)  # [B, P, qkv_dim], straight off the conv
        want = (
            per_token.narrow(2, 0, Q_WIDTH).reshape(b, p, num_k_heads, head_k_dim),
            per_token.narrow(2, Q_WIDTH, Q_WIDTH).reshape(b, p, num_k_heads, head_k_dim),
            per_token.narrow(2, 2 * Q_WIDTH, V_WIDTH).reshape(b, p, num_v_heads, head_v_dim),
        )
        assert all(torch.equal(gt, w) for gt, w in zip(grouped, want))


def test_draft_rows_are_independent() -> None:
    """Every row carries a distinct constant, so a token-axis mix-up -- invisible
    on rows that happen to agree -- cannot pass."""
    num_tokens = 6 * 4
    marks = torch.arange(1, num_tokens + 1, dtype=torch.float32, device=DEV).to(torch.bfloat16)
    mixed = marks.reshape(num_tokens, 1).expand(num_tokens, QKV_DIM).contiguous()
    q, k, v = op(mixed, *CKPT_HEADS)
    for out, width in ((q, Q_WIDTH), (k, Q_WIDTH), (v, V_WIDTH)):
        flat = out.reshape(num_tokens, -1)
        assert flat.shape[1] == width
        assert torch.equal(flat, marks.reshape(num_tokens, 1).expand(num_tokens, width))


def test_bitwise_gate_discriminates_wrong_variants() -> None:
    """Plausible ways to get the packing wrong; `_compare` rejects each."""
    g = torch.Generator(device=DEV).manual_seed(600)
    n = 24
    mixed = torch.randn(n, QKV_DIM, generator=g, device=DEV).to(torch.bfloat16)
    out = op(mixed, *CKPT_HEADS)
    q_ref, k_ref, v_ref = op.reference(mixed, *CKPT_HEADS)
    _compare(out, (q_ref, k_ref, v_ref))
    wrong = {
        "q_k_swapped": (k_ref, q_ref, v_ref),
        "v_from_q_offset": (
            q_ref,
            k_ref,
            mixed.narrow(1, 0, V_WIDTH).reshape(n, NUM_V_HEADS, HEAD_V_DIM).unsqueeze(0),
        ),
        "head_axis_transposed": (q_ref.transpose(2, 3).reshape(q_ref.shape), k_ref, v_ref),
        "token_order_reversed": (q_ref.flip(1), k_ref, v_ref),
    }
    for name, bad in wrong.items():
        with pytest.raises(AssertionError):
            _compare(out, bad)
        print(f"{name}: rejected")


@pytest.mark.parametrize("heads", [(16, 128, 48, 64), (16, 64, 48, 128)])
def test_op_rejects_head_dims_that_round_to_different_blocks(heads) -> None:
    """Loud at compile time, before any launch -- so not an `is_valid` guard."""
    qkv_dim = 2 * heads[0] * heads[1] + heads[2] * heads[3]
    mixed = torch.randn(13, qkv_dim, device=DEV).to(torch.bfloat16)
    with pytest.raises(CompilationError, match="Mismatched type for dim_mask"):
        op(mixed, *heads)


def test_a_short_row_reads_the_next_row_and_is_refused() -> None:
    """Unguarded, a `[32, 9216]` input at qkv_dim 10240 runs and silently reads
    the next row's first 1024 columns into the tail of `v`; the guard refuses it.

    The input is carved from a longer flat buffer so the over-read lands in
    known memory rather than in allocator garbage.
    """
    rows, width = 32, QKV_DIM - 1024
    flat = torch.randn(rows * width + 1024, device=DEV).to(torch.bfloat16)
    mixed = flat[: rows * width].view(rows, width)
    _, _, v = op(mixed, *CKPT_HEADS)
    torch.cuda.synchronize()
    tail = v[0].reshape(rows, V_WIDTH)[:, V_WIDTH - 1024 :]
    assert torch.equal(tail[:-1], mixed[1:, :1024]), "the hazard this guard covers is gone"

    with pytest.raises(AssertionError, match="must be at least qkv_dim=10240 wide"):
        with validating(op):
            op(mixed, *CKPT_HEADS)


def test_guard_refuses_over_declared_heads() -> None:
    mixed = torch.randn(8, QKV_DIM, device=DEV).to(torch.bfloat16)
    with pytest.raises(AssertionError, match="must be at least qkv_dim="):
        with validating(op):
            op(mixed, NUM_K_HEADS, HEAD_K_DIM, NUM_V_HEADS + 16, HEAD_V_DIM)


def test_guard_refuses_a_three_dimensional_input() -> None:
    """`[batch, draft, qkv_dim]` -- a caller who skipped the flatten. Unguarded
    it faults with an illegal memory access (measured; not replayed here, it
    poisons the process); the guard refuses it before the launch."""
    mixed = torch.randn(4, 2, QKV_DIM, device=DEV).to(torch.bfloat16)
    with pytest.raises(AssertionError, match="must be at least qkv_dim="):
        with validating(op):
            op(mixed, *CKPT_HEADS)
