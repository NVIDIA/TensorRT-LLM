# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the pack_gdn_decode_qkv entry.

The op is a pure copy, so every comparison is bitwise.
"""

import zlib

import pytest
import torch

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


def _build(spec, seed):
    """`(mixed_qkv, conv_out or None)`: the bf16 input, and for a verify batch the
    `[B, qkv_dim, P]` conv output it was flattened from."""
    g = torch.Generator(device=DEV).manual_seed(seed)

    def randn(*shape):
        return torch.randn(*shape, generator=g, device=DEV).to(torch.bfloat16)

    if "draft" in spec:
        b, p = spec["num_decodes"], spec["draft"]
        # The target: the conv result carries its `[n, C, p]` input view's
        # strides, so transposing back and flattening is a contiguous view.
        conv_out = randn(b, p, QKV_DIM).transpose(1, 2)
        mixed = conv_out.transpose(1, 2).reshape(b * p, QKV_DIM)
        assert mixed.is_contiguous() and mixed.data_ptr() == conv_out.data_ptr()
        return mixed, conv_out
    return randn(spec["num_tokens"], QKV_DIM), None


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    certifying(op)
    spec = cell.spec
    mixed, conv_out = _build(spec, seed=zlib.crc32(cell.why.encode()))
    snapshot = mixed.clone()
    out = op(mixed, *CKPT_HEADS)
    _compare(out, op.reference(mixed, *CKPT_HEADS))

    assert all(t.is_contiguous() for t in out)
    assert torch.equal(mixed, snapshot), "mixed_qkv is read-only"
    storages = {t.untyped_storage().data_ptr() for t in out}
    assert len(storages) == 3 and mixed.untyped_storage().data_ptr() not in storages

    if "draft" in spec:
        # The verify caller regroups each return as [B, P, heads, dim]: that is
        # a view of the same storage, and it lands on (sequence, draft) exactly.
        b, p = spec["num_decodes"], spec["draft"]
        grouped = (
            out[0].reshape(b, p, NUM_K_HEADS, HEAD_K_DIM),
            out[1].reshape(b, p, NUM_K_HEADS, HEAD_K_DIM),
            out[2].reshape(b, p, NUM_V_HEADS, HEAD_V_DIM),
        )
        assert all(gt.data_ptr() == t.data_ptr() for gt, t in zip(grouped, out))
        per_token = conv_out.permute(0, 2, 1)  # [B, P, qkv_dim], straight off the conv
        want = (
            per_token.narrow(2, 0, Q_WIDTH).reshape(b, p, NUM_K_HEADS, HEAD_K_DIM),
            per_token.narrow(2, Q_WIDTH, Q_WIDTH).reshape(b, p, NUM_K_HEADS, HEAD_K_DIM),
            per_token.narrow(2, 2 * Q_WIDTH, V_WIDTH).reshape(b, p, NUM_V_HEADS, HEAD_V_DIM),
        )
        assert all(torch.equal(gt, w) for gt, w in zip(grouped, want))


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
