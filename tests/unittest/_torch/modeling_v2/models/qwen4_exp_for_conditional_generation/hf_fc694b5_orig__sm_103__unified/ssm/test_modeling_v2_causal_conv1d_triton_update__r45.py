# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the causal_conv1d_triton_update entry."""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp_for_conditional_generation.hf_fc694b5_orig__sm_103__unified.ops.ssm.causal_conv1d_triton_update__r45 import (  # noqa: E501
    CKPT_X_STRIDE,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp_for_conditional_generation.hf_fc694b5_orig__sm_103__unified.ops.ssm.causal_conv1d_triton_update__r45 import (  # noqa: E501
    causal_conv1d_triton_update as op,
)

__extra_import_path__ = ["../../../.."]
from _validating import certifying, validating  # noqa: E402,F401 — path above; fixture by name

assert torch.cuda.is_available(), "causal_conv1d_triton_update requires a CUDA device"


def _compare(out: tuple, ref: tuple) -> None:
    """`(out, conv_state, intermediate_conv_window)` triples: the returned
    tensor with the two buffers as the call left them, against
    `reference`'s triple.

    The result at torch's default band over the rows the op writes (NaN in
    the reference marks the rest): against the operand-dtype-product
    reference bf16/fp16 were bit-exact at dim 512 and at dim 10240 off by
    one ulp in at most 9 of 2.6M elements (the fp32 `exp`); fp32 is off by
    1.4e-6. At a tep4 rank's 2560 channels the same one-ulp differences:
    0 of 10240 elements (1 row x 4), 2 of 655360 (128 x 2), 4 of 1310720
    (128 x 4), 10 of 2621440 (128 x 8); relative L2 at most 2.0e-5, cosine
    at least 0.99999988. The pool and the snapshots are copies, gated
    bitwise over the whole buffer -- the lock on every unnamed slot and
    snapshot row.
    """
    (act, state, inter), (ref_act, ref_state, ref_inter) = out, ref
    assert act.dtype == ref_act.dtype and act.shape == ref_act.shape
    written = ~torch.isnan(ref_act)
    torch.testing.assert_close(act[written], ref_act[written])
    assert torch.equal(state, ref_state), "the advanced state must be bitwise exact"
    assert (inter is None) == (ref_inter is None)
    if inter is not None:
        assert torch.equal(inter, ref_inter), "the snapshots must be bitwise exact"


PAD_SLOT_ID = -1


def _build(spec, seed):
    """`__call__`'s positional arguments for one spec, freshly allocated, plus the
    buffers its views are cut from.

    x is the q|k|v columns of an `[n * p, x_row_stride]` projection output
    (z | a | b past them), reshaped `[n, p, C]` and seen as `[n, C, p]`; the
    pool is the last layer's view of a `[slots, pool_layers, C, 3]` page, as the
    V2 manager coalesces the GDN layers' conv states inside each slot; the
    snapshot buffer is the first layer's view of a `[snapshot_layers, rows,
    steps, C, 3]` buffer, as the GDN speculative state holds every layer's.
    """
    g = torch.Generator(device="cuda").manual_seed(seed)
    dim, batch, seqlen, width = spec["dim"], spec["batch"], spec["seqlen"], 4
    wide = spec.get("x_row_stride", CKPT_X_STRIDE)
    layers = spec.get("pool_layers", 1)

    def randn(*shape):
        return torch.randn(*shape, generator=g, device="cuda").to(torch.bfloat16)

    buf = randn(batch * seqlen, wide)
    x = buf[:, :dim].reshape(batch, seqlen, dim).transpose(1, 2)
    assert x.stride() == (seqlen * wide, 1, wide)
    pool_buf = randn(spec["pool_slots"], layers, dim, width - 1)
    conv_state = pool_buf[:, layers - 1]
    assert conv_state.stride() == (layers * dim * (width - 1), width - 1, 1)
    weight = randn(dim, width)
    indices = torch.tensor(spec["slots"], dtype=torch.int32, device="cuda")
    steps = spec.get("snapshot_steps", seqlen)
    window_buf = torch.zeros(
        spec.get("snapshot_layers", 1),
        spec["snapshots"],
        steps,
        dim,
        width - 1,
        dtype=torch.bfloat16,
        device="cuda",
    )
    rows_idx = torch.arange(batch, dtype=torch.int32, device="cuda")
    args = [
        x,
        conv_state,
        weight,
        None,
        "silu",
        indices,
        PAD_SLOT_ID,
        None,
        window_buf[0],
        rows_idx,
        False,
    ]
    return args, dict(buf=buf, pool_buf=pool_buf, window_buf=window_buf)


def _run(args, extras):
    """Drive the op on copies of the pool page and the snapshot buffer, through
    the same views; return the triple compare takes."""
    got = list(args)
    pool_buf, window_buf = extras["pool_buf"].clone(), extras["window_buf"].clone()
    got[1], got[8] = pool_buf[:, pool_buf.shape[1] - 1], window_buf[0]
    buf = extras["buf"].clone()
    returned = op(*got)
    assert torch.equal(extras["buf"], buf), "x (and the columns past it) are read-only"
    assert torch.equal(pool_buf[:, :-1], extras["pool_buf"][:, :-1]), "another layer written"
    assert torch.equal(window_buf[1:], extras["window_buf"][1:]), "another layer's snapshots"
    return returned, got[1], got[8]


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    certifying(op)
    args, extras = _build(cell.spec, seed=zlib.crc32(cell.why.encode()))
    weight = args[2].clone()
    ref = op.reference(*args)
    got = _run(args, extras)
    assert got[0] is not args[0] and got[0].shape == args[0].shape
    assert torch.equal(args[2], weight), "weight is read-only"
    _compare(got, ref)
    # Non-vacuity of the whole-pool gate: every slot the batch named moved.
    for slot in cell.spec["slots"]:
        assert not torch.equal(got[1][slot], args[1][slot]), slot


def test_guard_refuses_undersized_buffers() -> None:
    """A snapshot buffer shallower than seqlen is accepted by the kernel and
    written past the row it was given (measured); the guard refuses it
    before the launch."""
    spec = dict(dim=512, batch=2, seqlen=2, pool_slots=8, slots=(0, 2), snapshots=2)
    args, _ = _build(spec, seed=15)
    args[8] = torch.zeros(2, 1, 512, 3, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(AssertionError, match="intermediate_conv_window must be at least"):
        with validating(op):
            op(*args)
