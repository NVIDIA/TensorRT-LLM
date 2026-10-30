# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the causal_conv1d_triton_update entry."""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp_for_conditional_generation.hf_fc694b5_orig__sm_103__unified.ops.ssm.causal_conv1d_triton_update__r45 import (  # noqa: E501
    ENGINE_SLOTS,
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
    1.4e-6. The pool and the snapshots are copies, gated bitwise over the
    whole buffer -- the lock on every unnamed slot and snapshot row.
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
    """`__call__`'s positional arguments for one spec, freshly allocated."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    dim, batch, seqlen, width = spec["dim"], spec["batch"], spec["seqlen"], 4

    def randn(*shape):
        return torch.randn(*shape, generator=g, device="cuda").to(torch.bfloat16)

    # The call site under the one input GEMM: the q|k|v columns of an
    # [n * p, C + 6240] projection, reshaped [n, p, C] and seen as [n, C, p].
    wide = dim + 6144 + 2 * 48
    x = randn(batch * seqlen, wide)[:, :dim].reshape(batch, seqlen, dim).transpose(1, 2)
    assert x.stride() == (seqlen * wide, 1, wide)
    conv_state = randn(spec["pool_slots"], dim, width - 1)
    weight = randn(dim, width)
    indices = torch.tensor(spec["slots"], dtype=torch.int32, device="cuda")
    steps = spec.get("snapshot_steps", seqlen)
    window = torch.zeros(
        spec["snapshots"], steps, dim, width - 1, dtype=torch.bfloat16, device="cuda"
    )
    rows_idx = torch.arange(batch, dtype=torch.int32, device="cuda")
    return [
        x,
        conv_state,
        weight,
        None,
        "silu",
        indices,
        PAD_SLOT_ID,
        None,
        window,
        rows_idx,
        False,
    ]


def _run(args):
    """Drive the op on copies of the two buffers; return the triple compare takes."""
    got = list(args)
    got[1], got[8] = args[1].clone(), args[8].clone()
    returned = op(*got)
    return returned, got[1], got[8]


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    certifying(op)
    args = _build(cell.spec, seed=zlib.crc32(cell.why.encode()))
    x, weight = args[0].clone(), args[2].clone()
    ref = op.reference(*args)
    got = _run(args)
    assert got[0] is not args[0] and got[0].shape == args[0].shape
    assert torch.equal(args[0], x) and torch.equal(args[2], weight), "x and weight are read-only"
    _compare(got, ref)
    if cell.spec["pool_slots"] == ENGINE_SLOTS:
        for slot in cell.spec["slots"]:  # non-vacuity of the whole-pool gate
            assert not torch.equal(got[1][slot], args[1][slot]), slot


def test_guard_refuses_undersized_buffers() -> None:
    """A snapshot buffer shallower than seqlen is accepted by the kernel and
    written past the row it was given (measured); the guard refuses it
    before the launch."""
    spec = dict(dim=512, batch=2, seqlen=2, pool_slots=8, slots=(0, 2), snapshots=2)
    args = _build(spec, seed=15)
    args[8] = torch.zeros(2, 1, 512, 3, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(AssertionError, match="intermediate_conv_window must be at least"):
        with validating(op):
            op(*args)
