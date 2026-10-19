# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the causal_conv1d_triton_update entry."""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1.ops.ssm.causal_conv1d_triton_update import (  # noqa: E501
    CKPT_DIM,
    ENGINE_SLOTS,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1.ops.ssm.causal_conv1d_triton_update import (  # noqa: E501
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
    dim, batch = spec["dim"], spec["batch"]
    width, state_len = spec.get("width", 4), spec.get("state_len", 3)
    seqlen, dtype = spec.get("seqlen", 2), spec.get("dtype", torch.bfloat16)
    x_form = spec.get("x_form", "token_major")

    def randn(*shape):
        return torch.randn(*shape, generator=g, device="cuda").to(dtype)

    if x_form == "2d":
        x = randn(batch, dim)
    elif x_form == "contiguous":
        x = randn(batch, dim, seqlen)
    else:  # the call site: an [n, p, C] block seen as [n, C, p]
        x = randn(batch, seqlen, dim).transpose(1, 2)
    conv_state = randn(spec["pool_slots"], dim, state_len)
    weight = randn(dim, width)
    bias = randn(dim) if spec.get("bias", False) else None
    slots = spec["slots"]
    if slots == "random":
        indices = torch.randperm(spec["pool_slots"], generator=g, device="cuda")[:batch]
        indices = indices.to(torch.int32)
    elif slots is None:
        indices = None
    else:
        indices = torch.tensor(slots, dtype=torch.int32, device="cuda")
    accepted = spec.get("accepted")
    if accepted is not None:
        accepted = torch.tensor(accepted, dtype=torch.int32, device="cuda")
    rows = spec.get("snapshots", 0)
    window = rows_idx = None
    if rows:
        steps = 1 if x_form == "2d" else seqlen
        window = torch.zeros(rows, steps, dim, width - 1, dtype=dtype, device="cuda")
        rows_idx = torch.arange(batch, dtype=torch.int32, device="cuda")
    return [
        x,
        conv_state,
        weight,
        bias,
        spec.get("activation", "silu"),
        indices,
        spec.get("pad_slot_id", PAD_SLOT_ID),
        accepted,
        window,
        rows_idx,
        spec.get("launch_dependent_kernels", False),
    ]


def _run(args):
    """Drive the op on copies of the two buffers; return the triple compare takes."""
    got = list(args)
    got[1] = args[1].clone()
    if args[8] is not None:
        got[8] = args[8].clone()
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


def test_the_last_snapshot_is_what_the_pool_holds() -> None:
    """Independent of the reference: snapshot `seqlen - 1` equals the advanced slot."""
    spec = dict(dim=512, batch=6, seqlen=4, pool_slots=16, slots="random", snapshots=6)
    args = _build(spec, seed=1)
    _, pool, snapshots = _run(args)
    for b, slot in enumerate(args[5].tolist()):
        assert torch.equal(snapshots[b, 3], pool[slot]), b


def test_fp32_products_are_the_wrong_reference() -> None:
    """Discrimination: the gate is default-tolerance tight, and the natural
    wrong guess -- multiplying in fp32 -- lands outside it.

    The rival is `reference` on fp32 copies of the operands, rounded once at
    the end. Measured at 10240 channels: the faithful reference differs on
    2.4e-5 of elements (one-ulp `exp` cases); the rival on 0.37 of them.
    """
    spec = dict(dim=CKPT_DIM, batch=4, pool_slots=128, slots="random")
    args = _build(spec, seed=0)
    got = _run(args)
    faithful = op.reference(*args)
    _compare(got, faithful)
    fp32_args = list(args)
    for i in (0, 1, 2):
        fp32_args[i] = args[i].float()
    rival_out = op.reference(*fp32_args)[0].to(args[0].dtype)
    rival = (rival_out, faithful[1], faithful[2])
    with pytest.raises(AssertionError):
        _compare(got, rival)
    faithful_diff = (got[0] != faithful[0]).float().mean().item()
    rival_diff = (got[0] != rival_out).float().mean().item()
    print(f"\nelements differing: faithful {faithful_diff:.3e}, rival {rival_diff:.3e}")
    assert faithful_diff < 1e-3 and rival_diff > 0.3, (faithful_diff, rival_diff)


def test_token_major_and_contiguous_x_are_bitwise_identical() -> None:
    args = _build(dict(dim=512, batch=3, seqlen=4, pool_slots=16, slots=(2, 5, 1)), seed=11)
    assert not args[0].is_contiguous()
    strided = _run(args)
    packed = _run([args[0].contiguous()] + args[1:])
    assert torch.equal(strided[0], packed[0]) and torch.equal(strided[1], packed[1])


def test_launch_dependent_kernels_is_bitwise_inert() -> None:
    spec = dict(dim=512, batch=4, pool_slots=16, slots="random", snapshots=4)
    off = _run(_build(spec, seed=12))
    on = _run(_build(dict(spec, launch_dependent_kernels=True), seed=12))
    assert all(torch.equal(a, b) for a, b in zip(off, on))


def test_width_five_is_silently_wrong_and_refused() -> None:
    """Unguarded, width 5 runs and misses the reference; the guard refuses it."""
    spec = dict(dim=512, batch=2, width=5, state_len=4, pool_slots=8, slots=(1, 3))
    args = _build(spec, seed=13)
    ref = op.reference(*args)
    with pytest.raises(AssertionError):
        _compare(_run(args), ref)
    with pytest.raises(AssertionError, match="width must be 2, 3 or 4"):
        with validating(op):
            op(*_build(spec, seed=13))


def test_a_short_pool_writes_into_the_next_slot_and_is_refused() -> None:
    """Unguarded, a 2-column pool at W 4 writes past its slot into an unnamed
    neighbour; the guard refuses it."""
    spec = dict(dim=512, batch=2, state_len=2, pool_slots=8, slots=(1, 4))
    args = _build(spec, seed=14)
    _, pool, _ = _run(args)
    unnamed = [s for s in range(8) if s not in (1, 4)]
    assert not torch.equal(pool[unnamed], args[1][unnamed]), "the hazard this guard covers is gone"
    with pytest.raises(AssertionError, match="must hold at least 3 columns"):
        with validating(op):
            op(*_build(spec, seed=14))


@pytest.mark.parametrize(
    "case,overrides,match",
    [
        ("narrow_pool", None, "must cover all 512 channels"),
        ("short_weight", None, "must cover all 512 channels"),
        ("short_snapshot_buffer", None, "intermediate_conv_window must be at least"),
        # The accepted-token form needs W-1 + (seqlen-1) = 4 columns at seqlen 2.
        ("accepted_needs_a_longer_pool", dict(accepted=(1, 1)), "must hold at least 4 columns"),
    ],
)
def test_guard_refuses_undersized_buffers(case, overrides, match) -> None:
    """Accepted by the kernel and read or written out of bounds (measured);
    refused by the guard before the launch."""
    spec = dict(dim=512, batch=2, pool_slots=8, slots=(0, 2), snapshots=2)
    spec.update(overrides or {})
    args = _build(spec, seed=15)
    if case == "narrow_pool":
        args[1] = torch.randn(8, 256, 3, dtype=torch.bfloat16, device="cuda")
    elif case == "short_weight":
        args[2] = torch.randn(256, 4, dtype=torch.bfloat16, device="cuda")
    elif case == "short_snapshot_buffer":
        args[8] = torch.zeros(2, 1, 512, 3, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(AssertionError, match=match):
        with validating(op):
            op(*args)
