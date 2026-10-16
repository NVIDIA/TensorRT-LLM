# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the causal_conv1d_fwd entry."""

import os
import zlib
from typing import Optional

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1.ops.ssm.causal_conv1d_fwd import (  # noqa: E501
    ENGINE_SLOTS,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1.ops.ssm.causal_conv1d_fwd import (  # noqa: E501
    causal_conv1d_fwd as op,
)

# Cells sized at the engine's pool get a non-vacuity check below.
assert ENGINE_SLOTS == 461

__extra_import_path__ = ["../../../.."]
from _validating import certifying, validating  # noqa: E402,F401 — path above; fixture by name

assert torch.cuda.is_available(), "causal_conv1d_fwd requires a CUDA device"


def _compare(
    out: tuple[torch.Tensor, Optional[torch.Tensor]],
    ref: tuple[torch.Tensor, Optional[torch.Tensor]],
) -> None:
    """`(activation, conv_states)` pairs: the returned activation with the
    pool as the call left it, against `reference`'s pair.

    Activation at torch's default band: the kernel accumulates the taps in
    fp32 and rounds once, so it sits within one ulp of the fp32 reference
    (checkpoint geometry: max abs 0.015625, bitwise fraction 0.999996; fp16
    one ulp; fp32 1.4e-6). The state is a copy of the input tail, so it is
    gated bitwise over the whole pool -- which is also the lock on every
    slot the batch did not name.
    """
    (act, states), (ref_act, ref_states) = out, ref
    assert act.dtype == ref_act.dtype and act.shape == ref_act.shape
    torch.testing.assert_close(act, ref_act)
    assert (states is None) == (ref_states is None)
    if states is not None:
        assert torch.equal(states, ref_states), "the published state must be bitwise exact"


PAD_SLOT_ID = -1


def _seed(text: str) -> int:
    return zlib.crc32(text.encode())


def _build(spec, seed):
    """`__call__`'s positional arguments for one spec, freshly allocated."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    dim, width = spec["dim"], spec.get("width", 4)
    dtype, layout = spec.get("dtype", torch.bfloat16), spec.get("layout", "in_place")

    def randn(*shape):
        return torch.randn(*shape, generator=g, device="cuda").to(dtype)

    out = None
    if "batch3d" in spec:
        batch, seqlen = spec["batch3d"]
        x = randn(batch, dim, seqlen)
        qsl = None
    else:
        lengths = spec["lengths"]
        total = sum(lengths)
        starts = [0]
        for length in lengths:
            starts.append(starts[-1] + length)
        qsl = torch.tensor(starts, dtype=torch.int32, device="cuda")
        token_major = randn(total, dim)
        if layout == "channel_last":
            x = token_major.t()
            assert x.stride() == (1, dim)
            out = torch.zeros(total, dim, dtype=dtype, device="cuda").t()
        else:
            x = token_major.t().contiguous()
            if layout == "out":
                out = torch.zeros_like(x)
    weight = randn(dim, width)
    bias = randn(dim) if spec.get("bias", False) else None
    pool = None if spec["pool_slots"] is None else randn(spec["pool_slots"], dim, width - 1)
    slots = spec["slots"]
    cache_indices = None if slots is None else torch.tensor(slots, dtype=torch.int32, device="cuda")
    initial = spec["initial"]
    his = None if initial is None else torch.tensor(initial, dtype=torch.bool, device="cuda")
    silu = spec.get("silu", True)
    pad = spec.get("pad_slot_id", PAD_SLOT_ID)
    return [x, weight, bias, pool, qsl, cache_indices, his, silu, pad, out]


def _clone(args):
    """The mutable operands cloned (x, conv_states, out); strides preserved."""
    got = list(args)
    for i in (0, 3, 9):
        if got[i] is not None:
            got[i] = got[i].clone()
    return got


def _run(args):
    """Drive the op on clones; return (returned activation, pool after, clones)."""
    got = _clone(args)
    returned = op(*got)
    return returned, got[3], got


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    certifying(op)
    args = _build(cell.spec, seed=_seed(cell.why))
    ref = op.reference(*args)
    returned, pool, got = _run(args)
    assert returned is (got[0] if got[9] is None else got[9]), "returns the tensor it wrote"
    if got[9] is not None:
        assert torch.equal(got[0], args[0]), "x is read-only when out is given"
    _compare((returned, pool), ref)
    if cell.spec["pool_slots"] == ENGINE_SLOTS:
        # The whole-pool bitwise gate above already locks the unnamed slots;
        # this keeps it from being vacuous on the large pool.
        for slot in cell.spec["slots"]:
            assert not torch.equal(pool[slot], args[3][slot]), slot


def test_in_place_and_out_of_place_are_bitwise_identical() -> None:
    spec = dict(dim=10240, lengths=(64, 9), pool_slots=4, slots=(1, 2), initial=(True, True))
    in_place, pool_a, _ = _run(_build(spec, seed=1))
    out_of_place, pool_b, _ = _run(_build(dict(spec, layout="out"), seed=1))
    assert torch.equal(in_place, out_of_place) and torch.equal(pool_a, pool_b)


def test_channel_last_and_channel_major_are_bitwise_identical() -> None:
    """The two kernels agree bitwise under the default SiLU form only."""
    spec = dict(
        dim=10240, lengths=(37, 1, 128), pool_slots=8, slots=(3, 0, 5), initial=(True, False, True)
    )
    major, pool_a, _ = _run(_build(spec, seed=11))
    last, pool_b, _ = _run(_build(dict(spec, layout="channel_last"), seed=11))
    assert torch.equal(pool_a, pool_b)
    if os.environ.get("TRTLLM_CAUSAL_CONV1D_SILU_MODE", "") in ("", "0", "1"):
        assert torch.equal(major, last)
    else:
        torch.testing.assert_close(major, last)


def test_the_published_state_is_the_raw_input_tail() -> None:
    """Discrimination: the state is the pre-convolution input, not the activation."""
    spec = dict(dim=512, lengths=(48,), pool_slots=1, slots=(0,), initial=(True,), layout="out")
    args = _build(spec, seed=12)
    returned, pool, _ = _run(args)
    x = args[0]
    assert torch.equal(pool[0], x[:, -3:])
    assert (pool[0].float() - returned[:, -3:].float()).abs().max().item() > 0.1


def test_the_initial_state_is_read() -> None:
    """Discrimination: flipping the carry moves the first W-1 columns, nothing else."""
    spec = dict(dim=512, lengths=(64,), pool_slots=1, slots=(0,))
    with_carry, _, _ = _run(_build(dict(spec, initial=(True,)), seed=13))
    without, _, _ = _run(_build(dict(spec, initial=(False,)), seed=13))
    assert (with_carry[:, :3].float() - without[:, :3].float()).abs().max().item() > 0.1
    assert torch.equal(with_carry[:, 3:], without[:, 3:])


_REJECTED = (
    "channel_last_in_place",
    "channel_last_with_channel_major_out",
    "channel_major_with_channel_last_out",
    "out_dtype",
    "out_shape",
    "bias_dtype",
    "x_float64",
    "query_start_loc_int64",
    "has_initial_state_length",
    "conv_states_dtype",
    "x_rank",
)


@pytest.mark.parametrize("case", _REJECTED)
def test_op_rejects_loudly_what_is_valid_does_not_restate(case: str) -> None:
    """Everything the op refuses itself -- the reason `is_valid` stays narrow."""
    dim, tokens = 512, 64
    base = dict(dim=dim, lengths=(30, 34), pool_slots=2, slots=(0, 1), initial=(True, True))
    args = _build(dict(base, pool_slots=None, slots=None, initial=None), seed=14)
    token_major = torch.randn(tokens, dim, dtype=torch.bfloat16, device="cuda")
    pool = torch.randn(2, dim, 3, dtype=torch.bfloat16, device="cuda")
    slots = torch.tensor([0, 1], dtype=torch.int32, device="cuda")
    if case == "channel_last_in_place":
        args[0] = token_major.t()
    elif case == "channel_last_with_channel_major_out":
        args[0] = token_major.t()
        args[9] = torch.empty(dim, tokens, dtype=torch.bfloat16, device="cuda")
    elif case == "channel_major_with_channel_last_out":
        args[9] = torch.empty(tokens, dim, dtype=torch.bfloat16, device="cuda").t()
    elif case == "out_dtype":
        args[9] = torch.empty(dim, tokens, dtype=torch.float16, device="cuda")
    elif case == "out_shape":
        args[9] = torch.empty(dim, tokens + 1, dtype=torch.bfloat16, device="cuda")
    elif case == "bias_dtype":
        args[2] = torch.randn(dim, dtype=torch.float32, device="cuda")
    elif case == "x_float64":
        args[0] = args[0].double()
        args[1] = args[1].double()
    elif case == "query_start_loc_int64":
        args[4] = args[4].long()
    elif case == "has_initial_state_length":
        args[3], args[5] = pool, slots
        args[6] = torch.ones(3, dtype=torch.bool, device="cuda")
    elif case == "conv_states_dtype":
        args[3], args[5] = pool.float(), slots
        args[6] = torch.ones(2, dtype=torch.bool, device="cuda")
    elif case == "x_rank":
        args[0] = torch.randn(2, dim, 32, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(RuntimeError):
        with validating(op):
            op(*args)


def _guard_case(case: str):
    """A fresh argument list exhibiting one silent violation."""
    dim, tokens = 512, 64
    spec = dict(dim=dim, lengths=(tokens,), pool_slots=1, slots=(0,), initial=(True,))
    if case in ("width_one", "width_five"):
        spec.update(
            width=1 if case == "width_one" else 5, pool_slots=None, slots=None, initial=None
        )
    args = _build(spec, seed=15)
    g = torch.Generator(device="cuda").manual_seed(16)
    if case == "weight_dtype":
        args[1] = args[1].float()
    elif case == "token_stride":
        wide = torch.randn(2 * tokens, dim, generator=g, device="cuda").to(torch.bfloat16)
        args[0] = wide.t().contiguous()[:, ::2]
        assert args[0].stride(1) == 2
    elif case == "state_stride":
        wide = torch.randn(1, dim, 6, generator=g, device="cuda").to(torch.bfloat16)
        args[3] = wide[:, :, ::2]
    return args


@pytest.mark.parametrize(
    "case", ["width_one", "width_five", "weight_dtype", "token_stride", "state_stride"]
)
def test_guard_refuses_what_the_op_answers_silently_wrong(case: str) -> None:
    """First the hazard, unguarded: the op returns without raising and its
    answer misses the reference. Then the guard refuses the same call."""
    ref = op.reference(*_guard_case(case))
    unguarded = _guard_case(case)  # strided views must not be cloned dense
    returned = op(*unguarded)
    with pytest.raises(AssertionError):
        _compare((returned, unguarded[3]), ref)

    with pytest.raises(AssertionError):
        with validating(op):
            op(*_guard_case(case))
