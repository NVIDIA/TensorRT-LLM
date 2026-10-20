# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the causal_conv1d_fwd catalog entry."""

import os
import zlib
from typing import Optional

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog.ssm.causal_conv1d_fwd import (
    QWEN4_EXP_SLOTS,
    ULTRA_TEP8_SLOTS_256,
    ULTRA_TP4_SLOTS,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.ssm.causal_conv1d_fwd import (
    causal_conv1d_fwd as op,
)

# Cells sized at an engine pool get a non-vacuity check below.
assert QWEN4_EXP_SLOTS == 461 and ULTRA_TP4_SLOTS == 35 and ULTRA_TEP8_SLOTS_256 == 271

__extra_import_path__ = [".."]
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
    (qwen4_exp geometry: max abs 0.015625, bitwise fraction 0.999996; Mamba2
    tep4 geometry: max abs 0.0039, bitwise fraction 0.999997; fp16 one ulp;
    fp32 1.4e-6). The state is a copy of the input tail, so it is gated
    bitwise over the whole pool view -- which is also the lock on every slot
    the batch did not name.
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


def _backing(t: torch.Tensor) -> torch.Tensor:
    """The whole buffer `t` views (`t` itself when it is not a view)."""
    return t if t._base is None else t._base


def _view_of(buffer: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    """`like`'s geometry laid over `buffer`, a same-shaped copy of `like`'s backing buffer."""
    base = _backing(like)
    return buffer.as_strided(
        like.size(), like.stride(), like.storage_offset() - base.storage_offset()
    )


def _fresh(t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    """`t` over a copy of its whole backing buffer, with `t`'s exact strides.

    A plain `clone()` of a view that is not dense comes back contiguous: for the
    projection slice that would turn a channel-last `x` into a channel-major
    one and drive the other kernel, and for one layer of the engine pool it
    would drop the slot stride under test.
    """
    if t is None:
        return None
    return _view_of(_backing(t).clone(), t)


def _build(spec, seed):
    """`__call__`'s positional arguments for one spec, freshly allocated.

    Operands the spec describes as views -- the projection's xBC columns, one
    layer of the engine pool -- are views of their own backing buffers, laid
    out exactly as the call site holds them.
    """
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
        if layout == "proj":
            proj_width, offset = spec["proj_width"], spec["proj_offset"]
            proj = randn(total, proj_width)
            x = proj[:, offset : offset + dim].t()
            assert x.stride() == (1, proj_width)
            out = torch.zeros(total, dim, dtype=dtype, device="cuda").t()
        else:
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
    if spec["pool_slots"] is None:
        pool = None
    elif "pool_layers" in spec:
        layers = randn(spec["pool_slots"], spec["pool_layers"], dim, width - 1)
        pool = layers[:, spec["pool_layer"]]
        assert pool.stride(0) == spec["pool_layers"] * dim * (width - 1)
    else:
        pool = randn(spec["pool_slots"], dim, width - 1)
    slots = spec["slots"]
    cache_indices = None if slots is None else torch.tensor(slots, dtype=torch.int32, device="cuda")
    initial = spec["initial"]
    his = None if initial is None else torch.tensor(initial, dtype=torch.bool, device="cuda")
    silu = spec.get("silu", True)
    pad = spec.get("pad_slot_id", PAD_SLOT_ID)
    return [x, weight, bias, pool, qsl, cache_indices, his, silu, pad, out]


def _run(args):
    """Drive the op on fresh copies of the mutable operands (x, conv_states,
    out) and their backing buffers; return (returned activation, pool after,
    the arguments it ran on)."""
    got = list(args)
    for i in (0, 3, 9):
        got[i] = _fresh(args[i])
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
        # The whole buffer x views: for the projection slice that is the z and
        # dt columns as well as xBC.
        assert torch.equal(_backing(got[0]), _backing(args[0])), "x is read-only when out is given"
    _compare((returned, pool), ref)
    if pool is not None and pool._base is not None:
        # One layer's view of a pool whose slots hold every layer: the other
        # layers' windows, interleaved with this one, must not move.
        expected = _backing(args[3]).clone()
        _view_of(expected, args[3]).copy_(ref[1])
        assert torch.equal(_backing(pool), expected), "only this layer's named slots may move"
    if cell.spec["pool_slots"] in (QWEN4_EXP_SLOTS, ULTRA_TP4_SLOTS, ULTRA_TEP8_SLOTS_256):
        # The whole-pool bitwise gate above already locks the unnamed slots;
        # this keeps it from being vacuous on the engine-sized pools.
        pad = cell.spec.get("pad_slot_id", PAD_SLOT_ID)
        for slot in cell.spec["slots"]:
            if slot != pad:
                assert not torch.equal(pool[slot], args[3][slot]), slot


def test_in_place_and_out_of_place_are_bitwise_identical() -> None:
    spec = dict(dim=10240, lengths=(64, 9), pool_slots=4, slots=(1, 2), initial=(True, True))
    in_place, pool_a, _ = _run(_build(spec, seed=1))
    out_of_place, pool_b, _ = _run(_build(dict(spec, layout="out"), seed=1))
    assert torch.equal(in_place, out_of_place) and torch.equal(pool_a, pool_b)


def _default_silu() -> bool:
    return os.environ.get("TRTLLM_CAUSAL_CONV1D_SILU_MODE", "") in ("", "0", "1")


def test_channel_last_and_channel_major_are_bitwise_identical() -> None:
    """The two kernels agree bitwise under the default SiLU form only."""
    spec = dict(
        dim=10240, lengths=(37, 1, 128), pool_slots=8, slots=(3, 0, 5), initial=(True, False, True)
    )
    major, pool_a, _ = _run(_build(spec, seed=11))
    last, pool_b, _ = _run(_build(dict(spec, layout="channel_last"), seed=11))
    assert torch.equal(pool_a, pool_b)
    if _default_silu():
        assert torch.equal(major, last)
    else:
        torch.testing.assert_close(major, last)


def test_projection_slice_and_a_channel_major_copy_are_bitwise_identical() -> None:
    """The Mamba2 call reads xBC in place from the projection (token stride
    8768) through the channel-last kernel; the same columns copied
    channel-major through the other kernel give the same activation and the
    same published state."""
    spec = next(c.spec for c in op.CELLS if c.why.startswith("Mamba2 tep4 prefill: 4 seq"))
    args = _build(spec, seed=21)
    proj_result, proj_pool, _ = _run(args)
    dense = list(args)
    dense[0] = args[0].contiguous()  # [dim, T], token axis contiguous
    dense[3] = args[3].clone()  # same content, plain [slots, dim, 3]
    dense[9] = None  # channel-major, in place
    major_result, major_pool, _ = _run(dense)
    assert torch.equal(proj_pool, major_pool)
    if _default_silu():
        assert torch.equal(proj_result, major_result)
    else:
        torch.testing.assert_close(proj_result, major_result)


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


_METADATA_STRIDE_CASES = (
    "query_start_loc_stride",
    "cache_indices_stride",
    "has_initial_state_stride",
)


def _stride_two(real: torch.Tensor, decoy: torch.Tensor) -> torch.Tensor:
    """`real` as a stride-2 view whose skipped entries hold `decoy`: a kernel
    that ignores the stride reads the decoys."""
    wide = torch.empty(2 * real.numel(), dtype=real.dtype, device=real.device)
    wide[0::2], wide[1::2] = real, decoy
    view = wide[0::2]
    assert torch.equal(view, real) and view.stride(0) == 2
    return view


def _guard_case(case: str):
    """A fresh argument list exhibiting one silent violation."""
    dim, tokens = 512, 64
    spec = dict(dim=dim, lengths=(tokens,), pool_slots=1, slots=(0,), initial=(True,))
    if case in ("width_one", "width_five"):
        spec.update(
            width=1 if case == "width_one" else 5, pool_slots=None, slots=None, initial=None
        )
    elif case in _METADATA_STRIDE_CASES:
        # The Mamba2 call-site layout, small: four sequences, mixed carry,
        # scattered slots of a six-slot pool.
        spec = dict(
            dim=dim,
            lengths=(16, 8, 24, 16),
            pool_slots=6,
            slots=(4, 0, 2, 5),
            initial=(True, False, True, False),
            bias=True,
            layout="proj",
            proj_width=2 * dim,
            proj_offset=dim // 2,
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
    elif case == "query_start_loc_stride":
        # Decoys between consecutive starts, so a misread still names
        # monotone, in-range spans -- just the wrong ones.
        qsl = args[4]
        mid = torch.cat([(qsl[:-1] + qsl[1:]) // 2, qsl[-1:]])
        args[4] = _stride_two(qsl, mid)
    elif case == "cache_indices_stride":
        args[5] = _stride_two(args[5], torch.tensor([1, 3, 1, 3], dtype=torch.int32, device="cuda"))
    elif case == "has_initial_state_stride":
        args[6] = _stride_two(args[6], ~args[6])
    return args


@pytest.mark.parametrize(
    "case",
    ["width_one", "width_five", "weight_dtype", "token_stride", "state_stride"]
    + list(_METADATA_STRIDE_CASES),
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
