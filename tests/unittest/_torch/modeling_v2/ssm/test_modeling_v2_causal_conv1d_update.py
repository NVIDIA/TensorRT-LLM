# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the causal_conv1d_update catalog entry."""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog.ssm.causal_conv1d_update import (
    QWEN4_EXP_SLOTS,
    ULTRA_TEP8_SLOTS_256,
    ULTRA_TP4_SLOTS,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.ssm.causal_conv1d_update import (
    causal_conv1d_update as op,
)

__extra_import_path__ = [".."]
from _validating import certifying, validating  # noqa: E402,F401 — path above; fixture by name

assert torch.cuda.is_available(), "causal_conv1d_update requires a CUDA device"


def _compare(
    out: tuple[torch.Tensor, torch.Tensor], ref: tuple[torch.Tensor, torch.Tensor]
) -> None:
    """`(x, conv_state)` pairs: the returned `x` with the pool as the call
    left it, against `reference`'s pair.

    Activation at torch's default band: bf16/fp16 were measured bitwise
    equal to the fp32 reference in nearly every cell (one bf16 ulp, 0.002,
    on a single element of one Mamba2 batch); fp32 differs by tap order
    (max 1.43e-6, inside the default fp32 atol). The pool is a copy of input
    values, so it is gated bitwise -- over the whole pool view, which is the
    lock on every slot the batch did not name.
    """
    (act, state), (ref_act, ref_state) = out, ref
    assert act.dtype == ref_act.dtype and act.shape == ref_act.shape
    torch.testing.assert_close(act, ref_act)
    assert torch.equal(state, ref_state), "the advanced state must be bitwise exact"


PAD_SLOT_ID = -1
#: Rows of prefill ahead of the decode rows in the qwen4_exp mixed-batch cell.
_PREFILL_ROWS = 37


def _backing(t: torch.Tensor) -> torch.Tensor:
    """The whole buffer `t` views (`t` itself when it is not a view)."""
    return t if t._base is None else t._base


def _view_of(buffer: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    """`like`'s geometry laid over `buffer`, a same-shaped copy of `like`'s backing buffer."""
    base = _backing(like)
    return buffer.as_strided(
        like.size(), like.stride(), like.storage_offset() - base.storage_offset()
    )


def _fresh(t: torch.Tensor) -> torch.Tensor:
    """`t` over a copy of its whole backing buffer, with `t`'s exact strides
    (a plain `clone()` of a non-dense view comes back contiguous)."""
    return _view_of(_backing(t).clone(), t)


def _assert_outside_unchanged(after: torch.Tensor, before: torch.Tensor, what: str) -> None:
    """Every element of the backing buffer outside the view is bitwise unchanged."""
    if before._base is None:
        return  # the view is the whole buffer
    got, want = _backing(after).clone(), _backing(before).clone()
    _view_of(got, before).zero_()
    _view_of(want, before).zero_()
    assert torch.equal(got, want), f"{what}: the op wrote outside the view it was handed"


def _build(spec, seed):
    """`__call__`'s arguments for one spec.

    Operands the spec describes as views -- a row slice or the xBC columns of
    a projection, one layer of the engine pool, a column-strided pool -- are
    views of their own backing buffers, as the call site holds them.
    """
    g = torch.Generator(device="cuda").manual_seed(seed)
    dim, batch = spec["dim"], spec["batch"]
    width, state_len, seqlen = spec.get("width", 4), spec.get("state_len", 3), spec.get("seqlen", 1)
    dtype, x_form = spec.get("dtype", torch.bfloat16), spec.get("x_form", "3d")

    def randn(*shape):
        return torch.randn(*shape, generator=g, device="cuda").to(dtype)

    if x_form == "2d":
        x = randn(batch, dim)
    elif x_form == "2d_row_slice":
        x = randn(_PREFILL_ROWS + batch, dim)[_PREFILL_ROWS:]
    elif x_form == "proj":
        ahead, proj_width, offset = (
            spec.get("rows_ahead", 0),
            spec["proj_width"],
            spec["proj_offset"],
        )
        x = randn(ahead + batch, proj_width)[ahead:, offset : offset + dim]
        assert x.stride() == (proj_width, 1)
    elif x_form == "token_major":
        x = randn(batch, seqlen, dim).transpose(1, 2)
        assert x.stride() == (seqlen * dim, 1, dim)
    else:
        x = randn(batch, dim, seqlen)
    if spec.get("strided_state", False):
        conv_state = randn(spec["pool_slots"], dim, 2 * state_len)[:, :, ::2]
    elif "pool_layers" in spec:
        layers = randn(spec["pool_slots"], spec["pool_layers"], dim, state_len)
        conv_state = layers[:, spec["pool_layer"]]
        assert conv_state.stride(0) == spec["pool_layers"] * dim * state_len
    else:
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
    cache_seqlens = spec.get("cache_seqlens")
    if cache_seqlens is not None:
        cache_seqlens = torch.tensor(cache_seqlens, dtype=torch.int32, device="cuda")
    silu, pad = spec.get("silu", True), spec.get("pad_slot_id", PAD_SLOT_ID)
    return [x, conv_state, weight, bias, silu, cache_seqlens, indices, pad]


def _run(args):
    """Drive the op on copies of x and the pool (and their backing buffers),
    through the same views; return (returned x, pool after, the arguments)."""
    got = list(args)
    got[0], got[1] = _fresh(args[0]), _fresh(args[1])
    returned = op(*got)
    assert returned is got[0], "returns the x it was handed"
    _assert_outside_unchanged(got[0], args[0], "x")
    _assert_outside_unchanged(got[1], args[1], "conv_state")
    return returned, got[1], got


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    certifying(op)
    spec = cell.spec
    args = _build(spec, seed=zlib.crc32(cell.why.encode()))
    weight, bias = args[2].clone(), None if args[3] is None else args[3].clone()
    ref_x, ref_state = op.reference(*args)
    returned, state, _ = _run(args)
    assert torch.equal(args[2], weight), "weight is read-only"
    assert bias is None or torch.equal(args[3], bias), "bias is read-only"
    named = [] if spec["slots"] in (None, "random") else list(spec["slots"])
    padding = spec.get("padding_rows", 0)
    if padding:
        # A graph-padded batch: the padding rows all name one dummy slot and
        # race on it by design. Gate the real rows, and every slot but that one.
        real, dummy = spec["batch"] - padding, spec["slots"][-1]
        assert set(spec["slots"][real:]) == {dummy} and dummy not in spec["slots"][:real]
        keep = [s for s in range(state.shape[0]) if s != dummy]
        _compare((returned[:real], state[keep]), (ref_x[:real], ref_state[keep]))
        named = named[:real]
    else:
        _compare((returned, state), (ref_x, ref_state))
    if spec["pool_slots"] in (QWEN4_EXP_SLOTS, ULTRA_TP4_SLOTS, ULTRA_TEP8_SLOTS_256):
        # Non-vacuity for the engine-sized pools: every named slot moved.
        for slot in named:
            assert not torch.equal(state[slot], args[1][slot]), slot


def test_cuda_graph_replay_reads_indices_and_inputs_on_device() -> None:
    """Decode runs captured. A replayed call must read the slot indices, x and
    the pool where they live at replay time rather than replaying what
    capture saw -- three replays with fresh indices and inputs, each against
    the reference of the state it started from."""
    spec = next(c.spec for c in op.CELLS if c.why.startswith("Mamba2 tep4 decode batch 8:"))
    args = _build(spec, seed=31)
    x, pool, indices = args[0], args[1], args[6]
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        op(*args)  # first launch outside capture
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        op(*args)
    g = torch.Generator(device="cuda").manual_seed(32)
    for _ in range(3):
        x.copy_(torch.randn(x.shape, generator=g, device="cuda").to(x.dtype))
        perm = torch.randperm(ULTRA_TP4_SLOTS, generator=g, device="cuda")[: indices.numel()]
        indices.copy_(perm.to(torch.int32))
        ref_x, ref_state = op.reference(*args)
        graph.replay()
        torch.cuda.synchronize()
        _compare((x, pool), (ref_x, ref_state))


def _small(**overrides):
    spec = dict(dim=512, batch=4, pool_slots=16, slots=(0, 1, 2, 3))
    spec.update(overrides)
    return _build(spec, seed=14)


_REJECTIONS = {
    "width_1": (
        "causal_conv1d only supports width between 2 and 4",
        dict(dim=128, width=1, state_len=1),
    ),
    "width_5": (
        "causal_conv1d only supports width between 2 and 4",
        dict(dim=128, width=5, state_len=4),
    ),
    "state_too_short": ("conv_state_len >= width - 1", dict(state_len=2)),
    "indices_length": ("conv_state_indices must have shape", None),
    "indices_int64": ("conv_state_indices.scalar_type", None),
    "indices_non_contiguous": ("is_contiguous", None),
    "cache_seqlens_int64": ("cache_seqlens.scalar_type", None),
    "no_indices_oversize_pool": ("conv_state must have shape", dict(slots=None)),
    "weight_shape": ("weight must have shape", None),
    "conv_state_dtype": ("conv_state.scalar_type", None),
    "weight_dtype": ("weight type must equal to input type", None),
    "bias_dtype": ("bias.scalar_type", None),
    "float64": ("input_type == at::ScalarType::Float", None),
}


@pytest.mark.parametrize("case", sorted(_REJECTIONS))
def test_op_rejects_every_metadata_violation_loudly(case: str) -> None:
    """Why the entry has no `is_valid`: every metadata violation probed raises."""
    fragment, overrides = _REJECTIONS[case]
    args = _small(**(overrides or {}))
    if case == "indices_length":
        args[6] = args[6][:3]
    elif case == "indices_int64":
        args[6] = args[6].long()
    elif case == "indices_non_contiguous":
        args[6] = torch.arange(8, dtype=torch.int32, device="cuda")[::2]
    elif case == "cache_seqlens_int64":
        args[5] = torch.zeros(4, dtype=torch.int64, device="cuda")
    elif case == "weight_shape":
        args[2] = torch.randn(256, 4, dtype=torch.bfloat16, device="cuda")
    elif case == "conv_state_dtype":
        args[1] = args[1].float()
    elif case == "weight_dtype":
        args[2] = args[2].float()
    elif case == "bias_dtype":
        args[3] = torch.randn(512, dtype=torch.float32, device="cuda")
    elif case == "float64":
        args[0], args[1], args[2] = args[0].double(), args[1].double(), args[2].double()
    with pytest.raises(RuntimeError, match=fragment.replace("(", r"\(").replace(")", r"\)")):
        with validating(op):
            op(*args)
