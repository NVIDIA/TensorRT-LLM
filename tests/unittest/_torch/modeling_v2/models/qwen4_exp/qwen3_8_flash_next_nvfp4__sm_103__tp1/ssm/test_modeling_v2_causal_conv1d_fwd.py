# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the causal_conv1d_fwd entry."""

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
    dim, width, dtype = spec["dim"], 4, torch.bfloat16

    def randn(*shape):
        return torch.randn(*shape, generator=g, device="cuda").to(dtype)

    lengths = spec["lengths"]
    starts = [0]
    for length in lengths:
        starts.append(starts[-1] + length)
    qsl = torch.tensor(starts, dtype=torch.int32, device="cuda")
    x = randn(sum(lengths), dim).t().contiguous()
    weight = randn(dim, width)
    pool = randn(spec["pool_slots"], dim, width - 1)
    cache_indices = torch.tensor(spec["slots"], dtype=torch.int32, device="cuda")
    his = torch.tensor(spec["initial"], dtype=torch.bool, device="cuda")
    return [x, weight, None, pool, qsl, cache_indices, his, True, PAD_SLOT_ID, None]


def _clone(args):
    """The mutable operands cloned (x, conv_states)."""
    got = list(args)
    for i in (0, 3):
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
    assert returned is got[0], "returns the tensor it wrote"
    _compare((returned, pool), ref)
    if cell.spec["pool_slots"] == ENGINE_SLOTS:
        # The whole-pool bitwise gate above already locks the unnamed slots;
        # this keeps it from being vacuous on the large pool.
        for slot in cell.spec["slots"]:
            assert not torch.equal(pool[slot], args[3][slot]), slot


def _token_strided_args():
    """A fresh argument list whose `x` has token stride 2: the channel-major
    kernel ignores that stride and answers silently wrong."""
    dim, tokens = 512, 64
    spec = dict(dim=dim, lengths=(tokens,), pool_slots=1, slots=(0,), initial=(True,))
    args = _build(spec, seed=15)
    g = torch.Generator(device="cuda").manual_seed(16)
    wide = torch.randn(2 * tokens, dim, generator=g, device="cuda").to(torch.bfloat16)
    args[0] = wide.t().contiguous()[:, ::2]
    assert args[0].stride(1) == 2
    return args


def test_guard_refuses_what_the_op_answers_silently_wrong() -> None:
    """First the hazard, unguarded: the op returns without raising and its
    answer misses the reference. Then the guard refuses the same call."""
    ref = op.reference(*_token_strided_args())
    unguarded = _token_strided_args()  # strided views must not be cloned dense
    returned = op(*unguarded)
    with pytest.raises(AssertionError):
        _compare((returned, unguarded[3]), ref)

    with pytest.raises(AssertionError):
        with validating(op):
            op(*_token_strided_args())
