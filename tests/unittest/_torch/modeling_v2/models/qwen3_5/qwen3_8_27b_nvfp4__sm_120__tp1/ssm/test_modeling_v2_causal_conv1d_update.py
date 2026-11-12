# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the qwen3.8-27b sm_120 tp1 causal_conv1d_update entry.

Each cell against the entry's float64 reference:

* `x` (the returned view, written in place) in torch's default bf16 band: the kernel's
  output is one bf16 rounding of an fp32 4-tap `fma` chain and `v / (1 + __expf(-v))`
  (an approximate exp, ~2^-21 relative), so it differs from the fp64 value by at most
  one bf16 rounding plus that;
* the pool bitwise, whole: the new window is a copy of inputs, and comparing every slot
  is the lock on the slots the call did not name (the pad row's included);
* the projection's other bytes untouched: the z columns past x, the context rows ahead
  of a mixed step's riders, the pad row's x, and (coalesced pool) the other layers.

Then the guard on shared slots.
"""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.qwen3_5.qwen3_8_27b_nvfp4__sm_120__tp1.ops.ssm.causal_conv1d_update import (  # noqa: E501
    CONV_DIM,
    POOL_SLOTS,
    QKVZ_DIM,
    WIDTH,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen3_5.qwen3_8_27b_nvfp4__sm_120__tp1.ops.ssm.causal_conv1d_update import (  # noqa: E501
    causal_conv1d_update as op,
)
from tensorrt_llm._utils import get_sm_version

__extra_import_path__ = ["../../../.."]
from _validating import certifying, validating  # noqa: E402,F401 — path above; fixture by name

pytestmark = pytest.mark.skipif(
    get_sm_version() != 120, reason="the qwen3.8-27b sm_120 tp1 entries are certified on sm_120"
)

DEV = "cuda"
PAD_SLOT_ID = -1
#: GDN layers whose conv states the cache manager v2 coalesces into one slot page.
_LAYERS = 48


def _build(spec, seed):
    """`(args, extras)`: `__call__`'s positional arguments and the buffers x and the pool
    view."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    rows, prefill = spec["rows"], spec.get("prefill_rows", 0)

    def randn(*shape):
        return torch.randn(*shape, generator=g, device=DEV).to(torch.bfloat16)

    proj = randn(prefill + rows, QKVZ_DIM)
    x = proj[prefill:, :CONV_DIM]
    if spec.get("pool") == "coalesced":
        backing = randn(POOL_SLOTS, _LAYERS, CONV_DIM, WIDTH - 1)
        conv_state = backing[:, 0]
    else:
        backing = randn(POOL_SLOTS, CONV_DIM, WIDTH - 1)
        conv_state = backing
    weight = (randn(CONV_DIM, WIDTH).float() * 0.5).to(torch.bfloat16)
    if spec.get("slots", "contiguous") == "scattered":
        slots = torch.randperm(POOL_SLOTS, generator=g, device=DEV)[:rows].tolist()
    else:
        slots = list(range(rows))
    if spec.get("pad"):
        slots[1] = PAD_SLOT_ID
    indices = torch.tensor(slots, dtype=torch.int32, device=DEV)
    args = [x, conv_state, weight, None, True, None, indices, PAD_SLOT_ID]
    return args, dict(proj=proj, backing=backing, prefill=prefill, slots=slots)


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — fixture by name
    certifying(op)
    args, extras = _build(cell.spec, seed=zlib.crc32(cell.why.encode()))
    x, conv_state, weight = args[:3]
    proj_before, backing_before = extras["proj"].clone(), extras["backing"].clone()
    x_before, weight_before, pool_before = x.clone(), weight.clone(), conv_state.clone()
    ref_x, ref_state = op.reference(*args)

    returned = op(*args)

    assert returned is x, "returns the x it was handed, written in place"
    assert x.stride() == (QKVZ_DIM, 1), "x is the projection's column view"
    torch.testing.assert_close(x, ref_x)
    assert torch.equal(conv_state, ref_state), "the advanced pool must be bitwise exact"
    assert torch.equal(weight, weight_before), "weight is read only"
    proj, prefill = extras["proj"], extras["prefill"]
    assert torch.equal(proj[:, CONV_DIM:], proj_before[:, CONV_DIM:]), "columns past x written"
    assert torch.equal(proj[:prefill], proj_before[:prefill]), "context rows written"
    for row, slot in enumerate(extras["slots"]):
        if slot == PAD_SLOT_ID:
            assert torch.equal(x[row], x_before[row]), "the pad row's x was written"
        else:  # non-vacuity: every named slot moved
            assert not torch.equal(conv_state[slot], pool_before[slot]), slot
    if cell.spec.get("pool") == "coalesced":
        assert torch.equal(extras["backing"][:, 1:], backing_before[:, 1:]), "another layer"


def test_guard_refuses_shared_slots() -> None:
    """Two rows naming one slot race on its window (each reads the old one, the last
    store wins); the guard refuses it."""
    args, _ = _build(dict(rows=4), seed=21)
    args[6] = torch.tensor([3, 7, 3, 9], dtype=torch.int32, device=DEV)
    with pytest.raises(AssertionError, match="two rows naming one slot"):
        with validating(op):
            op(*args)
