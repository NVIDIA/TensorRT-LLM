# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the Kimi K3.1 kda_mtp_commit_pending_drafts entry.

Each cell commits the cell's slots over padded-slot pools and replay caches laid out as the V2 manager carves
them, the counts request-indexed as the manager gathers them. The committed state rows are compared against an
fp32 sequential replay; every cache, every other slot, a slot with nothing pending and the slot padding must not
move at all.
"""

import pytest
import torch
from _kda_ops_util import (
    CONV_WIDTH,
    HEAD_DIM,
    assert_scaled_close,
    bits,
    generator,
    padded_slots,
    replay_caches,
)

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.ssm.kda_mtp_commit_pending_drafts import (
    kda_mtp_commit_pending_drafts as op,
)

__extra_import_path__ = ["../../../.."]
from _validating import certifying, validating  # noqa: E402,F401 — path above; fixture by name

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() not in ((10, 0), (10, 3), (10, 7)),
    reason="the KDA verify kernel needs sm_100, sm_103 or sm_107",
)

# One tensor-core pass over the replayed chunk against an fp32 sequential replay, relative to the largest entry.
# Measured over the cells on sm_103: 1.3e-5.
TOL_STATE = 2e-4
CACHES = ("cs_q", "cs_k", "cs_v", "k_cache", "g_cache", "v_cache", "beta_cache")


def _inputs(spec, gen):
    heads, num_spec = spec["heads"], spec["num_spec"]
    num_slots = max(spec["slots"]) + 3
    caches = replay_caches(num_slots, heads, num_spec, gen)
    state = padded_slots(num_slots, (heads, HEAD_DIM, HEAD_DIM), torch.float32)
    state.copy_(torch.randn(state.shape, generator=gen, device="cuda") * 0.05)
    slots = torch.tensor(spec["slots"], dtype=torch.int32, device="cuda")
    accepted = torch.tensor(spec["accepted"], dtype=torch.int32, device="cuda")
    return state, caches, slots, accepted


def _args(state, caches, slots, accepted, num_spec):
    return (state, *(caches[name] for name in CACHES), slots, accepted, num_spec, CONV_WIDTH)


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:60] for c in op.CELLS])
@torch.inference_mode()
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    certifying(op)
    spec = cell.spec
    state, caches, slots, accepted = _inputs(spec, generator(cell))
    # The storage range the slot-strided state view spans, padding between slots included (read through the
    # view: tensors made under inference mode keep no `_base`).
    row = state[0].numel()
    span = torch.as_strided(state, (state.shape[0], state.stride(0)), (state.stride(0), 1))
    before_span = span.clone()
    before_caches = {name: caches[name].clone() for name in CACHES}
    expected = state.clone()  # the reference leaves a slot with nothing to replay as it was

    op(*_args(state, caches, slots, accepted, spec["num_spec"]))
    op.reference(*_args(expected, caches, slots, accepted, spec["num_spec"]))

    pending = dict(zip(spec["slots"], spec["accepted"]))
    for slot in range(state.shape[0]):
        if pending.get(slot, 0):
            assert_scaled_close(state[slot], expected[slot], TOL_STATE, f"state[{slot}]")
        else:
            # Unselected, or selected with nothing pending: committed unchanged, bit for bit.
            assert torch.equal(bits(state[slot]), bits(expected[slot])), f"state[{slot}] moved"
    assert torch.equal(bits(span[:, row:]), bits(before_span[:, row:])), "slot padding written"
    for name in CACHES:
        assert torch.equal(bits(caches[name]), bits(before_caches[name])), (
            f"{name} must be read only"
        )


@torch.inference_mode()
def test_guard_refuses_repeated_slots() -> None:
    spec = dict(heads=6, num_spec=4, slots=(1, 1), accepted=(1, 2))
    state, caches, slots, accepted = _inputs(spec, torch.Generator("cuda").manual_seed(3))
    with pytest.raises(AssertionError, match="each slot once"):
        with validating(op):
            op(*_args(state, caches, slots, accepted, 4))


@torch.inference_mode()
def test_guard_refuses_slots_outside_the_pool() -> None:
    spec = dict(heads=6, num_spec=4, slots=(1,), accepted=(1,))
    state, caches, _, accepted = _inputs(spec, torch.Generator("cuda").manual_seed(4))
    slots = torch.tensor([state.shape[0]], dtype=torch.int32, device="cuda")
    with pytest.raises(AssertionError, match="outside the state pool"):
        with validating(op):
            op(*_args(state, caches, slots, accepted, 4))


@torch.inference_mode()
def test_guard_refuses_pending_beyond_num_spec() -> None:
    spec = dict(heads=6, num_spec=4, slots=(1, 2), accepted=(5, 0))
    state, caches, slots, accepted = _inputs(spec, torch.Generator("cuda").manual_seed(5))
    with pytest.raises(AssertionError, match=r"\[0, num_spec\]"):
        with validating(op):
            op(*_args(state, caches, slots, accepted, 4))
