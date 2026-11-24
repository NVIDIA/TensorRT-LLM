# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the Kimi K3.1 store_kda_replay_conv_window entry.

The live conv pool and the replay conv caches are padded-slot views laid out as the V2 manager carves them. The
live pool, bit for bit, against the reference on a same-layout copy; the replay caches, every other slot and the
slot padding must not move.
"""

import pytest
import torch
from _kda_ops_util import (
    COMMITTED,
    HEAD_DIM,
    bits,
    clone_strided,
    generator,
    padded_slots,
    replay_caches,
)

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.ssm.store_kda_replay_conv_window import (
    store_kda_replay_conv_window as op,
)

__extra_import_path__ = ["../../../.."]
from _validating import certifying, validating  # noqa: E402,F401 — path above; fixture by name

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="a Triton kernel: needs a GPU"
)

CACHES = ("cs_q", "cs_k", "cs_v")


def _inputs(spec, gen):
    heads, num_slots = spec["heads"], spec["pool_slots"]
    conv = padded_slots(num_slots, (3 * heads * HEAD_DIM, COMMITTED), torch.bfloat16)
    conv.copy_(torch.randn(conv.shape, generator=gen, device="cuda"))
    caches = replay_caches(num_slots, heads, spec["num_spec"], gen)
    indices = torch.tensor(spec["slots"], dtype=torch.int32, device="cuda")
    accepted = torch.tensor(spec["accepted"], dtype=torch.int32, device="cuda")
    return conv, [caches[name] for name in CACHES], indices, accepted


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:60] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    certifying(op)
    conv, caches, indices, accepted = _inputs(cell.spec, generator(cell))
    storage, before = conv._base, conv._base.clone()
    caches_before = [cache._base.clone() for cache in caches]
    expected = clone_strided(conv)

    op(conv, *caches, indices, accepted)
    op.reference(expected, *caches, indices, accepted)

    assert torch.equal(bits(conv), bits(expected)), "the live pool differs from the reference"
    row = conv[0].numel()
    assert torch.equal(bits(storage[:, row:]), bits(before[:, row:])), "slot padding written"
    selected = set(cell.spec["slots"])
    for slot in range(storage.shape[0]):
        if slot not in selected:
            assert torch.equal(bits(storage[slot]), bits(before[slot])), f"slot {slot} moved"
    for name, cache, cache_before in zip(CACHES, caches, caches_before):
        assert torch.equal(bits(cache._base), bits(cache_before)), f"{name} must be read only"


def _guard(slots, accepted, match) -> None:
    spec = dict(heads=6, num_spec=4, slots=slots, accepted=accepted, pool_slots=4)
    conv, caches, indices, counts = _inputs(spec, torch.Generator("cuda").manual_seed(3))
    with pytest.raises(AssertionError, match=match):
        with validating(op):
            op(conv, *caches, indices, counts)


def test_guard_refuses_slots_outside_the_pool() -> None:
    _guard((1, 4), (0, 1), "outside the pool")


def test_guard_refuses_repeated_slots() -> None:
    _guard((2, 2), (0, 3), "each slot once")
