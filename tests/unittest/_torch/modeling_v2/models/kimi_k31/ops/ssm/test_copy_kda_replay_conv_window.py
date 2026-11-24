# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the Kimi K3.1 copy_kda_replay_conv_window entry.

The live conv pool and the replay conv caches are padded-slot views laid out as the V2 manager carves them. The
caches, bit for bit, against the reference on same-layout copies; the live pool, the pending-draft tail columns,
every other slot and the slot padding must not move.
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

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.ssm.copy_kda_replay_conv_window import (
    copy_kda_replay_conv_window as op,
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
    dtype = torch.int64 if spec.get("index_dtype") == "int64" else torch.int32
    indices = torch.tensor(spec["slots"], dtype=dtype, device="cuda")
    return conv, [caches[name] for name in CACHES], indices


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:60] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    certifying(op)
    conv, caches, indices = _inputs(cell.spec, generator(cell))
    conv_storage, conv_before = conv._base, conv._base.clone()
    storages = [cache._base for cache in caches]
    storages_before = [storage.clone() for storage in storages]
    expected = [clone_strided(cache) for cache in caches]

    op(conv, *caches, indices)
    op.reference(conv, *expected, indices)

    for name, got, ref in zip(CACHES, caches, expected):
        assert torch.equal(bits(got), bits(ref)), f"{name} differs from the reference"
    # Everything outside the copied columns, including the slot padding, is untouched.
    selected = set(cell.spec["slots"])
    row = caches[0].shape[1] * caches[0].shape[2]
    for name, storage, before in zip(CACHES, storages, storages_before):
        assert torch.equal(bits(storage[:, row:]), bits(before[:, row:])), (
            f"{name}: padding written"
        )
        for slot in range(storage.shape[0]):
            if slot not in selected:
                assert torch.equal(bits(storage[slot]), bits(before[slot])), f"{name}[{slot}] moved"
    assert torch.equal(bits(conv_storage), bits(conv_before)), "the live pool must be read only"


def test_guard_refuses_slots_outside_the_pool() -> None:
    spec = dict(heads=6, num_spec=4, slots=(1, -1), pool_slots=4)
    conv, caches, indices = _inputs(spec, torch.Generator("cuda").manual_seed(2))
    with pytest.raises(AssertionError, match="outside the pool"):
        with validating(op):
            op(conv, *caches, indices)
