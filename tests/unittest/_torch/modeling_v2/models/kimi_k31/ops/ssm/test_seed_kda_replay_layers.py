# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the Kimi K3.1 seed_kda_replay_layers entry.

Each buffer kind lives in one slot-page pool that holds every layer's buffer of that kind, with padding between the
layers and between the slots, and the op gets the `[layers, slots, ...]` view over it, as the cache manager stacks
the V2 manager's per-layer views. Every written buffer, bit for bit, against the reference on same-layout copies;
the live pool, every other slot and the padding must not move.
"""

import math

import pytest
import torch
from _kda_ops_util import COMMITTED, HEAD_DIM, bits, generator

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.ssm.seed_kda_replay_layers import (
    seed_kda_replay_layers as op,
)

__extra_import_path__ = ["../../../.."]
from _validating import certifying, validating  # noqa: E402,F401 — path above; fixture by name

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

# Bytes between one layer's buffer and the next inside a slot page, and after the last one (multiples of 16, as
# the V2 manager keeps sub-pages 16-byte aligned).
LAYER_PAD = 32
SLOT_PAD = 48


def _stacked_pool(layers, slots, shape, dtype, gen):
    """(storage, `[layers, slots, *shape]` view) with every layer's buffer of a slot in one padded page."""
    itemsize = torch.empty((), dtype=dtype).element_size()
    elements = math.prod(shape)
    layer_stride = elements + LAYER_PAD // itemsize
    slot_stride = layers * layer_stride + SLOT_PAD // itemsize
    storage = torch.randn(slots * slot_stride, generator=gen, device="cuda").to(dtype)
    inner = [math.prod(shape[i + 1 :]) for i in range(len(shape))]
    view = storage.as_strided((layers, slots, *shape), (layer_stride, slot_stride, *inner))
    return storage, view


def _inputs(spec, gen):
    heads, layers, num_slots = spec["heads"], spec["layers"], spec["pool_slots"]
    num_spec = spec["num_spec"]
    dim = heads * HEAD_DIM
    storages = {}
    storages["conv"], conv = _stacked_pool(
        layers, num_slots, (3 * dim, COMMITTED), torch.bfloat16, gen
    )
    caches = []
    for name in ("cs_q", "cs_k", "cs_v"):
        storages[name], view = _stacked_pool(
            layers, num_slots, (COMMITTED + num_spec, dim), torch.bfloat16, gen
        )
        caches.append(view.transpose(-1, -2))
    zeroed = []
    for name, dtype in (
        ("k_cache", torch.bfloat16),
        ("g_cache", torch.float32),
        ("v_cache", torch.bfloat16),
    ):
        storages[name], view = _stacked_pool(layers, num_slots, (num_spec, dim), dtype, gen)
        zeroed.append(view)
    padded_heads = (heads + 7) // 8 * 8
    storages["beta_cache"], beta = _stacked_pool(
        layers, num_slots, (num_spec, padded_heads), torch.bfloat16, gen
    )
    zeroed.append(beta[..., :heads])
    indices = torch.tensor(spec["slots"], dtype=torch.int64, device="cuda")
    return storages, conv, caches, zeroed, indices


def _same_layout_copy(storages, conv, caches, zeroed):
    """The same views over cloned storages."""
    copies = {name: storage.clone() for name, storage in storages.items()}

    def rebase(view, name):
        return copies[name].as_strided(view.shape, view.stride(), view.storage_offset())

    names = ("cs_q", "cs_k", "cs_v")
    return (
        copies,
        rebase(conv, "conv"),
        [rebase(cache, name) for cache, name in zip(caches, names)],
        [
            rebase(buffer, name)
            for buffer, name in zip(zeroed, ("k_cache", "g_cache", "v_cache", "beta_cache"))
        ],
    )


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:60] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    certifying(op)
    storages, conv, caches, zeroed, indices = _inputs(cell.spec, generator(cell))
    before = {name: storage.clone() for name, storage in storages.items()}
    ref_storages, ref_conv, ref_caches, ref_zeroed = _same_layout_copy(
        storages, conv, caches, zeroed
    )

    with torch.inference_mode():
        op(conv, *caches, zeroed, indices)
    op.reference(ref_conv, *ref_caches, ref_zeroed, indices)
    torch.cuda.synchronize()

    for name, storage in storages.items():
        assert torch.equal(bits(storage), bits(ref_storages[name])), (
            f"{name} differs from the reference"
        )
    assert torch.equal(bits(storages["conv"]), bits(before["conv"])), (
        "the live pool must be read only"
    )
    # The seeded slots changed (the inputs are random), and nothing outside the viewed elements did: the
    # reference writes only the views, so equality above covers the padding and the other slots.
    for cache in caches:
        assert cache[:, indices, :, COMMITTED:].eq(0).all()
    for buffer in zeroed:
        assert buffer[:, indices].eq(0).all()


def test_guard_refuses_buffers_stacked_over_other_layer_counts() -> None:
    spec = dict(heads=6, num_spec=4, layers=3, slots=(1,), pool_slots=4)
    _, conv, caches, zeroed, indices = _inputs(spec, torch.Generator("cuda").manual_seed(3))
    with pytest.raises(AssertionError, match="same layer dimension"):
        with validating(op):
            op(conv, *caches, [zeroed[0][:2], *zeroed[1:]], indices)
