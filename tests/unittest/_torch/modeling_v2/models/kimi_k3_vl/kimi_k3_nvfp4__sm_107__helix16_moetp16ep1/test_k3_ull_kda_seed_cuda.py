# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CUDA tests of the ULL target's stacked KDA checkpoint seeding."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

_DIR = Path(__file__).resolve().parent


def _load_sibling(name: str) -> ModuleType:
    path = _DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_k3_ull_cuda_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_cache_tests = _load_sibling("test_k3_ull_cache_manager")
_ckpt_tests = _load_sibling("test_k3_ull_kda_checkpoints")


@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("wrapped", [False, True])
def test_transfer_seed_runs_on_all_layers_at_once_cuda(
    state_dtype: torch.dtype, wrapped: bool
) -> None:
    """On CUDA coalesced slots (including wrapped per-layer buffers), stacked seeding matches the
    layer-by-layer seed bitwise."""
    pools = []
    for stacked in (True, False):
        _, state, pool, _, _ = _cache_tests._bound_state(
            state_dtype, 16, device="cuda", wrapped=wrapped
        )
        assert state._stacked_seed is not None
        if not stacked:
            state._stacked_seed = None
        state.prev_num_accepted_tokens[[4, 1, 2]] = 2
        before = pool.slots.clone()
        state.seed_transferred_slots(torch.tensor([4, 1], dtype=torch.int32, device="cuda"))
        assert state.prev_num_accepted_tokens[[4, 1, 2]].tolist() == [0, 0, 2]
        assert not torch.equal(pool.slots, before)
        pools.append(pool.slots.clone())
    assert torch.equal(pools[0], pools[1])


@pytest.mark.parametrize(
    ("state_dtype", "ssm_dtype", "num_layers", "seeded"),
    [
        pytest.param(torch.float32, torch.float32, 1, [0], id="fp32-one-slot"),
        pytest.param(torch.bfloat16, torch.bfloat16, 1, [0], id="bf16-one-slot"),
        pytest.param(torch.bfloat16, torch.float32, 1, [0], id="cast-one-slot"),
        pytest.param(torch.float32, torch.float32, 69, [4, 0, 2], id="fp32-sparse-slots"),
        pytest.param(torch.bfloat16, torch.bfloat16, 69, [4, 0, 2], id="bf16-sparse-slots"),
        pytest.param(torch.bfloat16, torch.float32, 69, [4, 0, 2], id="cast-sparse-slots"),
        pytest.param(torch.bfloat16, torch.float32, 3, [3, 1], id="cast-intermediate-layers"),
        pytest.param(torch.float32, torch.float32, 3, [], id="empty-slots"),
    ],
)
def test_stacked_seed_is_the_layer_loop_bitwise_cuda(
    state_dtype: torch.dtype,
    ssm_dtype: torch.dtype,
    num_layers: int,
    seeded: list[int],
) -> None:
    """Stacked CUDA seeding writes the exact bytes of seed_kda_checkpoints layer by layer."""
    layout = _ckpt_tests._layout(state_dtype)
    loop = _ckpt_tests._coalesced_layers(
        num_layers, layout, ssm_dtype, "cuda", torch.Generator().manual_seed(7)
    )
    batched = _ckpt_tests._coalesced_layers(
        num_layers, layout, ssm_dtype, "cuda", torch.Generator().manual_seed(7)
    )
    (loop_backing, loop_slots), (batched_backing, batched_slots) = loop[0], batched[0]
    assert torch.equal(loop_slots, batched_slots)
    before = loop_slots.clone()
    slots = torch.tensor(seeded, dtype=torch.int64, device="cuda")

    _, ssm, conv, views = loop
    for i in range(num_layers):
        _ckpt_tests.ck.seed_kda_checkpoints(views[i].ckpt, views[i].vconv, ssm[i], conv[i], slots)
    _ckpt_tests._seed_stacked(*batched[1:], slots)

    assert torch.equal(loop_slots, batched_slots)
    assert torch.equal(loop_slots, before) == (not seeded)
    for backing, region in ((loop_backing, loop_slots), (batched_backing, batched_slots)):
        assert backing.sum(dtype=torch.int64) == region.sum(dtype=torch.int64)
