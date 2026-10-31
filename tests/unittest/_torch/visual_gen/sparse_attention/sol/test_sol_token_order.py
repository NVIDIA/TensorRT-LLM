# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the Morton (Z-order) and Hilbert token orderings used by SOL block routing.

These are CPU-only: each permutation is a fixed index tensor per latent grid.
The Wan forward equivalence (reordered output == native output under dense
attention) lives in ``test_wan_transformer.py``.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch
from pydantic import ValidationError

from tensorrt_llm._torch.visual_gen.token_order import (
    TokenReorder,
    hilbert3d_permutation,
    morton3d_permutation,
    require_native_token_order,
    resolve_token_order,
)
from tensorrt_llm.visual_gen.sparse_attention import SolAttentionConfig

_CPU_ONLY = pytest.mark.cpu_only


@_CPU_ONLY
def test_token_order_defaults_to_native_and_rejects_unknown():
    assert SolAttentionConfig(tau=2.0).token_order == "native"
    assert SolAttentionConfig(tau=2.0, token_order="morton").token_order == "morton"
    with pytest.raises(ValidationError):
        SolAttentionConfig(tau=2.0, token_order="peano")


@_CPU_ONLY
def test_resolve_token_order_reads_config_or_defaults():
    assert resolve_token_order(None) == "native"
    assert resolve_token_order(SolAttentionConfig(tau=2.0)) == "native"
    assert resolve_token_order(SolAttentionConfig(tau=2.0, token_order="morton")) == "morton"


@_CPU_ONLY
def test_require_native_token_order_rejects_morton_for_other_models():
    require_native_token_order(None, "FLUX")
    require_native_token_order(SolAttentionConfig(tau=2.0), "FLUX")
    with pytest.raises(ValueError, match="FLUX"):
        require_native_token_order(SolAttentionConfig(tau=2.0, token_order="morton"), "FLUX")


@_CPU_ONLY
def test_base_diffusion_model_rejects_non_native_order_unless_supported():
    """A model that does not declare ``_supports_token_order`` rejects a non-native
    order at construction; the native order is always accepted."""
    from tensorrt_llm._torch.visual_gen.models.modeling import BaseDiffusionModel
    from tensorrt_llm.visual_gen.args import AttentionConfig

    class _StubModel(BaseDiffusionModel):
        pass

    def _config(order: str) -> SimpleNamespace:
        return SimpleNamespace(
            parallel=SimpleNamespace(token_sharded_tp=False),
            attention=AttentionConfig(
                backend="CUTEDSL",
                sparse_attention_config=SolAttentionConfig(tau=2.0, token_order=order),
            ),
            component_name="transformer",
            pretrained_config=SimpleNamespace(),
        )

    _StubModel(_config("native"))
    for order in ("morton", "hilbert"):
        with pytest.raises(ValueError, match="_StubModel"):
            _StubModel(_config(order))


@_CPU_ONLY
def test_wan_transformer_declares_token_order_support():
    from tensorrt_llm._torch.visual_gen.models.wan.transformer_wan import WanTransformer3DModel

    assert WanTransformer3DModel._supports_token_order


@_CPU_ONLY
@pytest.mark.parametrize("grid", [(1, 32, 32), (4, 8, 8), (21, 30, 52), (3, 5, 7)])
def test_morton3d_permutation_round_trips(grid):
    perm, inverse = morton3d_permutation(grid, torch.device("cpu"))
    n = math.prod(grid)
    assert perm.shape == (n,) and inverse.shape == (n,)
    assert torch.equal(torch.sort(perm).values, torch.arange(n))
    assert torch.equal(perm[inverse], torch.arange(n))
    x = torch.randn(2, n, 3)
    assert torch.equal(x.index_select(1, perm).index_select(1, inverse), x)


@_CPU_ONLY
def test_morton3d_blocks_cover_compact_cubes():
    """On a power-of-two grid every 64-token block is exactly one 4x4x4 cube."""
    frames, height, width = 4, 8, 8
    perm, _ = morton3d_permutation((frames, height, width), torch.device("cpu"))
    t = perm // (height * width)
    h = (perm % (height * width)) // width
    w = perm % width
    for start in range(0, perm.numel(), 64):
        sl = slice(start, start + 64)
        for axis in (t, h, w):
            assert int(axis[sl].max() - axis[sl].min()) == 3
    # Native raster order, by contrast, keeps a 64-token block inside one frame.
    native = torch.arange(perm.numel())
    assert int((native[:64] // (height * width)).max()) == 0


@_CPU_ONLY
@pytest.mark.parametrize("grid", [(1, 32, 32), (4, 8, 8), (21, 30, 52), (3, 5, 7)])
def test_hilbert3d_permutation_round_trips(grid):
    perm, inverse = hilbert3d_permutation(grid, torch.device("cpu"))
    n = math.prod(grid)
    assert perm.shape == (n,) and inverse.shape == (n,)
    assert torch.equal(torch.sort(perm).values, torch.arange(n))
    assert torch.equal(perm[inverse], torch.arange(n))


@_CPU_ONLY
def test_hilbert3d_is_a_continuous_curve_on_a_cube():
    """On a power-of-two cube consecutive tokens are lattice neighbours (L1 distance 1),
    which Morton order violates at every octant jump; every 64-token block is a 4x4x4 cube."""
    side = 8
    perm, _ = hilbert3d_permutation((side, side, side), torch.device("cpu"))
    t = perm // (side * side)
    h = (perm % (side * side)) // side
    w = perm % side
    coords = torch.stack([t, h, w], dim=1)
    steps = (coords[1:] - coords[:-1]).abs().sum(dim=1)
    assert torch.equal(steps, torch.ones_like(steps))
    for start in range(0, perm.numel(), 64):
        sl = slice(start, start + 64)
        for axis in (t, h, w):
            assert int(axis[sl].max() - axis[sl].min()) == 3
    morton, _ = morton3d_permutation((side, side, side), torch.device("cpu"))
    mc = torch.stack(
        [morton // (side * side), (morton % (side * side)) // side, morton % side], dim=1
    )
    assert int((mc[1:] - mc[:-1]).abs().sum(dim=1).max()) > 1


@_CPU_ONLY
def test_token_reorder_hilbert_restores():
    x = torch.randn(1, 21 * 30 * 52, 5)
    hilbert = TokenReorder("hilbert")
    hilbert.prepare((21, 30, 52), x.device)
    assert hilbert.active
    y = hilbert.permute(x)
    assert not torch.equal(y, x)
    assert torch.equal(hilbert.restore(y), x)
    assert SolAttentionConfig(tau=2.0, token_order="hilbert").token_order == "hilbert"


@_CPU_ONLY
def test_token_reorder_native_is_identity_and_morton_restores():
    x = torch.randn(1, 4 * 8 * 8, 5)
    native = TokenReorder("native")
    native.prepare((4, 8, 8), x.device)
    assert not native.active
    assert native.permute(x) is x and native.restore(x) is x and native.permute(None) is None

    morton = TokenReorder("morton")
    morton.prepare((4, 8, 8), x.device)
    assert morton.active
    y = morton.permute(x)
    assert not torch.equal(y, x)
    assert torch.equal(morton.restore(y), x)


@_CPU_ONLY
def test_token_reorder_keeps_indices_of_every_prepared_grid():
    """A captured CUDA graph replays the index pointers it saw; switching grids must not free them."""
    morton = TokenReorder("morton")
    morton.prepare((4, 8, 8), torch.device("cpu"))
    first = morton._perm
    morton.prepare((2, 8, 8), torch.device("cpu"))
    assert morton._perm is not first
    morton.prepare((4, 8, 8), torch.device("cpu"))
    assert morton._perm is first
