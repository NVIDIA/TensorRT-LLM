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

"""Tests for the Morton / Z-order token ordering used by SOL block routing.

These are CPU-only: the permutation is a fixed index tensor per latent grid.
The Wan forward equivalence (morton output == native output under dense
attention) lives in ``test_wan_transformer.py``.
"""

from __future__ import annotations

import math

import pytest
import torch
from pydantic import ValidationError

from tensorrt_llm._torch.visual_gen.token_order import (
    TokenReorder,
    morton3d_permutation,
    require_native_token_order,
    resolve_token_order,
)
from tensorrt_llm.visual_gen.sparse_attention import SolAttentionConfig


def test_token_order_defaults_to_native_and_rejects_unknown():
    assert SolAttentionConfig(tau=2.0).token_order == "native"
    assert SolAttentionConfig(tau=2.0, token_order="morton").token_order == "morton"
    with pytest.raises(ValidationError):
        SolAttentionConfig(tau=2.0, token_order="hilbert")


def test_resolve_token_order_reads_config_or_defaults():
    assert resolve_token_order(None) == "native"
    assert resolve_token_order(SolAttentionConfig(tau=2.0)) == "native"
    assert resolve_token_order(SolAttentionConfig(tau=2.0, token_order="morton")) == "morton"


def test_require_native_token_order_rejects_morton_for_other_models():
    require_native_token_order(None, "FLUX")
    require_native_token_order(SolAttentionConfig(tau=2.0), "FLUX")
    with pytest.raises(ValueError, match="FLUX"):
        require_native_token_order(SolAttentionConfig(tau=2.0, token_order="morton"), "FLUX")


@pytest.mark.parametrize("grid", [(1, 32, 32), (4, 8, 8), (21, 30, 52), (3, 5, 7)])
def test_morton3d_permutation_round_trips(grid):
    perm, inverse = morton3d_permutation(grid, torch.device("cpu"))
    n = math.prod(grid)
    assert perm.shape == (n,) and inverse.shape == (n,)
    assert torch.equal(torch.sort(perm).values, torch.arange(n))
    assert torch.equal(perm[inverse], torch.arange(n))
    x = torch.randn(2, n, 3)
    assert torch.equal(x.index_select(1, perm).index_select(1, inverse), x)


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


def test_token_reorder_keeps_indices_of_every_prepared_grid():
    """A captured CUDA graph replays the index pointers it saw; switching grids must not free them."""
    morton = TokenReorder("morton")
    morton.prepare((4, 8, 8), torch.device("cpu"))
    first = morton._perm
    morton.prepare((2, 8, 8), torch.device("cpu"))
    assert morton._perm is not first
    morton.prepare((4, 8, 8), torch.device("cpu"))
    assert morton._perm is first
