# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The drafter's full-context decode (M3DrafterFullCtxFmha) is selected through FmhaManager, which caches a library's
support per cell of the generation request count (a ceiling bucket of _FMHA_CACHE_BATCH_SIZE_GRID, e.g. 41..47 -> 47).
The runner supports exact (query rows, requests) recipes, so every recipe's request count must be alone in its cell:
otherwise the first count seen in a cell decides for the others (a 46-request recipe would share 47's cell, and an
eager 41-request pass would reach the runner without a plan). CPU only."""

import pytest

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_107__tp2 import (
    modeling,
)
from tensorrt_llm._torch.attention.backends.fmha import manager

pytestmark = pytest.mark.cpu_only


def _cell(value: int, grid) -> int:
    return manager._normalize_fmha_cache_grid_value(value, grid)


def test_every_drafter_recipe_is_alone_in_its_cache_cells():
    keys = set(modeling.FULLCTX_DECODE_RECIPES)
    batch, qlen = manager._FMHA_CACHE_BATCH_SIZE_GRID, manager._FMHA_CACHE_SEQ_LEN_Q_GRID
    for q, n in keys:
        same_batch = {m for m in range(1, 4 * batch[-1]) if _cell(m, batch) == _cell(n, batch)}
        same_qlen = {r for r in range(1, 4 * qlen[-1]) if _cell(r, qlen) == _cell(q, qlen)}
        assert same_batch == {n}, (
            f"({q}, {n}): requests {sorted(same_batch)} share its FmhaManager cell"
        )
        assert same_qlen == {q}, (
            f"({q}, {n}): query rows {sorted(same_qlen)} share its FmhaManager cell"
        )
