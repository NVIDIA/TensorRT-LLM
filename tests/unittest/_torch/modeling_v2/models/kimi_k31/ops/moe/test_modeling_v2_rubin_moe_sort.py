# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""rubin_moe_sort, the CuteDSL context path's sort of the routed pairs into expert tiles.

On SM103 and SM107: a sample of the cells against the reference (SM103 runs the engine's Blackwell kernel,
whose tiles are 128 and 256).
"""

import zlib

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.moe.rubin_moe_sort import rubin_moe_sort

__extra_import_path__ = [".", "../../../..", "../.."]
import _rubin_nvfp4_moe_inputs as moe_inputs  # noqa: E402 — needs the path declared above
from _k31_utils import cpu_or_sm, on_sm, sample_cells
from _validating import certifying  # noqa: E402,F401 -- path above

pytestmark = cpu_or_sm(103, 107)
gpu_10x = on_sm(103, 107)


def _same_sort(got, want, tile):
    """Tile arrays and counts equal; within each expert's run the same set of pairs."""
    t2e, t2l, e2p, p2e, padded, n = got
    k = int(n[0])
    assert int(n[0]) == int(want[5][0]) and int(padded[0]) == int(want[4][0])
    assert torch.equal(t2e[:k].cpu(), want[0][:k].cpu()) and torch.equal(
        t2l[:k].cpu(), want[1][:k].cpu()
    )
    live = torch.arange(k * tile, device=p2e.device)
    live = live[live < t2l.long()[live // tile]]
    for tile_i in range(k):
        mine = live[live // tile == tile_i]
        assert sorted(p2e[mine].tolist()) == sorted(want[3].to(p2e.device)[mine].tolist())
    assert torch.equal((e2p >= 0).cpu(), (want[2] >= 0).cpu())


CELLS = sample_cells(rubin_moe_sort.CELLS, size_key="tokens")[:2]


@gpu_10x
@pytest.mark.parametrize("cell", CELLS, ids=[c.why[:40] for c in CELLS])
def test_sort_cells(cell, certifying):  # noqa: F811
    spec = cell.spec
    moe_inputs.skip_tiles_the_gpu_lacks(spec["tile_tokens_dim"])
    certifying(rubin_moe_sort)
    ids, weights = moe_inputs.route(
        spec["tokens"], spec["num_experts"], spec["top_k"], zlib.crc32(cell.why.encode()), "cuda"
    )
    args = (
        ids,
        weights,
        spec["num_experts"],
        spec["top_k"],
        spec["local_expert_offset"],
        spec["local_num_experts"],
        spec["tile_tokens_dim"],
    )
    _same_sort(rubin_moe_sort(*args), rubin_moe_sort.reference(*args), spec["tile_tokens_dim"])
