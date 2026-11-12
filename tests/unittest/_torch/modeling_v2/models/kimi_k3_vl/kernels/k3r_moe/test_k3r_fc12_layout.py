# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU checks of the Kimi K3 SM107 MoE helpers that launch nothing.

* ``fc12_counter_layout`` / ``fc12_counter_words``: the fused FC12 kernel's counter layout (one
  128-byte line per phase, two with die-local claiming) and the words a caller zeroing them ahead
  of a call allocates (``moe_sort``'s tile bound at the 128-row tile).
* ``fc12_die_sm_words``: die-oblivious claiming without a token floor, on 2-CTA clusters and below
  the floor (decided before the weights are looked at).
* ``tp_slice`` / ``ep_partition``: the experts' tensor- and expert-parallel slices.
* The family MoE sources import no V1 module layer (``_torch.moe.fused_moe``, ``_torch.modules``,
  ``_torch.models``), only kernels and ops.
"""

import ast
from pathlib import Path

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kernels.k3r_moe.fc12 import (
    fc12_counter_layout,
    fc12_counter_words,
    fc12_die_sm_words,
    fc12_max_tiles,
)

pytestmark = pytest.mark.cpu_only

_FAMILY = (
    Path(__file__).resolve().parents[8]
    / "tensorrt_llm/_torch/_experimental/modeling_v2/models/kimi_k3_vl"
)


@pytest.mark.parametrize(
    "num_tiles, die_sched, want",
    [
        (0, False, (0, 32, 64)),
        (1, False, (32, 64, 96)),
        (32, False, (32, 64, 96)),
        (33, False, (64, 96, 128)),
        (1, True, (32, 96, 160)),
        (100, True, (128, 192, 256)),
    ],
)
def test_counter_layout(num_tiles, die_sched, want):
    fc1_at, fc2_at, total = fc12_counter_layout(num_tiles, die_sched=die_sched)
    assert (fc1_at, fc2_at, total) == want
    assert fc1_at >= num_tiles and fc1_at % 32 == 0 and fc2_at % 32 == 0


@pytest.mark.parametrize(
    "num_local, num_tokens, top_k, want",
    [
        (56, 1, 16, 16),  # one token: at most 16 experts, one tile each
        (56, 32, 16, 56 + (512 - 56) // 128),
        (112, 2048, 16, 112 + (32768 - 112) // 128),
        (896, 8, 16, 128),
    ],
)
def test_max_tiles(num_local, num_tokens, top_k, want):
    assert fc12_max_tiles(num_local, num_tokens, top_k) == want


def test_counter_words_cover_every_tile_count():
    for num_local, num_tokens in ((56, 1), (56, 24), (112, 256), (112, 2048)):
        for die in (False, True):
            words = fc12_counter_words(num_local, num_tokens, 16, die)
            bound = fc12_max_tiles(num_local, num_tokens, 16)
            assert words == fc12_counter_layout(bound, die_sched=die)[2]
            for tiles in (1, bound):
                assert fc12_counter_layout(tiles, die_sched=die)[2] <= words


def test_die_words_need_floor_cluster_and_tokens():
    w = torch.empty(4, 8, dtype=torch.int64)
    assert fc12_die_sm_words(w, w, 1, 64, None) is None  # no floor: die-oblivious
    assert fc12_die_sm_words(w, w, 2, 64, 24) is None  # 2-CTA clusters
    assert fc12_die_sm_words(w, w, 1, 23, 24) is None  # below the floor


def test_tp_slice_and_ep_partition():
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.ops.moe.k3r_experts import (
        ep_partition,
        tp_slice,
    )

    t = torch.arange(10 * 3).view(10, 3)
    assert torch.equal(tp_slice(t, 4, 0, 0), t[0:3])
    assert torch.equal(tp_slice(t, 4, 3, 0), t[9:10])
    assert torch.equal(tp_slice(t, 2, 1, 1), t[:, 2:3])
    assert tp_slice(t, 1, 0, 0) is t
    assert tp_slice(t[:1], 4, 2, 0).shape == (1, 3)  # width 1 is not split
    assert ep_partition(896, 16, 7) == (392, 448)
    assert ep_partition(10, 4, 0) == (0, 3) and ep_partition(10, 4, 3) == (8, 10)
    with pytest.raises(ValueError):
        ep_partition(2, 4, 0)


_FAMILY_MOE_SOURCES = (
    "kernels/k3r_moe/fc12.py",
    "kernels/k3r_moe/fc12_runner.py",
    "kernels/k3r_moe/die_locality.py",
    "ops/moe/k3r_experts.py",
)
#: Module layers of V1 the family does not build on (kernels and torch ops are fine).
_V1_LAYERS = (
    "tensorrt_llm._torch.moe.fused_moe",
    "tensorrt_llm._torch.modules",
    "tensorrt_llm._torch.models",
    "tensorrt_llm._torch.moe.routing",
)


@pytest.mark.parametrize("rel", _FAMILY_MOE_SOURCES)
def test_family_moe_imports_no_v1_module_layer(rel):
    tree = ast.parse((_FAMILY / rel).read_text())
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names = [node.module]
        elif isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        else:
            continue
        found += [n for n in names if n.startswith(_V1_LAYERS)]
    assert not found, f"{rel} imports V1 module layers: {found}"
