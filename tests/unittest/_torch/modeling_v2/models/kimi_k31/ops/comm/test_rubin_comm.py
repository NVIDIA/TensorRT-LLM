# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The collective builders of ``ops/comm/rubin_comm.py`` take the group from the mapping given.

On helix8 the MoE runs on the 8 CP ranks as one TP group: the mapping
``Mapping.repurpose_helix_cp_to_tp`` returns. The helix mapping itself has a TP group of one rank
on every rank, and an all-reduce over it would return its input unreduced; the builders refuse it.
"""

from __future__ import annotations

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.comm import rubin_comm
from tensorrt_llm.mapping import CpType, Mapping

pytestmark = [
    pytest.mark.cpu_only,
    pytest.mark.skipif(
        torch.cuda.is_available() and torch.cuda.get_device_capability() != (10, 7),
        reason="certified on SM107 only",
    ),
]


def _helix8(rank: int) -> Mapping:
    return Mapping(
        world_size=8,
        rank=rank,
        tp_size=1,
        cp_size=8,
        cp_config={"cp_type": CpType.HELIX},
        moe_tp_size=2,
        moe_ep_size=4,
    )


class _FakeAllReduce:
    def __init__(self, mapping, strategy, dtype):
        self.mapping, self.strategy, self.dtype = mapping, strategy, dtype


@pytest.mark.parametrize("rank", range(8))
def test_helix8_moe_group_is_the_repurposed_tp8(monkeypatch, rank):
    """The repurposed mapping: TP group of all 8 ranks, MoE ranks (r // 4, r % 4), the expert
    window [224 (r % 4), +224); the all-reduce builder takes it and refuses the helix mapping."""
    monkeypatch.setattr(rubin_comm, "AllReduce", _FakeAllReduce)
    raw = _helix8(rank)
    tp8 = raw.repurpose_helix_cp_to_tp()
    assert (tp8.tp_size, tp8.tp_rank, list(tp8.tp_group)) == (8, rank, list(range(8)))
    assert (tp8.moe_tp_rank, tp8.moe_ep_rank) == (rank // 4, rank % 4)
    local = 896 // tp8.moe_ep_size
    assert (local * tp8.moe_ep_rank, local) == (224 * (rank % 4), 224)
    ar = rubin_comm.rubin_all_reduce(tp8, "AUTO", 8)
    assert ar.mapping is tp8 and ar.dtype == torch.bfloat16
    # The helix mapping: tp 1 on every rank, MoE ranks (0, 0) everywhere.
    assert (raw.tp_size, raw.moe_tp_rank, raw.moe_ep_rank) == (1, 0, 0)
    with pytest.raises(ValueError, match="1 ranks, the layout reduces over 8"):
        rubin_comm.rubin_all_reduce(raw, "AUTO", 8)
    with pytest.raises(ValueError, match="1 ranks, the layout reduces over 8"):
        rubin_comm.rubin_latent_ar_workspace(raw, 8, 3584, 128, 6291456)


def test_tep8_group():
    m = Mapping(world_size=8, rank=3, tp_size=8, moe_tp_size=1, moe_ep_size=8)
    rubin_comm._check_group(m, 8)
    with pytest.raises(ValueError):
        rubin_comm._check_group(m, 16)
