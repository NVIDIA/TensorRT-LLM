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
"""Nemotron-H weight mapper: Mamba2 in_proj/conv1d row rearrangement through
``Mamba2TpShard``, including B/C group replication when tp_size > n_groups."""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.models.checkpoints.hf.nemotron_h_weight_mapper import (
    NemotronHHfWeightMapper,
)
from tensorrt_llm._torch.modules.mamba.mamba2_tp import Mamba2TpShard
from tensorrt_llm._torch.utils import split

pytestmark = pytest.mark.cpu_only


def _row_ids(rows: int, cols: int) -> torch.Tensor:
    """Each row filled with its own row index, so row origins stay checkable
    after the mapper reorders/replicates them."""
    return torch.arange(rows).unsqueeze(1).expand(rows, cols).clone()


def _make_mapper(
    *,
    tp_size: int,
    tp_rank: int,
    n_groups: int,
    mamba_head_dim: int,
    mamba_num_heads: int,
    ssm_state_size: int,
) -> NemotronHHfWeightMapper:
    mapper = NemotronHHfWeightMapper()
    pretrained = SimpleNamespace(
        mamba_head_dim=mamba_head_dim,
        mamba_num_heads=mamba_num_heads,
        n_groups=n_groups,
        ssm_state_size=ssm_state_size,
        num_hidden_layers=2,
        num_key_value_heads=2,
        tie_word_embeddings=False,
    )
    mapping = SimpleNamespace(tp_size=tp_size, tp_rank=tp_rank, enable_attention_dp=False)
    model_config = SimpleNamespace(
        pretrained_config=pretrained, mapping=mapping, moe_backend="TRTLLM"
    )
    mapper._config = model_config
    mapper._model = SimpleNamespace(model_config=model_config, config=pretrained)
    mapper._tp_size = tp_size
    return mapper


def test_replicated_groups_tp4():
    """n_groups=1 < tp_size=4: every rank replicates the same (only) group."""
    tp_size, tp_rank = 4, 1
    n_groups, d_state, head_dim, nheads = 1, 3, 4, 8
    shard = Mamba2TpShard(
        tp_size=tp_size, nheads=nheads, n_groups=n_groups, head_dim=head_dim, d_state=d_state
    )
    mapper = _make_mapper(
        tp_size=tp_size,
        tp_rank=tp_rank,
        n_groups=n_groups,
        mamba_head_dim=head_dim,
        mamba_num_heads=nheads,
        ssm_state_size=d_state,
    )

    in_proj_rows = 2 * 32 + 2 * 3 + 8  # z + x + B + C + dt = 78
    conv1d_rows = 32 + 2 * 3  # x + B + C = 38

    weights = {
        "backbone.layers.0.mixer.in_proj.weight": _row_ids(in_proj_rows, 5),
        "backbone.layers.0.mixer.conv1d.weight": _row_ids(conv1d_rows, 4).unsqueeze(1),
        "backbone.layers.0.mixer.conv1d.bias": torch.arange(38.0),
        "backbone.layers.0.mixer.A_log": torch.arange(8.0) * 0.1,
        "backbone.layers.0.mixer.D": torch.arange(8.0),
        "backbone.layers.0.mixer.dt_bias": torch.arange(8.0),
        "backbone.layers.0.mixer.norm.weight": torch.arange(32.0),
    }

    out = mapper.preprocess_weights(weights)

    in_proj = out["model.layers.0.mixer.in_proj.weight"]
    assert in_proj.shape == (shard.padded_d_in_proj, 5) == (4 * 24, 5)
    for r in range(tp_size):
        start = r * shard.tp_d_in_proj
        block = in_proj[start : start + shard.tp_d_in_proj, 0].tolist()
        expected = (
            list(range(r * 8, (r + 1) * 8))
            + list(range(32 + r * 8, 32 + (r + 1) * 8))
            + [64, 65, 66]
            + [67, 68, 69]
            + [70 + 2 * r, 71 + 2 * r]
        )
        assert block == expected

    conv1d_w = out["model.layers.0.mixer.conv1d.weight"]
    assert conv1d_w.shape == (shard.padded_conv_dim, 4) == (4 * 14, 4)
    for r in range(tp_size):
        start = r * shard.tp_conv_dim
        block = conv1d_w[start : start + shard.tp_conv_dim, 0].tolist()
        expected = list(range(r * 8, (r + 1) * 8)) + [32, 33, 34] + [35, 36, 37]
        assert block == expected

    conv1d_b = out["model.layers.0.mixer.conv1d.bias"]
    assert conv1d_b.shape == (56,)
    for r in range(tp_size):
        start = r * shard.tp_conv_dim
        block = conv1d_b[start : start + shard.tp_conv_dim].tolist()
        expected = list(range(r * 8, (r + 1) * 8)) + [32, 33, 34] + [35, 36, 37]
        assert block == expected

    a = out["model.layers.0.mixer.A"]
    torch.testing.assert_close(a, -torch.exp(torch.arange(8.0) * 0.1)[2:4])
    assert a.dtype == torch.float32

    torch.testing.assert_close(out["model.layers.0.mixer.D"], torch.tensor([2.0, 3.0]))
    torch.testing.assert_close(out["model.layers.0.mixer.dt_bias"], torch.tensor([2.0, 3.0]))
    torch.testing.assert_close(out["model.layers.0.mixer.norm.weight"], torch.arange(8.0, 16.0))


def test_even_groups_match_legacy_split():
    """n_groups=8 % tp_size=4 == 0: must stay bit-identical to the legacy
    per-segment split-and-concat (no replication)."""
    tp_size, tp_rank = 4, 0
    n_groups, d_state, head_dim, nheads = 8, 3, 4, 8
    d_inner = head_dim * nheads  # 32
    grouped = n_groups * d_state  # 24

    mapper = _make_mapper(
        tp_size=tp_size,
        tp_rank=tp_rank,
        n_groups=n_groups,
        mamba_head_dim=head_dim,
        mamba_num_heads=nheads,
        ssm_state_size=d_state,
    )

    torch.manual_seed(0)
    in_proj_w = torch.randn(2 * d_inner + 2 * grouped + nheads, 5)
    conv1d_w = torch.randn(d_inner + 2 * grouped, 1, 4)

    out = mapper.preprocess_weights(
        {
            "backbone.layers.0.mixer.in_proj.weight": in_proj_w,
            "backbone.layers.0.mixer.conv1d.weight": conv1d_w,
        }
    )

    def _legacy_split(w: torch.Tensor, sizes: list[int]) -> torch.Tensor:
        segs = torch.split(w, sizes, dim=0)
        per_rank = [torch.cat([split(seg, tp_size, r) for seg in segs]) for r in range(tp_size)]
        return torch.cat(per_rank).contiguous()

    legacy_in_proj = _legacy_split(in_proj_w, [d_inner, d_inner, grouped, grouped, nheads])
    torch.testing.assert_close(out["model.layers.0.mixer.in_proj.weight"], legacy_in_proj)

    legacy_conv1d = _legacy_split(conv1d_w.squeeze(1), [d_inner, grouped, grouped])
    torch.testing.assert_close(out["model.layers.0.mixer.conv1d.weight"], legacy_conv1d)


def test_in_proj_per_row_scale_follows_weight():
    """A per-row in_proj scale (NVFP4/FP8 block scale) is rearranged exactly
    like the weight; a per-tensor scalar scale passes through untouched."""
    tp_size, tp_rank = 4, 1
    n_groups, d_state, head_dim, nheads = 1, 3, 4, 8
    shard = Mamba2TpShard(
        tp_size=tp_size, nheads=nheads, n_groups=n_groups, head_dim=head_dim, d_state=d_state
    )
    mapper = _make_mapper(
        tp_size=tp_size,
        tp_rank=tp_rank,
        n_groups=n_groups,
        mamba_head_dim=head_dim,
        mamba_num_heads=nheads,
        ssm_state_size=d_state,
    )

    in_proj_rows = 2 * 32 + 2 * 3 + 8  # 78, equals d_in_proj for this config
    out = mapper.preprocess_weights(
        {
            "backbone.layers.0.mixer.in_proj.weight_scale": _row_ids(in_proj_rows, 1),
            "backbone.layers.0.mixer.in_proj.weight_scale_2": torch.tensor(1.5),
        }
    )

    scale = out["model.layers.0.mixer.in_proj.weight_scale"]
    assert scale.shape == (shard.padded_d_in_proj, 1)
    for r in range(tp_size):
        start = r * shard.tp_d_in_proj
        block = scale[start : start + shard.tp_d_in_proj, 0].tolist()
        expected = (
            list(range(r * 8, (r + 1) * 8))
            + list(range(32 + r * 8, 32 + (r + 1) * 8))
            + [64, 65, 66]
            + [67, 68, 69]
            + [70 + 2 * r, 71 + 2 * r]
        )
        assert block == expected

    torch.testing.assert_close(
        out["model.layers.0.mixer.in_proj.weight_scale_2"], torch.tensor(1.5)
    )
