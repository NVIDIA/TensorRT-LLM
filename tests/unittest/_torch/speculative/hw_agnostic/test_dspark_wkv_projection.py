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
"""Packed backfill WKV projection matches the per-stage projections."""

import pytest
import torch
from torch import nn

from tensorrt_llm._torch.models.modeling_dspark import DSv4DSparkDraftModel
from tensorrt_llm._utils import get_sm_version


def _make_model(
    *,
    stages: int = 3,
    hidden: int = 12,
    head_dim: int = 8,
    window: int = 8,
) -> DSv4DSparkDraftModel:
    torch.manual_seed(41)
    model = DSv4DSparkDraftModel.__new__(DSv4DSparkDraftModel)
    nn.Module.__init__(model)
    model.uses_lagged_pre = True
    model.checkpoint_fp8_block_size = 32
    model._attn_params = {
        "head_dim": head_dim,
        "window_size": window,
        "rope_head_dim": min(head_dim, 64),
        "eps": 1e-6,
    }
    model._freqs_cap = 128
    model._freqs_table_cache = {}
    model._rope_theta = 10000.0
    model._backfill_wkv_weight = None
    model.mtp_layers = nn.ModuleList()
    for _ in range(stages):
        stage = nn.Module()
        stage.input_layernorm = nn.LayerNorm(hidden, dtype=torch.bfloat16)
        stage.main_proj = nn.Identity()
        stage.main_norm = nn.Identity()
        stage._dspark_attn = {
            "wkv": torch.randn(head_dim, hidden, dtype=torch.bfloat16) / hidden**0.5,
            "kv_norm_w": torch.randn(head_dim, dtype=torch.bfloat16),
        }
        model.mtp_layers.append(stage)
    return model


@pytest.mark.skipif(
    not (torch.cuda.is_available() and get_sm_version() in (100, 103, 107)),
    reason="Requires an SM100, SM103 or SM107 GPU",
)
def test_packed_backfill_matches_separate_projection_and_preserves_masked_slots() -> None:
    device = "cuda"
    window = 128
    model = _make_model(head_dim=512, window=window)
    for stage in model.mtp_layers:
        stage._dspark_attn = {k: v.to(device) for k, v in stage._dspark_attn.items()}
    model._pack_backfill_wkv_weights()
    main_hidden = torch.randn(3, 5, 12, dtype=torch.bfloat16, device=device)
    positions = torch.tensor(
        [
            [6, 7, 8, 9, 10],
            [17, 18, 100003, 100004, 100005],
            [100004, 100005, 100006, 100007, 100008],
        ],
        device=device,
    )
    slots = torch.tensor([3, 1, 4], device=device)
    mask = torch.tensor([[True] * 5, [True, True, False, False, False], [False] * 5], device=device)
    initial = torch.randn(6, 3, window, 512, dtype=torch.bfloat16, device=device)
    actual, expected = initial.clone(), initial.clone()
    with torch.inference_mode():
        model.write_context_windows_batched(main_hidden, positions, slots, mask, actual)
        for g, slot in enumerate(slots.tolist()):
            model.write_context_windows(
                main_hidden[g][mask[g]], positions[g][mask[g]], expected[slot]
            )
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
    changed = torch.zeros(6, window, dtype=torch.bool, device=device)
    changed[slots[:, None].expand_as(positions)[mask], (positions % window)[mask]] = True
    untouched = (~changed)[:, None, :, None].expand_as(initial)
    assert torch.equal(actual[untouched], initial[untouched])
