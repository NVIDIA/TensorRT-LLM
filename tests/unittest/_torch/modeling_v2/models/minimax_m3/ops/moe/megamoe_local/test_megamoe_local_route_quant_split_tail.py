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
"""Tests for the split padded tail of ``MegaMoECuteDslLocal.try_fused_route_quant``.

Above ``ROUTEQ_SPLIT_TAIL_MIN_CELLS`` padded cells the launch stages only the live rows and
two fills reset the tail; every staged byte must equal what the launch alone writes.
"""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.native import family_ops
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.ops.moe.megamoe_local import (
    local_impl as local_mod,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.ops.moe.megamoe_local import (
    local_plan as plan,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.ops.moe.megamoe_local.local_impl import (
    ROUTEQ_SPLIT_TAIL_MIN_CELLS,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.ops.moe.megamoe_local.local_impl import (
    TrtllmCutedslMegaMoeLocalNvfp4Impl as Local,
)


def _set_family_op(monkeypatch, name, fn):
    """Replace, for this test, one of the family's native ops the backend calls through ``local_impl.family_ops``."""
    ops = getattr(local_mod.family_ops, "fake_ops", None)
    if ops is None:
        ops = SimpleNamespace()

        def fake_family_ops():
            return ops

        fake_family_ops.fake_ops = ops
        monkeypatch.setattr(local_mod, "family_ops", fake_family_ops)
    setattr(ops, name, fn)


def _family_op_available(name):
    """Whether the family's native library is loaded here and carries the op ``name``."""
    try:
        return hasattr(family_ops(), name)
    except Exception:  # noqa: BLE001 - no library, no op
        return False


TOP_K, EXPERTS = 4, 128


def _leaf(monkeypatch, bufs, split_tail_enabled, device="cpu"):
    leaf = Local.__new__(Local)
    leaf.apply_topk_in_fc1 = False
    leaf._pad_mask = None
    leaf._last_staged_T = {}
    leaf._prestaged_route_quant = None
    leaf._route_quant_split_tail_enabled = split_tail_enabled
    leaf._route_quant_port_enabled = False
    leaf._route_quant_empty_stage = {}
    leaf._active_launch_max_tokens = bufs.topk_idx_local.shape[0]
    bias = (torch.arange(EXPERTS, dtype=torch.float32, device=device) % 7) * 0.01
    leaf.routing_method = SimpleNamespace(
        routing_impl=SimpleNamespace(top_k=TOP_K, routed_scaling_factor=2.0),
        e_score_correction_bias=bias,
    )
    leaf.fc31_input_scale = torch.ones(1, device=device)
    cap = bufs.topk_idx_local.shape[0]
    monkeypatch.setattr(leaf, "_route_quant_decline_reason", lambda x, logits: None)
    monkeypatch.setattr(leaf, "_select_launch_max_tokens", lambda max_tokens: cap)
    monkeypatch.setattr(leaf, "_acquire_buffers", lambda **kwargs: bufs)
    return leaf


def _bufs(cap, hidden, device="cpu"):
    bufs = SimpleNamespace(
        topk_idx_local=torch.empty(cap, TOP_K, dtype=torch.int32, device=device),
        topk_weights=torch.empty(cap, 8, dtype=torch.float32, device=device),
        activation=torch.empty(cap, hidden // 2, dtype=torch.uint8, device=device),
        activation_sf=torch.empty(cap, hidden // 16, dtype=torch.uint8, device=device),
    )
    bufs.topk_idx_local.fill_(7)
    bufs.topk_weights.fill_(0.5)
    bufs.activation.fill_(0x5A)
    bufs.activation_sf.fill_(0x3C)
    return bufs


@pytest.mark.parametrize(
    "cap, rows, enabled, split",
    [
        (2048, 6, True, True),
        (2048, 6, False, False),
        (ROUTEQ_SPLIT_TAIL_MIN_CELLS // TOP_K + 6, 6, True, False),
        (ROUTEQ_SPLIT_TAIL_MIN_CELLS // TOP_K + 7, 6, True, True),
        (2048, 2048, True, False),
    ],
)
def test_the_launch_stages_only_the_live_rows_above_the_cell_threshold(
    monkeypatch, cap, rows, enabled, split
):
    hidden = 64
    seen = {}

    def stage_out(*args, **kwargs):
        seen["idx_rows"], seen["w_rows"] = args[7].shape[0], args[8].shape[0]

    _set_family_op(monkeypatch, "noaux_tc_fp4_quantize_stage_out", stage_out)
    bufs = _bufs(cap, hidden)
    leaf = _leaf(monkeypatch, bufs, enabled)
    staged = leaf.try_fused_route_quant(
        torch.zeros(rows, hidden, dtype=torch.bfloat16), torch.zeros(rows, EXPERTS)
    )
    assert staged is not None
    staged_rows = rows if split else cap
    assert seen == {"idx_rows": staged_rows, "w_rows": staged_rows}
    tail_idx, tail_w = bufs.topk_idx_local[rows:], bufs.topk_weights[rows:]
    if split:
        assert bool((tail_idx == -1).all()) and bool((tail_w[:, :TOP_K] == 0).all())
        assert bool((tail_w[:, TOP_K:] == 0.5).all())
    else:
        assert bool((tail_idx == 7).all()) and bool((tail_w == 0.5).all())


@pytest.mark.parametrize(
    "environ, enabled",
    [
        ({}, True),
        ({"TRTLLM_M31_DISABLE_MEGAMOE_LOCAL_ROUTEQ_SPLIT_TAIL": "1"}, False),
        ({"TRTLLM_M31_DISABLE_MEGAMOE_LOCAL": "1"}, False),
    ],
)
def test_the_split_tail_lever_is_on_unless_disabled(environ, enabled):
    assert plan.lever_enabled("ROUTEQ_SPLIT_TAIL", environ) is enabled


def _route_quant_op_available():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10:
        return False
    import tensorrt_llm  # noqa: F401

    return _family_op_available("noaux_tc_fp4_quantize_stage_out")


@pytest.mark.skipif(not _route_quant_op_available(), reason="needs the SM100+ route-quant-stage op")
@pytest.mark.parametrize(
    "rows, padded", [(1, False), (8, True), (1024, False), (4096, True), (12288, False)]
)
def test_the_split_tail_stages_the_same_bytes_as_the_launch_alone(monkeypatch, rows, padded):
    cap, hidden = 16384, 6144
    gen = torch.Generator(device="cuda").manual_seed(rows)
    x = (torch.randn(rows, hidden, generator=gen, device="cuda") * 2).to(torch.bfloat16)
    logits = torch.randn(rows, EXPERTS, generator=gen, device="cuda")
    mask = None
    if padded:
        mask = torch.zeros(rows, dtype=torch.uint8, device="cuda")
        mask[1::3] = 1
    staged = {}
    for enabled in (False, True):
        bufs = _bufs(cap, hidden, device="cuda")
        leaf = _leaf(monkeypatch, bufs, enabled, device="cuda")
        leaf.fc31_input_scale = torch.tensor([448.0 * 6.0 / float(x.abs().max())], device="cuda")
        if mask is not None:
            leaf.set_pad_mask(mask)
        assert leaf.try_fused_route_quant(x, logits) is not None
        torch.cuda.synchronize()
        staged[enabled] = bufs
    for name in ("topk_idx_local", "topk_weights", "activation", "activation_sf"):
        assert torch.equal(getattr(staged[False], name), getattr(staged[True], name)), name
