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
"""Tests for the ROUTEQ_PORT form of ``MegaMoECuteDslLocal.try_fused_route_quant``.

From ``ROUTEQ_PORT_MIN_ROWS`` rows the seat is ``noaux_tc_route_stage_out`` plus FlashInfer's
NVFP4 quantizer writing the stage; every live byte must equal what the fused launch writes.
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
    ROUTEQ_PORT_MIN_ROWS,
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


def _leaf(monkeypatch, bufs, port_enabled, device="cpu"):
    leaf = Local.__new__(Local)
    leaf.apply_topk_in_fc1 = False
    leaf._pad_mask = None
    leaf._last_staged_T = {}
    leaf._prestaged_route_quant = None
    leaf._route_quant_split_tail_enabled = True
    leaf._route_quant_port_enabled = port_enabled
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


class _Quantizer:
    def __init__(self):
        self.calls = []

    def fp4_quantize(self, x, gs, fp4, sf, *flags):
        self.calls.append((x.shape[0], fp4.shape[0], sf.shape[0], flags[:5]))


@pytest.mark.parametrize(
    "cap, rows, enabled, ported",
    [
        (4096, ROUTEQ_PORT_MIN_ROWS, True, True),
        (4096, ROUTEQ_PORT_MIN_ROWS - 1, True, False),
        (4096, ROUTEQ_PORT_MIN_ROWS, False, False),
        (4096, 4096, True, True),
        (4100, 4099, True, False),
        (4112, 4099, True, True),
    ],
)
def test_the_port_runs_from_the_row_threshold_when_the_last_sf_block_fits(
    monkeypatch, cap, rows, enabled, ported
):
    hidden = 64
    seen = {}

    def route(*args, **kwargs):
        seen["route_rows"] = (args[4].shape[0], args[5].shape[0])

    def fused(*args, **kwargs):
        seen["fused"] = True

    quantizer = _Quantizer()
    _set_family_op(monkeypatch, "noaux_tc_route_stage_out", route)
    _set_family_op(monkeypatch, "noaux_tc_fp4_quantize_stage_out", fused)
    monkeypatch.setattr(local_mod, "_flashinfer_nvfp4_quantizer", lambda device: quantizer)
    bufs = _bufs(cap, hidden)
    leaf = _leaf(monkeypatch, bufs, enabled)
    staged = leaf.try_fused_route_quant(
        torch.zeros(rows, hidden, dtype=torch.bfloat16), torch.zeros(rows, EXPERTS)
    )
    assert staged is not None and staged[0].shape[0] == rows
    if ported:
        assert seen == {"route_rows": (cap, cap)}
        assert quantizer.calls == [(rows, rows, rows, (16, False, False, False, False))]
    else:
        assert seen == {"fused": True} and quantizer.calls == []


def test_the_port_falls_back_without_a_quantizer(monkeypatch):
    seen = {}
    _set_family_op(
        monkeypatch,
        "noaux_tc_fp4_quantize_stage_out",
        lambda *a, **k: seen.setdefault("fused", True),
    )
    monkeypatch.setattr(local_mod, "_flashinfer_nvfp4_quantizer", lambda device: None)
    bufs = _bufs(4096, 64)
    leaf = _leaf(monkeypatch, bufs, True)
    rows = ROUTEQ_PORT_MIN_ROWS
    assert leaf.try_fused_route_quant(
        torch.zeros(rows, 64, dtype=torch.bfloat16), torch.zeros(rows, EXPERTS)
    )
    assert seen == {"fused": True}


@pytest.mark.parametrize(
    "environ, enabled",
    [
        ({}, True),
        ({"TRTLLM_M31_DISABLE_MEGAMOE_LOCAL_ROUTEQ_PORT": "1"}, False),
        ({"TRTLLM_M31_DISABLE_MEGAMOE_LOCAL": "1"}, False),
    ],
)
def test_the_port_lever_is_on_unless_disabled(environ, enabled):
    assert plan.lever_enabled("ROUTEQ_PORT", environ) is enabled


def _port_available():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        return False
    import tensorrt_llm  # noqa: F401

    return _family_op_available("noaux_tc_route_stage_out") and (
        local_mod._flashinfer_nvfp4_quantizer(torch.device("cuda")) is not None
    )


@pytest.mark.skipif(
    not _port_available(), reason="needs SM10x, the route-stage op and FlashInfer's quantizer"
)
def test_a_flashinfer_nvfp4_knob_declines_the_quantizer(monkeypatch):
    from flashinfer.quantization import fp4_quantization

    for name in fp4_quantization.NVFP4_QUANT_ENV_VARS:
        monkeypatch.setenv(name, "1")
        assert local_mod._flashinfer_nvfp4_quantizer(torch.device("cuda")) is None
        monkeypatch.delenv(name)


def _staged(monkeypatch, enabled, x, logits, mask, cap, graph):
    bufs = _bufs(cap, x.shape[1], device="cuda")
    leaf = _leaf(monkeypatch, bufs, enabled, device="cuda")
    leaf.fc31_input_scale = torch.tensor([448.0 * 6.0 / float(x.abs().max())], device="cuda")
    if graph:
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            if mask is not None:
                leaf.set_pad_mask(mask)
            assert leaf.try_fused_route_quant(x, logits) is not None
        for t in vars(bufs).values():
            t.fill_(0x11)
        g.replay()
    else:
        if mask is not None:
            leaf.set_pad_mask(mask)
        assert leaf.try_fused_route_quant(x, logits) is not None
    torch.cuda.synchronize()
    return bufs


@pytest.mark.skipif(
    not _port_available(), reason="needs SM10x, the route-stage op and FlashInfer's quantizer"
)
@pytest.mark.parametrize(
    "rows, padded, graph",
    [
        (ROUTEQ_PORT_MIN_ROWS, False, False),
        (4096, True, False),
        (8200, False, False),
        (12288, True, False),
        (16384, False, False),
        (8200, True, True),
    ],
)
def test_the_port_stages_the_same_live_bytes_as_the_fused_launch(monkeypatch, rows, padded, graph):
    cap, hidden = 16384, 6144
    gen = torch.Generator(device="cuda").manual_seed(rows)
    x = (torch.randn(rows, hidden, generator=gen, device="cuda") * 2).to(torch.bfloat16)
    logits = torch.randn(rows, EXPERTS, generator=gen, device="cuda")
    mask = None
    if padded:
        mask = torch.zeros(rows, dtype=torch.uint8, device="cuda")
        mask[1::3] = 1
    fused = _staged(monkeypatch, False, x, logits, mask, cap, graph)
    ported = _staged(monkeypatch, True, x, logits, mask, cap, graph)
    assert torch.equal(fused.topk_idx_local, ported.topk_idx_local)
    assert torch.equal(fused.topk_weights.view(torch.int32), ported.topk_weights.view(torch.int32))
    assert torch.equal(fused.activation[:rows], ported.activation[:rows])
    assert torch.equal(fused.activation_sf[:rows], ported.activation_sf[:rows])
