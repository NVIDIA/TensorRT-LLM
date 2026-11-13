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
"""CPU tests for the MegaMoE-local pad mask: one mask per launch, applied by G1 or by the stock staging."""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.native import family_ops
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.ops.moe.megamoe_local import (
    local_impl as local_mod,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.ops.moe.megamoe_local.local_impl import (
    TrtllmCutedslMegaMoeLocalNvfp4Impl as Local,
)
from tensorrt_llm._torch.moe.fused_moe.mega_moe import TrtllmCutedslMegaMoeNvfp4Impl as Base

pytestmark = pytest.mark.cpu_only


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


ROWS, TOP_K, HIDDEN, CAP, EXPERTS = 6, 4, 64, 8, 16
PADDED = (1, 4)


def _leaf(apply_topk_in_fc1=False) -> Local:
    leaf = Local.__new__(Local)
    leaf.apply_topk_in_fc1 = apply_topk_in_fc1
    leaf._pad_mask = None
    leaf._last_staged_T = {}
    leaf._prestaged_route_quant = None
    leaf._route_quant_split_tail_enabled = True
    leaf._route_quant_port_enabled = True
    return leaf


def _mask(rows=ROWS) -> torch.Tensor:
    mask = torch.zeros(rows, dtype=torch.uint8)
    mask[list(PADDED)] = 1
    return mask


def _bufs() -> SimpleNamespace:
    return SimpleNamespace(
        topk_idx_local=torch.full((CAP, TOP_K), 9, dtype=torch.int32),
        topk_weights=torch.full((CAP, TOP_K), 0.5, dtype=torch.float32),
        activation=torch.zeros(CAP, HIDDEN // 2, dtype=torch.uint8),
        activation_sf=torch.zeros(CAP, HIDDEN // 16, dtype=torch.uint8),
    )


def test_a_launch_consumes_the_mask_once():
    leaf, mask = _leaf(), _mask()
    leaf.set_pad_mask(mask)
    assert leaf._take_pad_mask(ROWS) is mask
    assert leaf._take_pad_mask(ROWS) is None


def test_the_launch_must_have_the_mask_rows():
    leaf = _leaf()
    leaf.set_pad_mask(_mask())
    with pytest.raises(ValueError, match="rows"):
        leaf._take_pad_mask(ROWS + 1)
    assert leaf._pad_mask is None


@pytest.mark.parametrize(
    "bad",
    [
        torch.zeros(ROWS, dtype=torch.bool),
        torch.zeros(ROWS, dtype=torch.int32),
        torch.zeros(ROWS, 1, dtype=torch.uint8),
    ],
    ids=["bool", "int32", "2d"],
)
def test_the_mask_must_be_a_1d_uint8_tensor(bad):
    with pytest.raises(ValueError, match="uint8"):
        _leaf().set_pad_mask(bad)


def test_an_unweighted_reduce_declines_the_mask():
    leaf = _leaf(apply_topk_in_fc1=True)
    assert "unweighted" in leaf.pad_mask_decline_reason()
    with pytest.raises(ValueError, match="unweighted"):
        leaf.set_pad_mask(_mask())
    leaf.set_pad_mask(None)
    assert _leaf().pad_mask_decline_reason() is None


def test_a_launch_drops_a_mask_it_did_not_stage(monkeypatch):
    monkeypatch.setattr(Base, "_run_moe", lambda self, **kwargs: "launched")
    leaf = _leaf()
    leaf.set_pad_mask(_mask())
    assert leaf._run_moe(num_tokens=ROWS) == "launched"
    assert leaf._pad_mask is None

    def failed_launch(self, **kwargs):
        raise RuntimeError("launch failed")

    monkeypatch.setattr(Base, "_run_moe", failed_launch)
    leaf.set_pad_mask(_mask())
    with pytest.raises(RuntimeError, match="launch failed"):
        leaf._run_moe(num_tokens=ROWS)
    assert leaf._pad_mask is None


def test_the_stock_staging_masks_the_pad_rows_when_g1_declined(monkeypatch):
    def stock_stage(self, *, bufs, topk_idx, topk_weights, num_tokens, top_k, **kwargs):
        bufs.topk_idx_local[:num_tokens, :top_k].copy_(topk_idx)
        bufs.topk_weights[:num_tokens, :top_k].copy_(topk_weights)

    monkeypatch.setattr(Base, "_stage_inputs", stock_stage)
    leaf, bufs = _leaf(), _bufs()
    topk_idx = torch.arange(ROWS * TOP_K, dtype=torch.int32).view(ROWS, TOP_K)
    topk_weights = torch.full((ROWS, TOP_K), 0.25)
    leaf.set_pad_mask(_mask())
    leaf._stage_inputs(
        bufs=bufs,
        x=torch.zeros(ROWS, HIDDEN // 2, dtype=torch.uint8),
        x_sf=torch.zeros(ROWS, HIDDEN // 16, dtype=torch.uint8),
        topk_idx=topk_idx,
        topk_weights=topk_weights,
        num_tokens=ROWS,
        top_k=TOP_K,
        stage_activation=True,
    )
    pad = _mask().bool()
    assert bool((bufs.topk_idx_local[:ROWS][pad] == -1).all())
    assert bool((bufs.topk_weights[:ROWS][pad] == 0).all())
    assert torch.equal(bufs.topk_idx_local[:ROWS][~pad], topk_idx[~pad])
    assert torch.equal(bufs.topk_weights[:ROWS][~pad], topk_weights[~pad])
    assert leaf._pad_mask is None


def test_g1_takes_the_mask_into_its_launch(monkeypatch):
    seen = {}

    def stage_out(*args, pad_mask=None):
        seen["rows0"], seen["pad_mask"] = args[6], pad_mask

    _set_family_op(monkeypatch, "noaux_tc_fp4_quantize_stage_out", stage_out)
    leaf, bufs = _leaf(), _bufs()
    leaf.routing_method = SimpleNamespace(
        routing_impl=SimpleNamespace(top_k=TOP_K, routed_scaling_factor=2.0),
        e_score_correction_bias=torch.zeros(EXPERTS),
    )
    leaf.fc31_input_scale = torch.ones(1)
    leaf._active_launch_max_tokens = CAP
    leaf._route_quant_empty_stage = {}
    monkeypatch.setattr(leaf, "_route_quant_decline_reason", lambda x, logits: None)
    monkeypatch.setattr(leaf, "_select_launch_max_tokens", lambda max_tokens: CAP)
    monkeypatch.setattr(leaf, "_acquire_buffers", lambda **kwargs: bufs)
    mask = _mask()
    leaf.set_pad_mask(mask)
    staged = leaf.try_fused_route_quant(
        torch.zeros(ROWS, HIDDEN, dtype=torch.bfloat16), torch.zeros(ROWS, EXPERTS)
    )
    assert staged is not None
    assert seen == {"rows0": ROWS, "pad_mask": mask}
    assert leaf._pad_mask is None
