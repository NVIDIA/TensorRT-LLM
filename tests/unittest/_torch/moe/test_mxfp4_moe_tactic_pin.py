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
"""Tactic pins of the TRTLLM-Gen MXFP4 x MXFP8 MoE runner.

A pin registered for a runner shape restricts the autotuner's candidates to
the pinned tactic inside its token range and keys the tuning cache apart from
the unpinned runner. The Kimi K3.1 per-layout pins and row tactics must produce
the same output as the runner's default tactic.
"""

import pytest
import torch

from tensorrt_llm._torch.custom_ops import trtllm_gen_custom_ops as ops
from tensorrt_llm._torch.utils import ActType_TrtllmGen
from tensorrt_llm._utils import is_sm_100f

_NUM_EXPERTS = 896
_LOCAL = 448
_TOP_K = 16
_HIDDEN = 3584
_INTER = 768
_SITU = int(ActType_TrtllmGen.SiTu)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not is_sm_100f(), reason="needs SM100 family"
)


def _inputs(num_tokens, seed=0):
    torch.manual_seed(seed)
    dev = "cuda"
    # Random FP4 codes; E8M0 scales near 2^-7 keep FC1 / FC2 finite.
    w13 = torch.randint(0, 256, (_LOCAL, 2 * _INTER, _HIDDEN // 2), dtype=torch.uint8, device=dev)
    s13 = torch.randint(
        118, 122, (_LOCAL, 2 * _INTER, _HIDDEN // 32), dtype=torch.uint8, device=dev
    )
    w2 = torch.randint(0, 256, (_LOCAL, _HIDDEN, _INTER // 2), dtype=torch.uint8, device=dev)
    s2 = torch.randint(118, 122, (_LOCAL, _HIDDEN, _INTER // 32), dtype=torch.uint8, device=dev)
    x = torch.randint(0, 256, (num_tokens, _HIDDEN), dtype=torch.uint8, device=dev)
    x = x.bitwise_and(0x77).view(torch.float8_e4m3fn)  # finite e4m3 codes
    x_sf = torch.full((num_tokens * _HIDDEN // 32,), 127, dtype=torch.uint8, device=dev)
    ids = torch.stack(
        [torch.randperm(_NUM_EXPERTS, device=dev)[:_TOP_K] for _ in range(num_tokens)]
    ).to(torch.int32)
    weights = (torch.rand(num_tokens, _TOP_K, device=dev) + 0.5).to(torch.bfloat16)
    alpha = torch.full((_LOCAL,), 4.0 * 25.0, dtype=torch.float32, device=dev)
    beta = torch.full((_LOCAL,), 0.25, dtype=torch.float32, device=dev)
    return [None, None, x, x_sf, w13, s13, None, alpha, beta, None, w2, s2, None, weights, ids]


def _runner():
    return ops.MxE4m3MxE2m1BlockScaleMoERunner(
        _NUM_EXPERTS, _TOP_K, None, None, _INTER, _HIDDEN, None, 0, _LOCAL, 1.0, 2, _SITU
    )


@pytest.fixture
def pins():
    saved = dict(ops._MXE4M3_MXE2M1_TACTIC_PINS)
    yield ops._MXE4M3_MXE2M1_TACTIC_PINS
    ops._MXE4M3_MXE2M1_TACTIC_PINS.clear()
    ops._MXE4M3_MXE2M1_TACTIC_PINS.update(saved)


def test_pin_restricts_tactics_in_range(pins):
    pins.clear()
    runner = _runner()
    unpinned_id = runner.unique_id()
    full = {n: runner.get_valid_tactics(_inputs(n), None) for n in (8, 16, 64, 128)}
    assert all(any(list(t) == [16, 50] for t in full[n]) for n in (16, 64))
    ops.pin_mxe4m3_mxe2m1_moe_tactics(_INTER, _HIDDEN, _SITU, [(16, 64, 16, 50)])
    assert runner.unique_id() != unpinned_id
    for n in (16, 64):
        assert runner.get_valid_tactics(_inputs(n), None) == [[16, 50]]
    for n in (8, 128):
        assert runner.get_valid_tactics(_inputs(n), None) == full[n]
    # Another runner shape is not pinned.
    other = ops.MxE4m3MxE2m1BlockScaleMoERunner(
        _NUM_EXPERTS, _TOP_K, None, None, 2 * _INTER, _HIDDEN, None, 0, _LOCAL, 1.0, 2, _SITU
    )
    assert len(other.get_valid_tactics(_inputs(16), None)) > 1


def _ep_inputs(num_tokens, local, inter, seed=0):
    """Inputs of a runner without MoE TP: ``local`` experts of ``inter`` intermediate each."""
    torch.manual_seed(seed)
    dev = "cuda"
    w13 = torch.randint(0, 256, (local, 2 * inter, _HIDDEN // 2), dtype=torch.uint8, device=dev)
    s13 = torch.randint(118, 122, (local, 2 * inter, _HIDDEN // 32), dtype=torch.uint8, device=dev)
    w2 = torch.randint(0, 256, (local, _HIDDEN, inter // 2), dtype=torch.uint8, device=dev)
    s2 = torch.randint(118, 122, (local, _HIDDEN, inter // 32), dtype=torch.uint8, device=dev)
    x = torch.randint(0, 256, (num_tokens, _HIDDEN), dtype=torch.uint8, device=dev)
    x = x.bitwise_and(0x77).view(torch.float8_e4m3fn)
    x_sf = torch.full((num_tokens * _HIDDEN // 32,), 127, dtype=torch.uint8, device=dev)
    ids = torch.stack(
        [torch.randperm(_NUM_EXPERTS, device=dev)[:_TOP_K] for _ in range(num_tokens)]
    ).to(torch.int32)
    weights = (torch.rand(num_tokens, _TOP_K, device=dev) + 0.5).to(torch.bfloat16)
    alpha = torch.full((local,), 4.0 * 25.0, dtype=torch.float32, device=dev)
    beta = torch.full((local,), 0.25, dtype=torch.float32, device=dev)
    return [None, None, x, x_sf, w13, s13, None, alpha, beta, None, w2, s2, None, weights, ids]


def test_k31_pin_selection():
    """K3.1 pins: the per-layout tables for the deployed (MoE TP, local experts) layouts, attention DP only for the
    DEP16 / DEP32 tables; other layouts keep TP4 [16, 50] for 16..128-token buckets and [32, 34] for 256..512, and
    without MoE TP [16, 54] in the 64 bucket and [64, 11] in the 256 bucket (TP2: the 64 bucket only), TEP only."""
    from types import SimpleNamespace

    from tensorrt_llm._torch.models.modeling_kimi_linear import (
        _K31_LAYOUT_DECODE_MOE_TACTIC_PINS,
        _k31_moe_tactic_pins,
    )

    k31 = SimpleNamespace(dsa_config=object())
    k3 = SimpleNamespace(dsa_config=None)
    tables = _K31_LAYOUT_DECODE_MOE_TACTIC_PINS
    for (moe_tp, local), table in tables.items():
        assert _k31_moe_tactic_pins(k31, moe_tp, False, local) == table
        assert _k31_moe_tactic_pins(k3, moe_tp, False, local) is None
    assert _k31_moe_tactic_pins(k31, 1, True, 56) == tables[(1, 56)]
    assert _k31_moe_tactic_pins(k31, 1, True, 28) == tables[(1, 28)]
    assert _k31_moe_tactic_pins(k31, 1, True, 112) is None
    assert _k31_moe_tactic_pins(k31, 4, True, 224) is None
    # Every bucket of a table is covered once, buckets ascending.
    for table in tables.values():
        edges = [(lo, hi) for lo, hi, _, _ in table]
        assert all(lo <= hi for lo, hi in edges) and all(
            a[1] < b[0] for a, b in zip(edges, edges[1:])
        )
    # Layouts without a table.
    assert _k31_moe_tactic_pins(k31, 4) == ((16, 128, 16, 50), (256, 512, 32, 34))
    assert _k31_moe_tactic_pins(k31, 1) == ((64, 64, 16, 54), (256, 256, 64, 11))
    assert _k31_moe_tactic_pins(k31, 1, attention_dp=True) is None
    assert _k31_moe_tactic_pins(k31, 2) == ((64, 64, 16, 54),)
    assert _k31_moe_tactic_pins(k31, 8) is None
    assert _k31_moe_tactic_pins(k3, 1) is None
    assert _k31_moe_tactic_pins(k3, 4) is None


# (MoE TP, local experts) of the shipped K3.1 layouts: TEP8 ctx, TEP32 gen, TEP16 gen, DEP16.
_SHIPPED_LAYOUTS = frozenset({(1, 112), (4, 112), (4, 224), (1, 56)})


def _layout_pin_cases():
    """(moe_tp, local experts, intermediate per rank, bucket, pinned tactic) of every pin bucket of the shipped
    layouts."""
    from tensorrt_llm._torch.models.modeling_kimi_linear import _K31_LAYOUT_DECODE_MOE_TACTIC_PINS

    cases = []
    for (moe_tp, local), pins in sorted(_K31_LAYOUT_DECODE_MOE_TACTIC_PINS.items()):
        if (moe_tp, local) not in _SHIPPED_LAYOUTS:
            continue
        for lo, hi, tile_n, config in pins:
            bucket = lo
            while bucket <= hi:
                cases.append((moe_tp, local, 3072 // moe_tp, bucket, [tile_n, config]))
                bucket *= 2
    return cases


@pytest.mark.parametrize("moe_tp,local,inter,bucket,pinned", _layout_pin_cases())
def test_k31_layout_pins_are_valid_and_bitwise(moe_tp, local, inter, bucket, pinned):
    """Every layout pin is one of the runner's tactics at its bucket's profiled row count (so the autotuner takes it
    instead of profiling) and its output equals the runner's default tactic's at the top of the bucket."""
    runner = ops.MxE4m3MxE2m1BlockScaleMoERunner(
        _NUM_EXPERTS, _TOP_K, None, None, inter, _HIDDEN, None, 0, local, 1.0, 2, _SITU
    )
    tactics = [list(t) for t in runner.get_valid_tactics(_ep_inputs(bucket, local, inter), None)]
    assert pinned in tactics, (bucket, pinned)
    rows = 2 * bucket - 1
    inputs = _ep_inputs(rows, local, inter, seed=rows + local)
    out = runner.forward(inputs, tactic=pinned)
    ref = runner.forward(inputs, tactic=[-1, -1])
    torch.cuda.synchronize()
    assert torch.isfinite(out.float()).all()
    assert torch.equal(out.view(torch.int16), ref.view(torch.int16)), (rows, pinned)


def _row_tactic_cases():
    """(moe_tp, local experts, intermediate per rank, rows, row tactic) at both ends of every row range of the
    shipped layouts."""
    from tensorrt_llm._torch.models.modeling_kimi_linear import _K31_LAYOUT_DECODE_MOE_ROW_TACTICS

    return [
        (moe_tp, local, 3072 // moe_tp, rows, [tile_n, config])
        for (moe_tp, local), ranges in sorted(_K31_LAYOUT_DECODE_MOE_ROW_TACTICS.items())
        if (moe_tp, local) in _SHIPPED_LAYOUTS
        for lo, hi, tile_n, config in ranges
        for rows in (lo, hi)
    ]


@pytest.mark.parametrize("moe_tp,local,inter,rows,tactic", _row_tactic_cases())
def test_k31_row_tactics_are_bitwise(moe_tp, local, inter, rows, tactic):
    """Each row-range tactic (also outside the runner's tile window) gives the default tactic's output."""
    runner = ops.MxE4m3MxE2m1BlockScaleMoERunner(
        _NUM_EXPERTS, _TOP_K, None, None, inter, _HIDDEN, None, 0, local, 1.0, 2, _SITU
    )
    inputs = _ep_inputs(rows, local, inter, seed=rows + 7 * local)
    out = runner.forward(inputs, tactic=tactic)
    ref = runner.forward(inputs, tactic=[-1, -1])
    torch.cuda.synchronize()
    assert torch.isfinite(out.float()).all()
    assert torch.equal(out.view(torch.int16), ref.view(torch.int16)), (rows, tactic)


def test_row_tactic_lookup(monkeypatch):
    """The MXFP4 ops replace the bucket's tactic with the registered row-range tactic of the call's row count."""
    from types import SimpleNamespace

    from tensorrt_llm._torch.models.modeling_kimi_linear import (
        _K31_LAYOUT_DECODE_MOE_ROW_TACTICS,
        _K31_LAYOUT_DECODE_MOE_TACTIC_PINS,
        _k31_moe_row_tactics,
    )

    monkeypatch.setattr(ops, "_MXE4M3_MXE2M1_ROW_TACTICS", {})
    runner = SimpleNamespace(intermediate_size=3072, valid_hidden_size=_HIDDEN, act_type=_SITU)
    ops.pin_mxe4m3_mxe2m1_moe_row_tactics(
        3072, _HIDDEN, _SITU, ((128, 229, 16, 54), (230, 895, 32, 18))
    )
    assert ops._mxfp4_row_tactic(runner, 127, [64, 11]) == [64, 11]
    assert ops._mxfp4_row_tactic(runner, 128, [64, 11]) == [16, 54]
    assert ops._mxfp4_row_tactic(runner, 229, -1) == [16, 54]
    assert ops._mxfp4_row_tactic(runner, 230, [64, 11]) == [32, 18]
    assert ops._mxfp4_row_tactic(runner, 896, [128, 18]) == [128, 18]
    # Other shapes (and runners without a valid hidden size) are untouched.
    other = SimpleNamespace(intermediate_size=768, valid_hidden_size=_HIDDEN, act_type=_SITU)
    assert ops._mxfp4_row_tactic(other, 200, [32, 34]) == [32, 34]
    assert ops._mxfp4_row_tactic(SimpleNamespace(valid_hidden_size=None), 200, [1, 2]) == [1, 2]
    # The model hands them out only with its per-layout pin tables.
    k31 = SimpleNamespace(dsa_config=object())
    for layout, ranges in _K31_LAYOUT_DECODE_MOE_ROW_TACTICS.items():
        assert layout in _K31_LAYOUT_DECODE_MOE_TACTIC_PINS
        assert _k31_moe_row_tactics(k31, layout[0], layout == (1, 28), layout[1]) == ranges
    assert _k31_moe_row_tactics(k31, 4, True, 224) == ()
    assert _k31_moe_row_tactics(SimpleNamespace(dsa_config=None), 1, False, 56) == ()
    monkeypatch.setenv("KIMI_K31_MOE_ROW_TACTICS", "0")
    assert _k31_moe_row_tactics(k31, 1, False, 56) == ()
