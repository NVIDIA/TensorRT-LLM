# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The draft MoE's runner tactic by gathered rows at a dep8 rank (`DP_DRAFT_MOE_TILE_ENV`,
default on, CPU): the table, the build (where it engages, where it declines) and the
dispatch (`_draft_moe_tactic`: the table's tactic where the runner lists it, else the
AutoTuner's).

That every tactic computes the same bits is the op's GPU test
(`ops/moe/test_modeling_v2_fp8_block_scale_moe_runner.py`, the pinned cells and
`test_pinned_tactic_is_bitwise_the_autotuner_call`).
"""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_10x__default import (  # noqa: E501
    modeling as M,
)


def _core(**over):
    core = SimpleNamespace(
        mtp_enabled=True,
        mtp_local_experts=64,
        _mtp=SimpleNamespace(moe_gscale=None),
        dp_rank=3,
        dp_size=8,
        topk=10,
        hidden=2560,
        moe_inter=640,
        _draft_moe_tactics=None,
    )
    for k, v in over.items():
        setattr(core, k, v)
    return core


def _build(monkeypatch, core, value=None, dp=True):
    if value is None:
        monkeypatch.delenv(M.DP_DRAFT_MOE_TILE_ENV, raising=False)
    else:
        monkeypatch.setenv(M.DP_DRAFT_MOE_TILE_ENV, value)
    monkeypatch.setattr(M, "_dp_decode_rank", lambda core, device: dp)
    return M._draft_moe_tactics_build(core, torch.device("cpu"))


@pytest.mark.cpu_only
def test_table_rows_are_ordered_and_disjoint():
    for rules in M._DRAFT_MOE_TACTICS.values():
        prev = 0
        for lo, hi, tactic in rules:
            assert prev < lo <= hi and len(tactic) == 2
            prev = hi


@pytest.mark.cpu_only
@pytest.mark.parametrize("value", [None, "1", "on"])
def test_build_engages_at_a_dep8_rank(monkeypatch, value):
    assert _build(monkeypatch, _core(), value) == {}


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "over, value, dp",
    [
        ({}, "0", True),  # switched off
        ({}, None, False),  # not a measured attention-DP rank (or not SM103)
        ({"mtp_local_experts": 128}, None, True),  # dep4's 128-wide window: not measured
        ({"mtp_enabled": False}, None, True),
        ({"_mtp": SimpleNamespace(moe_gscale=torch.ones(1))}, None, True),  # NVFP4 draft experts
    ],
    ids=["off", "not-dp-rank", "dep4-window", "no-mtp", "nvfp4-experts"],
)
def test_build_declines(monkeypatch, over, value, dp):
    assert _build(monkeypatch, _core(**over), value, dp) is None


@pytest.mark.cpu_only
def test_dispatch_by_rows(monkeypatch):
    listed = []

    def valid(top_k, hidden, inter, local, tokens):
        listed.append((top_k, hidden, inter, local, tokens))
        return [[8, 8], [16, 8], [64, 0]] if tokens != 160 else [[8, 8], [64, 0]]

    monkeypatch.setattr(M, "fp8_moe_valid_tactics", valid)
    core = _core(_draft_moe_tactics={})
    assert M._draft_moe_tactic(core, 95) is None
    assert M._draft_moe_tactic(core, 96) == (16, 8)
    assert M._draft_moe_tactic(core, 192) == (16, 8)
    assert M._draft_moe_tactic(core, 193) is None
    # Not listed at this count: the AutoTuner's tactic.
    assert M._draft_moe_tactic(core, 160) is None
    # Settled once per row count: a second call asks the runner nothing.
    n = len(listed)
    assert M._draft_moe_tactic(core, 96) == (16, 8) and len(listed) == n
    assert listed[0] == (10, 2560, 640, 64, 96)


@pytest.mark.cpu_only
def test_dispatch_without_the_build_is_the_autotuner():
    assert M._draft_moe_tactic(_core(), 128) is None


@pytest.mark.cpu_only
def test_draft_moe_block_passes_the_tactic(monkeypatch):
    """The FP8 draft runner call carries `_draft_moe_tactic`'s answer for its chunk's rows."""
    seen = {}

    def runner(**kw):
        seen.update(kw)
        return torch.zeros(kw["hidden_states"].shape[0], 4)

    monkeypatch.setattr(M, "fp8_block_scale_moe_runner", runner)
    monkeypatch.setattr(M, "cublas_mm", lambda a, b, *r: torch.zeros(a.shape[0], 512))
    monkeypatch.setattr(
        M,
        "renorm_moe_routing_op",
        lambda logits, topk, output_dtype: (torch.zeros(logits.shape[0], topk), None),
    )
    monkeypatch.setattr(
        M, "fp8_quantize_1x128", lambda x, use_ue8m0: (x, torch.zeros(1, x.shape[0]))
    )
    monkeypatch.setattr(M, "_draft_moe_tactic", lambda core, rows: ("tactic", rows))
    core = _core(
        dp_size=1,
        tp_sum=lambda t: t,
        num_experts=512,
        mtp_expert_offset=0,
    )
    W = SimpleNamespace(
        moe_gscale=None, router=None, fc1_w=None, fc1_s=None, fc2_w=None, fc2_s=None
    )
    M._mtp_moe_block(core, W, torch.zeros(96, 4), routed_only=True)
    assert seen["tactic"] == ("tactic", 96)
