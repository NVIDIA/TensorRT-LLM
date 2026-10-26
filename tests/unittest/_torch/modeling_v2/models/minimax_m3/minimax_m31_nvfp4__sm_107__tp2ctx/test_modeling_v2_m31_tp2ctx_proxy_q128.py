# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TP2CTX_PROXY_Q128: the prefill proxy (indexer max-score) plan runs 128-row query tiles on long steps only.

MsaRuntimePolicy.prefill_proxy_qo_tile reaches fmha_sm100_plan(maxscore_qo_tile=...) through the MSA metadata
prepare only when the loaded planner accepts that argument and the step's longest proxy query span exceeds 128 rows;
decode-size steps (q <= 8) and every model without the policy keep the planner's own tile rule. CPU.
"""

import sys
import types
from pathlib import Path
from unittest.mock import Mock

import pytest
import torch

import tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3 as _family
from tensorrt_llm._torch.attention.backends.sparse.minimax_m3 import msa_indexer
from tensorrt_llm._torch.pyexecutor.runtime_hooks import MsaRuntimePolicy

_TARGET = Path(_family.__file__).resolve().parent / "minimax_m31_nvfp4__sm_107__tp2ctx"


def _plan_with_tile(*args, maxscore_qo_tile=-1, num_kv_splits=1, **kwargs):
    return None


def _plan_without_tile(*args, num_kv_splits=1, **kwargs):
    return None


def _kwargs(policy, plan, qo_lens, rows):
    fmha = Mock(spec_set=("fmha_sm100_plan",))
    fmha.fmha_sm100_plan = plan
    qo = torch.tensor(qo_lens, dtype=torch.int32)
    return msa_indexer.proxy_plan_kwargs(policy, fmha, qo, torch.zeros_like(qo), rows, 128)


_Q128 = MsaRuntimePolicy(prefill_proxy_qo_tile=128)


@pytest.mark.parametrize("q", [1, 4, 8, 128])
def test_decode_and_short_steps_keep_the_planners_rule(q):
    assert _kwargs(_Q128, _plan_with_tile, [q, q], (0, 2)) == {}


@pytest.mark.parametrize("q", [129, 1160, 16384])
def test_long_steps_take_the_128_row_tile(q):
    assert _kwargs(_Q128, _plan_with_tile, [1, q], (0, 2)) == {
        "maxscore_qo_tile": 128,
        "num_kv_splits": 1,
    }
    splits = MsaRuntimePolicy(prefill_proxy_qo_tile=128, prefill_proxy_kv_splits=4)
    assert _kwargs(splits, _plan_with_tile, [q], (0, 1)) == {
        "maxscore_qo_tile": 128,
        "num_kv_splits": 4,
    }


def test_only_the_planned_rows_count():
    # A long generation-side row outside the proxy rows does not switch the context rows' tile.
    assert _kwargs(_Q128, _plan_with_tile, [64, 16384], (0, 1)) == {}


def _wrapped_planner(monkeypatch, impl):
    """The MSA build's shape: the public planner forwards **kwargs to its module's _fmha_sm100_plan_impl."""
    module = types.ModuleType("_fake_fmha_sm100_api")
    module._fmha_sm100_plan_impl = impl

    def fmha_sm100_plan(qo_segment_lens, kv_segment_lens, *args, qo_offset=None, **kwargs):
        return None

    fmha_sm100_plan.__module__ = module.__name__
    module.fmha_sm100_plan = fmha_sm100_plan
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return fmha_sm100_plan


def test_a_kwargs_planner_is_probed_through_its_implementation(monkeypatch):
    plan = _wrapped_planner(monkeypatch, _plan_with_tile)
    assert _kwargs(_Q128, plan, [1, 16384], (0, 2)) == {"maxscore_qo_tile": 128, "num_kv_splits": 1}
    older = _wrapped_planner(monkeypatch, _plan_without_tile)
    assert _kwargs(_Q128, older, [1, 16384], (0, 2)) == {}


def test_no_policy_or_an_older_planner_changes_nothing():
    assert _kwargs(MsaRuntimePolicy(), _plan_with_tile, [16384], (0, 1)) == {}
    assert _kwargs(_Q128, _plan_without_tile, [16384], (0, 1)) == {}
    assert _kwargs(_Q128, _plan_with_tile, [16384], None) == {}


def test_tp2ctx_opts_in_with_a_kill_switch():
    src = (_TARGET / "modeling.py").read_text()
    assert 'prefill_proxy_qo_tile=None if m31_disabled("TP2CTX_PROXY_Q128") else 128,' in src
    # One kill switch covers the tile and the fixed split (no host readback, unlike the adaptive split); the OnlyScore
    # proxy modes are native on SM107, so nothing enables them here.
    assert 'prefill_proxy_kv_splits=None if m31_disabled("TP2CTX_PROXY_Q128") else 4,' in src
    assert "enable_proxy_modes_off_sm107" not in src
