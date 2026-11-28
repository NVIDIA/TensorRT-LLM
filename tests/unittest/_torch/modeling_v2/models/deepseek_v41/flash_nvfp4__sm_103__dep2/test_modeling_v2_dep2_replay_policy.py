# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""When the dep2 target runs the decoder half of a context chunk over the final window only (bounded decoder replay)."""

import importlib
from types import SimpleNamespace as NS

import pytest

modeling = importlib.import_module(
    "tensorrt_llm._torch.modeling_v2.models.deepseek_v41.flash_nvfp4__sm_103__dep2.modeling"
)

RATIOS = [0, 0] + [2] * 18 + [1] * 20 + [0, 0, 0]


def _cfg(**kw):
    base = dict(
        kv_source_layer_ids=[2, 8, 14, 20],
        compress_ratios=RATIOS,
        engram_layer_ids=[1, 14],
        sliding_window=128,
        num_hidden_layers=40,
    )
    base.update(kw)
    return NS(**base)


def _mc(adp=True, pp=1, cp=1, spec=None):
    return NS(mapping=NS(enable_attention_dp=adp, pp_size=pp, cp_size=cp), spec_config=spec)


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.delenv("TRTLLM_V41_DECODER_BOUNDED_REPLAY", raising=False)
    monkeypatch.delenv("TRTLLM_V41_BOUNDED_REPLAY_TOKENS", raising=False)


def test_released_checkpoint_replays_from_layer_20_over_140_rows():
    assert modeling._replay_policy(_cfg(), _mc()) == (20, 140)


def test_window_follows_the_env_and_must_exceed_the_sliding_window(monkeypatch):
    monkeypatch.setenv("TRTLLM_V41_BOUNDED_REPLAY_TOKENS", "200")
    assert modeling._replay_policy(_cfg(), _mc()) == (20, 200)
    monkeypatch.setenv("TRTLLM_V41_BOUNDED_REPLAY_TOKENS", "128")
    with pytest.raises(ValueError, match="must exceed"):
        modeling._replay_policy(_cfg(), _mc())


def test_switch_off(monkeypatch):
    monkeypatch.setenv("TRTLLM_V41_DECODER_BOUNDED_REPLAY", "0")
    assert modeling._replay_policy(_cfg(), _mc()) == (None, 0)


@pytest.mark.parametrize(
    "cfg,mc",
    [
        (_cfg(), _mc(adp=False)),  # the counts exchange is attention DP's
        (_cfg(), _mc(pp=2)),
        (_cfg(), _mc(spec=object())),  # speculative workers keep full prefill
        (_cfg(kv_source_layer_ids=[]), _mc()),
        (_cfg(kv_source_layer_ids=[2, 8, 14, 18]), _mc()),  # the last source is pooled (ratio 2)
        (_cfg(engram_layer_ids=[1, 24]), _mc()),  # an engram layer in the decoder half
    ],
    ids=["no-adp", "pp", "spec", "no-source", "pooled-last-source", "late-engram"],
)
def test_refusals_run_full_prefill(cfg, mc):
    assert modeling._replay_policy(cfg, mc) == (None, 0)
