# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``MuseSparkOpenMoE`` under DWDP: admission, expert construction, forward.

Every DWDP rank is a complete replica that prefetches its peers' experts, so
the MoE runs with no comm on this rank's own tokens. These run on CPU:
construction is stubbed around ``create_moe`` and the forward stops at the
expert call.
"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tensorrt_llm._torch.models import modeling_musespark_open
from tensorrt_llm._torch.models.modeling_musespark_open import MuseSparkOpenMoE
from tensorrt_llm._torch.pyexecutor.dwdp import get_global_dwdp_manager, set_global_dwdp_manager
from tensorrt_llm.mapping import Mapping

NUM_EXPERTS = 16


def _dwdp_mapping(*, tp, attention_dp):
    return Mapping(
        world_size=tp,
        rank=0,
        tp_size=tp,
        enable_attention_dp=attention_dp,
        dwdp_size=4,
        dwdp_rank=0,
    )


def _model_config(mapping, *, bf16_layer=False):
    pretrained_config = SimpleNamespace(
        hidden_size=16,
        expert_hidden_size=8,
        num_experts=NUM_EXPERTS,
        moe_topk=4,
        metap_multiplier=1.0,
        intermediate_size=32,
        torch_dtype=torch.bfloat16,
        rms_norm_eps=1e-5,
        rms_post_norm_eps=1e-5,
        router_score_division_eps=1e-20,
    )
    # The checkpoint quant-excludes whole layers, leaving their experts BF16.
    quant_config = SimpleNamespace(
        exclude_modules=["*layers.0.*"],
        is_module_excluded_from_quantization=lambda name: bf16_layer and name == "model.layers.0",
        kv_cache_quant_algo=None,
    )
    return SimpleNamespace(
        pretrained_config=pretrained_config,
        mapping=mapping,
        quant_config=quant_config,
        moe_backend="CUTEDSL",
    )


@pytest.mark.parametrize(
    "tp,attention_dp,supported",
    [
        # Aggregated serving: every attention-DP rank is a replica.
        (4, True, True),
        # An independently launched TP1 context worker.
        (1, False, True),
        # A TP-sharded replica cannot own an expert slice.
        (4, False, False),
    ],
)
def test_dwdp_admission(monkeypatch, tp, attention_dp, supported):
    class Admitted(Exception):
        pass

    def stop_before_weights(*args, **kwargs):
        raise Admitted

    monkeypatch.setattr(modeling_musespark_open, "Linear", stop_before_weights)
    config = _model_config(_dwdp_mapping(tp=tp, attention_dp=attention_dp))
    if supported:
        with pytest.raises(Admitted):
            MuseSparkOpenMoE(config, layer_idx=1)
    else:
        with pytest.raises(ValueError, match="MuseSparkOpen requires"):
            MuseSparkOpenMoE(config, layer_idx=1)


class _CreatedMoE:
    """What ``create_moe`` was asked for, and which DWDP manager it saw."""

    comm = None

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.dwdp_manager = get_global_dwdp_manager()


@pytest.fixture
def dwdp_manager(monkeypatch):
    """Build ``MuseSparkOpenMoE`` with stub layers under an installed DWDP manager."""
    monkeypatch.setattr(modeling_musespark_open, "Linear", lambda *a, **k: nn.Identity())
    monkeypatch.setattr(modeling_musespark_open, "RMSNorm", lambda *a, **k: nn.Identity())
    monkeypatch.setattr(modeling_musespark_open, "AllReduce", lambda *a, **k: None)
    monkeypatch.setattr(modeling_musespark_open, "create_moe", _CreatedMoE)
    assert get_global_dwdp_manager() is None
    manager = object()
    set_global_dwdp_manager(manager)
    yield manager
    set_global_dwdp_manager(None)


def test_dwdp_experts_are_built_without_attention_dp(dwdp_manager):
    moe = MuseSparkOpenMoE(_model_config(_dwdp_mapping(tp=4, attention_dp=True)), layer_idx=1)

    created = moe.experts
    expert_mapping = created.kwargs["model_config"].mapping
    # No comm strategy and no DP-padding truncation: this rank's tokens only.
    assert not expert_mapping.enable_attention_dp
    assert expert_mapping.dwdp_enabled and expert_mapping.tp_size == 4
    assert moe.mapping.enable_attention_dp
    # An NVFP4 layer is built under the manager, so its experts are prefetched.
    assert created.dwdp_manager is dwdp_manager
    assert created.kwargs["override_quant_config"] is None


def test_dwdp_bf16_layers_are_built_without_the_manager(dwdp_manager):
    moe = MuseSparkOpenMoE(
        _model_config(_dwdp_mapping(tp=4, attention_dp=True), bf16_layer=True), layer_idx=0
    )

    created = moe.experts
    # BF16 experts have no DWDP backend; seeing the manager would shrink the
    # layer to this rank's slots with nothing to prefetch the rest.
    assert created.dwdp_manager is None
    assert get_global_dwdp_manager() is dwdp_manager
    assert created.kwargs["override_quant_config"] is not None
    # CuteDslFusedMoE has no BF16 path that returns the unfinalized triple.
    assert created.kwargs["model_config"].moe_backend == "TRTLLM"
    assert not created.kwargs["model_config"].mapping.enable_attention_dp


def test_dwdp_manager_is_restored_when_a_bf16_layer_fails(dwdp_manager, monkeypatch):
    def fail(**kwargs):
        raise RuntimeError("create_moe failed")

    monkeypatch.setattr(modeling_musespark_open, "create_moe", fail)
    with pytest.raises(RuntimeError, match="create_moe failed"):
        MuseSparkOpenMoE(
            _model_config(_dwdp_mapping(tp=4, attention_dp=True), bf16_layer=True), layer_idx=0
        )
    assert get_global_dwdp_manager() is dwdp_manager


class _Nvtx:
    """``self.nvtx(name)`` is used as a context manager in ``forward``."""

    def __call__(self, _name):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False


class _ExpertsCalled(Exception):
    pass


@pytest.mark.parametrize("dwdp_local,expected", [(True, None), (False, [2, 5, 1, 3])])
def test_dwdp_forward_hands_the_experts_no_rank_token_counts(dwdp_local, expected):
    """The experts' non-DP mapping must not slice this rank's tokens by DP rank."""
    seen = {}

    class _Experts:
        comm = None
        routing_method = None

        def __call__(self, *_args, **kwargs):
            seen.update(kwargs)
            raise _ExpertsCalled

    mod = MuseSparkOpenMoE.__new__(MuseSparkOpenMoE)
    mod._dwdp_local = dwdp_local
    mod.experts = _Experts()
    mod.nvtx = _Nvtx()
    mod.prepare_moe_inputs = lambda h: (torch.zeros(h.shape[0], NUM_EXPERTS), h)

    with pytest.raises(_ExpertsCalled):
        mod.forward(
            torch.zeros(2, 3),
            attn_metadata=SimpleNamespace(all_rank_num_tokens=[2, 5, 1, 3]),
        )
    assert seen["do_finalize"] is False
    assert seen["all_rank_num_tokens"] == expected
