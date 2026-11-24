# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dense DWDP on the Kimi K3.1 modeling_v2 targets (the sm_103 DEP target and the general target): the dense roots,
the tables and the read-window hooks of ``K31DecoderLayer``, and the DEP target's KDA pair split under dense DWDP.

The decoder layers under test are shells of the target's ``K31DecoderLayer``: the class's own ``forward`` and
``dense_dwdp_roots`` run on recorders that stand in for the attention and the MoE block, with the norms and
attention-residual mixes replaced by plain arithmetic. A recording dense runtime checks that every covered layer opens
its read windows (H1) before its attention, closes the attention window (H2) after it and closes the MoE-dense window
(H3) after the MoE block.
"""

import types

import pytest
import torch
from torch import nn

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kimi_k31_pro_mxfp4__sm_10x__default import (
    modeling as general,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kimi_k31_pro_mxfp4__sm_10x__default import (
    weights as general_weights,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kimi_k31_pro_mxfp4__sm_103__dep import (
    modeling as dep,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kimi_k31_pro_mxfp4__sm_103__dep import (
    weights as dep_weights,
)
from tensorrt_llm._torch.modules.dwdp.dense.types import DenseMode

pytestmark = pytest.mark.cpu_only

_HIDDEN = 8
_TOKENS = 3
_TARGETS = pytest.mark.parametrize("target", [dep, general], ids=["dep", "general"])


class _Log(list):
    """The calls in program order: hooks, attention, the SEP attention and the MoE block."""


class _Dense:
    """A dense runtime that covers ``covered`` and logs every hook."""

    def __init__(self, log: _Log, covered) -> None:
        self.log = log
        self.covered = covered

    def covers(self, layer: int) -> bool:
        return layer in self.covered

    def wait_layer(self, layer: int) -> None:
        self.log.append(("H1", layer))

    def attention_done(self, layer: int) -> None:
        self.log.append(("H2", layer))

    def layer_done(self, layer: int) -> None:
        self.log.append(("H3", layer))


class _Attention(nn.Module):
    """Stands in for the KDA layer or the MLA runtime (with its ``mixer``)."""

    def __init__(self, log: _Log) -> None:
        super().__init__()
        self.log = log
        self.mixer = nn.Module()

    def forward(self, hidden_states: torch.Tensor, attn_metadata: object) -> torch.Tensor:
        self.log.append(("attn", None))
        return hidden_states * 2


class _MoE(nn.Module):
    """Stands in for ``K31MoE``: the children that hold its dense weights, the routed experts, and a forward that
    logs (the DEP target asks for the routed and shared parts)."""

    def __init__(self, log: _Log) -> None:
        super().__init__()
        self.log = log
        self.gate = nn.Linear(_HIDDEN, 4, bias=False)
        self.shared_experts = nn.Linear(_HIDDEN, _HIDDEN, bias=False)
        self.routed_expert_down_proj = nn.Linear(_HIDDEN, 4, bias=False)
        self.routed_expert_up_proj = nn.Linear(4, _HIDDEN, bias=False)
        self.routed_expert_norm = nn.LayerNorm(4)
        self.routed_experts = nn.Linear(4, 4, bias=False)

    def forward(self, hidden_states, all_rank_num_tokens=None, return_parts: bool = False):
        self.log.append(("moe", None))
        if return_parts:
            return hidden_states, torch.zeros_like(hidden_states)
        return hidden_states


class _Mlp(nn.Module):
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states

    def low_m_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states


def _norm() -> nn.Module:
    norm = nn.Module()
    norm.weight = None
    norm.variance_epsilon = 1e-6
    return norm


def _layer(target, log: _Log, layer_idx: int, *, is_kda=True, is_moe=True, covered=(1, 2, 3)):
    """A ``K31DecoderLayer`` shell of ``target`` whose submodules are recorders."""
    layer = target.K31DecoderLayer.__new__(target.K31DecoderLayer)
    nn.Module.__init__(layer)
    layer.layer_idx = layer_idx
    layer.is_kda = is_kda
    layer.is_moe = is_moe
    layer.attn_res_block_size = 1000
    for name in (
        "input_layernorm",
        "post_attention_layernorm",
        "self_attention_res_norm",
        "mlp_res_norm",
        "self_attention_res_proj",
        "mlp_res_proj",
    ):
        setattr(layer, name, _norm())
    if is_kda:
        layer.linear_attn = _Attention(log)
    else:
        layer.self_attn = _Attention(log)
    if is_moe:
        layer.block_sparse_moe = _MoE(log)
    else:
        layer.mlp = _Mlp()
    layer._dense_dwdp = types.SimpleNamespace(dense=_Dense(log, covered))
    return layer


@pytest.fixture
def plain_mixes(monkeypatch):
    """The norms and attention-residual mixes as plain arithmetic, for both targets."""
    for target in (dep, general):
        monkeypatch.setattr(target, "flashinfer_rmsnorm", lambda x, *args, **kwargs: x)
        monkeypatch.setattr(target, "_apply_attn_res_and_rmsnorm", lambda prefix, *args: prefix)
        monkeypatch.setattr(
            target,
            "_apply_attn_res_add_and_rmsnorm",
            lambda prefix, addend, *args, **kwargs: (prefix + addend, addend),
        )
    monkeypatch.setattr(dep, "_k31_residual_add_fits", lambda *tensors: False)


def _run(layer, metadata) -> torch.Tensor:
    """One layer forward on ones, with one snapshot in the bank; returns the new prefix sum."""
    out, _ = layer(torch.ones(_TOKENS, _HIDDEN), torch.zeros(2, _TOKENS, _HIDDEN), 1, metadata)
    return out


# ---------------------------------------------------------------------------------------------------------------------
# Hooks
# ---------------------------------------------------------------------------------------------------------------------


@_TARGETS
@pytest.mark.parametrize("is_kda", [True, False], ids=["kda", "mla"])
def test_a_covered_layer_runs_its_hooks_around_attention_and_moe(
    plain_mixes, target, is_kda
) -> None:
    log = _Log()
    layer = _layer(target, log, 2, is_kda=is_kda)
    out = _run(layer, types.SimpleNamespace(all_rank_num_tokens=None))
    assert log == [("H1", 2), ("attn", None), ("H2", 2), ("moe", None), ("H3", 2)]
    # prefix (1) + attention (2) + MoE of the mixed rows (2).
    assert torch.equal(out, torch.full((_TOKENS, _HIDDEN), 5.0))


@_TARGETS
def test_a_layer_that_dense_dwdp_does_not_cover_runs_no_hook(plain_mixes, target) -> None:
    """Layer 0 (the dense MLP) is not covered."""
    log = _Log()
    _run(_layer(target, log, 0, is_moe=False), types.SimpleNamespace(all_rank_num_tokens=None))
    assert log == [("attn", None)]


@_TARGETS
@pytest.mark.parametrize("manager", ["no_manager", "no_runtime"])
def test_without_dense_dwdp_a_layer_runs_no_hook(plain_mixes, target, manager) -> None:
    log = _Log()
    layer = _layer(target, log, 2)
    layer._dense_dwdp = None if manager == "no_manager" else types.SimpleNamespace(dense=None)
    _run(layer, types.SimpleNamespace(all_rank_num_tokens=None))
    assert log == [("attn", None), ("moe", None)]


# ---------------------------------------------------------------------------------------------------------------------
# Dense roots and tables
# ---------------------------------------------------------------------------------------------------------------------


@_TARGETS
@pytest.mark.parametrize("is_kda", [True, False], ids=["kda", "mla"])
def test_the_dense_roots_of_a_moe_layer(target, is_kda) -> None:
    layer = _layer(target, _Log(), 2, is_kda=is_kda)
    attention = ("linear_attn", layer.linear_attn) if is_kda else ("self_attn", layer.self_attn)
    assert layer.dense_dwdp_roots("attn") == [attention]
    moe = layer.block_sparse_moe
    assert layer.dense_dwdp_roots("moe") == [
        ("block_sparse_moe.gate", moe.gate),
        ("block_sparse_moe.shared_experts", moe.shared_experts),
        ("block_sparse_moe.routed_expert_down_proj", moe.routed_expert_down_proj),
        ("block_sparse_moe.routed_expert_up_proj", moe.routed_expert_up_proj),
        ("block_sparse_moe.routed_expert_norm", moe.routed_expert_norm),
    ]
    # The routed experts belong to the MoE all-to-all and are never a dense root.
    assert all(module is not moe.routed_experts for _, module in layer.dense_dwdp_roots("moe"))


@_TARGETS
def test_layers_without_moe_or_weights_stay_replicated(target) -> None:
    assert _layer(target, _Log(), 0, is_moe=False).dense_dwdp_roots("attn") == []
    layer = _layer(target, _Log(), 2)
    layer._weights_removed = True
    assert layer.dense_dwdp_roots("attn") == [] and layer.dense_dwdp_roots("moe") == []


@_TARGETS
def test_an_unknown_segment_raises(target) -> None:
    with pytest.raises(ValueError, match="segment must be one of"):
        _layer(target, _Log(), 2).dense_dwdp_roots("mlp")


@_TARGETS
def test_the_tables(target) -> None:
    allow = target.K31DecoderLayer.dense_dwdp_attr_allow
    exclude = target.K31DecoderLayer.dense_dwdp_exclude
    assert isinstance(allow, frozenset) and isinstance(exclude, frozenset)
    assert not allow & exclude
    # The derived weights the layer's attention reads directly move with the layer.
    assert {
        "_qkvg_proj_weight",
        "_bfa_proj_weight",
        "_packed_conv_weight",
        "_A_log_f32",
        "_dt_bias_f32",
        "_k31_packed_hidden_weight",
    } <= allow
    assert {"_o_dense", "e_score_correction_bias"} <= exclude
    assert target.K31DecoderLayer._dense_dwdp is None


# ---------------------------------------------------------------------------------------------------------------------
# The DEP target's KDA pair split
# ---------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "mode,standalone", [(None, False), (DenseMode.STANDALONE, True), (DenseMode.ATTACHED, False)]
)
def test_dense_dwdp_turns_the_kda_pair_split_off(monkeypatch, mode, standalone) -> None:
    """Under standalone dense DWDP the DEP target's KDA layers keep the whole fused [q | k | v | g] weight (one
    storage per layer, which dense DWDP shards) instead of the pair's halves in one pool across the layers."""
    registered = None if mode is None else types.SimpleNamespace(mode=mode)
    monkeypatch.setattr(dep, "get_global_dense_dwdp_manager", lambda: registered)
    assert dep._standalone_dense_dwdp() is standalone


# ---------------------------------------------------------------------------------------------------------------------
# Load guards
# ---------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("target_weights", [dep_weights, general_weights], ids=["dep", "general"])
def test_weight_loading_refuses_while_dense_dwdp_is_active(target_weights) -> None:
    """Once dense DWDP moved the dense weights into staging slots shared across layers, a load (the target's, before it
    touches the vision tower, or the text model's) is refused."""
    llm = types.SimpleNamespace(model=types.SimpleNamespace(_dense_dwdp_active=True))
    with pytest.raises(RuntimeError, match="dense DWDP is active"):
        target_weights.load(
            types.SimpleNamespace(llm=llm), {"vision_tower.patch_embed.weight": None}
        )
    with pytest.raises(RuntimeError, match="dense DWDP is active"):
        target_weights.load_text_model(llm, {})
