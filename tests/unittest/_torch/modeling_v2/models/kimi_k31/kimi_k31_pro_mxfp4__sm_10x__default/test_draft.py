# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Each target's DFlash2 drafter (``DraftModelBase`` in its ``modeling.py``) on the CPU: the shapes and tensor-parallel
shards its loader gives every backbone tensor under plain TP and under attention DP (the engine ``Linear``'s even
split, ``gate_up_proj`` = this rank's ``[gate | up]``), the context-only subset, and the drafter ``ModelConfig`` the
causal LM builds (``dflash2_draft_config``), and the one window the drafter and the KV cache manager read from it."""

import importlib
import json
import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from transformers import PretrainedConfig

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.gemm import bf16_linear
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.spec import (
    dflash2_draft_pages as pages_mod,
)
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.modules.linear import Linear
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization import QuantAlgo

pytestmark = pytest.mark.cpu_only

_FAMILY = "tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31"


@pytest.fixture(
    params=[
        "kimi_k31_pro_mxfp4__sm_103__tep",
        "kimi_k31_pro_mxfp4__sm_103__dep",
        "kimi_k31_pro_mxfp4__sm_10x__default",
    ],
    ids=["tep", "dep", "default"],
)
def target_modeling(request):
    """The target's ``modeling`` module, which holds its copy of the drafter."""
    return importlib.import_module(f"{_FAMILY}.{request.param}.modeling")


# A small drafter of the checkpoint's structure (heads and widths split evenly over up to 4 ranks).
_H, _NH, _QL, _KVL, _NOPE, _ROPE, _V, _I, _VOCAB, _L = 64, 8, 24, 16, 8, 8, 8, 48, 128, 2
_ROPE_PARAMS = dict(
    rope_type="yarn",
    rope_theta=50000.0,
    factor=32.0,
    original_max_position_embeddings=32768,
    beta_fast=32,
    beta_slow=1,
    mscale=1.0,
    mscale_all_dim=1.0,
)
_DFLASH = dict(
    attention_mode="mla",
    block_size=5,
    conv_kernel_size=2,
    conv_group_size=16,
    selector_rank=256,
    selector_top_k=32,
    mask_token_id=_VOCAB - 1,
    target_layer_ids=[0, 1],
)


def _pretrained() -> PretrainedConfig:
    cfg = PretrainedConfig(
        architectures=["DFlash2DraftModel"],
        hidden_size=_H,
        num_attention_heads=_NH,
        num_key_value_heads=_NH,
        q_lora_rank=_QL,
        kv_lora_rank=_KVL,
        qk_nope_head_dim=_NOPE,
        qk_rope_head_dim=_ROPE,
        v_head_dim=_V,
        intermediate_size=_I,
        num_hidden_layers=_L,
        rms_norm_eps=1e-5,
        vocab_size=_VOCAB,
        max_position_embeddings=4096,
        rope_parameters=dict(_ROPE_PARAMS),
        rope_theta=50000.0,
        sliding_window=8,
        layer_types=["sliding_attention"] * _L,
        is_causal=True,
        dflash_config=dict(_DFLASH),
    )
    cfg.model_type = "qwen3"
    cfg.torch_dtype = torch.bfloat16
    return cfg


def _weights() -> dict:
    """Every backbone tensor with distinct values, so a wrong slice cannot match."""
    shapes = {}
    for i in range(_L):
        p = f"layers.{i}."
        shapes.update(
            {
                p + "input_layernorm.weight": (_H,),
                p + "post_attention_layernorm.weight": (_H,),
                p + "self_attn.q_a_proj.weight": (_QL, _H),
                p + "self_attn.q_a_layernorm.weight": (_QL,),
                p + "self_attn.q_b_proj.weight": (_NH * (_NOPE + _ROPE), _QL),
                p + "self_attn.kv_a_proj_with_mqa.weight": (_KVL + _ROPE, _H),
                p + "self_attn.kv_a_layernorm.weight": (_KVL,),
                p + "self_attn.kv_b_proj.weight": (_NH * (_NOPE + _V), _KVL),
                p + "self_attn.o_proj.weight": (_H, _NH * _V),
                p + "mlp.gate_proj.weight": (_I, _H),
                p + "mlp.up_proj.weight": (_I, _H),
                p + "mlp.down_proj.weight": (_H, _I),
            }
        )
    shapes["norm.weight"] = (_H,)
    out, base = {}, 0
    for name, shape in shapes.items():
        n = int(torch.Size(shape).numel())
        out[name] = (
            (torch.arange(base, base + n, dtype=torch.float32) % 251).view(shape).to(torch.bfloat16)
        )
        base += n
    return out


def _rows(t: torch.Tensor, tp: int, rank: int, dim: int) -> torch.Tensor:
    """This rank's slice of ``t`` along ``dim`` by the engine ``Linear``'s split."""
    width = t.shape[dim]
    start, end = Linear._calc_shard(width, tp, rank), Linear._calc_shard(width, tp, rank + 1)
    return t.narrow(dim, start, end - start)


def _drafter(target_modeling, tp: int, rank: int, adp: bool, context_kv_only: bool = False):
    config = ModelConfig(
        pretrained_config=_pretrained(),
        mapping=Mapping(
            world_size=tp, tp_size=tp, rank=rank, gpus_per_node=8, enable_attention_dp=adp
        ),
    )
    stub = SimpleNamespace(mnnvl_allreduce=None, uses_nccl_symmetric_memory_window=lambda: False)
    # One process stands in for one rank: no TP group to build communication workspaces for.
    with (
        patch.object(bf16_linear, "tp_allreduce", return_value=stub),
        patch.object(target_modeling, "get_sm_version", return_value=100),
    ):
        return target_modeling.DraftModelBase(config, block_size=5, context_kv_only=context_kv_only)


@pytest.mark.parametrize("tp, rank", [(2, 0), (2, 1), (4, 3)])
def test_plain_tp_shards(target_modeling, tp, rank):
    drafter = _drafter(target_modeling, tp, rank, adp=False)
    weights = _weights()
    drafter._load_backbone_weights(weights, torch.device("cpu"))
    assert drafter._num_heads == _NH // tp
    for i, layer in enumerate(drafter.model.layers):
        p, attn, mlp = f"layers.{i}.", layer.self_attn, layer.mlp
        # Replicated: the *_a projections and every norm.
        for name in ("q_a_proj", "kv_a_proj_with_mqa", "q_a_layernorm", "kv_a_layernorm"):
            assert torch.equal(getattr(attn, name).weight, weights[p + f"self_attn.{name}.weight"])
        for name in ("input_layernorm", "post_attention_layernorm"):
            assert torch.equal(getattr(layer, name).weight, weights[p + f"{name}.weight"])
        # Column-parallel by head, row-parallel o_proj summed over the group.
        for name in ("q_b_proj", "kv_b_proj"):
            want = _rows(weights[p + f"self_attn.{name}.weight"], tp, rank, 0)
            assert torch.equal(getattr(attn, name).weight, want)
        assert torch.equal(
            attn.o_proj.weight, _rows(weights[p + "self_attn.o_proj.weight"], tp, rank, 1)
        )
        assert attn.o_proj.all_reduce is not None
        # The MLP: [gate_r | up_r] rows, the matching down_proj columns.
        gate_up = torch.cat(
            [
                _rows(weights[p + "mlp.gate_proj.weight"], tp, rank, 0),
                _rows(weights[p + "mlp.up_proj.weight"], tp, rank, 0),
            ]
        )
        assert torch.equal(mlp.gate_up_proj.weight, gate_up)
        assert torch.equal(
            mlp.down_proj.weight, _rows(weights[p + "mlp.down_proj.weight"], tp, rank, 1)
        )
        assert mlp.down_proj.all_reduce is not None and mlp.down_proj.tp_size == tp
    assert torch.equal(drafter.model.norm.weight, weights["norm.weight"])


def test_attention_dp_replicates(target_modeling):
    """Attention DP collapses the attention group to one rank and replicates the MLP (one-rank subgroups)."""
    drafter = _drafter(target_modeling, 4, 2, adp=True)
    weights = _weights()
    drafter._load_backbone_weights(weights, torch.device("cpu"))
    assert drafter._num_heads == _NH
    for i, layer in enumerate(drafter.model.layers):
        p, attn, mlp = f"layers.{i}.", layer.self_attn, layer.mlp
        for name in ("q_b_proj", "kv_b_proj", "o_proj"):
            assert torch.equal(getattr(attn, name).weight, weights[p + f"self_attn.{name}.weight"])
        assert attn.o_proj.all_reduce is None
        assert mlp.gate_up_proj.tp_size == mlp.down_proj.tp_size == 1
        gate_up = torch.cat(
            [weights[p + "mlp.gate_proj.weight"], weights[p + "mlp.up_proj.weight"]]
        )
        assert torch.equal(mlp.gate_up_proj.weight, gate_up)
        assert torch.equal(mlp.down_proj.weight, weights[p + "mlp.down_proj.weight"])


def test_context_only_takes_the_latent_projection(target_modeling):
    drafter = _drafter(target_modeling, 2, 1, adp=False, context_kv_only=True)
    weights = {
        k: v
        for k, v in _weights().items()
        if ".self_attn.kv_a_proj_with_mqa." in k or ".self_attn.kv_a_layernorm." in k
    }
    drafter._load_backbone_weights(weights, torch.device("cpu"))
    assert drafter.model.norm is None
    for i, layer in enumerate(drafter.model.layers):
        assert layer.mlp is None and layer.self_attn.q_b_proj is None
        key = f"layers.{i}.self_attn.kv_a_proj_with_mqa.weight"
        assert torch.equal(layer.self_attn.kv_a_proj_with_mqa.weight, weights[key])


def test_missing_backbone_tensor_is_named(target_modeling):
    drafter = _drafter(target_modeling, 2, 0, adp=False)
    weights = _weights()
    del weights["layers.1.mlp.up_proj.weight"]
    with pytest.raises(ValueError, match="layers.1.mlp.up_proj.weight"):
        drafter._load_backbone_weights(weights, torch.device("cpu"))


@pytest.mark.parametrize("adp", [False, True])
def test_draft_config_follows_the_target(target_modeling, tmp_path, adp):
    """The drafter's ModelConfig: its own checkpoint, the target's rows-wide mapping, KV dtype, max_seq_len and the
    target's shared extra_attrs."""
    checkpoint = _pretrained().to_dict()
    # to_dict writes the class's model_type (empty for a bare PretrainedConfig), not the instance's.
    checkpoint.update(model_type="qwen3", torch_dtype="bfloat16")
    with open(os.path.join(tmp_path, "config.json"), "w") as f:
        json.dump(checkpoint, f)
    mapping = Mapping(world_size=4, tp_size=4, rank=1, gpus_per_node=4, enable_attention_dp=adp)
    target = ModelConfig(
        pretrained_config=_pretrained(),
        mapping=mapping,
        quant_config=QuantConfig(kv_cache_quant_algo=QuantAlgo.FP8),
        max_seq_len=8192,
    )
    target.extra_attrs = {"marker": object()}
    draft_config = target_modeling.dflash2_draft_config(
        target, SimpleNamespace(speculative_model=str(tmp_path), moe_backend=None)
    )
    full_rows = mapping.mapping_for_full_rows()
    assert (draft_config.mapping.tp_size, draft_config.mapping.rank) == (
        full_rows.tp_size,
        full_rows.rank,
    )
    assert draft_config.mapping.enable_attention_dp == full_rows.enable_attention_dp
    assert draft_config.quant_config.kv_cache_quant_algo == QuantAlgo.FP8
    assert draft_config.extra_attrs is target.extra_attrs
    assert draft_config.max_seq_len == 8192
    assert draft_config.spec_config is None
    assert draft_config.pretrained_config.dflash_config["attention_mode"] == "mla"


def _write_drafter_checkpoint(path, **overrides) -> None:
    """A drafter ``config.json`` laid out like the released one (qwen3, every layer a sliding_attention layer)."""
    checkpoint = _pretrained().to_dict()
    # to_dict writes the class's model_type (empty for a bare PretrainedConfig), not the instance's.
    checkpoint.update(
        model_type="qwen3", torch_dtype="bfloat16", use_sliding_window=True, max_window_layers=_L
    )
    for key, value in overrides.items():
        if value is None:
            checkpoint.pop(key, None)
        else:
            checkpoint[key] = value
    with open(os.path.join(path, "config.json"), "w") as f:
        json.dump(checkpoint, f)


def _drafter_window_or_error(read):
    try:
        return read()
    except ValueError:
        return ValueError


@pytest.mark.parametrize(
    "overrides, window",
    [
        ({}, 8),
        # No layer_types: the config class derives them (every layer sliding above max_window_layers).
        (dict(layer_types=None, max_window_layers=0), 8),
        (dict(layer_types=None), ValueError),
        (dict(use_sliding_window=False), ValueError),
        (dict(layer_types=["full_attention"] + ["sliding_attention"] * (_L - 1)), ValueError),
        (dict(is_causal=False), ValueError),
        # Not an MLA drafter: no window for the KV cache manager (the drafter refuses it on its own).
        (dict(dflash_config=dict(_DFLASH, attention_mode="gqa")), None),
    ],
    ids=[
        "released",
        "derived_layer_types",
        "full_layers",
        "window_off",
        "full_layer",
        "not_causal",
        "gqa",
    ],
)
def test_drafter_and_kv_manager_read_one_window(target_modeling, tmp_path, overrides, window):
    """The drafter's window and the window the KV cache manager keeps of its context come from one reading of one
    parsed config."""
    _write_drafter_checkpoint(tmp_path, **overrides)
    pages_mod.dflash2_drafter_config.cache_clear()
    spec = SimpleNamespace(
        speculative_model=str(tmp_path),
        moe_backend=None,
        spec_dec_mode=SimpleNamespace(is_dflash=lambda: True),
    )
    target = ModelConfig(pretrained_config=_pretrained(), mapping=Mapping(), max_seq_len=8192)
    assert target_modeling.dflash2_sliding_window is pages_mod.dflash2_sliding_window
    from_drafter = _drafter_window_or_error(
        lambda: target_modeling.dflash2_sliding_window(
            target_modeling.dflash2_draft_config(target, spec).pretrained_config
        )
    )
    from_manager = _drafter_window_or_error(lambda: pages_mod.dflash2_draft_window(spec))
    assert from_drafter == from_manager == window


def test_drafter_decodes_with_the_shared_window(target_modeling):
    drafter = _drafter(target_modeling, 2, 0, adp=False)
    assert (
        drafter._window
        == drafter._ctx_ring_window
        == pages_mod.dflash2_sliding_window(_pretrained())
        == 8
    )


def test_draft_window_of_other_speculative_configs():
    assert pages_mod.dflash2_draft_window(None) is None
    other = SimpleNamespace(
        spec_dec_mode=SimpleNamespace(is_dflash=lambda: False), speculative_model="unused"
    )
    assert pages_mod.dflash2_draft_window(other) is None
