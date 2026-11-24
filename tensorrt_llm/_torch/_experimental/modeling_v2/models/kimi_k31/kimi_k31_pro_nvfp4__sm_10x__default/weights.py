# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Rank-local checkpoint loading for the Kimi K3.1 Pro NVFP4/MXFP8 target.

Stream complete NVFP4 routed-expert tuples into local TP/EP slots. Load MXFP8 KDA/shared
projections with their UE8M0 scales, preserving native code values through sharding and
fusion. Load unquantized text/vision tensors in their declared precision and retain FP32
KDA gate parameters. Check expected key coverage and complete local expert ownership.
"""

from __future__ import annotations

import copy
import glob
import json
import os
import threading
import time
import weakref
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    NamedTuple,
    Optional,
    Set,
    Tuple,
)

import torch
from safetensors import safe_open
from torch import nn

from tensorrt_llm._torch.models.checkpoints.base_weight_loader import ConsumableWeightsDict
from tensorrt_llm._torch.models.modeling_kimi_k25 import _has_meta_tensors
from tensorrt_llm._torch.models.modeling_utils import filter_weights, run_concurrently
from tensorrt_llm.logger import logger
from tensorrt_llm.models.modeling_utils import QuantAlgo

if TYPE_CHECKING:
    from .modeling import KimiK3MoERuntime  # noqa: E501


@dataclass
class DraftAttentionState:
    num_heads_tp: int
    num_heads_tp_cp: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    kv_lora_rank: int
    v_head_dim: int
    kv_cache_head_dim: int
    tp_rank: int
    output_rank: int
    output_all_reduce: Any


@dataclass
class DraftLayerState:
    self_attn: DraftAttentionState
    mlp_all_reduce: Any
    mlp_rank: int
    intermediate: int


def declare_draft(bank, cfg, model_config, helix):
    from tensorrt_llm._torch.distributed import AllReduce

    full = model_config.mapping
    attention = helix or full
    tp = 1 if attention.enable_attention_dp else attention.tp_size
    heads = cfg.num_attention_heads // tp
    if heads * tp != cfg.num_attention_heads:
        raise ValueError("Draft attention heads must divide its TP group")
    cp = attention.cp_size
    if heads % cp:
        raise ValueError("Draft attention heads must divide its CP group")
    # The repurposed TP mapping makes CP ranks replicas of each Q/KV head slice.
    attn_rank = full.rank % tp
    output_tp = 1 if full.enable_attention_dp else full.tp_size
    output_rank = full.tp_rank if output_tp > 1 else 0
    local_out_heads = cfg.num_attention_heads // output_tp
    mlp_tp = 1 if full.enable_attention_dp else full.tp_size
    if cfg.intermediate_size % mlp_tp:
        raise ValueError("Draft MLP intermediate dimension must divide its TP group")
    intermediate = cfg.intermediate_size // mlp_tp
    layers = []
    for i in range(cfg.num_hidden_layers):
        stem = f"l{i}_"
        shapes = {
            "qa": (cfg.q_lora_rank, cfg.hidden_size),
            "qn": (cfg.q_lora_rank,),
            "qb": (heads * (cfg.qk_nope_head_dim + cfg.qk_rope_head_dim), cfg.q_lora_rank),
            "kva": (cfg.kv_lora_rank + cfg.qk_rope_head_dim, cfg.hidden_size),
            "kvn": (cfg.kv_lora_rank,),
            "kvb": (heads * (cfg.qk_nope_head_dim + cfg.v_head_dim), cfg.kv_lora_rank),
            "out": (cfg.hidden_size, local_out_heads * cfg.v_head_dim),
            "gu": (2 * intermediate, cfg.hidden_size),
            "down": (cfg.hidden_size, intermediate),
            "norm1": (cfg.hidden_size,),
            "norm2": (cfg.hidden_size,),
        }
        for name, shape in shapes.items():
            bank[stem + name] = nn.Parameter(
                torch.empty(shape, dtype=cfg.torch_dtype), requires_grad=False
            )

        def reduce():
            if output_tp > 1:
                return AllReduce(
                    mapping=full, strategy=model_config.allreduce_strategy, dtype=cfg.torch_dtype
                )
            return None

        attn = DraftAttentionState(
            heads,
            heads // cp,
            cfg.qk_nope_head_dim,
            cfg.qk_rope_head_dim,
            cfg.kv_lora_rank,
            cfg.v_head_dim,
            cfg.kv_lora_rank + cfg.qk_rope_head_dim,
            attn_rank,
            output_rank,
            reduce(),
        )
        layers.append(
            DraftLayerState(attn, reduce(), full.tp_rank if mlp_tp > 1 else 0, intermediate)
        )
    bank["norm"] = nn.Parameter(
        torch.empty(cfg.hidden_size, dtype=cfg.torch_dtype), requires_grad=False
    )
    return layers


def draft_checkpoint_keys(model):
    keys = {"model.norm.weight"}
    for i in range(len(model.layers)):
        keys.update(
            f"model.layers.{i}.{suffix}.weight"
            for suffix in (
                "self_attn.q_a_proj",
                "self_attn.q_a_layernorm",
                "self_attn.q_b_proj",
                "self_attn.kv_a_proj_with_mqa",
                "self_attn.kv_a_layernorm",
                "self_attn.kv_b_proj",
                "self_attn.o_proj",
                "input_layernorm",
                "post_attention_layernorm",
                "mlp.gate_proj",
                "mlp.up_proj",
                "mlp.down_proj",
            )
        )
    return keys


def load_draft(model, weights):
    missing = draft_checkpoint_keys(model) - weights.keys()
    if missing:
        raise ValueError(f"Draft checkpoint is missing {sorted(missing)[:8]}")
    bank = model.w
    device = bank["norm"].device
    for i, layer in enumerate(model.layers):
        a = layer.self_attn
        stem, prefix = f"l{i}_", f"model.layers.{i}."

        def read(name):
            return _materialize(weights[prefix + name + ".weight"]).to(device=device)

        qdim = a.qk_nope_head_dim + a.qk_rope_head_dim
        kvdim = a.qk_nope_head_dim + a.v_head_dim
        out_size = bank[stem + "out"].shape[1]
        inter_lo = layer.mlp_rank * layer.intermediate
        values = {
            "qa": read("self_attn.q_a_proj"),
            "qn": read("self_attn.q_a_layernorm"),
            "qb": read("self_attn.q_b_proj").narrow(
                0, a.tp_rank * a.num_heads_tp * qdim, a.num_heads_tp * qdim
            ),
            "kva": read("self_attn.kv_a_proj_with_mqa"),
            "kvn": read("self_attn.kv_a_layernorm"),
            "kvb": read("self_attn.kv_b_proj").narrow(
                0, a.tp_rank * a.num_heads_tp * kvdim, a.num_heads_tp * kvdim
            ),
            "out": read("self_attn.o_proj").narrow(1, a.output_rank * out_size, out_size),
            "gu": torch.cat(
                [
                    read("mlp." + name).narrow(0, inter_lo, layer.intermediate)
                    for name in ("gate_proj", "up_proj")
                ]
            ),
            "down": read("mlp.down_proj").narrow(1, inter_lo, layer.intermediate),
            "norm1": read("input_layernorm"),
            "norm2": read("post_attention_layernorm"),
        }
        for name, value in values.items():
            dst = bank[stem + name]
            if dst.shape != value.shape:
                raise ValueError(f"{prefix}{name}: {value.shape} does not match {dst.shape}")
            dst.data.copy_(value)
    bank["norm"].data.copy_(_materialize(weights["model.norm.weight"]))
    bind_draft(model)


def bind_draft(model):
    """Refresh forward tensor views after loading or moving the parameter bank."""
    cfg = model.model_config.pretrained_config
    bank = model.w
    for i, layer in enumerate(model.layers):
        a = layer.self_attn
        stem = f"l{i}_"
        for field, name in (
            ("q_a_proj", "qa"),
            ("q_a_layernorm", "qn"),
            ("q_b_proj", "qb"),
            ("kv_a_proj_with_mqa", "kva"),
            ("kv_a_layernorm", "kvn"),
            ("kv_b_proj", "kvb"),
            ("o_proj", "out"),
        ):
            setattr(a, field, TensorWeights(bank[stem + name].detach(), cfg.rms_norm_eps))
        layer.input_layernorm = TensorWeights(bank[stem + "norm1"].detach(), cfg.rms_norm_eps)
        layer.post_attention_layernorm = TensorWeights(
            bank[stem + "norm2"].detach(), cfg.rms_norm_eps
        )
        layer.gate_up = bank[stem + "gu"].detach()
        layer.down = bank[stem + "down"].detach()
    model.norm = TensorWeights(bank["norm"].detach(), cfg.rms_norm_eps)


class TensorWeights(NamedTuple):
    weight: torch.Tensor
    eps: float = 0.0

    @property
    def variance_epsilon(self):
        return self.eps


def declare_auxiliary(bank, owner, field, source, shape, dtype, eps=0.0):
    key = source.replace(".", "__")
    bank[key] = nn.Parameter(torch.empty(shape, dtype=dtype), requires_grad=False)
    if not hasattr(bank, "checkpoint_sources"):
        bank.checkpoint_sources = {}
    bank.checkpoint_sources[key] = source
    if not hasattr(owner, "tensor_fields"):
        owner.tensor_fields = {}
    owner.tensor_fields[field] = (key, eps)
    setattr(owner, field, TensorWeights(bank[key], eps))


def bind_auxiliary(bank, owner):
    for field, (key, eps) in getattr(owner, "tensor_fields", {}).items():
        setattr(owner, field, TensorWeights(bank[key].detach(), eps))


def load_auxiliary(bank, weights, prefix=""):
    for key, source in bank.checkpoint_sources.items():
        value = _read_unquantized(weights, prefix + source)
        dst = bank[key]
        if value.shape != dst.shape:
            raise ValueError(f"{source}: {value.shape} does not match {dst.shape}")
        dst.data.copy_(value.to(dtype=dst.dtype))


@dataclass
class LayerState:
    layer_idx: int
    is_kda: bool
    is_moe: bool
    attn_res_block_size: int
    linear_attn: Any = None
    mla: Any = None
    mla_all_reduce: Any = None
    block_sparse_moe: Any = None
    mlp: Any = None
    mlp_weight_view: Any = None


def declare_layer(bank, model_config, cfg, layer_idx, streams):
    import math

    from tensorrt_llm._torch.distributed import AllReduce
    from tensorrt_llm._torch.utils import AuxStreamType

    from .modeling import KimiK3MoERuntime, _is_kda_layer, _is_mla_layer

    is_kda = _is_kda_layer(cfg, layer_idx)
    if is_kda == _is_mla_layer(cfg, layer_idx):
        raise ValueError(f"Layer {layer_idx} must select exactly one of KDA and MLA")
    is_moe = (
        cfg.num_experts is not None
        and layer_idx >= cfg.first_k_dense_replace
        and layer_idx % getattr(cfg, "moe_layer_freq", 1) == 0
    )
    state = LayerState(layer_idx, is_kda, is_moe, cfg.attn_res_block_size)
    mapping = model_config.mapping.mapping_for_full_rows()
    if is_kda:
        state.linear_attn = declare_kda(bank, cfg, layer_idx, model_config, mapping)
    else:
        state.mla = declare_mla(
            bank, cfg, layer_idx, model_config, streams[AuxStreamType.Attention]
        )
        if mapping.tp_size > 1 and not mapping.enable_attention_dp:
            state.mla_all_reduce = AllReduce(
                mapping=mapping, strategy=model_config.allreduce_strategy, dtype=torch.bfloat16
            )
    if is_moe:
        state.block_sparse_moe = KimiK3MoERuntime(model_config, cfg, layer_idx, streams, bank)
    else:
        tp = 1 if mapping.enable_attention_dp else math.gcd(cfg.intermediate_size, mapping.tp_size)
        if tp > mapping.gpus_per_node:
            tp = math.gcd(tp, mapping.gpus_per_node)
        mlp_config = copy.copy(model_config)
        mlp_config._frozen = False
        mlp_config.mapping = mapping
        state.mlp = declare_mlp(
            bank,
            layer_idx,
            cfg,
            mlp_config,
            shared=False,
            tp_size=tp,
            beta=getattr(cfg, "activation_situ_beta", None) or 1.0,
            linear_beta=getattr(cfg, "activation_situ_linear_beta", None),
        )
    for name in (
        "input_layernorm",
        "post_attention_layernorm",
        "self_attention_res_norm",
        "mlp_res_norm",
        "self_attention_res_proj",
        "mlp_res_proj",
    ):
        shape = (1, cfg.hidden_size) if name.endswith("proj") else (cfg.hidden_size,)
        declare_auxiliary(
            bank,
            state,
            name,
            f"model.layers.{layer_idx}.{name}.weight",
            shape,
            torch.bfloat16,
            cfg.rms_norm_eps,
        )
    return state


@dataclass
class MLAState:
    layer_idx: int
    layer_idx_str: str
    mqa: Any
    mapping: Any
    num_heads_tp: int
    num_heads_tp_cp: int
    num_heads_attn: int
    kv_lora_rank: int
    q_lora_rank: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    v_head_dim: int
    rms_norm_eps: float
    aux_stream: Any
    ln_events: Any
    tp_rank: int
    output_rank: int
    use_output_gate: bool
    k_b_proj_trans: Any = None
    v_b_proj: Any = None
    weight_view: Any = None
    out_scale: Any = None
    apply_rotary_emb: bool = False
    zero_padding_output_rows: bool = True

    @property
    def qk_head_dim(self):
        return self.qk_nope_head_dim + self.qk_rope_head_dim

    @property
    def attention_output_hidden_size(self):
        return self.num_heads_tp_cp * self.v_head_dim


def declare_mla(bank, cfg, layer_idx, model_config, aux_stream):
    from tensorrt_llm.models.modeling_utils import QuantConfig

    from .modeling import (
        _install_identity_rope_table,
        _make_pos_embd_params,
        padded_attention_heads,
    )
    from .ops.attention.backend import create_attention, k31_dsa_params
    from .ops.attention.cache_policy import get_kimi_k31_fp8_kv_layer_ids

    mapping = getattr(model_config, "_helix_mapping_with_cp", None) or model_config.mapping
    tp = 1 if mapping.enable_attention_dp else mapping.tp_size
    cp = mapping.cp_size
    if cfg.num_attention_heads % (tp * cp):
        raise ValueError("MLA heads must divide the TP/CP group")
    heads = cfg.num_attention_heads // tp
    local_heads = heads // cp
    maximum = int(os.environ.get("KIMI_K3_MLA_MAX_POSITIONS", cfg.max_position_embeddings))
    pos = _make_pos_embd_params(
        qk_rope_head_dim=cfg.qk_rope_head_dim, max_position_embeddings=maximum
    )
    sparse_cfg = model_config.sparse_attention_config
    sparse = k31_dsa_params(
        sparse_cfg.to_sparse_params(pretrained_config=cfg, layer_idx=layer_idx), cfg
    )
    algo = model_config.quant_config.kv_cache_quant_algo
    if layer_idx in get_kimi_k31_fp8_kv_layer_ids(cfg, nvfp4_enabled=algo == QuantAlgo.NVFP4):
        algo = QuantAlgo.FP8
    backend = create_attention(
        model_config.attn_backend,
        layer_idx,
        padded_attention_heads(heads),
        head_dim=cfg.kv_lora_rank + cfg.qk_rope_head_dim,
        num_kv_heads=1,
        pos_embd_params=pos,
        quant_config=QuantConfig(kv_cache_quant_algo=algo),
        q_scaling=1.0,
        is_mla_enable=True,
        q_lora_rank=cfg.q_lora_rank,
        kv_lora_rank=cfg.kv_lora_rank,
        qk_nope_head_dim=cfg.qk_nope_head_dim,
        qk_rope_head_dim=cfg.qk_rope_head_dim,
        hidden_size=cfg.hidden_size,
        predicted_tokens_per_seq=1,
        skip_create_weights_in_init=False,
        v_head_dim=cfg.kv_lora_rank,
        sparse_params=sparse,
        dtype=torch.bfloat16,
        aux_stream=aux_stream,
        rope_append=True,
        kv_cache_dtype=model_config.extra_attrs.get("kv_cache_dtype", "auto"),
        skip_correction_threshold=model_config.skip_correction_threshold,
        indexer_projection_free=True,
    )
    _install_identity_rope_table(backend)
    stem = f"l{layer_idx}_mla_"
    shapes = {
        "qkva": (cfg.q_lora_rank + cfg.kv_lora_rank + cfg.qk_rope_head_dim, cfg.hidden_size),
        "qn": (cfg.q_lora_rank,),
        "kvn": (cfg.kv_lora_rank,),
        "qb": (heads * (cfg.qk_nope_head_dim + cfg.qk_rope_head_dim), cfg.q_lora_rank),
        "kb": (heads, cfg.kv_lora_rank, cfg.qk_nope_head_dim),
        "vb": (local_heads, cfg.v_head_dim, cfg.kv_lora_rank),
        "out": (cfg.hidden_size, local_heads * cfg.v_head_dim),
    }
    if cfg.mla_use_output_gate:
        shapes["gate"] = (local_heads * cfg.v_head_dim, cfg.hidden_size)
    for name, shape in shapes.items():
        bank[stem + name] = nn.Parameter(
            torch.empty(shape, dtype=torch.bfloat16), requires_grad=False
        )
    indexer = backend.indexer
    if indexer is not None:
        if indexer.rope_dim:
            raise ValueError("Kimi K3.1 checkpoint uses a position-free indexer")
        for name, shape in {
            "iq": (indexer.n_heads * indexer.head_dim, cfg.q_lora_rank),
            "ikw": (indexer.head_dim + indexer.n_heads, cfg.hidden_size),
            "ikn": (indexer.head_dim,),
            "ikb": (indexer.head_dim,),
        }.items():
            bank[stem + name] = nn.Parameter(
                torch.empty(
                    shape, dtype=torch.float32 if name in ("ikn", "ikb") else torch.bfloat16
                ),
                requires_grad=False,
            )
        indexer.norm_eps = sparse.index_k_norm_eps if sparse.index_k_norm_eps is not None else 1e-6
        indexer.ln_events = [torch.cuda.Event(), torch.cuda.Event()]
        indexer.prev_topk_copy_events = [torch.cuda.Event(), torch.cuda.Event()]
    state = MLAState(
        layer_idx,
        str(layer_idx),
        backend,
        mapping,
        heads,
        local_heads,
        padded_attention_heads(heads),
        cfg.kv_lora_rank,
        cfg.q_lora_rank,
        cfg.qk_nope_head_dim,
        cfg.qk_rope_head_dim,
        cfg.v_head_dim,
        cfg.rms_norm_eps,
        aux_stream,
        [torch.cuda.Event(), torch.cuda.Event()],
        mapping.tp_rank if tp > 1 else 0,
        mapping.rank % (tp * cp),
        cfg.mla_use_output_gate,
    )
    registry = model_config.extra_attrs.setdefault("mla_layers", {})
    if state.layer_idx_str in registry:
        raise ValueError(f"MLA layer {layer_idx} already registered")
    registry[state.layer_idx_str] = weakref.ref(state)
    return state


def mla_checkpoint_keys(state, prefix=""):
    base = f"{prefix}model.layers.{state.layer_idx}.self_attn."
    names = [
        "q_a_proj",
        "kv_a_proj_with_mqa",
        "q_b_proj",
        "kv_b_proj",
        "q_a_layernorm",
        "kv_a_layernorm",
        "o_proj",
    ]
    if state.use_output_gate:
        names.append("g_proj")
    keys = {base + name + ".weight" for name in names}
    if state.mqa.indexer is not None:
        keys.update(
            base + "indexer." + name + ".weight"
            for name in ("wq_b", "wk", "weights_proj", "k_norm")
        )
        keys.add(base + "indexer.k_norm.bias")
    return keys


def load_mla(bank, state, weights, prefix=""):
    from .modeling import _IndexerWeights, _MLAWeights

    stem = f"l{state.layer_idx}_mla_"
    base = f"{prefix}model.layers.{state.layer_idx}.self_attn."
    device = bank[stem + "qkva"].device

    def read(name):
        return _read_unquantized(weights, base + name).to(device=device)

    heads, kv = state.num_heads_tp, state.kv_lora_rank
    nope, v = state.qk_nope_head_dim, state.v_head_dim
    head_lo = state.tp_rank * heads
    local = state.num_heads_tp_cp
    kvb = read("kv_b_proj.weight").view(-1, nope + v, kv)[head_lo : head_lo + heads]
    kb, vb = kvb.split((nope, v), dim=1)
    output_lo = state.output_rank * local * v
    values = {
        "qkva": torch.cat((read("q_a_proj.weight"), read("kv_a_proj_with_mqa.weight"))),
        "qn": read("q_a_layernorm.weight"),
        "kvn": read("kv_a_layernorm.weight"),
        "qb": read("q_b_proj.weight").narrow(
            0, head_lo * state.qk_head_dim, heads * state.qk_head_dim
        ),
        "kb": kb.transpose(1, 2),
        "vb": vb.narrow(0, state.mapping.cp_rank * local, local),
        "out": read("o_proj.weight").narrow(1, output_lo, local * v),
    }
    if stem + "gate" in bank:
        values["gate"] = read("g_proj.weight").narrow(0, output_lo, local * v)
    if state.mqa.indexer is not None:
        values.update(
            iq=read("indexer.wq_b.weight"),
            ikw=torch.cat((read("indexer.wk.weight"), read("indexer.weights_proj.weight"))),
            ikn=read("indexer.k_norm.weight"),
            ikb=read("indexer.k_norm.bias"),
        )
    for name, value in values.items():
        dst = bank[stem + name]
        if dst.shape != value.shape:
            raise ValueError(f"{base}{name}: {value.shape} does not match {dst.shape}")
        dst.data.copy_(value)
    state.k_b_proj_trans = bank[stem + "kb"].detach()
    state.v_b_proj = bank[stem + "vb"].detach()
    state.weight_view = _MLAWeights(
        bank[stem + "qkva"].detach().t(),
        bank[stem + "qn"].detach(),
        bank[stem + "kvn"].detach(),
        bank[stem + "qb"].detach(),
        bank[stem + "gate"].detach() if stem + "gate" in bank else None,
        bank[stem + "out"].detach(),
    )
    if state.mqa.indexer is not None:
        indexer = state.mqa.indexer
        indexer.weight_view = _IndexerWeights(
            bank[stem + "iq"].detach().t(),
            bank[stem + "ikw"].detach(),
            bank[stem + "ikn"].detach(),
            bank[stem + "ikb"].detach(),
            indexer.norm_eps,
        )


@dataclass
class MLPState:
    stem: str
    checkpoint: str
    hidden: int
    intermediate: int
    rank: int
    quantized: bool
    beta: float
    linear_beta: Optional[float]
    all_reduce: Any
    weight_view: Any = None


def declare_mlp(bank, layer_idx, cfg, model_config, *, shared, tp_size, beta, linear_beta):
    from tensorrt_llm._torch.distributed import AllReduce
    from tensorrt_llm.mapping import Mapping

    mapping = model_config.mapping
    intermediate = (
        cfg.moe_intermediate_size * cfg.num_shared_experts if shared else cfg.intermediate_size
    )
    if intermediate % tp_size:
        raise ValueError("MLP intermediate dimension must divide its tensor-parallel group")
    local = intermediate // tp_size
    stem = f"l{layer_idx}_mlp_"
    quantized = shared
    for name, rows, cols in (("gu", 2 * local, cfg.hidden_size), ("down", cfg.hidden_size, local)):
        bank[stem + name] = nn.Parameter(
            torch.empty((rows, cols), dtype=torch.float8_e4m3fn if quantized else torch.bfloat16),
            requires_grad=False,
        )
        if quantized:
            bank[stem + name + "_scale"] = nn.Parameter(
                torch.empty(
                    ((rows + 127) // 128 * 128) * ((cols // 32 + 3) // 4 * 4), dtype=torch.uint8
                ),
                requires_grad=False,
            )
    group_mapping = Mapping(
        world_size=mapping.tp_size * mapping.pp_size,
        rank=mapping.rank,
        tp_size=tp_size,
        pp_size=mapping.pp_size * mapping.tp_size // tp_size,
        gpus_per_node=mapping.gpus_per_node,
        enable_attention_dp=mapping.enable_attention_dp,
    )
    return MLPState(
        stem,
        f"model.layers.{layer_idx}." + ("block_sparse_moe.shared_experts" if shared else "mlp"),
        cfg.hidden_size,
        local,
        group_mapping.tp_rank,
        quantized,
        beta,
        linear_beta,
        AllReduce(
            mapping=group_mapping, strategy=model_config.allreduce_strategy, dtype=torch.bfloat16
        )
        if tp_size > 1
        else None,
    )


def mlp_checkpoint_keys(state, prefix=""):
    return {
        f"{prefix}{state.checkpoint}.{projection}.weight{suffix}"
        for projection in ("gate_proj", "up_proj", "down_proj")
        for suffix in (("", "_scale") if state.quantized else ("",))
    }


def load_mlp(bank, state, weights, prefix=""):
    from .modeling import _MLPWeights, _ProjectionWeights

    lo, size = state.rank * state.intermediate, state.intermediate
    device = bank[state.stem + "gu"].device

    def read(name):
        return _materialize(weights[f"{prefix}{state.checkpoint}.{name}"]).to(device=device)

    views = []
    alpha = torch.ones(1, dtype=torch.float32, device=device) if state.quantized else None
    for dest, projections, axis in (
        ("gu", ("gate_proj", "up_proj"), 0),
        ("down", ("down_proj",), 1),
    ):
        value = torch.cat(
            [read(name + ".weight").narrow(axis, lo, size) for name in projections], dim=0
        )
        dst = bank[state.stem + dest]
        if value.shape != dst.shape:
            raise ValueError(f"{state.checkpoint}.{dest}: {value.shape} does not match {dst.shape}")
        expected_dtype = torch.float8_e4m3fn if state.quantized else torch.bfloat16
        if value.dtype != expected_dtype:
            raise TypeError(
                f"{state.checkpoint}.{dest}: expected {expected_dtype}, got {value.dtype}"
            )
        dst.data.copy_(value)
        sf = None
        if state.quantized:
            scales = []
            for name in projections:
                scale = read(name + ".weight_scale")
                if scale.dtype != torch.uint8:
                    raise TypeError(f"{state.checkpoint}.{name}: expected UE8M0 scales")
                scales.append(
                    scale.narrow(axis, lo // 32 if axis else lo, size // 32 if axis else size)
                )
            sf = bank[state.stem + dest + "_scale"]
            sf.data.copy_(
                torch.ops.trtllm.block_scale_interleave(torch.cat(scales).contiguous()).flatten()
            )
        views.append(
            _ProjectionWeights(
                dst.detach(), sf.detach() if sf is not None else None, alpha, dst.shape[0]
            )
        )
    state.weight_view = _MLPWeights(*views, state.beta, state.linear_beta, state.all_reduce)


@dataclass
class KDAState:
    """Per-layer cache identity and static geometry; compute lives in modeling.forward."""

    layer_idx: int
    layer_idx_str: str
    heads: int
    head_dim: int
    hidden_size: int
    eps: float
    lower_bound: float
    _kda_tp_size: int
    _kda_tp_rank: int
    _o_allreduce: Any
    weight_view: Any = None


def declare_kda(bank, cfg, layer_idx, model_config, mapping):
    from tensorrt_llm._torch.distributed import AllReduce

    lin = cfg.linear_attn_config
    if lin["head_dim"] != 128 or lin["short_conv_kernel_size"] != 4:
        raise ValueError("Kimi K3.1 requires 128-channel KDA heads and width-four convolution")
    if not lin.get("use_full_rank_gate", True):
        raise ValueError("Kimi K3.1 requires full-rank KDA output gates")
    tp = mapping.tp_size if not mapping.enable_attention_dp else 1
    rank = mapping.tp_rank if tp > 1 else 0
    heads = lin["num_heads"] // tp
    if heads * tp != lin["num_heads"]:
        raise ValueError("KDA heads must divide the attention tensor-parallel group")
    d, hidden, width = lin["head_dim"], cfg.hidden_size, heads * lin["head_dim"]
    stem = f"l{layer_idx}_kda_"

    def declare(name, shape, dtype):
        bank[stem + name] = nn.Parameter(torch.empty(shape, dtype=dtype), requires_grad=False)

    for name, rows, cols in (
        ("qkvg", 4 * width, hidden),
        ("bfa", d + heads, hidden),
        ("out", hidden, width),
    ):
        declare(name, ((rows + 31) // 32 * 32, cols), torch.float8_e4m3fn)
        declare(
            name + "_scale",
            (((rows + 127) // 128 * 128) * ((cols // 32 + 3) // 4 * 4),),
            torch.uint8,
        )
    for name, shape, dtype in (
        ("fb", (width, d), torch.bfloat16),
        ("conv", (3 * width, 4), torch.bfloat16),
        ("a_log", (heads,), torch.float32),
        ("dt_bias", (width,), torch.float32),
        ("norm", (d,), torch.bfloat16),
    ):
        declare(name, shape, dtype)
    state = KDAState(
        layer_idx,
        str(layer_idx),
        heads,
        d,
        hidden,
        cfg.rms_norm_eps,
        lin["gate_lower_bound"],
        tp,
        rank,
        AllReduce(mapping=mapping, strategy=model_config.allreduce_strategy, dtype=torch.bfloat16)
        if tp > 1
        else None,
    )
    registry = model_config.extra_attrs.setdefault("kda_layers", {})
    if state.layer_idx_str in registry:
        raise ValueError(f"KDA layer {layer_idx} already registered")
    registry[state.layer_idx_str] = weakref.ref(state)
    return state


def kda_checkpoint_keys(layer_idx, prefix=""):
    base = f"{prefix}model.layers.{layer_idx}.self_attn."
    keys = {
        base + f"{name}.weight{suffix}"
        for name in ("q_proj", "k_proj", "v_proj", "g_proj", "f_a_proj", "b_proj", "o_proj")
        for suffix in ("", "_scale")
    }
    keys.update(
        base + name
        for name in (
            "f_b_proj.weight",
            "q_conv1d.weight",
            "k_conv1d.weight",
            "v_conv1d.weight",
            "A_log",
            "dt_bias",
            "o_norm.weight",
        )
    )
    return keys


def load_kda(bank, state, weights, prefix=""):
    """Shard native code/scale pairs before fusion; never requantize checkpoint weights."""
    from .modeling import _KDAWeights, _ProjectionWeights

    stem = f"l{state.layer_idx}_kda_"
    base = f"{prefix}model.layers.{state.layer_idx}.self_attn."
    d, h = state.head_dim, state.heads
    width = h * d
    device = bank[stem + "qkvg"].device
    lo = state._kda_tp_rank * width
    tp = state._kda_tp_size

    def read(name):
        return _materialize(weights[base + name]).to(device=device)

    def pair(name, start=None, size=None, axis=0):
        w, sf = read(name + ".weight"), read(name + ".weight_scale")
        if w.dtype != torch.float8_e4m3fn or sf.dtype != torch.uint8:
            raise TypeError(f"{base}{name}: expected native E4M3 weights and UE8M0 scales")
        if sf.shape != (w.shape[0], w.shape[1] // 32):
            raise ValueError(f"{base}{name}: invalid MXFP8 scale geometry")
        if start is not None:
            w = w.narrow(axis, start, size)
            sf = sf.narrow(axis, start // 32 if axis else start, size // 32 if axis else size)
        return w, sf

    def store(name, pairs, columns):
        w = torch.cat([p[0] for p in pairs], dim=0)
        sf = torch.cat([p[1] for p in pairs], dim=0)
        dst, scales = bank[stem + name], bank[stem + name + "_scale"]
        if w.shape != (columns, dst.shape[1]):
            raise ValueError(f"{base}{name}: shard shape {w.shape} does not match declared storage")
        dst.data.zero_()
        dst.data[:columns].copy_(w)
        scales.data.copy_(torch.ops.trtllm.block_scale_interleave(sf.contiguous()).flatten())
        return _ProjectionWeights(dst.detach(), scales.detach(), alpha, columns)

    alpha = torch.ones(1, dtype=torch.float32, device=device)
    qkvg = store(
        "qkvg",
        [pair(name, lo, width) for name in ("q_proj", "k_proj", "v_proj", "g_proj")],
        4 * width,
    )
    bfa = store("bfa", [pair("f_a_proj"), pair("b_proj", state._kda_tp_rank * h, h)], d + h)
    out = store("out", [pair("o_proj", lo, width, axis=1)], state.hidden_size)
    conv = torch.cat(
        [
            read(name + ".weight")[lo : lo + width].squeeze(1)
            for name in ("q_conv1d", "k_conv1d", "v_conv1d")
        ]
    )
    a_log = read("A_log")
    if tp == 1 and a_log.numel() > h and torch.count_nonzero(a_log[h:]).item():
        raise ValueError(f"{base}A_log has nonzero padding")
    values = {
        "fb": read("f_b_proj.weight")[lo : lo + width],
        "conv": conv,
        "a_log": a_log[state._kda_tp_rank * h : (state._kda_tp_rank + 1) * h],
        "dt_bias": read("dt_bias")[lo : lo + width],
        "norm": read("o_norm.weight"),
    }
    for name, value in values.items():
        dst = bank[stem + name]
        if value.shape != dst.shape:
            raise ValueError(f"{base}{name}: {value.shape} does not match {dst.shape}")
        dst.data.copy_(value)
    conv = bank[stem + "conv"].detach()
    norm = bank[stem + "norm"].detach()
    state.weight_view = _KDAWeights(
        qkvg,
        bfa,
        bank[stem + "fb"].detach(),
        out,
        conv,
        tuple(part.t().contiguous() for part in conv.split(width)),
        tuple(part.float().contiguous() for part in conv.split(width)),
        bank[stem + "a_log"].detach(),
        bank[stem + "dt_bias"].detach(),
        norm,
        norm.float().contiguous(),
        torch.zeros(width, dtype=torch.bfloat16, device=device),
        h,
        d,
        state.eps,
        state.lower_bound,
    )


LANG_PREFIX = "language_model."
VISION_PREFIXES = ("vision_tower.", "mm_projector.")


@contextmanager
def open_checkpoint_weights(checkpoint_dir: str) -> Iterator[Dict[str, Any]]:
    """Yield checkpoint name -> lazy safetensors slice (``slice[:]`` reads that tensor) for the load.

    Sharded files are read when a consolidated copy sits beside them. The weights also carry ``checkpoint_dir``: the
    expert loader re-opens the shard files itself. The files stay mapped until the load ends.
    """
    files = sorted(glob.glob(os.path.join(checkpoint_dir, "*.safetensors")))
    if not files:
        raise RuntimeError(f"No safetensors files in {checkpoint_dir}.")
    files = [f for f in files if "consolidated" not in os.path.basename(f)] or files
    with ExitStack() as stack:
        slices: Dict[str, Any] = {}
        for path in files:
            handle = stack.enter_context(safe_open(path, framework="pt", device="cpu"))
            for name in handle.keys():
                slices[name] = handle.get_slice(name)
        weights = ConsumableWeightsDict(slices)
        weights.checkpoint_dir = checkpoint_dir
        logger.info(
            f"Lazily opened {len(files)} safetensors files ({len(weights)} tensors) from {checkpoint_dir}"
        )
        try:
            yield weights
        finally:
            weights.clear()


def split_keys(keys) -> tuple[list[str], list[str], list[str]]:
    """``(language-model keys, vision keys, unknown keys)`` of a checkpoint's key set."""
    lang, vision, unknown = [], [], []
    for key in keys:
        if key.startswith(LANG_PREFIX):
            lang.append(key)
        elif key.startswith(VISION_PREFIXES):
            vision.append(key)
        else:
            unknown.append(key)
    return lang, vision, unknown


def load(model, weights) -> None:
    """Load the checkpoint into ``model`` (the target): the text model, then the vision tower."""
    lang, vision, unknown = split_keys(weights.keys())
    if unknown:
        unknown.sort()
        raise ValueError(
            f"Kimi K3.1 checkpoint: {len(unknown)} key(s) are neither language-model "
            f"({LANG_PREFIX}*) nor vision ({', '.join(p + '*' for p in VISION_PREFIXES)}) "
            f"weights, first {unknown[:5]}"
        )
    if not lang:
        raise ValueError(f"Kimi K3.1 checkpoint: no {LANG_PREFIX}* keys")

    if model._skip_mm_encoder:
        if vision:
            logger.info(
                f"Kimi K3.1: {len(vision)} vision tower / projector keys not loaded (no local vision tower)"
            )
    else:
        if model.mm_encoder is not None and _has_meta_tensors(model.mm_encoder):
            logger.info("Recreating deferred vision encoder after MetaInitMode")
            model.mm_encoder = None
        if model.mm_encoder is None:
            # model_config.pretrained_config points at the text config once the model is built.
            vision_model_config = copy.copy(model.model_config)
            vision_model_config._frozen = False
            vision_model_config.pretrained_config = model._vlm_pretrained_config
            vision_model_config._frozen = True
            model.mm_encoder = model._VISION_MODEL_CLS(vision_model_config)
        model.mm_encoder.load_weights(weights)

    lm_weights = ConsumableWeightsDict(filter_weights(LANG_PREFIX[:-1], weights))
    checkpoint_dir = getattr(weights, "checkpoint_dir", None)
    if checkpoint_dir is not None:
        lm_weights.checkpoint_dir = checkpoint_dir
    lm_weights.checkpoint_prefix = LANG_PREFIX
    load_text_model(model.llm, lm_weights)


# ---------------------------------------------------------------------------
# Text model: checkpoint mapping and loading.
# ---------------------------------------------------------------------------


def _has_weights(module: nn.Module) -> bool:
    """False once ``modeling_utils.remove_weights()`` has stripped a module.

    Post-load finalization walks every decoder layer, so it must skip layers
    whose parameters were dropped — the layer-wise benchmarks keep only the
    profiled slice resident.
    """
    return not getattr(module, "_weights_removed", False)


class _K3ExpertCkptSpec(NamedTuple):
    """How one routed-expert quantization is spelled and loaded."""

    # Per-``w{1,2,3}`` checkpoint tensor suffixes this layout stores. Every
    # combination is required: a missing one fails the key plan.
    kinds: Tuple[str, ...]
    loader: Callable[..., None]
    # Set of filled slots the loader maintains, checked after the load.
    loaded_slots_attr: str


def _load_nvfp4_expert(
    backend, base, ckpt_expert_idx, slot_expert_idx, local_slot_id, get_tensor
) -> None:
    prefix = f"{base}.{ckpt_expert_idx}"
    tensors = {}
    for projection in ("w1", "w2", "w3"):
        for kind in ("weight", "weight_scale", "weight_scale_2", "input_scale"):
            tensor = get_tensor(f"{prefix}.{projection}.{kind}")
            # Scalars remain staged until the layer is finalized; the shard-read buffer is reused.
            tensors[f"{projection}_{kind}"] = (
                tensor.clone() if kind in ("weight_scale_2", "input_scale") else tensor
            )
    for name, tensor in tensors.items():
        if name.endswith("input_scale") and (tensor.numel() != 1 or tensor.item() != 1.0):
            raise ValueError("Kimi K3.1 NVFP4 streaming requires unit expert input scales")
    backend.quant_method.load_streaming_nvfp4_expert(
        backend,
        global_expert_id=slot_expert_idx,
        local_slot_id=local_slot_id,
        **tensors,
    )


# The loader writes complete native NVFP4 tuples straight into each backend slot.
_K3_EXPERT_CKPT_SPECS = {
    QuantAlgo.NVFP4: _K3ExpertCkptSpec(
        kinds=("weight", "weight_scale", "weight_scale_2", "input_scale"),
        loader=_load_nvfp4_expert,
        loaded_slots_attr="_streamed_expert_slots",
    ),
}


def _k3_expert_ckpt_spec(quant_algo: Optional[QuantAlgo]) -> _K3ExpertCkptSpec:
    spec = _K3_EXPERT_CKPT_SPECS.get(quant_algo)
    if spec is None:
        raise NotImplementedError(
            f"Kimi K3 routed experts are quantized as {quant_algo}, for which "
            "no per-expert checkpoint layout is known. Supported: "
            f"{sorted(a.name for a in _K3_EXPERT_CKPT_SPECS)}."
        )
    return spec


def _read_unquantized(weights, ckpt_key: str) -> torch.Tensor:
    """Read trunk tensor ``ckpt_key``. The target loads BF16 / FP32 trunk weights, so an FP8 tensor fails
    the load rather than being cast without its scale."""
    src = _materialize(weights[ckpt_key])
    if src.dtype == torch.float8_e4m3fn:
        raise ValueError(
            f"Kimi K3.1: {ckpt_key} is FP8 E4M3; the target loads BF16 attention / MLP / LM head"
        )
    return src


def _materialize(value) -> torch.Tensor:
    """Materialize a (possibly lazy safetensors slice) weight value."""
    if isinstance(value, torch.Tensor):
        return value
    return value[()] if not value.get_shape() else value[:]


_SAFETENSORS_DTYPES = {
    "U8": torch.uint8,
    "I8": torch.int8,
    "I32": torch.int32,
    "I64": torch.int64,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "F32": torch.float32,
    "F8_E4M3": torch.float8_e4m3fn,
}


def read_safetensors_slices(
    path: str, names: Iterable[str], get_buffer: Callable[[int], torch.Tensor]
) -> Optional[Dict[str, torch.Tensor]]:
    """Read the named tensors of one safetensors file with a few large preadv calls.

    The tensors' byte ranges are merged across gaps under 1 MiB and read into the
    uint8 tensor ``get_buffer(nbytes)`` returns (at least ``nbytes`` long). The result
    maps every name found in the file to a view of that buffer, valid until the buffer
    is reused. Each range starts at a buffer offset congruent to its file offset mod
    64, so the views keep the file's alignment. Returns None when a requested tensor
    has a dtype this reader does not map or the file ends early.
    """
    with open(path, "rb") as fh:
        header_len = int.from_bytes(fh.read(8), "little")
        header = json.loads(fh.read(header_len))
    data_start = 8 + header_len
    entries = []
    for name in names:
        meta = header.get(name)
        if meta is None:
            continue
        dtype = _SAFETENSORS_DTYPES.get(meta["dtype"])
        if dtype is None:
            return None
        start, end = meta["data_offsets"]
        entries.append((start, end, name, dtype, meta["shape"]))
    entries.sort()
    ranges = []
    for start, end, *_ in entries:
        if ranges and start - ranges[-1][1] < (1 << 20):
            ranges[-1][1] = max(ranges[-1][1], end)
        else:
            ranges.append([start, end])
    placements = []
    buf_size = 0
    for start, end in ranges:
        buf_off = (buf_size + 63) // 64 * 64 + start % 64
        placements.append(buf_off)
        buf_size = buf_off + end - start
    buf = get_buffer(buf_size)
    view = memoryview(buf.numpy())
    fd = os.open(path, os.O_RDONLY)
    try:
        for (start, end), buf_off in zip(ranges, placements):
            done = 0
            while done < end - start:
                got = os.preadv(
                    fd,
                    [view[buf_off + done : buf_off + end - start]],
                    data_start + start + done,
                )
                if got <= 0:
                    return None
                done += got
    finally:
        os.close(fd)
    tensors = {}
    range_idx = 0
    for start, end, name, dtype, shape in entries:
        while end > ranges[range_idx][1]:
            range_idx += 1
        buf_off = placements[range_idx] + start - ranges[range_idx][0]
        tensors[name] = buf[buf_off : buf_off + end - start].view(dtype).view(shape)
    return tensors


# ------------------------------------------------------------------
# Weight loading (streams the 1.5TB checkpoint; only the rank-local
# expert slice of each MoE layer is kept: whole experts under MoE EP,
# the intra-expert intermediate shard of ALL experts under MoE TP —
# in the TP case every expert tensor is read and sliced, so expect a
# correspondingly longer load).
# ------------------------------------------------------------------


def trunk_parameters(llm) -> Dict[str, torch.nn.Parameter]:
    """Shell parameters outside the flat tensor bank and separate draft checkpoint."""
    return {
        name: param
        for name, param in llm.named_parameters()
        if not name.startswith(("draft_model.", "model.w."))
    }


def checkpoint_name_plan(
    llm, prefix: str
) -> Tuple[Dict[str, str], Set[str], List[Tuple[int, KimiK3MoERuntime, str]]]:
    """Return ``(name_map, expected_keys, expert_jobs)``.

    ``name_map`` maps shell parameters to checkpoint keys. ``expected_keys``
    additionally covers the flat trunk tensors and rank-local NVFP4 experts.
    ``expert_jobs`` lists ``(layer_idx, moe_module, key_base)``
    for backend-owned expert slots. Exposed separately so the weight-name
    mapping can be dry-run without touching any tensor data.
    """
    params = trunk_parameters(llm)
    expected_keys = set()
    name_map: Dict[str, str] = {}
    for name in params:
        # ConfigurableMoE's backend owns already-packed runtime weights,
        # generated zero biases, and SiTU constants. They do not have
        # one-to-one checkpoint parameter names.
        if ".routed_experts.backend." in name or name.startswith("model.moe_runners."):
            continue
        ckpt_key = prefix + name
        name_map[name] = ckpt_key
        expected_keys.add(ckpt_key)

    # Backend-owned expert slots (per-expert checkpoint tensors; the
    # rank-local id range — an EP slice of whole experts, or ALL experts
    # when the routed MoE is TP-sharded (moe_ep=1 -> ids 0..num_experts)).
    expert_jobs = []
    for layer_idx, layer in enumerate(llm.model.layers):
        if not getattr(layer, "is_moe", False) or not _has_weights(layer):
            continue
        moe = layer.block_sparse_moe
        base = f"{prefix}model.layers.{layer_idx}.block_sparse_moe.experts"
        kinds = moe.expert_ckpt_spec.kinds
        for expert_idx in moe.ckpt_expert_ids:
            for w in ("w1", "w2", "w3"):
                for kind in kinds:
                    expected_keys.add(f"{base}.{expert_idx}.{w}.{kind}")
        expert_jobs.append((layer_idx, moe, base))
    for index, layer in enumerate(llm.model.layers):
        if layer.is_kda:
            expected_keys.update(kda_checkpoint_keys(index, prefix))
        else:
            expected_keys.update(mla_checkpoint_keys(layer.mla, prefix))
        mlp = layer.block_sparse_moe.shared_experts if layer.is_moe else layer.mlp
        expected_keys.update(mlp_checkpoint_keys(mlp, prefix))
    expected_keys.update(prefix + source for source in llm.model.w.checkpoint_sources.values())
    return name_map, expected_keys, expert_jobs


def load_text_model(llm, weights: Dict[str, torch.Tensor]) -> None:
    prefix = "language_model." if any(k.startswith("language_model.") for k in weights) else ""
    params = trunk_parameters(llm)
    name_map, expected_keys, expert_jobs = checkpoint_name_plan(llm, prefix)

    _validate_checkpoint_keys(llm, weights, expected_keys, prefix)
    load_auxiliary(llm.model.w, weights, prefix)
    for layer in llm.model.layers:
        if layer.is_kda:
            load_kda(llm.model.w, layer.linear_attn, weights, prefix)
        else:
            load_mla(llm.model.w, layer.mla, weights, prefix)
        mlp = layer.block_sparse_moe.shared_experts if layer.is_moe else layer.mlp
        load_mlp(llm.model.w, mlp, weights, prefix)
    num_params = _load_trunk_params(llm, weights, params, name_map)
    if llm.model_config.moe_backend == "MEGAMOE_CUTEDSL":
        # MegaMoE keeps packed destinations and temporary source weights.
        # Packing each layer releases its sources before the next allocation.
        for job in expert_jobs:
            _load_expert_slices(llm, weights, [job])
    else:
        _load_expert_slices(llm, weights, expert_jobs)
    _finalize_weight_load(llm, num_params, len(expert_jobs))
    device = next(llm.parameters()).device
    if device.type == "cuda":
        # Lazy source mappings are load-scoped; finish nonblocking H2D
        # work before the caller can release the weights container.
        torch.cuda.synchronize(device)


def _validate_checkpoint_keys(
    llm, weights: Dict[str, torch.Tensor], expected_keys: Set[str], prefix: str
) -> None:
    """Key-set validation (both directions): every expected key must be
    present; unmatched checkpoint keys (beyond the expected leftovers)
    only warn."""
    ckpt_keys = set(weights.keys())
    relevant_ckpt_keys = {
        k for k in ckpt_keys if not (k.startswith("vision_tower.") or k.startswith("mm_projector."))
    }
    missing = sorted(expected_keys - ckpt_keys)
    if missing:
        raise KeyError(
            f"Kimi K3 load_weights: {len(missing)} expected checkpoint "
            f"keys are missing, e.g. {missing[:10]}"
        )
    unexpected = relevant_ckpt_keys - expected_keys
    # Non-local experts and (in layer-truncated debug mode) extra layers
    # are expected leftovers.
    surprising = sorted(
        k
        for k in unexpected
        if ".block_sparse_moe.experts." not in k and not k.startswith(f"{prefix}model.layers.")
    )
    if surprising:
        logger.warning(
            f"Kimi K3 load_weights: {len(surprising)} unmatched "
            f"checkpoint keys, e.g. {surprising[:10]}"
        )


def _load_trunk_params(llm, weights, params, name_map) -> int:
    device = next(llm.parameters()).device

    def load_param(name, param):
        if device.type == "cuda":
            torch.cuda.set_device(device)
        value = _read_unquantized(weights, name_map[name])
        if name == "lm_head.weight":
            llm.lm_head.load_weights(weights=[{"weight": value}])
            return
        if value.shape != param.shape:
            raise ValueError(f"{name}: {value.shape} does not match {param.shape}")
        param.data.copy_(value.to(param.dtype))

    run_concurrently(load_param, [(name, params[name]) for name in name_map], num_workers=8)
    return len(name_map)


def _load_expert_slices(
    llm,
    weights: Dict[str, torch.Tensor],
    expert_jobs: List[Tuple[int, KimiK3MoERuntime, str]],
) -> None:
    """Load the rank-local NVFP4 expert slices of every MoE layer into
    the backend expert slots, then verify every slot was filled."""
    device = next(llm.parameters()).device
    # Startup accounting, logged once below: checkpoint reads (bytes and
    # thread-summed time) and the backend's per-expert loader (reads, H2D
    # copy, shuffle, scale interleave; thread-summed).
    # TLLM_K3_EXPERT_LOAD_THREADS sets the loader threads.
    load_threads = int(os.environ.get("TLLM_K3_EXPERT_LOAD_THREADS", "4"))
    load_stats = {
        "read_s": 0.0,
        "read_bytes": 0,
        "expert_s": 0.0,
        "experts": 0,
    }
    stats_lock = threading.Lock()
    load_start = time.perf_counter()

    def timed_read(get):
        def read(key):
            start = time.perf_counter()
            tensor = get(key)
            elapsed = time.perf_counter() - start
            nbytes = tensor.numel() * tensor.element_size()
            with stats_lock:
                load_stats["read_s"] += elapsed
                load_stats["read_bytes"] += nbytes
            return tensor

        return read

    def load_expert(
        moe: KimiK3MoERuntime,
        base: str,
        local_slot_id: int,
        expert_idx: int,
        get_tensor,
    ):
        """Load checkpoint expert ``expert_idx`` into local slot ``local_slot_id``."""
        if device.type == "cuda":
            torch.cuda.set_device(device)
        start = time.perf_counter()
        moe.expert_ckpt_spec.loader(
            moe.routed_experts.backend,
            base,
            expert_idx,
            moe.local_expert_ids[local_slot_id],
            local_slot_id,
            timed_read(get_tensor),
        )
        with stats_lock:
            load_stats["expert_s"] += time.perf_counter() - start
            load_stats["experts"] += 1

    def load_experts_from_weights(layer_idx: int, moe: KimiK3MoERuntime, base: str):
        del layer_idx
        for local_slot_id, expert_idx in enumerate(moe.ckpt_expert_ids):
            load_expert(
                moe,
                base,
                local_slot_id,
                expert_idx,
                lambda key: _materialize(weights[key]),
            )

    # ---- backend expert slots: file-grouped streaming ----
    for _, moe, _ in expert_jobs:
        if moe.expert_ckpt_spec.loaded_slots_attr == "_streamed_expert_slots":
            backend = moe.routed_experts.backend
            backend.quant_method.prepare_streaming_expert_load(backend)
    # The shared lazy ``weights`` dict keeps every shard mmapped for the
    # whole load, so pages it touches cannot be dropped until the load
    # ends (fadvise skips mapped pages). The expert slices are ~90 GB of
    # DISTINCT pages per rank — with 4 ranks/node that overruns the job
    # cgroup and OOM-kills the step (observed repeatedly on GB300
    # trays). Instead, group the rank-local expert tensors by shard file
    # and stream each file through a short-lived handle:
    # open -> copy -> close (unmap) -> fadvise(DONTNEED).
    # The lazy loader records the directory it opened; prefer it over
    # ``_name_or_path``, which transformers no longer populates (it is
    # empty on transformers 5.x, which silently sent the whole load down
    # the fallback below and OOM-killed the step).
    ckpt_dir = getattr(weights, "checkpoint_dir", None) or getattr(
        llm.model_config.pretrained_config, "_name_or_path", None
    )
    index_path = os.path.join(ckpt_dir or "", "model.safetensors.index.json")
    checkpoint_prefix = getattr(weights, "checkpoint_prefix", "")

    def checkpoint_key(key: str) -> str:
        return f"{checkpoint_prefix}{key}"

    if expert_jobs and ckpt_dir and os.path.isfile(index_path):
        with open(index_path) as f:
            weight_map = json.load(f)["weight_map"]
        per_file: Dict[str, list] = {}
        per_file_keys: Dict[str, list] = {}
        split_file_jobs = []
        for layer_idx, moe, base in expert_jobs:
            del layer_idx
            spec = moe.expert_ckpt_spec
            for local_slot_id, expert_idx in enumerate(moe.ckpt_expert_ids):
                keys = [
                    f"{base}.{expert_idx}.{w}.{kind}"
                    for w in ("w1", "w2", "w3")
                    for kind in spec.kinds
                ]
                files = {weight_map[checkpoint_key(key)] for key in keys}
                job = (moe, base, local_slot_id, expert_idx)
                if len(files) == 1:
                    file_name = files.pop()
                    per_file.setdefault(file_name, []).append(job)
                    per_file_keys.setdefault(file_name, []).extend(keys)
                else:
                    split_file_jobs.append((job, files))

        def drop_file_pages(file_name: str):
            path = os.path.join(ckpt_dir, file_name)
            try:
                fd = os.open(path, os.O_RDONLY)
                try:
                    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                finally:
                    os.close(fd)
            except OSError:
                pass

        # Kimi K3.1: read this rank's byte ranges of a shard
        # (merged across gaps under 1 MiB) with large preadv calls into a
        # per-thread pinned host buffer and serve its expert tensors as
        # views of it. Tensors are stored sorted by name, so experts
        # 0..N/ep-1 (rank 0) are spread over the whole shard (experts 0, 1,
        # 10-19, 100-...) while every other rank's slice is one compact
        # region. Copying them to the GPU straight out of the safetensors
        # mapping faults the file in page by page, which is random access
        # for rank 0: its loads took 3x as long as the other ranks' on a
        # GB300 TEP8 worker. Any problem (an unknown dtype, a short read)
        # falls back to the mapping.
        from .modeling import _get_text_config  # noqa: E501

        pread_experts = (
            getattr(_get_text_config(llm.model_config.pretrained_config), "dsa_config", None)
            is not None
        )
        pread_buffers = threading.local()

        def pread_buffer(nbytes: int) -> torch.Tensor:
            buf = getattr(pread_buffers, "buf", None)
            if buf is None or buf.numel() < nbytes:
                pread_buffers.buf = None
                buf = torch.empty(nbytes, dtype=torch.uint8, pin_memory=device.type == "cuda")
                pread_buffers.buf = buf
            return buf

        def read_file_tensors(file_name: str) -> Optional[Dict[str, torch.Tensor]]:
            return read_safetensors_slices(
                os.path.join(ckpt_dir, file_name),
                [checkpoint_key(key) for key in per_file_keys[file_name]],
                pread_buffer,
            )

        def load_expert_file(file_name: str, jobs: list):
            if device.type == "cuda":
                torch.cuda.set_device(device)
            if pread_experts:
                read_start = time.perf_counter()
                try:
                    tensors = read_file_tensors(file_name)
                except (OSError, ValueError, KeyError, RuntimeError):
                    tensors = None
                if tensors is not None:
                    # Bytes are counted when the loaders take the views.
                    with stats_lock:
                        load_stats["read_s"] += time.perf_counter() - read_start
                    for moe, base, local_slot_id, expert_idx in jobs:
                        load_expert(
                            moe,
                            base,
                            local_slot_id,
                            expert_idx,
                            lambda key: tensors[checkpoint_key(key)],
                        )
                    if device.type == "cuda":
                        # Copies out of the buffer may still be in flight;
                        # the next shard overwrites it.
                        torch.cuda.current_stream(device).synchronize()
                    drop_file_pages(file_name)
                    return
            path = os.path.join(ckpt_dir, file_name)
            with safe_open(path, framework="pt", device="cpu") as fh:
                for moe, base, local_slot_id, expert_idx in jobs:
                    load_expert(
                        moe,
                        base,
                        local_slot_id,
                        expert_idx,
                        lambda key: fh.get_tensor(checkpoint_key(key)),
                    )
            # Handle closed -> pages unmapped -> the drop takes effect.
            drop_file_pages(file_name)

        def load_split_file_expert(job, files):
            if device.type == "cuda":
                torch.cuda.set_device(device)
            with ExitStack() as stack:
                handles = {
                    file_name: stack.enter_context(
                        safe_open(os.path.join(ckpt_dir, file_name), framework="pt", device="cpu")
                    )
                    for file_name in files
                }

                def get_tensor(key):
                    source_key = checkpoint_key(key)
                    return handles[weight_map[source_key]].get_tensor(source_key)

                load_expert(*job, get_tensor)
            for file_name in files:
                drop_file_pages(file_name)

        run_concurrently(load_expert_file, sorted(per_file.items()), num_workers=load_threads)
        run_concurrently(load_split_file_expert, split_file_jobs, num_workers=load_threads)
    else:
        # Falling back is a silent loss of the whole point of the block
        # above: the shared lazy dict keeps every shard mapped, which is
        # the OOM this streaming path exists to avoid. Say so.
        if expert_jobs:
            logger.warning(
                f"Kimi K3: no safetensors index at '{index_path}', so routed "
                "experts are loaded from the shared lazy weight dict instead "
                "of being streamed per shard file. Every shard stays mapped "
                "for the whole load, which OOM-kills the step at DEP8 scale."
            )
        run_concurrently(load_experts_from_weights, expert_jobs, num_workers=load_threads)

    for _, moe, _ in expert_jobs:
        spec = moe.expert_ckpt_spec
        backend = moe.routed_experts.backend
        loaded_slots = getattr(backend, spec.loaded_slots_attr, set())
        expected_slots = set(range(backend.expert_size_per_partition))
        if loaded_slots != expected_slots:
            missing_slots = sorted(expected_slots - loaded_slots)
            raise RuntimeError(
                "Kimi K3 streaming expert loading did not fill all backend "
                f"slots; missing {missing_slots[:10]}."
            )
        if spec.loaded_slots_attr == "_streamed_expert_slots":
            backend.process_weights_after_loading()
        backend._weights_transformed = False
    wall = time.perf_counter() - load_start
    logger.info(
        f"Kimi K3: expert slices: {load_stats['experts']} experts, "
        f"{load_stats['read_bytes'] / 2**30:.1f} GiB read in {load_stats['read_s']:.1f}s, "
        f"per-expert load {load_stats['expert_s']:.1f}s (incl. reads, thread-summed), "
        f"wall {wall:.1f}s, "
        f"{load_threads} threads"
    )


def _finalize_weight_load(llm, num_params: int, num_moe_layers: int) -> None:
    """Build the KDA layers' derived decode weights once the checkpoint values are in place."""
    bind_auxiliary(llm.model.w, llm.model)
    for layer in llm.model.layers:
        if not _has_weights(layer):
            continue
        bind_auxiliary(llm.model.w, layer)
        if layer.is_moe:
            moe = layer.block_sparse_moe
            bind_auxiliary(llm.model.w, moe)
            moe.shared_weight_view = moe.shared_experts.weight_view
        else:
            layer.mlp_weight_view = layer.mlp.weight_view
    logger.info(
        f"Kimi K3: loaded {num_params} parameters and expert slices of {num_moe_layers} layers"
    )
