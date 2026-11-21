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
"""MX post-transform qualification tests for the GLM-5 DSA + FP8 block-scale MoE root on SM90."""

from unittest.mock import MagicMock

import pytest
import torch
from torch import nn
from transformers import PretrainedConfig
from utils.post_transform_qualification import (
    AllReduceStub,
    PostTransformQualificationCase,
    assert_every_parameter_was_loaded,
    assert_post_transform_lifecycle_equivalent,
    keep_factory_parameters,
    module_state_snapshot,
    mx_canonical_parameter_catalog,
)
from utils.util import getSMVersion

import tensorrt_llm.mapping as mapping_mod
from tensorrt_llm._torch import distributed as distributed_mod
from tensorrt_llm._torch import model_config as model_config_mod
from tensorrt_llm._torch.attention.mla import MLA
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models import modeling_deepseekv3 as modeling_deepseekv3_mod
from tensorrt_llm._torch.models.checkpoints import model_hooks as model_hooks_mod
from tensorrt_llm._torch.models.checkpoints.mx import model_loading as mx_loading_mod
from tensorrt_llm._torch.weight_sharing import (
    ARTIFACT_IDENTITY_FORMAT_VERSION,
    DEEPSEEK_V3_MLA_CUTLASS_POST_TRANSFORM_LAYOUT_ABI_V1,
    GLM5_DSA_FP8_BLOCK_SM90_POST_TRANSFORM_LAYOUT_ABI_V1,
    ArtifactIdentity,
    IdentityCheckPolicy,
    PostTransformConfigIdentity,
    PostTransformFeature,
    PostTransformProfile,
    PostTransformQualificationDecision,
    PostTransformQualificationReason,
    PostTransformRuntimeConfig,
    PostTransformTransferScope,
    SourceIdentity,
    check_weight_sharing_compatibility,
)
from tensorrt_llm.llmapi.llm_args import DeepSeekSparseAttentionConfig
from tensorrt_llm.models.modeling_utils import QuantAlgo, QuantConfig

_PROFILE_ID = "glm-moe-dsa-for-causal-lm-fp8-block-sm90-cutlass-target-v1"
_NUM_EXPERTS = 4
# GLM-5 runs dense MLPs in its first three layers and MoE in every later layer.
_FIRST_K_DENSE_REPLACE = 3
_NUM_LAYERS = _FIRST_K_DENSE_REPLACE + 1
_MOE_LAYER = _FIRST_K_DENSE_REPLACE
_FP8_BLOCK_SIZE = 128

# The profile admits only the SM90 FP8 block-scale layouts, and the routed
# experts run the DeepGEMM Hopper kernels.
_requires_sm90 = pytest.mark.skipif(
    not torch.cuda.is_available() or getSMVersion() != 90,
    reason="the GLM-5 profile qualifies the SM90 FP8 block-scale layout only",
)


class _UnqualifiedGlmMoeDsaForCausalLM(modeling_deepseekv3_mod.GlmMoeDsaForCausalLM):
    """Subclass that must not inherit the exact-root GLM-5 profile."""


def _hf_config(**overrides: object) -> PretrainedConfig:
    """Tiny four-layer config with the GLM-5 MLA, DSA indexer, and FP8 geometry.

    Layers 0 to 2 run dense MLPs and layer 3 is MoE with four routed experts
    (top-2) and one shared expert, as `first_k_dense_replace=3` declares. The
    MLA and indexer head dimensions and the quantization config are those of
    GLM-5-FP8, and every layer owns an indexer unless `index_topk_pattern`
    marks a layer `S` (shared).
    """
    values: dict[str, object] = {
        "architectures": ["GlmMoeDsaForCausalLM"],
        "model_type": "glm_moe_dsa",
        "vocab_size": 256,
        "hidden_size": 256,
        "intermediate_size": 256,
        "moe_intermediate_size": 256,
        "num_hidden_layers": _NUM_LAYERS,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "q_lora_rank": 128,
        "kv_lora_rank": 512,
        "qk_nope_head_dim": 192,
        "qk_rope_head_dim": 64,
        "qk_head_dim": 256,
        "v_head_dim": 256,
        "n_routed_experts": _NUM_EXPERTS,
        "n_shared_experts": 1,
        "num_experts_per_tok": 2,
        "first_k_dense_replace": _FIRST_K_DENSE_REPLACE,
        "moe_layer_freq": 1,
        "n_group": 1,
        "topk_group": 1,
        "routed_scaling_factor": 2.5,
        "scoring_func": "sigmoid",
        "topk_method": "noaux_tc",
        "norm_topk_prob": True,
        "max_position_embeddings": 256,
        "rope_theta": 1000000,
        "rope_scaling": None,
        "rope_parameters": {"rope_theta": 1000000, "rope_type": "default"},
        "rope_interleave": True,
        "rms_norm_eps": 1e-5,
        "hidden_act": "silu",
        "tie_word_embeddings": False,
        "num_nextn_predict_layers": 1,
        "index_n_heads": 32,
        "index_head_dim": 128,
        "index_topk": 64,
        "indexer_rope_interleave": True,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "pad_token_id": None,
        "torch_dtype": torch.bfloat16,
        # The quantization config GLM-5-FP8 declares.
        "quantization_config": {
            "activation_scheme": "dynamic",
            "fmt": "e4m3",
            "quant_method": "fp8",
            "weight_block_size": [_FP8_BLOCK_SIZE, _FP8_BLOCK_SIZE],
        },
    }
    values.update(overrides)
    return PretrainedConfig(**values)


def _quant_config() -> QuantConfig:
    # Mirrors what load_hf_quant_config() builds for GLM-5-FP8: its
    # modules_to_not_convert, as globs over the layers, plus the default FP8
    # block-scale exclusions that keep `kv_b_proj` in BF16.
    return QuantConfig(
        quant_algo=QuantAlgo.FP8_BLOCK_SCALES,
        group_size=_FP8_BLOCK_SIZE,
        exclude_modules=[
            "lm_head",
            "model.embed_tokens",
            "model.norm",
            "*input_layernorm",
            "*post_attention_layernorm",
            "*kv_a_layernorm",
            "*q_a_layernorm",
            "*mlp.gate",
            "*mlp.gate.e_score_correction_bias",
            "*self_attn.indexer.k_norm",
            "*self_attn.indexer.k_norm.bias",
            "*self_attn.indexers_proj",
            "*kv_b_proj*",
            "*k_b_proj*",
            "*eh_proj",
        ],
    )


def _sparse_attention_config() -> DeepSeekSparseAttentionConfig:
    # The config ModelConfig.from_pretrained() builds for a GLM-5 checkpoint
    # when the user sets none.
    return DeepSeekSparseAttentionConfig(
        index_n_heads=32,
        index_head_dim=128,
        index_topk=64,
        indexer_rope_interleave=True,
    )


def _tiny_model(
    monkeypatch: pytest.MonkeyPatch,
    *,
    model_class: type[nn.Module] = modeling_deepseekv3_mod.GlmMoeDsaForCausalLM,
    tp_size: int = 1,
    rank: int = 0,
    moe_ep_size: int = -1,
    filled: bool = True,
    config_overrides: dict[str, object] | None = None,
) -> nn.Module:
    """Build a four-layer GLM-5 model on CUDA with deterministic weights.

    `moe_ep_size=2` at `tp_size=2` selects the MoE-EP partition; the default
    leaves MoE TP equal to TP. TP=2 builds one rank of the group with
    collectives stubbed out. The weights are created after construction, as in
    a production load, so the quantization exclusions apply before any weight
    exists. The unit activation scales of the dynamic-activation checkpoint
    keep their value.
    """
    if tp_size > 1:
        monkeypatch.setattr(mapping_mod, "mpi_disabled", lambda: False)
        monkeypatch.setattr(distributed_mod, "AllReduce", AllReduceStub)
        monkeypatch.setattr(modeling_deepseekv3_mod, "AllReduce", AllReduceStub)
        monkeypatch.setattr(modeling_deepseekv3_mod, "MoEAllReduce", AllReduceStub)
        monkeypatch.setattr(modeling_deepseekv3_mod, "can_access_peer", lambda _mapping: False)
    model = model_class(
        ModelConfig(
            pretrained_config=_hf_config(**(config_overrides or {})),
            quant_config=_quant_config(),
            sparse_attention_config=_sparse_attention_config(),
            mapping=mapping_mod.Mapping(
                world_size=tp_size,
                rank=rank,
                tp_size=tp_size,
                moe_ep_size=moe_ep_size,
            ),
            max_num_tokens=16,
            max_seq_len=64,
            skip_create_weights_in_init=True,
        )
    ).cuda()
    if filled:
        with torch.no_grad():
            for index, (name, parameter) in enumerate(model.named_parameters()):
                if name.endswith("input_scale"):
                    continue
                values = torch.arange(
                    parameter.numel(),
                    dtype=torch.float32,
                    device=parameter.device,
                ).reshape(parameter.shape)
                # Centered on zero, so no positive checkpoint tensor can equal it.
                parameter.copy_(((((values + index) % 17) - 8) / 64).to(parameter.dtype))
    return model


def _fp8_weight(shape: tuple[int, int], index: int) -> torch.Tensor:
    values = torch.arange(torch.Size(shape).numel(), dtype=torch.float32).reshape(shape)
    return (((values + index) % 17 + 1) / 16).to(torch.float8_e4m3fn)


def _block_scale(shape: tuple[int, int], index: int) -> torch.Tensor:
    rows = -(-shape[0] // _FP8_BLOCK_SIZE)
    columns = -(-shape[1] // _FP8_BLOCK_SIZE)
    values = torch.arange(rows * columns, dtype=torch.float32).reshape(rows, columns)
    return ((values + index) % 5 + 1) / 48


def _hf_state_dict(**config_overrides: object) -> dict[str, torch.Tensor]:
    """Return the tiny config's checkpoint tensors in the GLM-5-FP8 Hugging Face layout.

    Every Linear that the checkpoint quantizes stores E4M3 weights and FP32
    `weight_scale_inv` block scales, including `kv_b_proj` and the indexer's
    `wk`; the router, the indexer's `weights_proj` and the norms are BF16, and
    routed experts are stored per expert. Every value is positive, so no
    loaded tensor can equal the factory's fill, which is centered on zero.
    """
    config = _hf_config(**config_overrides)
    hidden = config.hidden_size
    heads = config.num_attention_heads
    qk_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim
    kv_lora = config.kv_lora_rank
    moe_inter = config.moe_intermediate_size
    pattern = config.to_dict().get("index_topk_pattern") or "F" * config.num_hidden_layers
    fp8_shapes: dict[str, tuple[int, int]] = {}
    bf16_shapes: dict[str, tuple[int, ...]] = {
        "model.embed_tokens.weight": (config.vocab_size, hidden),
        "model.norm.weight": (hidden,),
        "lm_head.weight": (config.vocab_size, hidden),
    }
    for layer in range(config.num_hidden_layers):
        prefix = f"model.layers.{layer}."
        attention = prefix + "self_attn."
        bf16_shapes |= {
            prefix + "input_layernorm.weight": (hidden,),
            prefix + "post_attention_layernorm.weight": (hidden,),
            attention + "kv_a_layernorm.weight": (kv_lora,),
            attention + "q_a_layernorm.weight": (config.q_lora_rank,),
        }
        fp8_shapes |= {
            attention + "kv_a_proj_with_mqa": (kv_lora + config.qk_rope_head_dim, hidden),
            attention + "kv_b_proj": (
                heads * (config.qk_nope_head_dim + config.v_head_dim),
                kv_lora,
            ),
            attention + "o_proj": (hidden, heads * config.v_head_dim),
            attention + "q_a_proj": (config.q_lora_rank, hidden),
            attention + "q_b_proj": (heads * qk_head_dim, config.q_lora_rank),
        }
        if str(pattern[layer]).upper() != "S":
            indexer = attention + "indexer."
            fp8_shapes |= {
                indexer + "wq_b": (
                    config.index_n_heads * config.index_head_dim,
                    config.q_lora_rank,
                ),
                indexer + "wk": (config.index_head_dim, hidden),
            }
            bf16_shapes |= {
                indexer + "weights_proj.weight": (config.index_n_heads, hidden),
                indexer + "k_norm.weight": (config.index_head_dim,),
                indexer + "k_norm.bias": (config.index_head_dim,),
            }
        if layer < config.first_k_dense_replace:
            fp8_shapes |= {
                prefix + "mlp.gate_proj": (config.intermediate_size, hidden),
                prefix + "mlp.up_proj": (config.intermediate_size, hidden),
                prefix + "mlp.down_proj": (hidden, config.intermediate_size),
            }
            continue
        bf16_shapes[prefix + "mlp.gate.weight"] = (config.n_routed_experts, hidden)
        shared_inter = moe_inter * config.n_shared_experts
        fp8_shapes |= {
            prefix + "mlp.shared_experts.gate_proj": (shared_inter, hidden),
            prefix + "mlp.shared_experts.up_proj": (shared_inter, hidden),
            prefix + "mlp.shared_experts.down_proj": (hidden, shared_inter),
        }
        for expert in range(config.n_routed_experts):
            fp8_shapes |= {
                prefix + f"mlp.experts.{expert}.gate_proj": (moe_inter, hidden),
                prefix + f"mlp.experts.{expert}.up_proj": (moe_inter, hidden),
                prefix + f"mlp.experts.{expert}.down_proj": (hidden, moe_inter),
            }
    state_dict = {}
    for index, (name, shape) in enumerate(bf16_shapes.items()):
        values = torch.arange(torch.Size(shape).numel(), dtype=torch.float32).reshape(shape)
        state_dict[name] = (((values + index) % 17 + 1) / 256).to(torch.bfloat16)
    for index, (name, shape) in enumerate(fp8_shapes.items()):
        state_dict[name + ".weight"] = _fp8_weight(shape, index)
        state_dict[name + ".weight_scale_inv"] = _block_scale(shape, index)
    for layer in range(config.first_k_dense_replace, config.num_hidden_layers):
        state_dict[f"model.layers.{layer}.mlp.gate.e_score_correction_bias"] = (
            torch.arange(1, config.n_routed_experts + 1, dtype=torch.float32) / 16
        )
    return state_dict


def _hf_checkpoint_loader(state_dict: dict[str, torch.Tensor]):
    """Return a `prepare_producer` hook that runs the real `DeepseekV3WeightLoader`."""

    def _load(model: nn.Module) -> None:
        model.load_weights(dict(state_dict))

    return _load


# MX captures the identity before construction, because the GLM-5 constructor
# rewrites the config's model type to `deepseek_v32`.
_CONFIG_IDENTITY = PostTransformConfigIdentity.from_model_config(
    ModelConfig(pretrained_config=_hf_config())
)


def _qualify(model: nn.Module) -> PostTransformQualificationDecision:
    return mx_loading_mod.qualify_post_transform_profile(
        model,
        config_identity=_CONFIG_IDENTITY,
        speculative_mode=None,
        loads_draft_weights=False,
    )


def _profile() -> PostTransformProfile:
    (profile,) = (
        candidate
        for candidate in mx_loading_mod.MX_POST_TRANSFORM_PROFILE_REGISTRY.profiles
        if candidate.profile_id == _PROFILE_ID
    )
    return profile


def _runtime_config(**overrides: object) -> PostTransformRuntimeConfig:
    """The realized runtime config of a qualified tiny model at TP1 on SM90, with overrides."""
    values = {
        "dtype": "bfloat16",
        "quant_algorithm": "FP8_BLOCK_SCALES",
        "kv_cache_quant_algorithm": "none",
        "layerwise_quantization": False,
        "force_dynamic_quantization": False,
        "lora_enabled": False,
        "sparse_attention_enabled": True,
        "attention_backend": "TRTLLM",
        "moe_backend": "CUTLASS",
        "tp_size": 1,
        "pp_size": 1,
        "cp_size": 1,
        "moe_tp_size": 1,
        "moe_ep_size": 1,
        "moe_cluster_size": 1,
        "attention_tp_size": 1,
        "attention_cp_size": 1,
        "attention_dp": False,
        "multi_node": False,
        "tied_word_embeddings": False,
        "rope_type": "default",
        "rope_fusion": True,
        "sliding_window": None,
        "sm_version": 90,
        "weight_block_size": (_FP8_BLOCK_SIZE, _FP8_BLOCK_SIZE),
        "first_k_dense_replace": _FIRST_K_DENSE_REPLACE,
        "moe_layer_freq": 1,
        "dsa_indexer_sharing": False,
        "dsa_indexer_projection_dtype": "float32",
    }
    values.update(overrides)
    return PostTransformRuntimeConfig(**values)


def _v_b_proj_offset_bytes(attention: MLA) -> int:
    """Byte offset of `v_b_proj` from the start of `kv_b_proj.weight`."""
    return attention.v_b_proj.data_ptr() - attention.kv_b_proj.weight.data_ptr()


def _layout_state(model: nn.Module) -> dict[str, object]:
    layers = model.model.layers
    attention = layers[_MOE_LAYER].self_attn
    indexer = attention.indexer
    moe = layers[_MOE_LAYER].mlp
    experts = moe.experts
    backend = experts.backend
    return {
        "attention_type": type(attention).__name__,
        "rope_fusion": attention.rope_fusion,
        "q_b_proj": (attention.q_b_proj.weight.dtype, tuple(attention.q_b_proj.weight.shape)),
        "q_b_proj_scale_shape": tuple(attention.q_b_proj.weight_scale.shape),
        "kv_a_proj": (
            attention.kv_a_proj_with_mqa.weight.dtype,
            tuple(attention.kv_a_proj_with_mqa.weight.shape),
        ),
        "o_proj": (attention.o_proj.weight.dtype, tuple(attention.o_proj.weight.shape)),
        "kv_b_proj": (attention.kv_b_proj.weight.dtype, tuple(attention.kv_b_proj.weight.shape)),
        "kv_b_proj_has_weight_scale": hasattr(attention.kv_b_proj, "weight_scale"),
        "k_b_proj_trans": (attention.k_b_proj_trans.dtype, tuple(attention.k_b_proj_trans.shape)),
        "v_b_proj_shape": tuple(attention.v_b_proj.shape),
        "v_b_proj_offsets_bytes": tuple(
            _v_b_proj_offset_bytes(layer.self_attn) for layer in layers
        ),
        "v_b_proj_strides": tuple(layer.self_attn.v_b_proj.stride() for layer in layers),
        "layer_indexers": tuple(layer.self_attn.indexer is not None for layer in layers),
        "indexer_wq_b": (indexer.wq_b.weight.dtype, tuple(indexer.wq_b.weight.shape)),
        "indexer_wk": (indexer.wk.weight.dtype, tuple(indexer.wk.weight.shape)),
        "indexer_weights_proj": (
            indexer.weights_proj.weight.dtype,
            tuple(indexer.weights_proj.weight.shape),
        ),
        "indexer_k_norm_dtype": indexer.k_norm.weight.dtype,
        "dense_mlp_types": tuple(
            type(layer.mlp).__name__ for layer in layers[:_FIRST_K_DENSE_REPLACE]
        ),
        "dense_gate_up": (
            layers[0].mlp.gate_up_proj.weight.dtype,
            tuple(layers[0].mlp.gate_up_proj.weight.shape),
        ),
        "moe_type": type(moe).__name__,
        "router_weight": (moe.gate.weight.dtype, tuple(moe.gate.weight.shape)),
        "router_bias_dtype": moe.gate.e_score_correction_bias.dtype,
        "shared_experts_type": type(moe.shared_experts).__name__,
        "shared_gate_up_dtype": moe.shared_experts.gate_up_proj.weight.dtype,
        "num_fused_shared_expert": getattr(experts, "num_fused_shared_expert", 0),
        "experts_type": type(experts).__name__,
        "backend_type": type(backend).__name__,
        "backend_id": type(backend).descriptor.identity.canonical(),
        "quant_method_type": type(backend.quant_method).__name__,
        "w3_w1_weight": (backend.w3_w1_weight.dtype, tuple(backend.w3_w1_weight.shape)),
        "w2_weight": (backend.w2_weight.dtype, tuple(backend.w2_weight.shape)),
        "w3_w1_scale_shape": tuple(backend.w3_w1_weight_scaling_factor.shape),
        "w2_scale_shape": tuple(backend.w2_weight_scaling_factor.shape),
        "initial_local_expert_ids": tuple(backend.initial_local_expert_ids),
        "layer_load_balancer": backend.layer_load_balancer,
        "tied_lm_head": model.lm_head.weight is model.model.embed_tokens.weight,
    }


def _pattern(model: nn.Module, *shape: int) -> torch.Tensor:
    """Fixed bf16 values in `[-0.5, 0.5]` on the model's device."""
    values = torch.arange(
        torch.Size(shape).numel(), dtype=torch.float32, device=model.lm_head.weight.device
    ).reshape(shape)
    return (((values % 17) - 8) / 17).to(torch.bfloat16)


def _embedding_logits(model: nn.Module) -> torch.Tensor:
    token_ids = torch.arange(8, device=model.lm_head.weight.device)
    return model.lm_head(model.model.embed_tokens(token_ids))


def _indexer_output(model: nn.Module) -> list[torch.Tensor]:
    """The indexer's fused `wk`/`weights_proj` projection, its `wq_b`, and its `k_norm`."""
    outputs = []
    for layer in model.model.layers:
        indexer = layer.self_attn.indexer
        hidden_states = _pattern(model, 3, model.config.hidden_size)
        outputs.append(
            torch.nn.functional.linear(
                hidden_states.to(indexer._fused_wk_wp_weight.dtype), indexer._fused_wk_wp_weight
            )
        )
        outputs.append(indexer.wq_b(_pattern(model, 3, model.config.q_lora_rank)))
        outputs.append(indexer.k_norm(_pattern(model, 3, indexer.head_dim)))
    return outputs


def _context_kv_b_proj_output(model: nn.Module) -> list[torch.Tensor]:
    """The MLA context path expands the compressed KV through `kv_b_proj`."""
    return [
        layer.self_attn.kv_b_proj(_pattern(model, 3, model.config.kv_lora_rank))
        for layer in model.model.layers
    ]


def _absorption_output(model: nn.Module) -> list[torch.Tensor]:
    """The MLA absorption path applies `k_b_proj_trans` and `v_b_proj` per head."""
    outputs = []
    for layer in model.model.layers:
        attention = layer.self_attn
        heads = attention.k_b_proj_trans.shape[0]
        q_nope = _pattern(model, heads, 3, attention.qk_nope_head_dim)
        latent = _pattern(model, heads, 3, attention.kv_lora_rank)
        outputs.append(torch.einsum("htd,hcd->htc", q_nope, attention.k_b_proj_trans))
        outputs.append(torch.einsum("htc,hvc->htv", latent, attention.v_b_proj))
    return outputs


def _fp8_attention_projection_output(model: nn.Module) -> list[torch.Tensor]:
    """The FP8 block-scale GEMMs of the fused `q_a`/`kv_a`, `q_b`, and `o` projections."""
    outputs = []
    for layer in model.model.layers:
        attention = layer.self_attn
        outputs.append(attention.kv_a_proj_with_mqa(_pattern(model, 3, model.config.hidden_size)))
        outputs.append(attention.q_b_proj(_pattern(model, 3, model.config.q_lora_rank)))
        outputs.append(attention.o_proj(_pattern(model, 3, attention.o_proj.in_features)))
    return outputs


def _dense_mlp_output(model: nn.Module) -> list[torch.Tensor]:
    return [
        layer.mlp(_pattern(model, 3, model.config.hidden_size))
        for layer in model.model.layers[:_FIRST_K_DENSE_REPLACE]
    ]


def _router_output(model: nn.Module) -> tuple[torch.Tensor, ...]:
    """Router logits and the routing decision, which depends on the correction bias."""
    gate = model.model.layers[_MOE_LAYER].mlp.gate
    logits = gate(_pattern(model, 3, model.config.hidden_size))
    indices, scales = gate.apply(logits)
    return logits, indices, scales


def _shared_experts_output(model: nn.Module) -> torch.Tensor:
    return model.model.layers[_MOE_LAYER].mlp.shared_experts(
        _pattern(model, 3, model.config.hidden_size)
    )


def _routed_expert_output(model: nn.Module) -> torch.Tensor:
    """Run the MoE layer's routed experts on fixed hidden states.

    The synthetic router logits send the three tokens to experts {0, 1}, {2, 3}
    and {0, 3}, so every expert, and both ranks of the MoE-EP partition, does
    work. Under MoE TP or EP the result is this rank's partial sum.
    """
    hidden_states = _pattern(model, 3, model.config.hidden_size)
    router_logits = torch.tensor(
        [[4.0, 3.0, 0.0, 0.0], [0.0, 0.0, 4.0, 3.0], [3.0, 0.0, 0.0, 4.0]],
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    output = model.model.layers[_MOE_LAYER].mlp.experts(
        hidden_states, router_logits, use_dp_padding=False
    )
    torch.cuda.synchronize()
    return output


# The embedding and LM head of a simulated TP2 rank gather across ranks, which
# a single process cannot run, so only TP1 probes them.
_TP2_OUTPUT_PROBES = (
    ("indexer", _indexer_output),
    ("context-kv-b-proj", _context_kv_b_proj_output),
    ("absorption", _absorption_output),
    ("fp8-attention-projections", _fp8_attention_projection_output),
    ("dense-mlp", _dense_mlp_output),
    ("router", _router_output),
    ("shared-experts", _shared_experts_output),
)
_TP1_OUTPUT_PROBES = (("embedding-logits", _embedding_logits), *_TP2_OUTPUT_PROBES)


def _without_activation_scales(factory_parameters: dict[str, torch.Tensor]) -> dict:
    """Drop the unit activation scales, which a dynamic-activation checkpoint never writes."""
    return {
        name: parameter
        for name, parameter in factory_parameters.items()
        if not name.endswith("input_scale")
    }


def _lifecycle_case(
    monkeypatch: pytest.MonkeyPatch,
    *,
    tp_size: int = 1,
    rank: int = 0,
    moe_ep_size: int = -1,
    output_probes: tuple | None = None,
) -> tuple[PostTransformQualificationCase, dict[str, torch.Tensor]]:
    if output_probes is None:
        output_probes = _TP1_OUTPUT_PROBES if tp_size == 1 else _TP2_OUTPUT_PROBES
    prepare_producer, factory_parameters = keep_factory_parameters(
        _hf_checkpoint_loader(_hf_state_dict())
    )

    def factory(model_class: type[nn.Module] = modeling_deepseekv3_mod.GlmMoeDsaForCausalLM):
        return _tiny_model(
            monkeypatch,
            model_class=model_class,
            tp_size=tp_size,
            rank=rank,
            moe_ep_size=moe_ep_size,
        )

    case = PostTransformQualificationCase(
        profile_id=_PROFILE_ID,
        model_factory=factory,
        unqualified_model_factory=lambda: factory(_UnqualifiedGlmMoeDsaForCausalLM),
        qualify_model=_qualify,
        state_probes=(
            ("layout", _layout_state),
            ("module-state", module_state_snapshot),
        ),
        output_probes=output_probes,
        prepare_producer=prepare_producer,
    )
    return case, factory_parameters


@_requires_sm90
def test_glm5_profile_qualifies_full_staged_lifecycle(monkeypatch: pytest.MonkeyPatch) -> None:
    """A staged receiver matches a producer that ran the real `DeepseekV3WeightLoader`.

    The producer loads a GLM-5-FP8-layout state dict through the production
    weight loader and then runs every post-load hook; the receiver only gets
    the producer's parameters and the staged finalization walks. The
    routed-expert output has its own case.
    """
    case, factory_parameters = _lifecycle_case(monkeypatch)

    producer, receiver = assert_post_transform_lifecycle_equivalent(case)

    # `v_b_proj` is a view of `kv_b_proj.weight`, so the loader overwrites it
    # through the V rows even though no factory byte of it is checked on its own.
    assert_every_parameter_was_loaded(producer, _without_activation_scales(factory_parameters))
    decision = _qualify(receiver)
    assert decision.transform_abi_id == GLM5_DSA_FP8_BLOCK_SM90_POST_TRANSFORM_LAYOUT_ABI_V1
    assert (
        PostTransformRuntimeConfig.from_model_config(
            producer.model_config, model=producer, sm_version=90
        )
        == _runtime_config()
    )
    config = _hf_config()
    heads = config.num_attention_heads
    nope, v_dim, kv_lora = config.qk_nope_head_dim, config.v_head_dim, config.kv_lora_rank
    hidden = config.hidden_size
    index_rows = config.index_head_dim + config.index_n_heads
    layout = _layout_state(producer)
    assert layout["attention_type"] == modeling_deepseekv3_mod.DeepseekV32Attention.__name__
    assert layout["rope_fusion"] is True
    assert layout["q_b_proj"] == (torch.float8_e4m3fn, (heads * (nope + 64), config.q_lora_rank))
    assert layout["kv_a_proj"] == (
        torch.float8_e4m3fn,
        (config.q_lora_rank + kv_lora + config.qk_rope_head_dim, hidden),
    )
    # `kv_b_proj` is dequantized to BF16 and keeps no FP8 block scale.
    assert layout["kv_b_proj"] == (torch.bfloat16, (heads * (nope + v_dim), kv_lora))
    assert layout["kv_b_proj_has_weight_scale"] is False
    assert layout["k_b_proj_trans"] == (torch.bfloat16, (heads, kv_lora, nope))
    assert layout["v_b_proj_shape"] == (heads, v_dim, kv_lora)
    # The V rows follow all K-nope rows of the re-laid-out `kv_b_proj.weight`.
    assert layout["v_b_proj_offsets_bytes"] == (heads * nope * kv_lora * 2,) * _NUM_LAYERS
    assert layout["layer_indexers"] == (True,) * _NUM_LAYERS
    assert layout["indexer_wq_b"] == (
        torch.float8_e4m3fn,
        (config.index_n_heads * config.index_head_dim, config.q_lora_rank),
    )
    assert layout["indexer_wk"] == (torch.float32, (config.index_head_dim, hidden))
    assert layout["indexer_weights_proj"] == (torch.float32, (config.index_n_heads, hidden))
    assert layout["indexer_k_norm_dtype"] == torch.float32
    assert layout["dense_mlp_types"] == ("GatedMLP",) * _FIRST_K_DENSE_REPLACE
    assert layout["dense_gate_up"] == (torch.float8_e4m3fn, (2 * config.intermediate_size, hidden))
    assert layout["moe_type"] == modeling_deepseekv3_mod.Deepseekv3MoE.__name__
    assert layout["router_weight"] == (torch.bfloat16, (_NUM_EXPERTS, hidden))
    assert layout["router_bias_dtype"] == torch.float32
    assert layout["shared_experts_type"] == "GatedMLP"
    assert layout["shared_gate_up_dtype"] == torch.float8_e4m3fn
    assert layout["num_fused_shared_expert"] == 0
    assert layout["backend_id"] == "deepgemm.cuda.hopper_grouped_gemm.fp8_block_scales"
    assert layout["quant_method_type"] == "DeepSeekFP8BlockScalesFusedMoEMethod"
    moe_inter = config.moe_intermediate_size
    assert layout["w3_w1_weight"] == (torch.float8_e4m3fn, (_NUM_EXPERTS, 2 * moe_inter, hidden))
    assert layout["w2_weight"] == (torch.float8_e4m3fn, (_NUM_EXPERTS, hidden, moe_inter))
    assert layout["w3_w1_scale_shape"] == (_NUM_EXPERTS, 2 * moe_inter // 128, hidden // 128)
    assert layout["w2_scale_shape"] == (_NUM_EXPERTS, hidden // 128, moe_inter // 128)
    assert layout["layer_load_balancer"] is None
    # The fused indexer projection is rebuilt on the receiver, never transferred.
    for layer in receiver.model.layers:
        fused_weight = layer.self_attn.indexer._fused_wk_wp_weight
        assert fused_weight.dtype == torch.float32
        assert tuple(fused_weight.shape) == (index_rows, hidden)
    assert not any(
        "_fused_wk_wp_weight" in name for name in mx_canonical_parameter_catalog(receiver)
    )
    # The receiver keeps its shared experts as a separate module.
    assert receiver.model.layers[_MOE_LAYER].mlp.shared_experts is not None


# The admitted TP2 partitions as `(MoE EP size, MoE TP size)`; `moe_ep_size=-1`
# leaves MoE TP equal to TP.
_TP2_PARTITIONS = (
    pytest.param(-1, 2, id="moe-tp2"),
    pytest.param(2, 1, id="moe-ep2"),
)


@_requires_sm90
@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("moe_ep_size, moe_tp_size", _TP2_PARTITIONS)
def test_glm5_profile_qualifies_tp2_rank_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
    rank: int,
    moe_ep_size: int,
    moe_tp_size: int,
) -> None:
    """Each simulated TP2 rank's staged receiver matches its producer that ran the real load."""
    case, factory_parameters = _lifecycle_case(
        monkeypatch, tp_size=2, rank=rank, moe_ep_size=moe_ep_size
    )

    producer, _receiver = assert_post_transform_lifecycle_equivalent(case)

    assert_every_parameter_was_loaded(producer, _without_activation_scales(factory_parameters))
    resolved_moe_ep_size = 2 // moe_tp_size
    assert PostTransformRuntimeConfig.from_model_config(
        producer.model_config, model=producer, sm_version=90
    ) == _runtime_config(
        tp_size=2,
        moe_tp_size=moe_tp_size,
        moe_ep_size=resolved_moe_ep_size,
        attention_tp_size=2,
    )
    config = _hf_config()
    local_heads = config.num_attention_heads // 2
    nope, v_dim, kv_lora = config.qk_nope_head_dim, config.v_head_dim, config.kv_lora_rank
    hidden = config.hidden_size
    local_experts = _NUM_EXPERTS // resolved_moe_ep_size
    local_intermediate_size = config.moe_intermediate_size // moe_tp_size
    layout = _layout_state(producer)
    # Attention TP splits the heads over the two ranks; every rank holds a full indexer.
    assert layout["kv_b_proj"] == (torch.bfloat16, (local_heads * (nope + v_dim), kv_lora))
    assert layout["v_b_proj_shape"] == (local_heads, v_dim, kv_lora)
    assert layout["v_b_proj_offsets_bytes"] == (local_heads * nope * kv_lora * 2,) * _NUM_LAYERS
    assert layout["indexer_wk"] == (torch.float32, (config.index_head_dim, hidden))
    assert layout["indexer_wq_b"] == (
        torch.float8_e4m3fn,
        (config.index_n_heads * config.index_head_dim, config.q_lora_rank),
    )
    assert layout["router_weight"] == (torch.bfloat16, (_NUM_EXPERTS, hidden))
    assert layout["w3_w1_weight"] == (
        torch.float8_e4m3fn,
        (local_experts, 2 * local_intermediate_size, hidden),
    )
    assert layout["w2_weight"] == (
        torch.float8_e4m3fn,
        (local_experts, hidden, local_intermediate_size),
    )
    expert_rank = rank if resolved_moe_ep_size > 1 else 0
    assert layout["initial_local_expert_ids"] == tuple(
        range(expert_rank * local_experts, (expert_rank + 1) * local_experts)
    )


@_requires_sm90
@pytest.mark.parametrize(
    "tp_size, rank, moe_ep_size",
    [
        pytest.param(1, 0, -1, id="tp1"),
        pytest.param(2, 0, -1, id="moe-tp2-rank0"),
        pytest.param(2, 1, -1, id="moe-tp2-rank1"),
        pytest.param(2, 0, 2, id="moe-ep2-rank0"),
        pytest.param(2, 1, 2, id="moe-ep2-rank1"),
    ],
)
def test_glm5_staged_receiver_matches_routed_expert_output(
    monkeypatch: pytest.MonkeyPatch,
    tp_size: int,
    rank: int,
    moe_ep_size: int,
) -> None:
    """A staged receiver's DeepGEMM Hopper routed experts compute exactly what its producer's do."""
    case, _factory_parameters = _lifecycle_case(
        monkeypatch,
        tp_size=tp_size,
        rank=rank,
        moe_ep_size=moe_ep_size,
        output_probes=(("routed-experts", _routed_expert_output),),
    )

    assert_post_transform_lifecycle_equivalent(case)


@_requires_sm90
def test_glm5_receiver_restores_v_b_proj_alias_before_transfer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The inherited receiver hook binds `v_b_proj` to the V rows of the GLM-5 `kv_b_proj`.

    GLM-5's K-nope head dimension is 192, so the V rows start past
    `heads * 192` rows of `kv_lora_rank` BF16 values. Storage offsets and alias
    groups match the donor's before the transfer and in the MX parameter
    catalog.
    """
    producer = _tiny_model(monkeypatch)
    receiver = _tiny_model(monkeypatch)
    _hf_checkpoint_loader(_hf_state_dict())(producer)
    model_hooks_mod.walk_full_post_load(producer)
    donor_catalog = mx_canonical_parameter_catalog(producer)

    model_hooks_mod.prepare_post_transform_receiver(receiver)

    config = _hf_config()
    aliased_offset_bytes = (
        config.num_attention_heads * config.qk_nope_head_dim * config.kv_lora_rank * 2
    )
    for layer in receiver.model.layers:
        attention = layer.self_attn
        assert (
            attention.v_b_proj.untyped_storage().data_ptr()
            == attention.kv_b_proj.weight.untyped_storage().data_ptr()
        )
        assert _v_b_proj_offset_bytes(attention) == aliased_offset_bytes
    receiver_catalog = mx_canonical_parameter_catalog(receiver)
    assert receiver_catalog.keys() == donor_catalog.keys()
    assert {name: entry[1:] for name, entry in receiver_catalog.items()} == {
        name: entry[1:] for name, entry in donor_catalog.items()
    }


@_requires_sm90
def test_staged_glm5_finalization_preserves_mx_tensor_catalog(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """MLA, indexer, and routed-expert tensors stay registered, with unchanged storage."""
    model = _tiny_model(monkeypatch)

    model_hooks_mod.prepare_post_transform_receiver(model)
    registered_catalog = mx_canonical_parameter_catalog(model)

    model_hooks_mod.setup_aliases(model)
    model_hooks_mod.mark_weights_transformed(model)
    model_hooks_mod.walk_cache_state(model)
    published_catalog = mx_canonical_parameter_catalog(model)

    assert published_catalog == registered_catalog
    prefix = f"model.layers.{_MOE_LAYER}."
    assert {
        f"{prefix}self_attn.kv_b_proj.weight",
        f"{prefix}self_attn.k_b_proj_trans",
        f"{prefix}self_attn.v_b_proj",
        f"{prefix}self_attn.q_b_proj.weight_scale",
        f"{prefix}self_attn.indexer.wq_b.weight",
        f"{prefix}self_attn.indexer.wq_b.weight_scale",
        f"{prefix}self_attn.indexer.wk.weight",
        f"{prefix}self_attn.indexer.weights_proj.weight",
        f"{prefix}self_attn.indexer.k_norm.weight",
        f"{prefix}mlp.gate.weight",
        f"{prefix}mlp.gate.e_score_correction_bias",
        f"{prefix}mlp.experts.backend.w3_w1_weight",
        f"{prefix}mlp.experts.backend.w3_w1_weight_scaling_factor",
        f"{prefix}mlp.experts.backend.w2_weight",
        f"{prefix}mlp.experts.backend.w2_weight_scaling_factor",
        f"{prefix}mlp.shared_experts.gate_up_proj.weight",
    } <= registered_catalog.keys()


@_requires_sm90
def test_glm5_qualification_uses_the_pre_construction_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The constructed model reports `deepseek_v32`, so only the captured identity qualifies."""
    model = _tiny_model(monkeypatch, filled=False)

    constructed_identity = PostTransformConfigIdentity.from_model_config(model.model_config)
    decision = mx_loading_mod.qualify_post_transform_profile(
        model, speculative_mode=None, loads_draft_weights=False
    )

    assert constructed_identity.model_type == "deepseek_v32"
    assert not decision.qualified
    assert decision.reason is PostTransformQualificationReason.MODEL_TYPE_NOT_REGISTERED
    assert _qualify(model).qualified


@_requires_sm90
def test_mx_prepares_the_qualified_glm5_receiver(monkeypatch: pytest.MonkeyPatch) -> None:
    """MX asks for receiver preparation, and so the `v_b_proj` restore, for the GLM-5 root."""
    model = _tiny_model(monkeypatch, filled=False)
    decision = _qualify(model)

    kwargs = mx_loading_mod.get_mx_load_weights_kwargs(
        model_config=model.model_config,
        llm_args=MagicMock(name="llm_args"),
        qualification=decision,
    )

    assert decision.qualified
    assert decision.profile is _profile()
    assert kwargs["prepare_post_transform_receiver"] is (
        model_hooks_mod.prepare_post_transform_receiver
    )


@_requires_sm90
def test_glm5_profile_rejects_shared_dsa_indexers(monkeypatch: pytest.MonkeyPatch) -> None:
    """A checkpoint whose layers reuse another layer's top-k falls back.

    Layer 1 of the `FSFF` pattern owns no indexer, so the transferred tensor
    set differs from the ABI's indexer in every layer.
    """
    model = _tiny_model(
        monkeypatch, filled=False, config_overrides={"index_topk_pattern": list("FSFF")}
    )

    decision = _qualify(model)

    assert tuple(layer.self_attn.indexer is not None for layer in model.model.layers) == (
        True,
        False,
        True,
        True,
    )
    assert not decision.qualified
    assert decision.reason is PostTransformQualificationReason.RUNTIME_CONFIG_NOT_SUPPORTED
    assert decision.unsupported_runtime_dimensions == frozenset({"dsa_indexer_sharing"})


@_requires_sm90
def test_glm5_profile_rejects_bf16_indexer_projections(monkeypatch: pytest.MonkeyPatch) -> None:
    """`TRTLLM_DSA_INDEXER_BF16=1` builds BF16 indexer projections, which the ABI does not hold."""
    monkeypatch.setenv("TRTLLM_DSA_INDEXER_BF16", "1")
    model = _tiny_model(monkeypatch, filled=False)

    decision = _qualify(model)

    indexer = model.model.layers[0].self_attn.indexer
    assert (indexer.wk.weight.dtype, indexer.weights_proj.weight.dtype) == (
        torch.bfloat16,
        torch.bfloat16,
    )
    assert not decision.qualified
    assert decision.reason is PostTransformQualificationReason.RUNTIME_CONFIG_NOT_SUPPORTED
    assert decision.unsupported_runtime_dimensions == frozenset({"dsa_indexer_projection_dtype"})


@_requires_sm90
@pytest.mark.parametrize("sm_version", [100, 103, 120])
def test_glm5_profile_rejects_a_load_on_another_sm(
    monkeypatch: pytest.MonkeyPatch, sm_version: int
) -> None:
    """Qualification reads the SM of the running GPU, and only SM90 qualifies."""
    model = _tiny_model(monkeypatch, filled=False)
    monkeypatch.setattr(mx_loading_mod, "get_sm_version", lambda: sm_version)

    decision = _qualify(model)

    assert not decision.qualified
    assert decision.reason is PostTransformQualificationReason.RUNTIME_CONFIG_NOT_SUPPORTED
    assert decision.unsupported_runtime_dimensions == frozenset({"sm_version"})


@_requires_sm90
def test_glm5_source_identity_separates_transform_abis(monkeypatch: pytest.MonkeyPatch) -> None:
    """The same model under the DeepSeek-V3 transform ABI does not share weights."""
    model = _tiny_model(monkeypatch)
    artifact_identity = ArtifactIdentity(
        format_version=ARTIFACT_IDENTITY_FORMAT_VERSION,
        scheme="checkpoint_manifest_sha256",
        digest="0" * 64,
    )
    local, source = (
        SourceIdentity.from_model_config(
            model.model_config,
            model,
            artifact_identity=artifact_identity,
            transform_abi_id=abi,
        )
        for abi in (
            GLM5_DSA_FP8_BLOCK_SM90_POST_TRANSFORM_LAYOUT_ABI_V1,
            DEEPSEEK_V3_MLA_CUTLASS_POST_TRANSFORM_LAYOUT_ABI_V1,
        )
    )

    decision = check_weight_sharing_compatibility(local, source, IdentityCheckPolicy.WARN_FALLBACK)

    assert not decision.should_share
    assert decision.match_result.mismatched_fields == ["transform_abi_id"]


def _qualify_registry(**overrides: object) -> PostTransformQualificationDecision:
    """Qualify the registered GLM-5 identity directly, with `overrides` applied."""
    arguments = {
        "root_model_class": modeling_deepseekv3_mod.GlmMoeDsaForCausalLM,
        "architecture": "GlmMoeDsaForCausalLM",
        "model_type": "glm_moe_dsa",
        "speculative_mode": None,
        "protocol_version": mx_loading_mod.MX_STAGED_RECEIVER_TRANSFORM_PROTOCOL_VERSION,
        "transfer_scope": PostTransformTransferScope.TARGET_MODEL,
        "runtime_config": _runtime_config(),
    }
    arguments.update(overrides)
    return mx_loading_mod.MX_POST_TRANSFORM_PROFILE_REGISTRY.qualify(**arguments)


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "overrides",
    [
        pytest.param({}, id="tp1"),
        pytest.param(
            {"tp_size": 2, "moe_tp_size": 2, "attention_tp_size": 2},
            id="moe-tp2",
        ),
        pytest.param(
            {"tp_size": 2, "moe_tp_size": 1, "moe_ep_size": 2, "attention_tp_size": 2},
            id="moe-ep2",
        ),
    ],
)
def test_glm5_profile_matches_its_exact_identity(overrides: dict[str, object]) -> None:
    decision = _qualify_registry(runtime_config=_runtime_config(**overrides))

    assert decision.qualified
    assert decision.profile is _profile()
    assert decision.transform_abi_id == GLM5_DSA_FP8_BLOCK_SM90_POST_TRANSFORM_LAYOUT_ABI_V1


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "overrides, reason",
    [
        pytest.param(
            {"root_model_class": modeling_deepseekv3_mod.DeepseekV3ForCausalLM},
            PostTransformQualificationReason.ARCHITECTURE_NOT_REGISTERED,
            id="deepseek-v3-root",
        ),
        pytest.param(
            {"root_model_class": _UnqualifiedGlmMoeDsaForCausalLM},
            PostTransformQualificationReason.ROOT_MODEL_CLASS_NOT_REGISTERED,
            id="subclass-root",
        ),
        pytest.param(
            {"architecture": "DeepseekV32ForCausalLM"},
            PostTransformQualificationReason.ARCHITECTURE_NOT_REGISTERED,
            id="deepseek-v3-2-architecture",
        ),
        pytest.param(
            {"model_type": "deepseek_v32"},
            PostTransformQualificationReason.MODEL_TYPE_NOT_REGISTERED,
            id="constructed-model-type",
        ),
        pytest.param(
            {"speculative_mode": "MTP"},
            PostTransformQualificationReason.SPECULATIVE_MODE_NOT_REGISTERED,
            id="mtp-speculative-mode",
        ),
        pytest.param(
            {"transfer_scope": PostTransformTransferScope.LANGUAGE_MODEL},
            PostTransformQualificationReason.TRANSFER_SCOPE_NOT_REGISTERED,
            id="language-model-scope",
        ),
    ],
)
def test_glm5_profile_rejects_other_identities(
    overrides: dict[str, object], reason: PostTransformQualificationReason
) -> None:
    """Only the exact root, architecture, and pre-construction model type of GLM-5 qualify."""
    decision = _qualify_registry(**overrides)

    assert not decision.qualified
    assert decision.reason is reason


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "overrides, expected_dimensions",
    [
        pytest.param({"sm_version": 100}, {"sm_version"}, id="sm100"),
        pytest.param({"sm_version": 103}, {"sm_version"}, id="sm103"),
        pytest.param({"sm_version": 120}, {"sm_version"}, id="sm120"),
        pytest.param({"sm_version": None}, {"sm_version"}, id="unknown-sm"),
        pytest.param({"quant_algorithm": "none"}, {"quant_algorithm"}, id="bf16-weights"),
        pytest.param({"quant_algorithm": "NVFP4"}, {"quant_algorithm"}, id="nvfp4"),
        pytest.param({"weight_block_size": None}, {"weight_block_size"}, id="undeclared-block"),
        pytest.param({"weight_block_size": (64, 64)}, {"weight_block_size"}, id="64x64-blocks"),
        pytest.param(
            {"kv_cache_quant_algorithm": "FP8"}, {"kv_cache_quant_algorithm"}, id="fp8-kv-cache"
        ),
        pytest.param(
            {"sparse_attention_enabled": False}, {"sparse_attention_enabled"}, id="dense-attention"
        ),
        pytest.param({"dsa_indexer_sharing": True}, {"dsa_indexer_sharing"}, id="shared-indexers"),
        pytest.param({"dsa_indexer_sharing": None}, {"dsa_indexer_sharing"}, id="no-dsa-indexers"),
        pytest.param(
            {"dsa_indexer_projection_dtype": "bfloat16"},
            {"dsa_indexer_projection_dtype"},
            id="bf16-indexer-projections",
        ),
        pytest.param(
            {"dsa_indexer_projection_dtype": None},
            {"dsa_indexer_projection_dtype"},
            id="mixed-indexer-projections",
        ),
        pytest.param({"first_k_dense_replace": 1}, {"first_k_dense_replace"}, id="one-dense-layer"),
        pytest.param({"moe_layer_freq": 2}, {"moe_layer_freq"}, id="moe-every-other-layer"),
        pytest.param({"moe_backend": "TRTLLM"}, {"moe_backend"}, id="trtllm-moe-backend"),
        pytest.param({"moe_backend": "DEEPGEMM"}, {"moe_backend"}, id="deepgemm-moe-backend"),
        pytest.param({"rope_type": "yarn"}, {"rope_type"}, id="yarn-rope"),
        pytest.param({"rope_fusion": False}, {"rope_fusion"}, id="unfused-rope"),
        pytest.param({"attention_dp": True}, {"attention_dp"}, id="attention-dp"),
        pytest.param(
            {"cp_size": 2, "attention_cp_size": 2}, {"cp_size", "attention_cp_size"}, id="cp2"
        ),
        pytest.param(
            {"tp_size": 4, "moe_tp_size": 4, "attention_tp_size": 4},
            {"tp_size", "moe_tp_size", "attention_tp_size"},
            id="tp4",
        ),
        pytest.param(
            {"tp_size": 8, "moe_tp_size": 1, "moe_ep_size": 8, "attention_tp_size": 8},
            {"tp_size", "moe_ep_size", "attention_tp_size"},
            id="tp8-ep8",
        ),
        pytest.param(
            {
                "tp_size": 2,
                "moe_tp_size": 1,
                "moe_ep_size": 1,
                "moe_cluster_size": 2,
                "attention_tp_size": 2,
            },
            {"moe_cluster_size"},
            id="moe-cluster2",
        ),
    ],
)
def test_glm5_profile_rejects_unqualified_runtime_variants(
    overrides: dict[str, object], expected_dimensions: set[str]
) -> None:
    """The profile admits only SM90 FP8 128x128 weights, BF16 KV, and the GLM-5 layer layout.

    It also rejects shared DSA indexers, BF16 indexer projections, other MoE
    backends, attention DP, CP, and wider TP.
    """
    decision = _qualify_registry(runtime_config=_runtime_config(**overrides))

    assert not decision.qualified
    assert decision.reason is PostTransformQualificationReason.RUNTIME_CONFIG_NOT_SUPPORTED
    assert decision.unsupported_runtime_dimensions == frozenset(expected_dimensions)


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "feature", [PostTransformFeature.MOE_LOAD_BALANCER, PostTransformFeature.DWDP]
)
def test_glm5_profile_rejects_unsupported_moe_features(feature: PostTransformFeature) -> None:
    decision = _qualify_registry(enabled_features=frozenset({feature}))

    assert not decision.qualified
    assert decision.reason is PostTransformQualificationReason.FEATURE_NOT_SUPPORTED
    assert decision.unsupported_features == frozenset({feature})


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "sm_100f, expected_backend",
    [
        pytest.param(False, "CUTLASS", id="sm90"),
        pytest.param(True, "TRTLLM", id="sm100f"),
    ],
)
def test_glm5_auto_backend_resolves_to_the_pinned_literal_only_on_hopper(
    monkeypatch: pytest.MonkeyPatch, sm_100f: bool, expected_backend: str
) -> None:
    """`AUTO` resolves GLM-5 FP8 to the pinned `CUTLASS` on Hopper and to `TRTLLM` on Blackwell."""
    monkeypatch.setattr(model_config_mod, "is_sm_100f", lambda *_args, **_kwargs: sm_100f)

    resolved = ModelConfig.resolve_moe_backend(
        "AUTO",
        "GlmMoeDsaForCausalLM",
        QuantConfig(quant_algo=QuantAlgo.FP8_BLOCK_SCALES, group_size=_FP8_BLOCK_SIZE),
    )

    assert resolved == expected_backend
