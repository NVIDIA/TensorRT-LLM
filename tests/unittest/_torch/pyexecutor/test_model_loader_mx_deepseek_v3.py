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
"""MX post-transform qualification tests for the DeepSeek-V3 MLA + CUTLASS MoE root."""

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
from tensorrt_llm._torch.attention.mla import MLA
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models import modeling_deepseekv3 as modeling_deepseekv3_mod
from tensorrt_llm._torch.models import modeling_mistral_large3 as modeling_mistral_large3_mod
from tensorrt_llm._torch.moe.fused_moe.configurable_moe import ConfigurableMoE
from tensorrt_llm._torch.moe.fused_moe.cutlass import TrtllmCutlassUnquantizedImpl
from tensorrt_llm._torch.moe.fused_moe.quantization import UnquantizedFusedMoEMethod
from tensorrt_llm._torch.pyexecutor.model_loader import ModelLoader
from tensorrt_llm._torch.weight_sharing import (
    ARTIFACT_IDENTITY_FORMAT_VERSION,
    DEEPSEEK_V3_MLA_CUTLASS_POST_TRANSFORM_LAYOUT_ABI_V1,
    QWEN3_MOE_CUTLASS_POST_TRANSFORM_LAYOUT_ABI_V1,
    ArtifactIdentity,
    IdentityCheckPolicy,
    PostTransformFeature,
    PostTransformProfile,
    PostTransformQualificationDecision,
    PostTransformQualificationReason,
    PostTransformRuntimeConfig,
    PostTransformTransferScope,
    SourceIdentity,
    check_weight_sharing_compatibility,
)
from tensorrt_llm.models.modeling_utils import QuantConfig

_PROFILE_ID = "deepseek-v3-for-causal-lm-bf16-cutlass-target-v1"
_NUM_EXPERTS = 4
_NUM_LAYERS = 2

# The loop below runs the real CUTLASS MoE leaf, which supports SM80 and newer.
_requires_cutlass_bf16_moe = pytest.mark.skipif(
    not torch.cuda.is_available() or getSMVersion() < 80,
    reason="the DeepSeek-V3 fixture needs CUDA and the SM80+ CUTLASS MoE leaf",
)
# The routed-expert forward of this fixture is validated only on SM90 and newer.
_requires_sm90_routed_expert_forward = pytest.mark.skipif(
    not torch.cuda.is_available() or getSMVersion() < 90,
    reason="the DeepSeek-V3 fixture's routed-expert forward is validated only on SM90 and newer",
)


class _UnqualifiedDeepseekV3ForCausalLM(modeling_deepseekv3_mod.DeepseekV3ForCausalLM):
    """Subclass that must not inherit the exact-root DeepSeek-V3 profile."""


def _hf_config(*, q_lora_rank: int | None = None, **overrides: object) -> PretrainedConfig:
    """Tiny two-layer config with the real MLA geometry.

    Layer 0 is a dense MLP and layer 1 is MoE with four routed experts (top-2)
    and one shared expert. `q_lora_rank=None` is the DeepSeek-V3-Lite query
    path; a value selects the low-rank query path of the full model.
    """
    values: dict[str, object] = {
        "architectures": ["DeepseekV3ForCausalLM"],
        "model_type": "deepseek_v3",
        "vocab_size": 64,
        "hidden_size": 256,
        "intermediate_size": 256,
        "moe_intermediate_size": 256,
        "num_hidden_layers": _NUM_LAYERS,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "q_lora_rank": q_lora_rank,
        "kv_lora_rank": 512,
        "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64,
        "v_head_dim": 128,
        "n_routed_experts": _NUM_EXPERTS,
        "n_shared_experts": 1,
        "num_experts_per_tok": 2,
        "first_k_dense_replace": 1,
        "moe_layer_freq": 1,
        "n_group": 1,
        "topk_group": 1,
        "routed_scaling_factor": 2.0,
        "scoring_func": "sigmoid",
        "topk_method": "noaux_tc",
        "norm_topk_prob": True,
        "max_position_embeddings": 64,
        "rope_theta": 10000.0,
        "rope_scaling": None,
        "rms_norm_eps": 1e-6,
        "hidden_act": "silu",
        "tie_word_embeddings": False,
        "num_nextn_predict_layers": 1,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "pad_token_id": None,
        "torch_dtype": torch.bfloat16,
    }
    values.update(overrides)
    return PretrainedConfig(**values)


_LITE = pytest.param(None, id="lite")
_Q_LORA = pytest.param(96, id="q-lora")


def _tiny_model(
    monkeypatch: pytest.MonkeyPatch,
    *,
    q_lora_rank: int | None = None,
    model_class: type[nn.Module] = modeling_deepseekv3_mod.DeepseekV3ForCausalLM,
    tp_size: int = 1,
    rank: int = 0,
    moe_ep_size: int = -1,
    filled: bool = True,
) -> nn.Module:
    """Build a two-layer DeepSeek-V3 model on CUDA with deterministic weights.

    `moe_ep_size=2` at `tp_size=2` selects the MoE-EP partition; the default
    leaves MoE TP equal to TP. TP=2 builds one rank of the group with
    collectives stubbed out. The model is built on the CPU and moved to CUDA,
    because the attention backend creates its rotary tables with NumPy.
    """
    if tp_size > 1:
        monkeypatch.setattr(mapping_mod, "mpi_disabled", lambda: False)
        monkeypatch.setattr(distributed_mod, "AllReduce", AllReduceStub)
        monkeypatch.setattr(modeling_deepseekv3_mod, "AllReduce", AllReduceStub)
        monkeypatch.setattr(modeling_deepseekv3_mod, "MoEAllReduce", AllReduceStub)
        monkeypatch.setattr(modeling_deepseekv3_mod, "can_access_peer", lambda _mapping: False)
    model = model_class(
        ModelConfig(
            pretrained_config=_hf_config(q_lora_rank=q_lora_rank),
            mapping=mapping_mod.Mapping(
                world_size=tp_size,
                rank=rank,
                tp_size=tp_size,
                moe_ep_size=moe_ep_size,
            ),
            max_num_tokens=16,
            max_seq_len=16,
        )
    ).cuda()
    if filled:
        with torch.no_grad():
            for index, parameter in enumerate(model.parameters()):
                values = torch.arange(
                    parameter.numel(),
                    dtype=torch.float32,
                    device=parameter.device,
                ).reshape(parameter.shape)
                # Centered and scaled so the routed-expert outputs stay well
                # inside the BF16 range.
                parameter.copy_((((values + index) % 17) - 8).to(parameter.dtype) / 64)
    return model


def _hf_state_dict(q_lora_rank: int | None) -> dict[str, torch.Tensor]:
    """Return the tiny config's checkpoint tensors in the Hugging Face layout.

    Routed experts are stored per expert, as in a checkpoint on disk. Every
    value is positive, so no loaded tensor can equal the factory's fill, which
    is centered on zero.
    """
    config = _hf_config(q_lora_rank=q_lora_rank)
    hidden = config.hidden_size
    heads = config.num_attention_heads
    qk_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim
    kv_lora = config.kv_lora_rank
    moe_inter = config.moe_intermediate_size
    shared_inter = moe_inter * config.n_shared_experts
    shapes = {
        "model.embed_tokens.weight": (config.vocab_size, hidden),
        "model.norm.weight": (hidden,),
        "lm_head.weight": (config.vocab_size, hidden),
    }
    for layer in range(config.num_hidden_layers):
        prefix = f"model.layers.{layer}."
        shapes |= {
            prefix + "input_layernorm.weight": (hidden,),
            prefix + "post_attention_layernorm.weight": (hidden,),
            prefix + "self_attn.kv_a_proj_with_mqa.weight": (
                kv_lora + config.qk_rope_head_dim,
                hidden,
            ),
            prefix + "self_attn.kv_a_layernorm.weight": (kv_lora,),
            prefix + "self_attn.kv_b_proj.weight": (
                heads * (config.qk_nope_head_dim + config.v_head_dim),
                kv_lora,
            ),
            prefix + "self_attn.o_proj.weight": (hidden, heads * config.v_head_dim),
        }
        if config.q_lora_rank is None:
            shapes[prefix + "self_attn.q_proj.weight"] = (heads * qk_head_dim, hidden)
        else:
            shapes |= {
                prefix + "self_attn.q_a_proj.weight": (config.q_lora_rank, hidden),
                prefix + "self_attn.q_a_layernorm.weight": (config.q_lora_rank,),
                prefix + "self_attn.q_b_proj.weight": (heads * qk_head_dim, config.q_lora_rank),
            }
        if layer < config.first_k_dense_replace:
            shapes |= {
                prefix + "mlp.gate_proj.weight": (config.intermediate_size, hidden),
                prefix + "mlp.up_proj.weight": (config.intermediate_size, hidden),
                prefix + "mlp.down_proj.weight": (hidden, config.intermediate_size),
            }
            continue
        shapes |= {
            prefix + "mlp.gate.weight": (config.n_routed_experts, hidden),
            prefix + "mlp.gate.e_score_correction_bias": (config.n_routed_experts,),
            prefix + "mlp.shared_experts.gate_proj.weight": (shared_inter, hidden),
            prefix + "mlp.shared_experts.up_proj.weight": (shared_inter, hidden),
            prefix + "mlp.shared_experts.down_proj.weight": (hidden, shared_inter),
        }
        for expert in range(config.n_routed_experts):
            shapes |= {
                prefix + f"mlp.experts.{expert}.gate_proj.weight": (moe_inter, hidden),
                prefix + f"mlp.experts.{expert}.up_proj.weight": (moe_inter, hidden),
                prefix + f"mlp.experts.{expert}.down_proj.weight": (hidden, moe_inter),
            }
    state_dict = {}
    for index, (name, shape) in enumerate(shapes.items()):
        values = torch.arange(torch.Size(shape).numel(), dtype=torch.float32).reshape(shape)
        state_dict[name] = (((values + index) % 17 + 1) / 256).to(torch.bfloat16)
    return state_dict


def _hf_checkpoint_loader(state_dict: dict[str, torch.Tensor]):
    """Return a `prepare_producer` hook that runs the real `DeepseekV3WeightLoader`."""

    def _load(model: nn.Module) -> None:
        model.load_weights(dict(state_dict))

    return _load


def _qualify(model: nn.Module) -> PostTransformQualificationDecision:
    return ModelLoader._qualify_post_transform_profile(
        model,
        speculative_mode=None,
        loads_draft_weights=False,
    )


def _profile() -> PostTransformProfile:
    (profile,) = (
        candidate
        for candidate in ModelLoader._post_transform_profile_registry().profiles
        if candidate.profile_id == _PROFILE_ID
    )
    return profile


def _runtime_config(**overrides: object) -> PostTransformRuntimeConfig:
    """The realized runtime config of a qualified tiny model at TP1, with overrides."""
    values = {
        "dtype": "bfloat16",
        "quant_algorithm": "none",
        "kv_cache_quant_algorithm": "none",
        "layerwise_quantization": False,
        "force_dynamic_quantization": False,
        "lora_enabled": False,
        "sparse_attention_enabled": False,
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
    }
    values.update(overrides)
    return PostTransformRuntimeConfig(**values)


def _v_b_proj_offset_bytes(attention: MLA) -> int:
    """Byte offset of `v_b_proj` from the start of `kv_b_proj.weight`."""
    return attention.v_b_proj.data_ptr() - attention.kv_b_proj.weight.data_ptr()


def _layout_state(model: nn.Module) -> dict[str, object]:
    layers = model.model.layers
    attention = layers[1].self_attn
    moe = layers[1].mlp
    experts = moe.experts
    backend = experts.backend
    return {
        "attention_type": type(attention).__name__,
        "rope_fusion": attention.rope_fusion,
        "q_lora_rank": attention.q_lora_rank,
        "q_b_proj_shape": tuple(attention.q_b_proj.weight.shape),
        "kv_a_proj_shape": tuple(attention.kv_a_proj_with_mqa.weight.shape),
        "kv_b_proj_shape": tuple(attention.kv_b_proj.weight.shape),
        "k_b_proj_trans_shape": tuple(attention.k_b_proj_trans.shape),
        "v_b_proj_shape": tuple(attention.v_b_proj.shape),
        "v_b_proj_offsets_bytes": tuple(
            _v_b_proj_offset_bytes(layer.self_attn) for layer in layers
        ),
        "v_b_proj_strides": tuple(layer.self_attn.v_b_proj.stride() for layer in layers),
        "first_layer_mlp_type": type(layers[0].mlp).__name__,
        "moe_type": type(moe).__name__,
        "router_weight_shape": tuple(moe.gate.weight.shape),
        "router_bias_dtype": moe.gate.e_score_correction_bias.dtype,
        "router_moe_backend": moe.gate.moe_backend,
        "shared_experts_type": type(moe.shared_experts).__name__,
        "num_fused_shared_expert": getattr(experts, "num_fused_shared_expert", 0),
        "experts_type": type(experts).__name__,
        "backend_type": type(backend).__name__,
        "backend_id": type(backend).descriptor.identity.canonical(),
        "quant_method_type": type(backend.quant_method).__name__,
        "quant_scales": backend.quant_scales,
        "routing_method_type": type(experts.routing_method).__name__,
        "w3_w1_weight_shape": tuple(backend.w3_w1_weight.shape),
        "w2_weight_shape": tuple(backend.w2_weight.shape),
        "expert_size_per_partition": backend.expert_size_per_partition,
        "initial_local_expert_ids": tuple(backend.initial_local_expert_ids),
        "layer_load_balancer": backend.layer_load_balancer,
        "layer0_next_norm": layers[0].next_layer_layernorm is layers[1].input_layernorm,
        "layer1_next_norm": layers[1].next_layer_layernorm is model.model.norm,
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


def _dense_mlp_output(model: nn.Module) -> torch.Tensor:
    return model.model.layers[0].mlp(_pattern(model, 3, model.config.hidden_size))


def _router_output(model: nn.Module) -> tuple[torch.Tensor, ...]:
    """Router logits and the routing decision, which depends on the correction bias."""
    gate = model.model.layers[1].mlp.gate
    logits = gate(_pattern(model, 3, model.config.hidden_size))
    indices, scales = gate.apply(logits)
    return logits, indices, scales


def _shared_experts_output(model: nn.Module) -> torch.Tensor:
    return model.model.layers[1].mlp.shared_experts(_pattern(model, 3, model.config.hidden_size))


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
    output = model.model.layers[1].mlp.experts(hidden_states, router_logits, use_dp_padding=False)
    torch.cuda.synchronize()
    return output


# The embedding and LM head of a simulated TP2 rank gather across ranks, which
# a single process cannot run, so only TP1 probes them.
_TP2_OUTPUT_PROBES = (
    ("context-kv-b-proj", _context_kv_b_proj_output),
    ("absorption", _absorption_output),
    ("dense-mlp", _dense_mlp_output),
    ("router", _router_output),
    ("shared-experts", _shared_experts_output),
)
_TP1_OUTPUT_PROBES = (("embedding-logits", _embedding_logits), *_TP2_OUTPUT_PROBES)


def _lifecycle_case(
    monkeypatch: pytest.MonkeyPatch,
    q_lora_rank: int | None,
    *,
    tp_size: int = 1,
    rank: int = 0,
    moe_ep_size: int = -1,
    output_probes: tuple | None = None,
) -> tuple[PostTransformQualificationCase, dict[str, torch.Tensor]]:
    if output_probes is None:
        output_probes = _TP1_OUTPUT_PROBES if tp_size == 1 else _TP2_OUTPUT_PROBES
    prepare_producer, factory_parameters = keep_factory_parameters(
        _hf_checkpoint_loader(_hf_state_dict(q_lora_rank))
    )

    def factory(model_class: type[nn.Module] = modeling_deepseekv3_mod.DeepseekV3ForCausalLM):
        return _tiny_model(
            monkeypatch,
            q_lora_rank=q_lora_rank,
            model_class=model_class,
            tp_size=tp_size,
            rank=rank,
            moe_ep_size=moe_ep_size,
        )

    case = PostTransformQualificationCase(
        profile_id=_PROFILE_ID,
        model_factory=factory,
        unqualified_model_factory=lambda: factory(_UnqualifiedDeepseekV3ForCausalLM),
        qualify_model=_qualify,
        state_probes=(
            ("layout", _layout_state),
            ("module-state", module_state_snapshot),
        ),
        output_probes=output_probes,
        prepare_producer=prepare_producer,
    )
    return case, factory_parameters


@_requires_cutlass_bf16_moe
@pytest.mark.parametrize("q_lora_rank", [_LITE, _Q_LORA])
def test_deepseek_v3_profile_qualifies_full_staged_lifecycle(
    monkeypatch: pytest.MonkeyPatch, q_lora_rank: int | None
) -> None:
    """A staged receiver matches a producer that ran the real `DeepseekV3WeightLoader`.

    The producer loads a checkpoint-format state dict through the production
    weight loader and then runs every post-load hook; the receiver only gets
    the producer's parameters and the staged finalization walks. The
    routed-expert output has its own SM90+ case.
    """
    case, factory_parameters = _lifecycle_case(monkeypatch, q_lora_rank)

    producer, receiver = assert_post_transform_lifecycle_equivalent(case)

    # `v_b_proj` is a view of `kv_b_proj.weight`, so the loader overwrites it
    # through the V rows even though no factory byte of it is checked on its own.
    assert_every_parameter_was_loaded(producer, factory_parameters)
    decision = _qualify(receiver)
    assert decision.transform_abi_id == DEEPSEEK_V3_MLA_CUTLASS_POST_TRANSFORM_LAYOUT_ABI_V1
    assert (
        PostTransformRuntimeConfig.from_model_config(producer.model_config, model=producer)
        == _runtime_config()
    )
    config = _hf_config(q_lora_rank=q_lora_rank)
    heads = config.num_attention_heads
    nope, v_dim, kv_lora = config.qk_nope_head_dim, config.v_head_dim, config.kv_lora_rank
    layout = _layout_state(producer)
    assert layout["attention_type"] == modeling_deepseekv3_mod.DeepseekV3Attention.__name__
    assert layout["rope_fusion"] is True
    assert layout["kv_b_proj_shape"] == (heads * (nope + v_dim), kv_lora)
    assert layout["k_b_proj_trans_shape"] == (heads, kv_lora, nope)
    assert layout["v_b_proj_shape"] == (heads, v_dim, kv_lora)
    # The V rows follow all K-nope rows of the re-laid-out `kv_b_proj.weight`.
    assert layout["v_b_proj_offsets_bytes"] == (heads * nope * kv_lora * 2,) * _NUM_LAYERS
    assert layout["first_layer_mlp_type"] == "GatedMLP"
    assert layout["moe_type"] == modeling_deepseekv3_mod.Deepseekv3MoE.__name__
    assert layout["router_bias_dtype"] == torch.float32
    assert layout["shared_experts_type"] == "GatedMLP"
    assert layout["num_fused_shared_expert"] == 0
    assert layout["experts_type"] == ConfigurableMoE.__name__
    assert layout["backend_type"] == TrtllmCutlassUnquantizedImpl.__name__
    assert layout["backend_id"] == "trtllm.cutlass.grouped_gemm.none"
    assert layout["quant_method_type"] == UnquantizedFusedMoEMethod.__name__
    assert layout["layer_load_balancer"] is None
    # The receiver keeps its shared experts as a separate module.
    assert receiver.model.layers[1].mlp.shared_experts is not None


# The admitted TP2 partitions as `(MoE EP size, MoE TP size)`; `moe_ep_size=-1`
# leaves MoE TP equal to TP.
_TP2_PARTITIONS = (
    pytest.param(-1, 2, id="moe-tp2"),
    pytest.param(2, 1, id="moe-ep2"),
)


@_requires_cutlass_bf16_moe
@pytest.mark.parametrize("q_lora_rank", [_LITE, _Q_LORA])
@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("moe_ep_size, moe_tp_size", _TP2_PARTITIONS)
def test_deepseek_v3_profile_qualifies_tp2_rank_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
    q_lora_rank: int | None,
    rank: int,
    moe_ep_size: int,
    moe_tp_size: int,
) -> None:
    """Each simulated TP2 rank's staged receiver matches its producer that ran the real load."""
    case, factory_parameters = _lifecycle_case(
        monkeypatch, q_lora_rank, tp_size=2, rank=rank, moe_ep_size=moe_ep_size
    )

    producer, _receiver = assert_post_transform_lifecycle_equivalent(case)

    assert_every_parameter_was_loaded(producer, factory_parameters)
    resolved_moe_ep_size = 2 // moe_tp_size
    assert PostTransformRuntimeConfig.from_model_config(
        producer.model_config, model=producer
    ) == _runtime_config(
        tp_size=2,
        moe_tp_size=moe_tp_size,
        moe_ep_size=resolved_moe_ep_size,
        attention_tp_size=2,
    )
    config = _hf_config(q_lora_rank=q_lora_rank)
    local_heads = config.num_attention_heads // 2
    nope, v_dim, kv_lora = config.qk_nope_head_dim, config.v_head_dim, config.kv_lora_rank
    local_experts = _NUM_EXPERTS // resolved_moe_ep_size
    local_intermediate_size = config.moe_intermediate_size // moe_tp_size
    layout = _layout_state(producer)
    # Attention TP splits the heads over the two ranks.
    assert layout["kv_b_proj_shape"] == (local_heads * (nope + v_dim), kv_lora)
    assert layout["v_b_proj_shape"] == (local_heads, v_dim, kv_lora)
    assert layout["v_b_proj_offsets_bytes"] == (local_heads * nope * kv_lora * 2,) * _NUM_LAYERS
    assert layout["router_weight_shape"] == (_NUM_EXPERTS, config.hidden_size)
    assert layout["w3_w1_weight_shape"] == (
        local_experts,
        2 * local_intermediate_size,
        config.hidden_size,
    )
    assert layout["w2_weight_shape"] == (local_experts, config.hidden_size, local_intermediate_size)
    expert_rank = rank if resolved_moe_ep_size > 1 else 0
    assert layout["initial_local_expert_ids"] == tuple(
        range(expert_rank * local_experts, (expert_rank + 1) * local_experts)
    )


@_requires_cutlass_bf16_moe
@_requires_sm90_routed_expert_forward
@pytest.mark.parametrize("q_lora_rank", [_LITE])
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
def test_deepseek_v3_staged_receiver_matches_routed_expert_output(
    monkeypatch: pytest.MonkeyPatch,
    q_lora_rank: int | None,
    tp_size: int,
    rank: int,
    moe_ep_size: int,
) -> None:
    """A staged receiver's routed experts compute exactly what its producer's compute.

    The other lifecycle cases compare tensors, state, guards, and the dense
    outputs on every SM80+ GPU. This case adds the routed-expert output, which
    this fixture validates only on SM90 and newer.
    """
    case, _factory_parameters = _lifecycle_case(
        monkeypatch,
        q_lora_rank,
        tp_size=tp_size,
        rank=rank,
        moe_ep_size=moe_ep_size,
        output_probes=(("routed-experts", _routed_expert_output),),
    )

    assert_post_transform_lifecycle_equivalent(case)


@_requires_cutlass_bf16_moe
@pytest.mark.parametrize("q_lora_rank", [_LITE, _Q_LORA])
def test_deepseek_v3_receiver_restores_v_b_proj_alias_before_transfer(
    monkeypatch: pytest.MonkeyPatch, q_lora_rank: int | None
) -> None:
    """A receiver that never loads still binds `v_b_proj` to the V rows of `kv_b_proj`.

    The loader binds that view; a staged receiver gets it from
    `setup_aliases()` instead, so storage offsets and alias groups match the
    donor's before the transfer and in the MX parameter catalog.
    """
    producer = _tiny_model(monkeypatch, q_lora_rank=q_lora_rank)
    receiver = _tiny_model(monkeypatch, q_lora_rank=q_lora_rank)
    _hf_checkpoint_loader(_hf_state_dict(q_lora_rank))(producer)
    ModelLoader._walk_full_post_load(producer)
    # A fresh model has a separate `v_b_proj` tensor.
    config = _hf_config(q_lora_rank=q_lora_rank)
    aliased_offset_bytes = (
        config.num_attention_heads * config.qk_nope_head_dim * config.kv_lora_rank * 2
    )
    assert all(
        _v_b_proj_offset_bytes(layer.self_attn) != aliased_offset_bytes
        for layer in receiver.model.layers
    )
    donor_catalog = mx_canonical_parameter_catalog(producer)

    ModelLoader._setup_aliases(receiver)
    bound = [layer.self_attn.v_b_proj for layer in receiver.model.layers]
    ModelLoader._setup_aliases(receiver)

    for layer, parameter in zip(receiver.model.layers, bound):
        attention = layer.self_attn
        assert attention.v_b_proj is parameter, "setup_aliases() must be idempotent"
        assert not attention._weights_transformed
        # The same storage as `kv_b_proj.weight`, past the K-nope rows.
        assert (
            attention.v_b_proj.untyped_storage().data_ptr()
            == attention.kv_b_proj.weight.untyped_storage().data_ptr()
        )
        assert _v_b_proj_offset_bytes(attention) == (
            attention.num_heads_tp * attention.qk_nope_head_dim * attention.kv_lora_rank * 2
        )
    receiver_catalog = mx_canonical_parameter_catalog(receiver)
    assert receiver_catalog.keys() == donor_catalog.keys()
    assert {name: entry[1:] for name, entry in receiver_catalog.items()} == {
        name: entry[1:] for name, entry in donor_catalog.items()
    }
    # Parameter-to-parameter byte offsets inside the catalog match the donor's.
    for name in receiver_catalog:
        if name.endswith("self_attn.v_b_proj"):
            kv_b_name = name.replace("v_b_proj", "kv_b_proj.weight")
            assert (
                receiver_catalog[name][0] - receiver_catalog[kv_b_name][0]
                == donor_catalog[name][0] - donor_catalog[kv_b_name][0]
            )


@_requires_cutlass_bf16_moe
def test_deepseek_v3_setup_aliases_leaves_a_loaded_producer_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """After the loader bound `v_b_proj`, `setup_aliases()` rebinds nothing."""
    producer = _tiny_model(monkeypatch)
    _hf_checkpoint_loader(_hf_state_dict(None))(producer)
    ModelLoader._walk_full_post_load(producer)
    before = [
        (
            layer.self_attn.v_b_proj,
            layer.self_attn.v_b_proj.data_ptr(),
            layer.self_attn._weights_transformed,
        )
        for layer in producer.model.layers
    ]

    ModelLoader._setup_aliases(producer)

    after = [
        (
            layer.self_attn.v_b_proj,
            layer.self_attn.v_b_proj.data_ptr(),
            layer.self_attn._weights_transformed,
        )
        for layer in producer.model.layers
    ]
    assert all(a[0] is b[0] and a[1:] == b[1:] for a, b in zip(before, after))


@_requires_cutlass_bf16_moe
def test_deepseek_v3_cutlass_pin_keeps_shared_experts_unfused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The qualified backend never fuses shared experts into routed slots.

    `setup_aliases()` can fuse them and drop `shared_experts`, but only for
    backends that report fused shared-expert slots; the CUTLASS pin reports none.
    """
    model = _tiny_model(monkeypatch)
    experts = model.model.layers[1].mlp.experts

    ModelLoader._setup_aliases(model)

    assert getattr(experts, "num_fused_shared_expert", 0) == 0
    assert model.model.layers[1].mlp.shared_experts is not None


@_requires_cutlass_bf16_moe
def test_staged_deepseek_v3_finalization_preserves_mx_tensor_catalog(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Routed-expert and MLA tensors stay registered, with unchanged storage, through finalization."""
    model = _tiny_model(monkeypatch)

    ModelLoader._setup_aliases(model)
    registered_catalog = mx_canonical_parameter_catalog(model)

    ModelLoader._setup_aliases(model)
    ModelLoader._mark_weights_transformed(model)
    ModelLoader._walk_cache_state(model)
    published_catalog = mx_canonical_parameter_catalog(model)

    assert published_catalog == registered_catalog
    prefix = "model.layers.1."
    assert {
        f"{prefix}self_attn.kv_b_proj.weight",
        f"{prefix}self_attn.k_b_proj_trans",
        f"{prefix}self_attn.v_b_proj",
        f"{prefix}mlp.gate.weight",
        f"{prefix}mlp.gate.e_score_correction_bias",
        f"{prefix}mlp.experts.backend.w3_w1_weight",
        f"{prefix}mlp.experts.backend.w2_weight",
        f"{prefix}mlp.shared_experts.gate_up_proj.weight",
    } <= registered_catalog.keys()


def _qualify_registry(**overrides: object) -> PostTransformQualificationDecision:
    """Qualify the registered DeepSeek-V3 identity directly, with `overrides` applied."""
    arguments = {
        "root_model_class": modeling_deepseekv3_mod.DeepseekV3ForCausalLM,
        "architecture": "DeepseekV3ForCausalLM",
        "model_type": "deepseek_v3",
        "speculative_mode": None,
        "protocol_version": ModelLoader._MX_STAGED_RECEIVER_TRANSFORM_PROTOCOL_VERSION,
        "transfer_scope": PostTransformTransferScope.TARGET_MODEL,
        "runtime_config": _runtime_config(),
    }
    arguments.update(overrides)
    return ModelLoader._post_transform_profile_registry().qualify(**arguments)


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
def test_deepseek_v3_profile_matches_its_exact_identity(overrides: dict[str, object]) -> None:
    decision = _qualify_registry(runtime_config=_runtime_config(**overrides))

    assert decision.qualified
    assert decision.profile is _profile()
    assert decision.transform_abi_id == DEEPSEEK_V3_MLA_CUTLASS_POST_TRANSFORM_LAYOUT_ABI_V1


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "overrides, reason",
    [
        pytest.param(
            {"root_model_class": modeling_deepseekv3_mod.GlmMoeDsaForCausalLM},
            PostTransformQualificationReason.ROOT_MODEL_CLASS_NOT_REGISTERED,
            id="glm-5-root",
        ),
        pytest.param(
            {"root_model_class": modeling_mistral_large3_mod.MistralLarge3ForCausalLM},
            PostTransformQualificationReason.ROOT_MODEL_CLASS_NOT_REGISTERED,
            id="mistral-large-3-root",
        ),
        pytest.param(
            {"root_model_class": _UnqualifiedDeepseekV3ForCausalLM},
            PostTransformQualificationReason.ROOT_MODEL_CLASS_NOT_REGISTERED,
            id="subclass-root",
        ),
        pytest.param(
            {"architecture": "DeepseekV32ForCausalLM"},
            PostTransformQualificationReason.ARCHITECTURE_NOT_REGISTERED,
            id="deepseek-v3-2-architecture",
        ),
        pytest.param(
            {"architecture": "GlmMoeDsaForCausalLM"},
            PostTransformQualificationReason.ARCHITECTURE_NOT_REGISTERED,
            id="glm-5-architecture",
        ),
        pytest.param(
            {"model_type": "kimi_k2"},
            PostTransformQualificationReason.MODEL_TYPE_NOT_REGISTERED,
            id="kimi-k2-model-type",
        ),
        pytest.param(
            {"model_type": "deepseek_v32"},
            PostTransformQualificationReason.MODEL_TYPE_NOT_REGISTERED,
            id="deepseek-v3-2-model-type",
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
def test_deepseek_v3_profile_rejects_other_identities(
    overrides: dict[str, object], reason: PostTransformQualificationReason
) -> None:
    """Only the exact root, architecture, and model type of DeepSeek-V3 qualify.

    Full DeepSeek-V3, R1 and V3.1 share this identity and are excluded by the
    runtime pins below; the other DeepSeek-family roots do not share it.
    """
    decision = _qualify_registry(**overrides)

    assert not decision.qualified
    assert decision.reason is reason


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "overrides, expected_dimensions",
    [
        pytest.param({"rope_type": "yarn"}, {"rope_type"}, id="yarn-rope"),
        pytest.param({"rope_fusion": False}, {"rope_fusion"}, id="unfused-rope"),
        pytest.param(
            {"quant_algorithm": "FP8_BLOCK_SCALES"}, {"quant_algorithm"}, id="fp8-block-scales"
        ),
        pytest.param({"quant_algorithm": "NVFP4"}, {"quant_algorithm"}, id="nvfp4"),
        pytest.param(
            {"kv_cache_quant_algorithm": "FP8"}, {"kv_cache_quant_algorithm"}, id="fp8-kv-cache"
        ),
        pytest.param({"moe_backend": "TRTLLM"}, {"moe_backend"}, id="trtllm-moe-backend"),
        pytest.param({"moe_backend": "DEEPGEMM"}, {"moe_backend"}, id="deepgemm-moe-backend"),
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
def test_deepseek_v3_profile_rejects_unqualified_runtime_variants(
    overrides: dict[str, object], expected_dimensions: set[str]
) -> None:
    """The profile rejects YaRN, FP8, other MoE backends, attention DP, CP, and wider TP.

    `test_bf16_profiles_reject_unqualified_runtime_variants` covers the
    variants that every BF16 profile rejects.
    """
    decision = _qualify_registry(runtime_config=_runtime_config(**overrides))

    assert not decision.qualified
    assert decision.reason is PostTransformQualificationReason.RUNTIME_CONFIG_NOT_SUPPORTED
    assert decision.unsupported_runtime_dimensions == frozenset(expected_dimensions)


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "feature", [PostTransformFeature.MOE_LOAD_BALANCER, PostTransformFeature.DWDP]
)
def test_deepseek_v3_profile_rejects_unsupported_moe_features(
    feature: PostTransformFeature,
) -> None:
    decision = _qualify_registry(enabled_features=frozenset({feature}))

    assert not decision.qualified
    assert decision.reason is PostTransformQualificationReason.FEATURE_NOT_SUPPORTED
    assert decision.unsupported_features == frozenset({feature})


@pytest.mark.cpu_only
def test_deepseek_v3_auto_backend_resolves_to_qualified_cutlass() -> None:
    """`AUTO` resolves to the pinned literal for BF16 DeepSeek-V3."""
    resolved = ModelConfig.resolve_moe_backend("AUTO", "DeepseekV3ForCausalLM", QuantConfig())

    assert resolved == "CUTLASS"


@_requires_cutlass_bf16_moe
def test_deepseek_v3_source_identity_separates_transform_abis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The same model under another transform ABI does not share weights."""
    model = _tiny_model(monkeypatch)
    artifact_identity = ArtifactIdentity(
        format_version=ARTIFACT_IDENTITY_FORMAT_VERSION,
        scheme="checkpoint_manifest_sha256",
        digest="0" * 64,
    )
    identities = [
        SourceIdentity.from_model_config(
            model.model_config,
            model,
            artifact_identity=artifact_identity,
            transform_abi_id=abi,
        )
        for abi in (
            DEEPSEEK_V3_MLA_CUTLASS_POST_TRANSFORM_LAYOUT_ABI_V1,
            QWEN3_MOE_CUTLASS_POST_TRANSFORM_LAYOUT_ABI_V1,
        )
    ]

    decision = check_weight_sharing_compatibility(*identities, IdentityCheckPolicy.WARN_FALLBACK)

    assert not decision.should_share
    assert decision.match_result.mismatched_fields == ["transform_abi_id"]
