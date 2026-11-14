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
"""MX post-transform qualification tests for the Qwen3.5 MoE vision-language root.

The root combines the dense Qwen3.5 vision-language profile (see
`test_model_loader_mx_qwen3_5_vl.py`, whose helpers this file reuses) with the
Qwen3 MoE CUTLASS profile: every decoder layer holds a sparse block with a
replicated router, CUTLASS routed experts, a fused shared expert, and a
replicated shared-expert gate.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import transformers
from test_model_loader_mx_qwen3_5_vl import (
    _derived_tensor_caches,
    _embedding_logits,
    _hf_checkpoint_loader,
    _linear_outputs,
    _qwen3_5_vl_layout_state,
    _qwen3_5_vl_runtime_config,
    _tiny_hf_config_dict,
    _tiny_qwen3_5_vl_model,
    _vision_patch_embed_output,
    _write_tiny_checkpoint_config,
)
from torch import nn
from utils.post_transform_qualification import (
    AllReduceStub,
    PostTransformQualificationCase,
    assert_every_parameter_was_loaded,
    assert_post_transform_lifecycle_equivalent,
    keep_factory_parameters,
    module_state_snapshot,
    mx_canonical_parameter_catalog,
)

import tensorrt_llm.mapping as mapping_mod
from tensorrt_llm._torch import distributed as distributed_mod
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models import modeling_qwen3_5 as modeling_qwen3_5_mod
from tensorrt_llm._torch.models import modeling_qwen3_moe as modeling_qwen3_moe_mod
from tensorrt_llm._torch.models import modeling_qwen3_next as modeling_qwen3_next_mod
from tensorrt_llm._torch.models import modeling_qwen3vl_moe as modeling_qwen3vl_moe_mod
from tensorrt_llm._torch.moe.fused_moe.configurable_moe import ConfigurableMoE
from tensorrt_llm._torch.moe.fused_moe.interface import MoEWeightLoadingMode
from tensorrt_llm._torch.pyexecutor.config_utils import load_pretrained_config
from tensorrt_llm._torch.pyexecutor.model_loader import ModelLoader
from tensorrt_llm._torch.weight_sharing import (
    ARTIFACT_IDENTITY_FORMAT_VERSION,
    QWEN3_5_DENSE_VL_POST_TRANSFORM_LAYOUT_ABI_V1,
    QWEN3_5_MOE_VL_CUTLASS_POST_TRANSFORM_LAYOUT_ABI_V1,
    QWEN3_MOE_CUTLASS_POST_TRANSFORM_LAYOUT_ABI_V1,
    ArtifactIdentity,
    IdentityCheckPolicy,
    PostTransformConfigIdentity,
    PostTransformFeature,
    PostTransformQualificationReason,
    PostTransformRuntimeConfig,
    PostTransformTransferScope,
    SourceIdentity,
    check_weight_sharing_compatibility,
)
from tensorrt_llm._utils import get_sm_version

_PROFILE_ID = "qwen3_5-moe-for-conditional-generation-bf16-cutlass-complete-v1"
# ModelLoader captures this identity before construction; the constructed root
# rewrites its own `pretrained_config` to the text config.
_CONFIG_IDENTITY = PostTransformConfigIdentity("Qwen3_5MoeForConditionalGeneration", "qwen3_5_moe")
_ARTIFACT_IDENTITY = ArtifactIdentity(
    format_version=ARTIFACT_IDENTITY_FORMAT_VERSION,
    scheme="checkpoint_manifest_sha256",
    digest="0" * 64,
)
_MAX_NUM_TOKENS = 16
_NUM_EXPERTS = 4
_HIDDEN_SIZE = 128
_MOE_INTERMEDIATE_SIZE = 256
_SHARED_INTERMEDIATE_SIZE = 256

_requires_cutlass_bf16_moe = pytest.mark.skipif(
    not torch.cuda.is_available() or get_sm_version() < 80,
    reason="the routed experts run through the SM80+ CUTLASS MoE leaf on CUDA",
)
# The routed-expert forward of this fixture is validated only on SM90 and newer,
# as for the Qwen3 MoE fixture; the tensor, state, and guard checks of the other
# cases still run on every SM80+ GPU.
_requires_sm90_routed_expert_forward = pytest.mark.skipif(
    not torch.cuda.is_available() or get_sm_version() < 90,
    reason="the routed-expert forward is validated only on SM90 and newer",
)

# The admitted TP2 partitions as `(MoE EP size, MoE TP size)`; `moe_ep_size=-1`
# leaves MoE TP equal to TP.
_TP2_PARTITIONS = (
    pytest.param(-1, 2, id="moe-tp2"),
    pytest.param(2, 1, id="moe-ep2"),
)


class _UnqualifiedQwen3_5MoeVLModel(modeling_qwen3_5_mod.Qwen3_5MoeVLModel):
    """Subclass that must not inherit the exact-root Qwen3.5 MoE profile."""


def _tiny_moe_hf_config_dict() -> dict:
    """Tiny Qwen3.5 MoE VLM config derived from the dense tiny config.

    The routed experts keep every per-rank GEMM dimension at least 128 in both
    the MoE-TP and MoE-EP partitions, which the CUTLASS leaf needs.
    """
    config = _tiny_hf_config_dict()
    config["architectures"] = ["Qwen3_5MoeForConditionalGeneration"]
    config["model_type"] = "qwen3_5_moe"
    text_config = config["text_config"]
    del text_config["intermediate_size"]
    text_config.update(
        {
            "hidden_size": _HIDDEN_SIZE,
            "model_type": "qwen3_5_moe_text",
            "moe_intermediate_size": _MOE_INTERMEDIATE_SIZE,
            "norm_topk_prob": True,
            "num_experts": _NUM_EXPERTS,
            "num_experts_per_tok": 2,
            "shared_expert_intermediate_size": _SHARED_INTERMEDIATE_SIZE,
        }
    )
    config["vision_config"]["model_type"] = "qwen3_5_moe"
    config["vision_config"]["out_hidden_size"] = _HIDDEN_SIZE
    return config


def _write_tiny_moe_checkpoint_config(path: Path) -> Path:
    """Write the tiny MoE `config.json` into `path` and return `path`."""
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text(json.dumps(_tiny_moe_hf_config_dict()), encoding="utf-8")
    return path


def _tiny_moe_vl_model(
    checkpoint_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    model_class: type[nn.Module] = modeling_qwen3_5_mod.Qwen3_5MoeVLModel,
    tp_size: int = 1,
    rank: int = 0,
    moe_ep_size: int = -1,
    disable_mm_encoder: bool = False,
) -> nn.Module:
    """Build a tiny MoE root through the production config loader on CUDA.

    `moe_ep_size=2` at `tp_size=2` selects the MoE-EP partition; the default
    leaves MoE TP equal to TP. TP2 builds one rank of the group with the
    collectives stubbed out, as the dense TP2 tests do.
    """
    if tp_size > 1:
        monkeypatch.setattr(mapping_mod, "mpi_disabled", lambda: False)
        monkeypatch.setattr(distributed_mod, "AllReduce", AllReduceStub)
        monkeypatch.setattr(modeling_qwen3_next_mod, "AllReduce", AllReduceStub)
    model_config = ModelConfig(
        pretrained_config=load_pretrained_config(str(checkpoint_dir)),
        mapping=mapping_mod.Mapping(
            world_size=tp_size, rank=rank, tp_size=tp_size, moe_ep_size=moe_ep_size
        ),
        max_num_tokens=_MAX_NUM_TOKENS,
        max_seq_len=_MAX_NUM_TOKENS,
        disable_mm_encoder=disable_mm_encoder,
    )
    with torch.device("cuda"):
        model = model_class(model_config)
    # Distinct small values, so a missed copy cannot pass by accident and the
    # routed-expert outputs stay well inside the BF16 range.
    with torch.no_grad():
        for index, parameter in enumerate(model.parameters()):
            values = torch.arange(
                parameter.numel(), dtype=torch.float32, device=parameter.device
            ).reshape(parameter.shape)
            parameter.copy_((((values + index) % 17) - 8).to(parameter.dtype) / 64)
    return model


def _tiny_moe_hf_state_dict() -> dict[str, torch.Tensor]:
    """Hugging Face-format checkpoint tensors for the tiny MoE config.

    Routed experts are stacked (`experts.gate_up_proj` and `experts.down_proj`),
    as in the real BF16 checkpoints. Every value is positive, so no loaded
    tensor can equal the factory's fill, which is centered on zero.
    """
    hf_config = transformers.Qwen3_5MoeConfig.from_dict(_tiny_moe_hf_config_dict())
    hf_model = transformers.Qwen3_5MoeForConditionalGeneration(hf_config).to(torch.bfloat16)
    state_dict = {}
    for index, (name, tensor) in enumerate(hf_model.state_dict().items()):
        values = torch.arange(tensor.numel(), dtype=torch.float32).reshape(tensor.shape)
        state_dict[name] = (((values + index) % 17 + 1) / 256).to(tensor.dtype)
    return state_dict


def _qualify(model: nn.Module):
    """Qualify `model` with the identity that `ModelLoader` captures before construction."""
    return ModelLoader._qualify_post_transform_profile(
        model,
        config_identity=_CONFIG_IDENTITY,
        speculative_mode=None,
        loads_draft_weights=False,
    )


def _moe_block(model: nn.Module, layer: int = 0) -> nn.Module:
    return model.llm.model.layers[layer].mlp


def _moe_vl_layout_state(model: nn.Module) -> dict[str, object]:
    """Dense vision-language layout facts plus the sparse block's realized layout."""
    state = _qwen3_5_vl_layout_state(model)
    block = _moe_block(model)
    experts = block.experts
    backend = experts.backend
    state.update(
        {
            "router_weight_shape": tuple(block.gate.weight.shape),
            "router_weight_dtype": block.gate.weight.dtype,
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
            "shared_gate_up_weight_shape": tuple(block.shared_expert.gate_up_proj.weight.shape),
            "shared_down_weight_shape": tuple(block.shared_expert.down_proj.weight.shape),
            "shared_expert_gate_weight_shape": tuple(block.shared_expert_gate.weight.shape),
            "can_defer_finalize": block.can_defer_finalize,
            "shared_expert_low_m_is_none": block._shared_expert_low_m is None,
        }
    )
    return state


def _module_state_without_load_inputs(model: nn.Module) -> dict[str, object]:
    """`module_state_snapshot` without the expert backends' `weight_loading_mode`.

    The weight mapper sets that attribute to the layout of the tensors it is
    about to hand to `load_weights()`, and only `load_weights()` reads it. A
    staged receiver never loads, so it keeps the construction default, which
    the lifecycle outputs show forward does not depend on. The attribute of the
    `ConfigurableMoE` wrapper is never changed by the mapper and stays compared.
    """
    return {
        key: value
        for key, value in module_state_snapshot(model).items()
        if not key.endswith(".experts.backend.weight_loading_mode")
    }


def _hidden_states(model: nn.Module) -> torch.Tensor:
    router_weight = _moe_block(model).gate.weight
    values = torch.arange(
        3 * _HIDDEN_SIZE, dtype=torch.float32, device=router_weight.device
    ).reshape(3, _HIDDEN_SIZE)
    return ((values % 17) - 8).to(router_weight.dtype) / 17


def _router_logits(model: nn.Module) -> torch.Tensor:
    return _moe_block(model).gate(_hidden_states(model))


def _shared_expert_output(model: nn.Module) -> torch.Tensor:
    """Run the first layer's shared expert and its gate on fixed hidden states."""
    block = _moe_block(model)
    hidden_states = _hidden_states(model)
    return torch.cat(
        [block.shared_expert(hidden_states), block.shared_expert_gate(hidden_states)], dim=-1
    )


def _routed_expert_output(model: nn.Module) -> torch.Tensor:
    """Run the first layer's routed experts on fixed hidden states.

    The synthetic router logits send the three tokens to experts {0, 1}, {2, 3}
    and {0, 3}, so every expert, and both ranks of the MoE-EP partition, does
    work. Top-2 routing keeps the CUTLASS finalize deterministic. Under MoE TP
    or EP the result is this rank's partial sum; the decoder layer reduces it
    later.
    """
    hidden_states = _hidden_states(model)
    router_logits = torch.tensor(
        [[4.0, 3.0, 0.0, 0.0], [0.0, 0.0, 4.0, 3.0], [3.0, 0.0, 0.0, 4.0]],
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    output = _moe_block(model).experts(hidden_states, router_logits, use_dp_padding=False)
    torch.cuda.synchronize()
    return output


def _lifecycle_case(
    checkpoint_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    state_dict: dict[str, torch.Tensor],
    *,
    tp_size: int = 1,
    rank: int = 0,
    moe_ep_size: int = -1,
    routed_expert_output: bool = False,
) -> tuple[PostTransformQualificationCase, dict[str, torch.Tensor]]:
    """Build the lifecycle case whose producer runs the real HF load of `state_dict`.

    Also returns the factory parameters that the load overwrites, for
    `assert_every_parameter_was_loaded`.
    """

    def factory(model_class: type[nn.Module] = modeling_qwen3_5_mod.Qwen3_5MoeVLModel) -> nn.Module:
        return _tiny_moe_vl_model(
            checkpoint_dir,
            monkeypatch,
            model_class=model_class,
            tp_size=tp_size,
            rank=rank,
            moe_ep_size=moe_ep_size,
        )

    output_probes = [
        ("linear-outputs", _linear_outputs),
        ("vision-patch-embed", _vision_patch_embed_output),
        ("derived-tensor-caches", _derived_tensor_caches),
        ("router-logits", _router_logits),
        ("shared-expert", _shared_expert_output),
    ]
    if tp_size == 1:
        output_probes.append(("embedding-logits", _embedding_logits))
    if routed_expert_output:
        output_probes.append(("routed-experts", _routed_expert_output))
    prepare_producer, factory_parameters = keep_factory_parameters(
        _hf_checkpoint_loader(state_dict)
    )
    case = PostTransformQualificationCase(
        profile_id=_PROFILE_ID,
        model_factory=factory,
        unqualified_model_factory=lambda: factory(_UnqualifiedQwen3_5MoeVLModel),
        qualify_model=_qualify,
        state_probes=(
            ("layout", _moe_vl_layout_state),
            ("module-state", _module_state_without_load_inputs),
        ),
        output_probes=tuple(output_probes),
        prepare_producer=prepare_producer,
    )
    return case, factory_parameters


def _runtime_config(model: nn.Module) -> PostTransformRuntimeConfig:
    """Realize the runtime config of a constructed root."""
    return PostTransformRuntimeConfig.from_model_config(model.model_config, model=model)


@_requires_cutlass_bf16_moe
def test_qwen3_5_moe_vl_profile_qualifies_full_staged_lifecycle(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A staged complete-model receiver matches a producer that ran the real load path.

    The producer loads a Hugging Face-format state dict with stacked experts
    through the production weight mapper and then runs every post-load hook; the
    receiver only gets the producer's parameters and the staged finalization
    walks. The routed-expert output has its own SM90+ case,
    `test_qwen3_5_moe_vl_staged_receiver_matches_routed_expert_output`.
    """
    checkpoint_dir = _write_tiny_moe_checkpoint_config(tmp_path / "ckpt")
    state_dict = _tiny_moe_hf_state_dict()

    case, factory_parameters = _lifecycle_case(checkpoint_dir, monkeypatch, state_dict)
    producer, receiver = assert_post_transform_lifecycle_equivalent(case)

    assert_every_parameter_was_loaded(producer, factory_parameters)

    # The producer's bytes come from the checkpoint through the weight mapper.
    for trtllm_name, hf_name in (
        ("llm.model.embed_tokens.weight", "model.language_model.embed_tokens.weight"),
        ("llm.lm_head.weight", "lm_head.weight"),
        ("mm_encoder.visual.patch_embed.proj.weight", "model.visual.patch_embed.proj.weight"),
        (
            "llm.model.layers.0.mlp.gate.weight",
            "model.language_model.layers.0.mlp.gate.weight",
        ),
        (
            "llm.model.layers.0.mlp.shared_expert_gate.weight",
            "model.language_model.layers.0.mlp.shared_expert_gate.weight",
        ),
    ):
        assert torch.equal(producer.get_parameter(trtllm_name).cpu(), state_dict[hf_name]), (
            trtllm_name
        )
    hf_prefix = "model.language_model.layers.0.mlp."
    shared = _moe_block(producer).shared_expert
    assert torch.equal(
        shared.gate_up_proj.weight.cpu(),
        torch.cat(
            [
                state_dict[hf_prefix + "shared_expert.gate_proj.weight"],
                state_dict[hf_prefix + "shared_expert.up_proj.weight"],
            ]
        ),
    )
    assert torch.equal(
        shared.down_proj.weight.cpu(), state_dict[hf_prefix + "shared_expert.down_proj.weight"]
    )
    producer_backend = _moe_block(producer).experts.backend
    gate_up = state_dict[hf_prefix + "experts.gate_up_proj"]
    down = state_dict[hf_prefix + "experts.down_proj"]
    for expert in range(_NUM_EXPERTS):
        gate, up = gate_up[expert].chunk(2, dim=0)
        # `w3_w1_weight` stacks the up projection before the gate projection.
        assert torch.equal(producer_backend.w3_w1_weight[expert].cpu(), torch.cat([up, gate]))
        assert torch.equal(producer_backend.w2_weight[expert].cpu(), down[expert])

    decision = _qualify(receiver)
    assert decision.transform_abi_id == QWEN3_5_MOE_VL_CUTLASS_POST_TRANSFORM_LAYOUT_ABI_V1
    assert (
        ModelLoader._post_transform_transfer_scope(receiver)
        is PostTransformTransferScope.COMPLETE_MODEL
    )
    assert _runtime_config(producer) == _qwen3_5_vl_runtime_config()
    # The receiver marks the backend behind the `_weights_removed` wrapper
    # without loading, which a later transform walk must not undo.
    assert _moe_block(receiver).experts.backend._weights_transformed is True
    # The mapper switches the backend to the per-expert layout it unfused the
    # stacked tensors into, right before it loads; the receiver never loads.
    assert _moe_block(producer).experts.backend.weight_loading_mode is MoEWeightLoadingMode.VANILLA
    assert (
        _moe_block(receiver).experts.backend.weight_loading_mode
        is MoEWeightLoadingMode.FUSED_GATE_UP_PROJ
    )
    # The vision tower is part of the transfer: its parameters are not owned
    # by the language model.
    language_model_parameters = {id(parameter) for parameter in producer.llm.parameters()}
    assert any(
        id(parameter) not in language_model_parameters
        for parameter in producer.mm_encoder.parameters()
    )
    layout = _moe_vl_layout_state(producer)
    assert layout["layer_types"] == (
        "Qwen3NextLinearDecoderLayer",
        "Qwen3NextFullAttentionDecoderLayer",
    )
    assert layout["next_layer_layernorm_aliases"] == (True, True)
    assert layout["share_gate_up_quantize"] == (False, False)
    assert layout["fuse_qk_norm_rope"] is True
    assert layout["rope_fusion"] is False
    assert layout["tied_lm_head"] is False
    assert layout["experts_type"] == ConfigurableMoE.__name__
    assert layout["backend_type"] == "TrtllmCutlassUnquantizedImpl"
    assert layout["backend_id"] == "trtllm.cutlass.grouped_gemm.none"
    assert layout["quant_method_type"] == "UnquantizedFusedMoEMethod"
    assert layout["quant_scales"] == ()
    assert layout["routing_method_type"] == "RenormalizeMoeRoutingMethod"
    assert layout["router_weight_shape"] == (_NUM_EXPERTS, _HIDDEN_SIZE)
    assert layout["router_weight_dtype"] == torch.bfloat16
    assert layout["w3_w1_weight_shape"] == (_NUM_EXPERTS, 2 * _MOE_INTERMEDIATE_SIZE, _HIDDEN_SIZE)
    assert layout["w2_weight_shape"] == (_NUM_EXPERTS, _HIDDEN_SIZE, _MOE_INTERMEDIATE_SIZE)
    assert layout["expert_size_per_partition"] == _NUM_EXPERTS
    assert layout["initial_local_expert_ids"] == tuple(range(_NUM_EXPERTS))
    assert layout["layer_load_balancer"] is None
    assert layout["shared_gate_up_weight_shape"] == (2 * _SHARED_INTERMEDIATE_SIZE, _HIDDEN_SIZE)
    assert layout["shared_down_weight_shape"] == (_HIDDEN_SIZE, _SHARED_INTERMEDIATE_SIZE)
    assert layout["shared_expert_gate_weight_shape"] == (1, _HIDDEN_SIZE)
    for model in (producer, receiver):
        workspace = model.mrope_rotary_cos_sin_workspace
        assert workspace.shape[-1] % _MAX_NUM_TOKENS == 0
        assert not workspace.any()


@_requires_cutlass_bf16_moe
@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("moe_ep_size, moe_tp_size", _TP2_PARTITIONS)
def test_qwen3_5_moe_vl_profile_qualifies_tp2_rank_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    rank: int,
    moe_ep_size: int,
    moe_tp_size: int,
) -> None:
    """Each simulated TP2 rank's staged receiver matches its producer that ran the real load."""
    checkpoint_dir = _write_tiny_moe_checkpoint_config(tmp_path / "ckpt")
    case, factory_parameters = _lifecycle_case(
        checkpoint_dir,
        monkeypatch,
        _tiny_moe_hf_state_dict(),
        tp_size=2,
        rank=rank,
        moe_ep_size=moe_ep_size,
    )

    producer, _receiver = assert_post_transform_lifecycle_equivalent(case)

    assert_every_parameter_was_loaded(producer, factory_parameters)
    resolved_moe_ep_size = 2 // moe_tp_size
    assert _runtime_config(producer) == _qwen3_5_vl_runtime_config(
        tp_size=2,
        moe_tp_size=moe_tp_size,
        moe_ep_size=resolved_moe_ep_size,
        attention_tp_size=2,
    )
    local_experts = _NUM_EXPERTS // resolved_moe_ep_size
    local_intermediate_size = _MOE_INTERMEDIATE_SIZE // moe_tp_size
    layout = _moe_vl_layout_state(producer)
    # The router and the shared-expert gate are replicated on every rank.
    assert layout["router_weight_shape"] == (_NUM_EXPERTS, _HIDDEN_SIZE)
    assert layout["shared_expert_gate_weight_shape"] == (1, _HIDDEN_SIZE)
    assert layout["w3_w1_weight_shape"] == (
        local_experts,
        2 * local_intermediate_size,
        _HIDDEN_SIZE,
    )
    assert layout["w2_weight_shape"] == (local_experts, _HIDDEN_SIZE, local_intermediate_size)
    # The shared expert is a TP-sharded GatedMLP: gate and up rows, and down columns, split by TP.
    assert layout["shared_gate_up_weight_shape"] == (_SHARED_INTERMEDIATE_SIZE, _HIDDEN_SIZE)
    assert layout["shared_down_weight_shape"] == (_HIDDEN_SIZE, _SHARED_INTERMEDIATE_SIZE // 2)
    expert_rank = rank if resolved_moe_ep_size > 1 else 0
    assert layout["initial_local_expert_ids"] == tuple(
        range(expert_rank * local_experts, (expert_rank + 1) * local_experts)
    )
    full_attention = producer.llm.model.layers[1].self_attn
    assert full_attention.qkv_proj.weight.shape[0] == (4 * 2 + 2 + 2) * 256 // 2


@_requires_cutlass_bf16_moe
@_requires_sm90_routed_expert_forward
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
def test_qwen3_5_moe_vl_staged_receiver_matches_routed_expert_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    tp_size: int,
    rank: int,
    moe_ep_size: int,
) -> None:
    """A staged receiver's routed experts compute exactly what its producer's compute.

    The receiver keeps the construction-time `weight_loading_mode`, which the
    producer's load switched, so equal outputs show that forward never reads it.
    """
    case, _factory_parameters = _lifecycle_case(
        _write_tiny_moe_checkpoint_config(tmp_path / "ckpt"),
        monkeypatch,
        _tiny_moe_hf_state_dict(),
        tp_size=tp_size,
        rank=rank,
        moe_ep_size=moe_ep_size,
        routed_expert_output=True,
    )

    assert_post_transform_lifecycle_equivalent(case)


@_requires_cutlass_bf16_moe
def test_staged_qwen3_5_moe_vl_finalization_preserves_mx_tensor_catalog(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Routed-expert, shared-expert and vision tensors stay in the MX catalog.

    `ConfigurableMoE` is `_weights_removed`, so loader walks skip it, but the
    expert weights live in its backend child and must be registered for
    transfer and keep their storage through staged finalization. MX registers
    tensors after the receiver's alias preparation and publishes them after
    finalization, so the catalog must not change in between and must match the
    producer's names, shapes, and dtypes.
    """
    checkpoint_dir = _write_tiny_moe_checkpoint_config(tmp_path / "ckpt")
    producer = _tiny_moe_vl_model(checkpoint_dir, monkeypatch)
    _hf_checkpoint_loader(_tiny_moe_hf_state_dict())(producer)
    ModelLoader._walk_full_post_load(producer)
    receiver = _tiny_moe_vl_model(checkpoint_dir, monkeypatch)

    ModelLoader._setup_aliases(receiver)
    registered_catalog = mx_canonical_parameter_catalog(receiver)
    ModelLoader._setup_aliases(receiver)
    ModelLoader._mark_weights_transformed(receiver)
    ModelLoader._walk_cache_state(receiver)
    published_catalog = mx_canonical_parameter_catalog(receiver)

    assert published_catalog == registered_catalog
    assert any(name.startswith("mm_encoder.") for name in registered_catalog)
    for layer_index in range(2):
        prefix = f"llm.model.layers.{layer_index}.mlp."
        assert {
            f"{prefix}gate.weight",
            f"{prefix}experts.backend.w3_w1_weight",
            f"{prefix}experts.backend.w2_weight",
            f"{prefix}shared_expert.gate_up_proj.weight",
            f"{prefix}shared_expert.down_proj.weight",
            f"{prefix}shared_expert_gate.weight",
        } <= registered_catalog.keys()
    producer_layout = {
        name: entry[1:] for name, entry in mx_canonical_parameter_catalog(producer).items()
    }
    assert {name: entry[1:] for name, entry in registered_catalog.items()} == producer_layout


@_requires_cutlass_bf16_moe
def test_qwen3_5_moe_vl_qualification_requires_preconstruction_identity(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The constructed root reports the inner text identity, which never qualifies.

    `ModelLoader` therefore qualifies with the identity it captured before
    construction.
    """
    model = _tiny_moe_vl_model(_write_tiny_moe_checkpoint_config(tmp_path / "ckpt"), monkeypatch)

    post_construction_identity = PostTransformConfigIdentity.from_model_config(model.model_config)
    decision = ModelLoader._qualify_post_transform_profile(
        model,
        config_identity=post_construction_identity,
        speculative_mode=None,
        loads_draft_weights=False,
    )

    assert post_construction_identity == PostTransformConfigIdentity(
        "Qwen3_5MoeForCausalLM", "qwen3_5_moe_text"
    )
    assert decision.reason is PostTransformQualificationReason.ARCHITECTURE_NOT_REGISTERED
    assert _qualify(model).qualified


@_requires_cutlass_bf16_moe
@pytest.mark.parametrize("encoder_mode", ["disable_mm_encoder", "mm_disaggregated"])
def test_qwen3_5_moe_vl_without_local_encoder_falls_back(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, encoder_mode: str
) -> None:
    """Without its local encoder the root carries only the language model and falls back."""
    if encoder_mode == "mm_disaggregated":
        monkeypatch.setenv("TLLM_MULTIMODAL_DISAGGREGATED", "1")
    model = _tiny_moe_vl_model(
        _write_tiny_moe_checkpoint_config(tmp_path / "ckpt"),
        monkeypatch,
        disable_mm_encoder=encoder_mode == "disable_mm_encoder",
    )

    assert model.mm_encoder is None
    assert (
        ModelLoader._post_transform_transfer_scope(model)
        is PostTransformTransferScope.LANGUAGE_MODEL
    )
    assert _qualify(model).reason is PostTransformQualificationReason.TRANSFER_SCOPE_NOT_REGISTERED


@_requires_cutlass_bf16_moe
def test_qwen3_5_moe_vl_source_identity_keeps_dense_and_moe_abis_apart(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A dense Qwen3.5 donor, a Qwen3 MoE donor and this root never pair.

    `ENFORCE` skips the config and shard fingerprints, so the layout ABI alone
    separates the profiles even when everything else about two sources agrees.
    """
    moe_model = _tiny_moe_vl_model(_write_tiny_moe_checkpoint_config(tmp_path / "moe"), monkeypatch)
    dense_model = _tiny_qwen3_5_vl_model(_write_tiny_checkpoint_config(tmp_path / "dense"))

    def identity(model: nn.Module, transform_abi_id: str) -> SourceIdentity:
        return SourceIdentity.from_model_config(
            model.model_config,
            model,
            artifact_identity=_ARTIFACT_IDENTITY,
            transform_abi_id=transform_abi_id,
        )

    moe_identity = identity(moe_model, QWEN3_5_MOE_VL_CUTLASS_POST_TRANSFORM_LAYOUT_ABI_V1)
    for other in (
        identity(moe_model, QWEN3_5_DENSE_VL_POST_TRANSFORM_LAYOUT_ABI_V1),
        identity(moe_model, QWEN3_MOE_CUTLASS_POST_TRANSFORM_LAYOUT_ABI_V1),
        identity(dense_model, QWEN3_5_DENSE_VL_POST_TRANSFORM_LAYOUT_ABI_V1),
    ):
        decision = check_weight_sharing_compatibility(
            moe_identity, other, IdentityCheckPolicy.ENFORCE
        )
        assert not decision.should_share
        assert "transform_abi_id" in decision.match_result.mismatched_fields
    assert check_weight_sharing_compatibility(
        moe_identity, moe_identity, IdentityCheckPolicy.ENFORCE
    ).should_share


def _qualify_registry(**overrides: object):
    """Qualify the registered Qwen3.5 MoE identity directly, with `overrides` applied."""
    arguments = {
        "root_model_class": modeling_qwen3_5_mod.Qwen3_5MoeVLModel,
        "architecture": "Qwen3_5MoeForConditionalGeneration",
        "model_type": "qwen3_5_moe",
        "speculative_mode": None,
        "protocol_version": ModelLoader._MX_STAGED_RECEIVER_TRANSFORM_PROTOCOL_VERSION,
        "transfer_scope": PostTransformTransferScope.COMPLETE_MODEL,
        "runtime_config": _qwen3_5_vl_runtime_config(),
    }
    arguments.update(overrides)
    return ModelLoader._post_transform_profile_registry().qualify(**arguments)


@pytest.mark.cpu_only
def test_qwen3_5_moe_vl_profile_matches_its_exact_identity() -> None:
    """The registered identity qualifies with the Qwen3.5 MoE vision-language ABI."""
    decision = _qualify_registry()

    assert decision.qualified
    assert decision.profile.profile_id == _PROFILE_ID
    assert decision.transform_abi_id == QWEN3_5_MOE_VL_CUTLASS_POST_TRANSFORM_LAYOUT_ABI_V1
    assert QWEN3_5_MOE_VL_CUTLASS_POST_TRANSFORM_LAYOUT_ABI_V1 not in (
        QWEN3_5_DENSE_VL_POST_TRANSFORM_LAYOUT_ABI_V1,
        QWEN3_MOE_CUTLASS_POST_TRANSFORM_LAYOUT_ABI_V1,
    )


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "runtime_overrides",
    [
        {},
        {"tp_size": 2, "moe_tp_size": 2, "attention_tp_size": 2},
        {"tp_size": 2, "moe_tp_size": 1, "moe_ep_size": 2, "attention_tp_size": 2},
    ],
    ids=["tp1", "moe-tp2", "moe-ep2"],
)
@pytest.mark.parametrize("sm_version", [None, 80, 90, 100, 120])
def test_qwen3_5_moe_vl_profile_qualifies_admitted_partitions_on_any_sm(
    runtime_overrides: dict[str, object], sm_version: int | None
) -> None:
    """TP 1, MoE TP 2 and MoE EP 2 qualify, and no SM version blocks the BF16 profile."""
    decision = _qualify_registry(
        runtime_config=_qwen3_5_vl_runtime_config(sm_version=sm_version, **runtime_overrides)
    )

    assert decision.qualified
    assert decision.profile.runtime_constraints.sm_versions is None


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        pytest.param(
            {"root_model_class": modeling_qwen3_5_mod.Qwen3_5ForCausalLM},
            PostTransformQualificationReason.ROOT_MODEL_CLASS_NOT_REGISTERED,
            id="dense-text-only-root",
        ),
        pytest.param(
            {"root_model_class": modeling_qwen3_5_mod.Qwen3_5MoeForCausalLM},
            PostTransformQualificationReason.ROOT_MODEL_CLASS_NOT_REGISTERED,
            id="moe-text-only-root",
        ),
        pytest.param(
            {
                "root_model_class": modeling_qwen3_5_mod.Qwen3_5MoeForCausalLM,
                "architecture": "Qwen3_5MoeForCausalLM",
                "model_type": "qwen3_5_moe_text",
                "transfer_scope": PostTransformTransferScope.TARGET_MODEL,
            },
            PostTransformQualificationReason.ROOT_MODEL_CLASS_NOT_REGISTERED,
            id="moe-text-only-root-and-identity",
        ),
        pytest.param(
            {"root_model_class": modeling_qwen3_moe_mod.Qwen3MoeForCausalLM},
            PostTransformQualificationReason.ARCHITECTURE_NOT_REGISTERED,
            id="qwen3-moe-root",
        ),
        pytest.param(
            {"root_model_class": modeling_qwen3vl_moe_mod.Qwen3MoeVLModel},
            PostTransformQualificationReason.ROOT_MODEL_CLASS_NOT_REGISTERED,
            id="qwen3-vl-moe-root",
        ),
        pytest.param(
            {"root_model_class": _UnqualifiedQwen3_5MoeVLModel},
            PostTransformQualificationReason.ROOT_MODEL_CLASS_NOT_REGISTERED,
            id="subclass-of-the-root",
        ),
        pytest.param(
            {"root_model_class": modeling_qwen3_5_mod.Qwen3_5VLModel},
            PostTransformQualificationReason.ARCHITECTURE_NOT_REGISTERED,
            id="dense-root-with-moe-identity",
        ),
        pytest.param(
            {"architecture": "Qwen3_5ForConditionalGeneration", "model_type": "qwen3_5"},
            PostTransformQualificationReason.ARCHITECTURE_NOT_REGISTERED,
            id="moe-root-with-dense-identity",
        ),
        pytest.param(
            {"architecture": "Qwen3_5MoeForCausalLM"},
            PostTransformQualificationReason.ARCHITECTURE_NOT_REGISTERED,
            id="text-only-architecture",
        ),
        pytest.param(
            {"model_type": "qwen3_5_moe_text"},
            PostTransformQualificationReason.MODEL_TYPE_NOT_REGISTERED,
            id="text-model-type",
        ),
        pytest.param(
            {"transfer_scope": PostTransformTransferScope.TARGET_MODEL},
            PostTransformQualificationReason.TRANSFER_SCOPE_NOT_REGISTERED,
            id="target-model-scope",
        ),
        pytest.param(
            {"transfer_scope": PostTransformTransferScope.LANGUAGE_MODEL},
            PostTransformQualificationReason.TRANSFER_SCOPE_NOT_REGISTERED,
            id="language-model-scope",
        ),
        pytest.param(
            {"speculative_mode": "mtp"},
            PostTransformQualificationReason.SPECULATIVE_MODE_NOT_REGISTERED,
            id="mtp",
        ),
    ],
)
def test_qwen3_5_moe_vl_profile_is_exact(
    overrides: dict[str, object], reason: PostTransformQualificationReason
) -> None:
    """Neighboring roots, identities, speculative modes, and scopes do not qualify.

    The `language-model-scope` case is what `disable_mm_encoder` and multimodal
    encoder/prefill disaggregation realize.
    """
    decision = _qualify_registry(**overrides)

    assert not decision.qualified
    assert decision.reason is reason


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    ("runtime_overrides", "dimensions"),
    [
        pytest.param({"moe_backend": "TRTLLM"}, {"moe_backend"}, id="trtllm-moe-backend"),
        pytest.param({"moe_backend": "CUTEDSL"}, {"moe_backend"}, id="cutedsl-moe-backend"),
        pytest.param({"moe_backend": "DEEPGEMM"}, {"moe_backend"}, id="deepgemm-moe-backend"),
        pytest.param({"moe_backend": "TRITON"}, {"moe_backend"}, id="triton-moe-backend"),
        pytest.param({"moe_backend": "MARLIN"}, {"moe_backend"}, id="marlin-moe-backend"),
        pytest.param({"moe_backend": "VANILLA"}, {"moe_backend"}, id="vanilla-moe-backend"),
        pytest.param({"moe_backend": None}, {"moe_backend"}, id="unresolved-moe-backend"),
        pytest.param(
            {"tp_size": 4, "moe_tp_size": 4, "attention_tp_size": 4},
            {"tp_size", "moe_tp_size", "attention_tp_size"},
            id="tp4-moe-tp4",
        ),
        pytest.param(
            {"tp_size": 4, "moe_tp_size": 1, "moe_ep_size": 4, "attention_tp_size": 4},
            {"tp_size", "moe_ep_size", "attention_tp_size"},
            id="tp4-moe-ep4",
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
        pytest.param({"dtype": "float16"}, {"dtype"}, id="float16"),
        pytest.param({"quant_algorithm": "FP8"}, {"quant_algorithm"}, id="fp8"),
        pytest.param(
            {"quant_algorithm": "FP8_BLOCK_SCALES"}, {"quant_algorithm"}, id="fp8-block-scales"
        ),
        pytest.param({"quant_algorithm": "NVFP4"}, {"quant_algorithm"}, id="nvfp4"),
        pytest.param({"quant_algorithm": "W4A16_GPTQ"}, {"quant_algorithm"}, id="gptq"),
        pytest.param(
            {"kv_cache_quant_algorithm": "FP8"}, {"kv_cache_quant_algorithm"}, id="fp8-kv"
        ),
        pytest.param({"layerwise_quantization": True}, {"layerwise_quantization"}, id="layerwise"),
        pytest.param({"lora_enabled": True}, {"lora_enabled"}, id="lora"),
        pytest.param({"sparse_attention_enabled": True}, {"sparse_attention_enabled"}, id="sparse"),
        pytest.param({"attention_backend": "FLASHINFER"}, {"attention_backend"}, id="flashinfer"),
        pytest.param({"pp_size": 2}, {"pp_size"}, id="pp2"),
        pytest.param({"cp_size": 2}, {"cp_size"}, id="cp2"),
        pytest.param({"attention_dp": True}, {"attention_dp"}, id="attention-dp"),
        pytest.param({"multi_node": True}, {"multi_node"}, id="multi-node"),
        pytest.param({"rope_type": "default"}, {"rope_type"}, id="rope-default"),
        pytest.param({"rope_type": "yarn"}, {"rope_type"}, id="rope-yarn"),
        pytest.param({"rope_fusion": True}, {"rope_fusion"}, id="rope-fusion"),
        pytest.param({"tied_word_embeddings": True}, {"tied_word_embeddings"}, id="tied"),
    ],
)
def test_qwen3_5_moe_vl_profile_rejects_unqualified_runtime_variants(
    runtime_overrides: dict[str, object], dimensions: set[str]
) -> None:
    """Each unqualified runtime variant reports exactly the dimensions that block it."""
    decision = _qualify_registry(runtime_config=_qwen3_5_vl_runtime_config(**runtime_overrides))

    assert not decision.qualified
    assert decision.reason is PostTransformQualificationReason.RUNTIME_CONFIG_NOT_SUPPORTED
    assert decision.unsupported_runtime_dimensions == frozenset(dimensions)


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "runtime_overrides",
    [
        pytest.param({}, id="tp1"),
        pytest.param({"tp_size": 2, "moe_tp_size": 2, "attention_tp_size": 2}, id="moe-tp2"),
        pytest.param(
            {"tp_size": 2, "moe_tp_size": 1, "moe_ep_size": 2, "attention_tp_size": 2},
            id="moe-ep2",
        ),
    ],
)
@pytest.mark.parametrize(
    "feature",
    [
        PostTransformFeature.MOE_LOAD_BALANCER,
        PostTransformFeature.DWDP,
        PostTransformFeature.SEPARATE_DRAFT_MODEL,
        PostTransformFeature.LOCALITY_DOMAIN_LOCALIZED_WEIGHTS,
    ],
)
def test_qwen3_5_moe_vl_profile_rejects_unsupported_features(
    runtime_overrides: dict[str, object], feature: PostTransformFeature
) -> None:
    """The MoE load balancer (EPLB), DWDP, a separate draft model and localized weights fall back."""
    runtime_config = _qwen3_5_vl_runtime_config(**runtime_overrides)
    assert _qualify_registry(runtime_config=runtime_config).qualified

    decision = _qualify_registry(
        runtime_config=runtime_config, enabled_features=frozenset({feature})
    )

    assert not decision.qualified
    assert decision.reason is PostTransformQualificationReason.FEATURE_NOT_SUPPORTED
    assert decision.unsupported_features == frozenset({feature})


@pytest.mark.cpu_only
def test_qwen3_5_moe_vl_config_realizes_the_pinned_text_config(tmp_path: Path) -> None:
    """The production config loader keeps the VLM identity and M-RoPE text config.

    The checkpoint's top-level identity is what `ModelLoader` qualifies with;
    the text config only carries the MoE and M-RoPE realization.
    """
    config = load_pretrained_config(str(_write_tiny_moe_checkpoint_config(tmp_path / "ckpt")))

    assert config.architectures == ["Qwen3_5MoeForConditionalGeneration"]
    assert config.model_type == "qwen3_5_moe"
    assert config.text_config.model_type == "qwen3_5_moe_text"
    assert config.text_config.num_experts == _NUM_EXPERTS
    realized = PostTransformRuntimeConfig.from_model_config(
        SimpleNamespace(pretrained_config=config.text_config)
    )
    assert realized.rope_type == "mrope"
