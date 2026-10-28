# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for MX-specific ModelLoader branches."""

import ast
import re
from collections.abc import Callable
from contextlib import contextmanager, nullcontext
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock

import pytest
import torch
import yaml
from torch import nn
from transformers import (
    LlamaConfig,
    MistralConfig,
    Phi3Config,
    Qwen2Config,
    Qwen3Config,
    Qwen3MoeConfig,
)
from utils.post_transform_qualification import (
    AllReduceStub,
    PostTransformQualificationCase,
    assert_every_parameter_was_loaded,
    assert_post_transform_lifecycle_equivalent,
    keep_factory_parameters,
    module_state_snapshot,
)
from utils.util import getSMVersion

import tensorrt_llm._torch.weight_sharing as weight_sharing_mod
import tensorrt_llm.mapping as mapping_mod
from tensorrt_llm._torch import distributed as distributed_mod
from tensorrt_llm._torch import model_config as model_config_mod
from tensorrt_llm._torch.attention import mla as mla_mod
from tensorrt_llm._torch.attention.mla import MLA
from tensorrt_llm._torch.locality_domain.policy import LocalityDomainPolicy
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models import modeling_llama as modeling_llama_mod
from tensorrt_llm._torch.models import modeling_mistral as modeling_mistral_mod
from tensorrt_llm._torch.models import modeling_phi3 as modeling_phi3_mod
from tensorrt_llm._torch.models import modeling_qwen as modeling_qwen_mod
from tensorrt_llm._torch.models import modeling_qwen3 as modeling_qwen3_mod
from tensorrt_llm._torch.models import modeling_qwen3_moe as modeling_qwen3_moe_mod
from tensorrt_llm._torch.models.checkpoints import memory_init as memory_init_mod
from tensorrt_llm._torch.models.checkpoints import model_hooks as model_hooks_mod
from tensorrt_llm._torch.models.checkpoints.gms import model_loading as gms_loading_mod
from tensorrt_llm._torch.models.checkpoints.hf.qwen3_moe_weight_mapper import Qwen3MoeHfWeightMapper
from tensorrt_llm._torch.models.checkpoints.mx import model_loading as mx_loading_mod
from tensorrt_llm._torch.models.checkpoints.mx.checkpoint_loader import MXCheckpointLoader
from tensorrt_llm._torch.models.modeling_multimodal_mixin import MultimodalModelMixin
from tensorrt_llm._torch.models.modeling_utils import MetaInitMode, get_registered_model_class
from tensorrt_llm._torch.modules import linear as linear_mod
from tensorrt_llm._torch.modules.linear import Linear, WeightMode
from tensorrt_llm._torch.moe.fused_moe.configurable_moe import ConfigurableMoE
from tensorrt_llm._torch.moe.fused_moe.cutlass import TrtllmCutlassUnquantizedImpl
from tensorrt_llm._torch.moe.fused_moe.impl_contract import MoEEnvironment
from tensorrt_llm._torch.moe.fused_moe.impl_environment import MoEDep, override_moe_environment
from tensorrt_llm._torch.moe.fused_moe.interface import MoEWeightLoadingMode
from tensorrt_llm._torch.moe.fused_moe.moe_resolution import resolve_moe_impl
from tensorrt_llm._torch.moe.fused_moe.quantization import UnquantizedFusedMoEMethod
from tensorrt_llm._torch.pyexecutor import model_loader as model_loader_mod
from tensorrt_llm._torch.pyexecutor.model_loader import ModelLoader
from tensorrt_llm._torch.weight_sharing import (
    ARTIFACT_IDENTITY_FORMAT_VERSION,
    LLAMA_POST_TRANSFORM_LAYOUT_ABI_V1,
    PHI3_DENSE_POST_TRANSFORM_LAYOUT_ABI_V1,
    QWEN3_DENSE_FP8_BLOCK128_FP32SCALE_POST_TRANSFORM_LAYOUT_ABI_V1,
    QWEN3_DENSE_POST_TRANSFORM_LAYOUT_ABI_V1,
    QWEN3_DENSE_TIED_POST_TRANSFORM_LAYOUT_ABI_V1,
    QWEN3_MOE_CUTLASS_POST_TRANSFORM_LAYOUT_ABI_V1,
    SOURCE_IDENTITY_FORMAT_VERSION,
    WEIGHT_MANIFEST_DIR_ENV,
    WEIGHT_MANIFEST_ROLE_ENV,
    ArtifactIdentity,
    IdentityCheckPolicy,
    PostTransformFeature,
    PostTransformProfile,
    PostTransformProfileRegistry,
    PostTransformQualificationDecision,
    PostTransformQualificationReason,
    PostTransformRuntimeConfig,
    PostTransformRuntimeConstraints,
    PostTransformTransferScope,
    SourceIdentity,
    check_weight_sharing_compatibility,
    load_weight_manifest,
)
from tensorrt_llm.llmapi.llm_args import LoadFormat
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo

_SOURCE_IDENTITY = model_loader_mod.SourceIdentity(
    format_version=SOURCE_IDENTITY_FORMAT_VERSION,
    artifact_identity=ArtifactIdentity(
        format_version=ARTIFACT_IDENTITY_FORMAT_VERSION,
        scheme="checkpoint_manifest_sha256",
        digest="0" * 64,
    ),
    model_fingerprint="model",
    quant_fingerprint="quant",
    backend_fingerprint="backend",
    parallel_fingerprint="parallel",
    rank=0,
    shard_fingerprint="shard",
)


class _LinearStub(nn.Module):
    def __init__(self):
        super().__init__()
        self._weights_transformed = False

    def post_load_weights(self):
        pass


class _AllReduceStub(nn.Module):
    def __init__(self, *_args: object, **_kwargs: object) -> None:
        super().__init__()

    def uses_nccl_symmetric_memory_window(self) -> bool:
        return False

    def forward(self, tensor: torch.Tensor, **_kwargs: object) -> torch.Tensor:
        return tensor


def _make_draft_model_config():
    pretrained_config = SimpleNamespace(
        architectures=["DraftArch"],
        num_attention_heads=1,
        num_key_value_heads=1,
        tie_word_embeddings=False,
        torch_dtype=torch.float16,
    )
    return ModelConfig(pretrained_config=pretrained_config)


class _DraftModel(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        self.model_config = model_config
        self.config = model_config.pretrained_config
        self.linear = _LinearStub()


class _TinyModel(nn.Module):
    def __init__(self, events):
        super().__init__()
        self.model_config = SimpleNamespace(
            pretrained_config=SimpleNamespace(architectures=["TinyForCausalLM"], model_type="tiny"),
            mapping=mapping_mod.Mapping(),
        )
        self._weights_transformed = False
        self.linear = _LinearStub()
        self.draft_config = _make_draft_model_config()
        self.draft_model = _DraftModel(self.draft_config)
        self._events = events

    def _apply(self, fn):
        # The test is about ModelLoader's MX branching, not CUDA allocation.
        return self

    def to(self, *args, **kwargs):
        return self

    def load_weights(self, weights, mapper):
        self._events.append("load_weights")

    def load_draft_weights(self, weights, mapper):
        self._events.append("load_draft_weights")

    def setup_aliases(self):
        self._events.append("setup_aliases")

    def cache_derived_state(self):
        self._events.append("cache_derived_state")

    def post_load_weights(self):
        self._events.append("post_load_weights")


class _ManifestTinyModel(_TinyModel):
    """`_TinyModel` plus real CPU tensors so a weight manifest has something to hash."""

    def __init__(self, events: list[str]) -> None:
        super().__init__(events)
        self.weight = nn.Parameter(torch.arange(8, dtype=torch.float32).reshape(2, 4))
        self.register_buffer("scale", torch.tensor([0.5, 0.25]))


@contextmanager
def _moe_context(config, mapping):
    yield None


class _UnqualifiedLlamaForCausalLM(modeling_llama_mod.LlamaForCausalLM):
    pass


class _UnqualifiedQwen2ForCausalLM(modeling_qwen_mod.Qwen2ForCausalLM):
    pass


class _UnqualifiedQwen3ForCausalLM(modeling_qwen3_mod.Qwen3ForCausalLM):
    pass


class _UnqualifiedMistralForCausalLM(modeling_mistral_mod.MistralForCausalLM):
    pass


class _UnqualifiedPhi3ForCausalLM(modeling_phi3_mod.Phi3ForCausalLM):
    pass


class _UnqualifiedQwen3MoeForCausalLM(modeling_qwen3_moe_mod.Qwen3MoeForCausalLM):
    pass


def _tiny_llama_model(
    monkeypatch: pytest.MonkeyPatch,
    *,
    model_class: type[nn.Module] = modeling_llama_mod.LlamaForCausalLM,
    tp_size: int = 1,
    rank: int = 0,
) -> nn.Module:
    monkeypatch.setattr(modeling_llama_mod, "get_sm_version", lambda: 90)
    llama_config = LlamaConfig(
        architectures=["LlamaForCausalLM"],
        attention_bias=False,
        hidden_act="silu",
        hidden_size=16,
        intermediate_size=32,
        max_position_embeddings=16,
        mlp_bias=False,
        num_attention_heads=2,
        num_hidden_layers=2,
        num_key_value_heads=2,
        rms_norm_eps=1e-5,
        tie_word_embeddings=False,
        torch_dtype=torch.bfloat16,
        vocab_size=32,
    )
    model = model_class(
        ModelConfig(
            pretrained_config=llama_config,
            mapping=mapping_mod.Mapping(
                world_size=tp_size,
                rank=rank,
                tp_size=tp_size,
            ),
            max_num_tokens=16,
            max_seq_len=16,
        )
    )
    with torch.no_grad():
        for index, parameter in enumerate(model.parameters()):
            values = torch.arange(
                parameter.numel(),
                dtype=torch.float32,
                device=parameter.device,
            ).reshape(parameter.shape)
            parameter.copy_(((values + index) % 17).to(parameter.dtype) / 17)
    return model


def _llama_alias_state(model):
    layers = model.model.layers
    return {
        "skip_norm": model.model.skip_norm,
        "layer0_next_norm": layers[0].next_layer_layernorm is layers[1].input_layernorm,
        "layer0_next_attn": layers[0].next_attn is layers[1].self_attn,
        "layer1_skip_input_norm": layers[1].skip_input_layernorm,
        "layer1_next_norm": layers[1].next_layer_layernorm is model.model.norm,
        "layer1_next_attn": layers[1].next_attn is None,
    }


def _mx_canonical_parameter_catalog(model: nn.Module):
    """Mirror MX's canonical TRT-LLM parameter view without importing MX."""
    catalog = {}
    seen_storages = set()
    runtime_alias_components = {"next_attn", "next_layer_layernorm"}
    for name, parameter in model.named_parameters(remove_duplicate=False):
        storage = (
            parameter.device.type,
            parameter.device.index,
            parameter.data_ptr(),
        )
        if runtime_alias_components.intersection(name.split(".")):
            continue
        if storage in seen_storages:
            continue
        seen_storages.add(storage)
        catalog[name] = (
            parameter.data_ptr(),
            tuple(parameter.shape),
            parameter.dtype,
        )
    return catalog


def _llama_input_embeddings(model: nn.Module) -> torch.Tensor:
    input_ids = torch.tensor(
        [0, 1, 2],
        dtype=torch.long,
        device=model.model.embed_tokens.weight.device,
    )
    return model.model.embed_tokens(input_ids)


def _llama_embedding_logits(model: nn.Module) -> torch.Tensor:
    return model.lm_head(_llama_input_embeddings(model))


def _tiny_qwen2_model(
    *,
    model_class: type[nn.Module] = modeling_qwen_mod.Qwen2ForCausalLM,
    tp_size: int = 1,
    rank: int = 0,
    kv_cache_compression_config: object | None = None,
) -> nn.Module:
    qwen2_config = Qwen2Config(
        architectures=["Qwen2ForCausalLM"],
        attention_bias=True,
        hidden_act="silu",
        hidden_size=16,
        intermediate_size=32,
        max_position_embeddings=16,
        mlp_bias=False,
        num_attention_heads=4,
        num_hidden_layers=2,
        num_key_value_heads=2,
        rms_norm_eps=1e-5,
        tie_word_embeddings=False,
        torch_dtype=torch.bfloat16,
        vocab_size=32,
    )
    model = model_class(
        ModelConfig(
            pretrained_config=qwen2_config,
            mapping=mapping_mod.Mapping(
                world_size=tp_size,
                rank=rank,
                tp_size=tp_size,
            ),
            max_num_tokens=16,
            max_seq_len=16,
            kv_cache_compression_config=kv_cache_compression_config,
        )
    )
    with torch.no_grad():
        for index, parameter in enumerate(model.parameters()):
            values = torch.arange(
                parameter.numel(),
                dtype=torch.float32,
                device=parameter.device,
            ).reshape(parameter.shape)
            parameter.copy_(((values + index) % 17).to(parameter.dtype) / 17)
    return model


def _tiny_qwen3_model(
    monkeypatch: pytest.MonkeyPatch,
    *,
    model_class: type[nn.Module] = modeling_qwen3_mod.Qwen3ForCausalLM,
    tp_size: int = 1,
    rank: int = 0,
    tie_word_embeddings: bool = False,
) -> nn.Module:
    monkeypatch.setattr(torch.cuda, "Stream", lambda *_args, **_kwargs: MagicMock())
    monkeypatch.setattr(torch.cuda, "Event", lambda *_args, **_kwargs: MagicMock())
    qwen3_config = _tiny_qwen3_config(tie_word_embeddings=tie_word_embeddings)
    model = model_class(
        ModelConfig(
            pretrained_config=qwen3_config,
            mapping=mapping_mod.Mapping(
                world_size=tp_size,
                rank=rank,
                tp_size=tp_size,
            ),
            max_num_tokens=16,
            max_seq_len=16,
        )
    )
    with torch.no_grad():
        for index, parameter in enumerate(model.parameters()):
            values = torch.arange(
                parameter.numel(),
                dtype=torch.float32,
                device=parameter.device,
            ).reshape(parameter.shape)
            parameter.copy_(((values + index) % 17).to(parameter.dtype) / 17)
    return model


def _tiny_qwen3_config(*, tie_word_embeddings: bool = False) -> Qwen3Config:
    return Qwen3Config(
        architectures=["Qwen3ForCausalLM"],
        attention_bias=False,
        head_dim=4,
        hidden_act="silu",
        hidden_size=16,
        intermediate_size=32,
        max_position_embeddings=16,
        mlp_bias=False,
        num_attention_heads=4,
        num_hidden_layers=2,
        num_key_value_heads=2,
        rms_norm_eps=1e-6,
        rope_scaling=None,
        tie_word_embeddings=tie_word_embeddings,
        torch_dtype=torch.bfloat16,
        vocab_size=32,
    )


_QWEN3_FP8_BLOCK_PROFILE_ID = "qwen3-for-causal-lm-fp8-block-sm90-target-v1"
_FP8_BLOCK_SIZE = 128


def _simulate_sm(monkeypatch: pytest.MonkeyPatch, sm_version: int) -> None:
    """Make FP8 block-scale transforms and MX qualification see `sm_version`.

    Executor unit tests run on CPU, Hopper, and Blackwell runners. FP8
    block-scale layouts depend on the GPU architecture, so tests that describe
    one architecture's layout pin it instead of inheriting the runner's.
    """
    monkeypatch.setattr(linear_mod, "get_sm_version", lambda: sm_version)
    monkeypatch.setattr(linear_mod, "is_sm_100f", lambda *_args, **_kwargs: 100 <= sm_version < 110)
    monkeypatch.setattr(linear_mod, "is_sm_120f", lambda *_args, **_kwargs: 120 <= sm_version < 130)
    monkeypatch.setattr(mx_loading_mod, "get_sm_version", lambda: sm_version)


def _fp8_block_quant_config() -> QuantConfig:
    # Mirrors what load_hf_quant_config() builds for a `quant_method: fp8`
    # checkpoint with 128x128 weight blocks and no modules_to_not_convert.
    return QuantConfig(
        quant_algo=QuantAlgo.FP8_BLOCK_SCALES,
        group_size=_FP8_BLOCK_SIZE,
        exclude_modules=["*kv_b_proj*", "*k_b_proj*", "*eh_proj"],
    )


def _tiny_qwen3_fp8_block_model(
    monkeypatch: pytest.MonkeyPatch,
    *,
    model_class: type[nn.Module] = modeling_qwen3_mod.Qwen3ForCausalLM,
    tp_size: int = 1,
    rank: int = 0,
) -> nn.Module:
    monkeypatch.setattr(torch.cuda, "Stream", lambda *_args, **_kwargs: MagicMock())
    monkeypatch.setattr(torch.cuda, "Event", lambda *_args, **_kwargs: MagicMock())
    # Every per-rank q, k, v, o, gate, up, and down span is a whole number of
    # 128-wide scale blocks at TP1 and TP2, as in the Qwen3 FP8 checkpoints.
    qwen3_config = Qwen3Config(
        architectures=["Qwen3ForCausalLM"],
        attention_bias=False,
        head_dim=128,
        hidden_act="silu",
        hidden_size=128,
        intermediate_size=256,
        max_position_embeddings=16,
        mlp_bias=False,
        num_attention_heads=4,
        num_hidden_layers=2,
        num_key_value_heads=2,
        # The quantization config the Qwen3 FP8 checkpoints declare.
        quantization_config={
            "activation_scheme": "dynamic",
            "fmt": "e4m3",
            "quant_method": "fp8",
            "weight_block_size": [_FP8_BLOCK_SIZE, _FP8_BLOCK_SIZE],
        },
        rms_norm_eps=1e-6,
        rope_scaling=None,
        tie_word_embeddings=False,
        torch_dtype=torch.bfloat16,
        vocab_size=32,
    )
    model = model_class(
        ModelConfig(
            pretrained_config=qwen3_config,
            quant_config=_fp8_block_quant_config(),
            mapping=mapping_mod.Mapping(
                world_size=tp_size,
                rank=rank,
                tp_size=tp_size,
            ),
            max_num_tokens=16,
            max_seq_len=16,
        )
    )
    with torch.no_grad():
        for index, (name, parameter) in enumerate(model.named_parameters()):
            if name.endswith("input_scale"):
                # Dynamic-activation checkpoints keep the unit activation scales.
                continue
            values = torch.arange(
                parameter.numel(),
                dtype=torch.float32,
                device=parameter.device,
            ).reshape(parameter.shape)
            # Positive values keep every FP32 block scale a valid scale.
            parameter.copy_((((values + index) % 17 + 1) / 17).to(parameter.dtype))
    return model


def _fp8_block_projections(model: nn.Module) -> dict[str, Linear]:
    layer = model.model.layers[0]
    return {
        "qkv": layer.self_attn.qkv_proj,
        "o": layer.self_attn.o_proj,
        "gate_up": layer.mlp.gate_up_proj,
        "down": layer.mlp.down_proj,
    }


def _qwen3_fp8_block_layout_state(model: nn.Module) -> dict[str, object]:
    state = _qwen3_layout_state(model)
    for name, projection in _fp8_block_projections(model).items():
        state[f"{name}_quant_method"] = type(projection.quant_method).__name__
        state[f"{name}_weight"] = (projection.weight.dtype, tuple(projection.weight.shape))
        state[f"{name}_weight_scale"] = (
            projection.weight_scale.dtype,
            tuple(projection.weight_scale.shape),
        )
        state[f"{name}_input_scales"] = (
            (projection.input_scale.dtype, tuple(projection.input_scale.shape)),
            (projection.inv_input_scale.dtype, tuple(projection.inv_input_scale.shape)),
        )
        state[f"{name}_relayout_parameters"] = tuple(sorted(projection.rebuild_tensor_metadata))
        # The default range stays unless the DeepGEMM relayout in
        # transform_weights() replaces it, a step a staged receiver skips.
        state[f"{name}_low_m_gemm"] = projection.fp8_low_m_gemm
    state["attention_has_quant_scale"] = model.model.layers[0].self_attn.has_quant_scale
    state["lm_head_weight_dtype"] = model.lm_head.weight.dtype
    state["lm_head_has_weight_scale"] = hasattr(model.lm_head, "weight_scale")
    # GatedMLP derives this flag in post_load_weights() only; the fused Qwen3
    # projections never share a quantized activation, so both paths agree.
    state["share_gate_up_quantization"] = tuple(
        layer.mlp._maybe_share_gate_up_quantize for layer in model.model.layers
    )
    return state


def _fp8_block_dequantized_weight(projection: Linear) -> torch.Tensor:
    weight = projection.weight.to(torch.float32)
    out_features, in_features = weight.shape
    scale = projection.weight_scale.repeat_interleave(_FP8_BLOCK_SIZE, dim=0)[:out_features]
    return weight * scale.repeat_interleave(_FP8_BLOCK_SIZE, dim=1)[:, :in_features]


def _fp8_block_projection_output(projection: Linear) -> torch.Tensor:
    # FP8 GEMMs need CUDA, so the probe dequantizes on the CPU in FP32.
    in_features = projection.weight.shape[1]
    values = torch.arange(
        3 * in_features,
        dtype=torch.float32,
        device=projection.weight.device,
    ).reshape(3, in_features)
    return ((values % 17) / 17) @ _fp8_block_dequantized_weight(projection).T


def _fp8_block_output_probes() -> tuple[tuple[str, Callable[[nn.Module], torch.Tensor]], ...]:
    return tuple(
        (
            f"{name}-dequantized-output",
            lambda model, name=name: _fp8_block_projection_output(
                _fp8_block_projections(model)[name]
            ),
        )
        for name in ("qkv", "o", "gate_up", "down")
    )


def _tiny_mistral_model(
    *,
    model_class: type[nn.Module] = modeling_mistral_mod.MistralForCausalLM,
    tp_size: int = 1,
    rank: int = 0,
    sliding_window: int | None = None,
    layer_types: tuple[str, ...] | None = None,
) -> nn.Module:
    # transformers defaults `MistralConfig.sliding_window` to a positive window,
    # so the full-attention fixture has to request `None` explicitly.
    mistral_config = MistralConfig(
        architectures=["MistralForCausalLM"],
        head_dim=4,
        hidden_act="silu",
        hidden_size=16,
        intermediate_size=32,
        max_position_embeddings=16,
        num_attention_heads=4,
        num_hidden_layers=2,
        num_key_value_heads=2,
        rms_norm_eps=1e-5,
        sliding_window=sliding_window,
        tie_word_embeddings=False,
        torch_dtype=torch.bfloat16,
        vocab_size=32,
    )
    if layer_types is not None:
        # Ministral-style checkpoints mark sliding and full-attention layers.
        mistral_config.layer_types = list(layer_types)
    model = model_class(
        ModelConfig(
            pretrained_config=mistral_config,
            mapping=mapping_mod.Mapping(
                world_size=tp_size,
                rank=rank,
                tp_size=tp_size,
            ),
            max_num_tokens=16,
            max_seq_len=16,
        )
    )
    with torch.no_grad():
        for index, parameter in enumerate(model.parameters()):
            values = torch.arange(
                parameter.numel(),
                dtype=torch.float32,
                device=parameter.device,
            ).reshape(parameter.shape)
            parameter.copy_(((values + index) % 17).to(parameter.dtype) / 17)
    return model


def _tiny_phi3_model(
    *,
    model_class: type[nn.Module] = modeling_phi3_mod.Phi3ForCausalLM,
    tp_size: int = 1,
    rank: int = 0,
    sliding_window: int | None = None,
    max_seq_len: int = 16,
    rope_scaling: dict[str, object] | None = None,
    tie_word_embeddings: bool = False,
) -> nn.Module:
    """Build a tiny deterministic `Phi3ForCausalLM` for the staged-lifecycle tests.

    Defaults mirror the Phi-4 checkpoint structure: GQA attention, bias-free
    fused QKV and gate-up projections, default RoPE, and untied embeddings.
    """
    phi3_config = Phi3Config(
        architectures=["Phi3ForCausalLM"],
        attention_bias=False,
        bos_token_id=1,
        eos_token_id=2,
        hidden_act="silu",
        hidden_size=16,
        intermediate_size=32,
        max_position_embeddings=16,
        num_attention_heads=4,
        num_hidden_layers=2,
        num_key_value_heads=2,
        original_max_position_embeddings=16,
        pad_token_id=0,
        rms_norm_eps=1e-5,
        rope_scaling=rope_scaling,
        sliding_window=sliding_window,
        tie_word_embeddings=tie_word_embeddings,
        torch_dtype=torch.bfloat16,
        vocab_size=32,
    )
    model = model_class(
        ModelConfig(
            pretrained_config=phi3_config,
            mapping=mapping_mod.Mapping(
                world_size=tp_size,
                rank=rank,
                tp_size=tp_size,
            ),
            max_num_tokens=16,
            max_seq_len=max_seq_len,
        )
    )
    with torch.no_grad():
        for index, parameter in enumerate(model.parameters()):
            values = torch.arange(
                parameter.numel(),
                dtype=torch.float32,
                device=parameter.device,
            ).reshape(parameter.shape)
            parameter.copy_(((values + index) % 17).to(parameter.dtype) / 17)
    return model


_QWEN3_MOE_PROFILE_ID = "qwen3-moe-for-causal-lm-bf16-cutlass-target-v1"
_QWEN3_MOE_NUM_EXPERTS = 4


def _qwen3_moe_hf_config(**overrides: object) -> Qwen3MoeConfig:
    values: dict[str, object] = {
        "architectures": ["Qwen3MoeForCausalLM"],
        "attention_bias": False,
        "decoder_sparse_step": 1,
        "head_dim": 32,
        "hidden_act": "silu",
        "hidden_size": 128,
        "intermediate_size": 256,
        "max_position_embeddings": 64,
        "mlp_only_layers": [],
        "moe_intermediate_size": 256,
        "norm_topk_prob": True,
        "num_attention_heads": 4,
        "num_experts": _QWEN3_MOE_NUM_EXPERTS,
        "num_experts_per_tok": 2,
        "num_hidden_layers": 2,
        "num_key_value_heads": 2,
        "rms_norm_eps": 1e-6,
        "rope_scaling": None,
        "tie_word_embeddings": False,
        "torch_dtype": torch.bfloat16,
        "vocab_size": 64,
    }
    values.update(overrides)
    return Qwen3MoeConfig(**values)


# `_tiny_qwen3_moe_model` builds its routed experts on CUDA through the CUTLASS
# leaf for unquantized weights, which supports SM80 and newer.
_requires_cutlass_bf16_moe = pytest.mark.skipif(
    not torch.cuda.is_available() or getSMVersion() < 80,
    reason="the Qwen3 MoE fixture needs CUDA and the SM80+ CUTLASS MoE leaf",
)


def _tiny_qwen3_moe_model(
    monkeypatch: pytest.MonkeyPatch,
    *,
    model_class: type[nn.Module] = modeling_qwen3_moe_mod.Qwen3MoeForCausalLM,
    tp_size: int = 1,
    rank: int = 0,
    moe_ep_size: int = -1,
) -> nn.Module:
    """Build a two-layer Qwen3 MoE model on CUDA with deterministic weights.

    The routed experts run through `ConfigurableMoE` over the CUTLASS leaf for
    unquantized weights, whose kernels need CUDA tensors. `moe_ep_size=2` at
    `tp_size=2` selects the MoE-EP partition; the default leaves MoE TP equal
    to TP. TP=2 builds one rank of the group with collectives stubbed out, as
    the dense TP=2 tests do. Every per-rank expert GEMM dimension stays at
    least 128 in both partitions.
    """
    if tp_size > 1:
        monkeypatch.setattr(mapping_mod, "mpi_disabled", lambda: False)
        monkeypatch.setattr(distributed_mod, "AllReduce", AllReduceStub)
        monkeypatch.setattr(modeling_qwen3_moe_mod, "AllReduce", AllReduceStub)
        monkeypatch.setattr(modeling_qwen3_moe_mod, "MoEAllReduce", AllReduceStub)
        monkeypatch.setattr(modeling_qwen3_moe_mod, "can_access_peer", lambda _mapping: False)
    with torch.device("cuda"):
        model = model_class(
            ModelConfig(
                pretrained_config=_qwen3_moe_hf_config(),
                mapping=mapping_mod.Mapping(
                    world_size=tp_size,
                    rank=rank,
                    tp_size=tp_size,
                    moe_ep_size=moe_ep_size,
                ),
                max_num_tokens=16,
                max_seq_len=16,
            )
        )
    with torch.no_grad():
        for index, parameter in enumerate(model.parameters()):
            values = torch.arange(
                parameter.numel(),
                dtype=torch.float32,
                device=parameter.device,
            ).reshape(parameter.shape)
            # Centered and scaled so the routed-expert outputs stay well inside
            # the BF16 range.
            parameter.copy_((((values + index) % 17) - 8).to(parameter.dtype) / 64)
    return model


def _qwen3_moe_layout_state(model: nn.Module) -> dict[str, object]:
    layers = model.model.layers
    layer = layers[0]
    experts = layer.mlp.experts
    backend = experts.backend
    return {
        "attention_type": type(layer.self_attn).__name__,
        "qkv_weight_mode": layer.self_attn.qkv_proj.weights_loading_config.weight_mode,
        "qkv_weight_shape": tuple(layer.self_attn.qkv_proj.weight.shape),
        "q_norm_weight_shape": tuple(layer.self_attn.q_norm.weight.shape),
        "k_norm_weight_shape": tuple(layer.self_attn.k_norm.weight.shape),
        "fuse_qk_norm_rope": layer.self_attn.fuse_qk_norm_rope,
        "rope_fusion": layer.self_attn.rope_fusion,
        "router_weight_shape": tuple(layer.mlp.gate.weight.shape),
        "router_weight_dtype": layer.mlp.gate.weight.dtype,
        "experts_type": type(experts).__name__,
        "backend_type": type(backend).__name__,
        "backend_id": type(backend).descriptor.identity.canonical(),
        "quant_method_type": type(backend.quant_method).__name__,
        "quant_scales": backend.quant_scales,
        "weight_loading_mode": backend.weight_loading_mode,
        "routing_method_type": type(experts.routing_method).__name__,
        "routing_scales_dtype": getattr(experts.routing_method, "output_dtype", None),
        "w3_w1_weight_shape": tuple(backend.w3_w1_weight.shape),
        "w2_weight_shape": tuple(backend.w2_weight.shape),
        "expert_size_per_partition": backend.expert_size_per_partition,
        "initial_local_expert_ids": tuple(backend.initial_local_expert_ids),
        "layer_load_balancer": backend.layer_load_balancer,
        "layer0_next_norm": layers[0].next_layer_layernorm is layers[1].input_layernorm,
        "layer1_next_norm": layers[1].next_layer_layernorm is model.model.norm,
        "tied_lm_head": model.lm_head.weight is model.model.embed_tokens.weight,
    }


def _qwen3_moe_hidden_states(model: nn.Module) -> torch.Tensor:
    router_weight = model.model.layers[0].mlp.gate.weight
    values = torch.arange(
        3 * model.config.hidden_size,
        dtype=torch.float32,
        device=router_weight.device,
    ).reshape(3, model.config.hidden_size)
    return ((values % 17) - 8).to(router_weight.dtype) / 17


def _qwen3_moe_router_logits(model: nn.Module) -> torch.Tensor:
    return model.model.layers[0].mlp.gate(_qwen3_moe_hidden_states(model))


def _qwen3_moe_routed_expert_output(model: nn.Module) -> torch.Tensor:
    """Run the first layer's routed experts on fixed hidden states.

    The synthetic router logits send the three tokens to experts {0, 1}, {2, 3}
    and {0, 3}, so every expert, and both ranks of the MoE-EP partition, does
    work. Top-2 routing keeps the CUTLASS finalize deterministic. Under MoE TP or
    EP the result is this rank's partial sum; the decoder layer reduces it later.
    """
    hidden_states = _qwen3_moe_hidden_states(model)
    router_logits = torch.tensor(
        [[4.0, 3.0, 0.0, 0.0], [0.0, 0.0, 4.0, 3.0], [3.0, 0.0, 0.0, 4.0]],
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    output = model.model.layers[0].mlp.experts(hidden_states, router_logits, use_dp_padding=False)
    torch.cuda.synchronize()
    return output


# This fixture's routed-expert forward has not been validated below SM90, so
# the lifecycle cases that compare its output skip there with this reason; the
# tensor, state, and guard checks of the other cases still run on every GPU.
_requires_sm90_routed_expert_forward = pytest.mark.skipif(
    not torch.cuda.is_available() or getSMVersion() < 90,
    reason="the Qwen3 MoE fixture's routed-expert forward is validated only on SM90 and newer",
)


def _qwen3_moe_hf_state_dict() -> dict[str, torch.Tensor]:
    """Return the tiny config's checkpoint tensors in the Hugging Face layout.

    Routed experts are stored per expert, as in a checkpoint on disk. Every
    value is positive, so no loaded tensor can equal the factory's fill, which
    is centered on zero.
    """
    config = _qwen3_moe_hf_config()
    hidden, head_dim = config.hidden_size, config.head_dim
    q_rows = config.num_attention_heads * head_dim
    kv_rows = config.num_key_value_heads * head_dim
    inter = config.moe_intermediate_size
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
            prefix + "self_attn.q_proj.weight": (q_rows, hidden),
            prefix + "self_attn.k_proj.weight": (kv_rows, hidden),
            prefix + "self_attn.v_proj.weight": (kv_rows, hidden),
            prefix + "self_attn.o_proj.weight": (hidden, q_rows),
            prefix + "self_attn.q_norm.weight": (head_dim,),
            prefix + "self_attn.k_norm.weight": (head_dim,),
            prefix + "mlp.gate.weight": (config.num_experts, hidden),
        }
        for expert in range(config.num_experts):
            shapes |= {
                prefix + f"mlp.experts.{expert}.gate_proj.weight": (inter, hidden),
                prefix + f"mlp.experts.{expert}.up_proj.weight": (inter, hidden),
                prefix + f"mlp.experts.{expert}.down_proj.weight": (hidden, inter),
            }
    state_dict = {}
    for index, (name, shape) in enumerate(shapes.items()):
        values = torch.arange(torch.Size(shape).numel(), dtype=torch.float32).reshape(shape)
        state_dict[name] = (((values + index) % 17 + 1) / 256).to(torch.bfloat16)
    return state_dict


def _qwen3_moe_hf_checkpoint_loader(
    state_dict: dict[str, torch.Tensor],
) -> Callable[[nn.Module], None]:
    """Return a `prepare_producer` hook that loads `state_dict` like a real HF load."""

    def _load(model: nn.Module) -> None:
        mapper = Qwen3MoeHfWeightMapper()
        mapper.init_model_and_config(model, model.model_config)
        model.load_weights(dict(state_dict), mapper)

    return _load


def _qwen3_moe_runtime_config(**overrides: object) -> PostTransformRuntimeConfig:
    values: dict[str, object] = {"rope_fusion": False}
    values.update(overrides)
    return _bf16_dense_runtime_config(**values)


def _bf16_dense_runtime_config(**overrides: object) -> PostTransformRuntimeConfig:
    """Return the runtime config a qualified BF16 dense tiny model realizes at TP1.

    Keyword arguments override individual fields, for example the TP sizes of a
    TP2 rank or a dimension that a negative test expects the registry to reject.
    """
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
        # Llama, Qwen, and Phi3 attention modules expose no window state, so their
        # real tiny models realize `None`; Mistral tests override this to `"none"`.
        "sliding_window": None,
    }
    values.update(overrides)
    return PostTransformRuntimeConfig(**values)


def _fp8_block_runtime_config(**overrides: object) -> PostTransformRuntimeConfig:
    values: dict[str, object] = {
        "quant_algorithm": QuantAlgo.FP8_BLOCK_SCALES.value,
        "rope_fusion": False,
        "sm_version": 90,
        "weight_block_size": (_FP8_BLOCK_SIZE, _FP8_BLOCK_SIZE),
    }
    values.update(overrides)
    return _bf16_dense_runtime_config(**values)


def _qwen2_layout_state(model: nn.Module) -> dict[str, object]:
    layer = model.model.layers[0]
    return {
        "attention_type": type(layer.self_attn).__name__,
        "qkv_weight_mode": layer.self_attn.qkv_proj.weights_loading_config.weight_mode,
        "qkv_weight_shape": tuple(layer.self_attn.qkv_proj.weight.shape),
        "gate_up_weight_mode": layer.mlp.gate_up_proj.weights_loading_config.weight_mode,
        "gate_up_weight_shape": tuple(layer.mlp.gate_up_proj.weight.shape),
        "qkv_bias": layer.self_attn.qkv_proj.bias is not None,
        "rope_fusion": layer.self_attn.rope_fusion,
        "rotary_embedding": layer.self_attn.rotary_emb,
        "tied_lm_head": model.lm_head.weight is model.model.embed_tokens.weight,
    }


def _qwen3_layout_state(model: nn.Module) -> dict[str, object]:
    layer = model.model.layers[0]
    return {
        "attention_type": type(layer.self_attn).__name__,
        "qkv_weight_mode": layer.self_attn.qkv_proj.weights_loading_config.weight_mode,
        "qkv_weight_shape": tuple(layer.self_attn.qkv_proj.weight.shape),
        "gate_up_weight_mode": layer.mlp.gate_up_proj.weights_loading_config.weight_mode,
        "gate_up_weight_shape": tuple(layer.mlp.gate_up_proj.weight.shape),
        "qkv_bias": layer.self_attn.qkv_proj.bias is not None,
        "q_norm_weight_shape": tuple(layer.self_attn.q_norm.weight.shape),
        "k_norm_weight_shape": tuple(layer.self_attn.k_norm.weight.shape),
        "fuse_qk_norm_rope": layer.self_attn.fuse_qk_norm_rope,
        "rope_fusion": layer.self_attn.rope_fusion,
        "rotary_embedding_present": layer.self_attn.rotary_emb is not None,
        "tied_lm_head": model.lm_head.weight is model.model.embed_tokens.weight,
    }


def _mistral_layout_state(model: nn.Module) -> dict[str, object]:
    layer = model.model.layers[0]
    return {
        "attention_type": type(layer.self_attn).__name__,
        "qkv_weight_mode": layer.self_attn.qkv_proj.weights_loading_config.weight_mode,
        "qkv_weight_shape": tuple(layer.self_attn.qkv_proj.weight.shape),
        "gate_up_weight_mode": layer.mlp.gate_up_proj.weights_loading_config.weight_mode,
        "gate_up_weight_shape": tuple(layer.mlp.gate_up_proj.weight.shape),
        "qkv_bias": layer.self_attn.qkv_proj.bias is not None,
        "rope_fusion": layer.self_attn.rope_fusion,
        "rotary_embedding_present": layer.self_attn.rotary_emb is not None,
        "attention_window_size": layer.self_attn.attention_window_size,
        "tied_lm_head": model.lm_head.weight is model.model.embed_tokens.weight,
    }


def _phi3_layout_state(model: nn.Module) -> dict[str, object]:
    """Return the Phi3 layout and derived state that a staged receiver must reproduce."""
    layer = model.model.layers[0]
    return {
        "attention_type": type(layer.self_attn).__name__,
        "qkv_weight_mode": layer.self_attn.qkv_proj.weights_loading_config.weight_mode,
        "qkv_weight_shape": tuple(layer.self_attn.qkv_proj.weight.shape),
        "gate_up_weight_mode": layer.mlp.gate_up_proj.weights_loading_config.weight_mode,
        "gate_up_weight_shape": tuple(layer.mlp.gate_up_proj.weight.shape),
        "qkv_bias": layer.self_attn.qkv_proj.bias is not None,
        "rope_fusion": layer.self_attn.rope_fusion,
        "rotary_embedding_present": layer.self_attn.rotary_emb is not None,
        # `GatedMLP.post_load_weights()` derives this flag on the full path only;
        # the staged receiver must keep the same value on every layer.
        "share_gate_up_quantize": tuple(
            decoder_layer.mlp._maybe_share_gate_up_quantize for decoder_layer in model.model.layers
        ),
        "tied_lm_head": model.lm_head.weight is model.model.embed_tokens.weight,
    }


def _dense_input_embeddings(model: nn.Module) -> torch.Tensor:
    input_ids = torch.tensor(
        [0, 1, 2],
        dtype=torch.long,
        device=model.model.embed_tokens.weight.device,
    )
    return model.model.embed_tokens(input_ids)


def _dense_embedding_logits(model: nn.Module) -> torch.Tensor:
    return model.lm_head(_dense_input_embeddings(model))


def _dense_hidden_states(model: nn.Module) -> torch.Tensor:
    qkv_weight = model.model.layers[0].self_attn.qkv_proj.weight
    values = torch.arange(
        3 * model.config.hidden_size,
        dtype=torch.float32,
        device=qkv_weight.device,
    ).reshape(3, model.config.hidden_size)
    return (values % 17).to(qkv_weight.dtype) / 17


def _dense_fused_qkv_output(model: nn.Module) -> torch.Tensor:
    return model.model.layers[0].self_attn.qkv_proj(_dense_hidden_states(model))


def _dense_fused_gate_up_output(model: nn.Module) -> torch.Tensor:
    return model.model.layers[0].mlp.gate_up_proj(_dense_hidden_states(model))


def _tiny_profile_registry(*, speculative_mode: str | None = None) -> PostTransformProfileRegistry:
    return PostTransformProfileRegistry(
        profiles=(
            PostTransformProfile(
                profile_id="tiny-for-causal-lm-target-v1",
                root_model_class=_TinyModel,
                architecture="TinyForCausalLM",
                model_type="tiny",
                speculative_mode=speculative_mode,
                protocol_version=mx_loading_mod.MX_STAGED_RECEIVER_TRANSFORM_PROTOCOL_VERSION,
                transform_abi_id=LLAMA_POST_TRANSFORM_LAYOUT_ABI_V1,
                transfer_scope=PostTransformTransferScope.TARGET_MODEL,
                runtime_constraints=PostTransformRuntimeConstraints(),
            ),
        )
    )


def _make_loader(monkeypatch, *, events, spec_config=None):
    llm_args = SimpleNamespace(load_format=LoadFormat.AUTO, lm_head_output_dtype="auto")
    loader = ModelLoader(
        llm_args=llm_args,
        mapping=MagicMock(name="mapping"),
        spec_config=spec_config,
        sparse_attention_config=None,
        max_num_tokens=128,
        max_seq_len=128,
    )
    # Record each weight copy; tests read the calls from `model_hooks_mod.call_load_weights`.
    monkeypatch.setattr(
        model_hooks_mod,
        "call_load_weights",
        MagicMock(side_effect=lambda fn, weights, mapper, **kwargs: fn(weights, mapper)),
    )
    loader._load_and_validate_config = MagicMock(
        return_value=SimpleNamespace(
            name="config",
            mapping=SimpleNamespace(),
            pretrained_config=SimpleNamespace(
                architectures=["TinyForCausalLM"],
                model_type="tiny",
            ),
        )
    )

    monkeypatch.setattr(model_loader_mod, "timing_metric", lambda *_args, **_kwargs: nullcontext())
    monkeypatch.setattr(model_loader_mod, "maybe_create_moe_load_balancer", _moe_context)
    monkeypatch.setattr(memory_init_mod, "MetaInitMode", lambda: nullcontext())

    # These tests stub ModelConfig, while SourceIdentity has dedicated
    # coverage. Keep this file focused on ModelLoader MX branch behavior.

    def _build_artifact_identity(_cls, checkpoint_dir):
        assert checkpoint_dir == "/ckpt"
        return _SOURCE_IDENTITY.artifact_identity

    monkeypatch.setattr(
        model_loader_mod.ArtifactIdentity,
        "from_checkpoint",
        classmethod(_build_artifact_identity),
    )

    def _build_source_identity(_cls, *_args, **kwargs):
        assert kwargs["artifact_identity"] is _SOURCE_IDENTITY.artifact_identity
        return replace(
            _SOURCE_IDENTITY,
            transform_abi_id=kwargs["transform_abi_id"],
        )

    monkeypatch.setattr(
        model_loader_mod.SourceIdentity,
        "from_model_config",
        classmethod(_build_source_identity),
    )
    monkeypatch.setattr(
        memory_init_mod.AutoModelForCausalLM,
        "from_config",
        MagicMock(return_value=_TinyModel(events)),
    )
    monkeypatch.setattr(memory_init_mod, "get_rank_model_storage", lambda _model: 0)
    return loader


def _mock_mx_checkpoint_loader() -> MagicMock:
    """Return a mocked MX checkpoint loader; `ModelLoader.load()` calls its `load_weights`."""
    checkpoint_loader = MagicMock(spec=MXCheckpointLoader, name="checkpoint_loader")
    checkpoint_loader.checkpoint_format = "MX"
    return checkpoint_loader


@pytest.mark.cpu_only
def test_construct_checkpoint_loader_passes_mx_config():
    mx_config = SimpleNamespace(
        server_url="http://mx:8001",
        server_query_timeout_s=17,
    )

    checkpoint_loader = model_loader_mod.construct_checkpoint_loader(
        None,
        "MX",
        mx_config=mx_config,
    )

    assert isinstance(checkpoint_loader, MXCheckpointLoader)
    assert checkpoint_loader.mx_server_url == "http://mx:8001"


def _format_documented_values(
    values: frozenset[object] | None,
    *,
    labels: dict[object, str] | None = None,
) -> str:
    assert values is not None
    assert None not in values
    labels = labels or {}
    return " or ".join(labels.get(value, str(value)) for value in sorted(values, key=str))


# Constraint phrases of the public support table, keyed by registry values.
# A profile that needs a new phrase adds it here rather than special-casing
# its model type.
_DOCUMENTED_SCOPES = {
    PostTransformTransferScope.TARGET_MODEL: "{family_kind} {dtypes}",
    PostTransformTransferScope.COMPLETE_MODEL: (
        "{family_kind} {dtypes} vision-language model with its vision encoder"
    ),
}
# Keyed by the profile's (`rope_types`, `rope_fusion`) constraints.
_DOCUMENTED_ROPE = {
    (frozenset({"default"}), frozenset({True})): "default fused RoPE",
    (frozenset({"default"}), frozenset({False})): "default fused QK-norm/RoPE",
    (frozenset({"mrope"}), frozenset({False})): "fused QK-norm/M-RoPE",
}
_DOCUMENTED_SLIDING_WINDOWS = {
    None: "",
    frozenset({"none"}): ", no sliding window",
}
# Documented weight-quantization envelopes, keyed by the quant_algorithms,
# sm_versions, and weight_block_sizes constraints of a dense row. A row with any
# other combination needs its own description here and in the support table.
_DOCUMENTED_DENSE_QUANTIZATION = {
    (frozenset({"none"}), None, None): "unquantized weights and KV cache",
    (
        frozenset({QuantAlgo.FP8_BLOCK_SCALES.value}),
        frozenset({90}),
        frozenset({(128, 128)}),
    ): "FP8 block-scale (128x128) weights and unquantized KV cache, SM90 only",
}


def _documented_constraints(profile: PostTransformProfile) -> str:
    """Render the support-table constraint text that a registry profile implies.

    A MoE profile also names its pinned MoE backend and admitted routed-expert
    partitions, and adds the MoE load balancer and DWDP to the exclusions.
    """
    constraints = profile.runtime_constraints
    quantization = _DOCUMENTED_DENSE_QUANTIZATION[
        (constraints.quant_algorithms, constraints.sm_versions, constraints.weight_block_sizes)
    ]
    assert constraints.kv_cache_quant_algorithms == frozenset({"none"})
    assert constraints.layerwise_quantization == frozenset({False})
    assert constraints.force_dynamic_quantization == frozenset({False})
    assert constraints.lora_enabled == frozenset({False})
    assert constraints.sparse_attention_enabled == frozenset({False})
    assert constraints.attention_dp == frozenset({False})
    assert constraints.multi_node == frozenset({False})
    # Each profile pins one embedding layout. A family that also qualifies tied
    # checkpoints, such as Qwen3, does so through a sibling profile with its own
    # layout ABI.
    assert constraints.tied_word_embeddings in (frozenset({False}), frozenset({True}))
    moe_profile = constraints.moe_backends is not None
    if moe_profile:
        assert constraints.moe_tp_sizes == constraints.tp_sizes
        assert constraints.moe_ep_sizes == constraints.tp_sizes
        assert constraints.moe_cluster_sizes == frozenset({1})
    else:
        assert constraints.moe_tp_sizes is None
        assert constraints.moe_ep_sizes is None
        assert constraints.moe_cluster_sizes is None
    assert constraints.attention_tp_sizes == constraints.tp_sizes
    assert constraints.attention_cp_sizes == constraints.cp_sizes
    assert constraints.pp_sizes == constraints.cp_sizes
    assert profile.speculative_mode is None
    assert profile.supported_features == frozenset()

    dtypes = _format_documented_values(
        constraints.dtypes,
        labels={"bfloat16": "BF16"},
    )
    attention_backends = _format_documented_values(constraints.attention_backends)
    tp_sizes = _format_documented_values(constraints.tp_sizes)
    pp_cp_sizes = _format_documented_values(constraints.pp_sizes)
    if moe_profile:
        family_kind = "MoE"
        moe_backend_description = (
            f"{_format_documented_values(constraints.moe_backends)} MoE backend, "
        )
        partition_description = (
            f"TP={tp_sizes}, MoE TP={_format_documented_values(constraints.moe_tp_sizes)} and "
            f"MoE EP={_format_documented_values(constraints.moe_ep_sizes)} with "
            "MoE TP x MoE EP = TP, "
            f"MoE cluster={_format_documented_values(constraints.moe_cluster_sizes)}"
        )
        moe_exclusions = "MoE load balancer, DWDP, "
    else:
        family_kind = "dense"
        moe_backend_description = ""
        partition_description = f"TP={tp_sizes}"
        moe_exclusions = ""
    model_description = _DOCUMENTED_SCOPES[profile.transfer_scope].format(
        family_kind=family_kind, dtypes=dtypes
    )
    rope_description = _DOCUMENTED_ROPE[(constraints.rope_types, constraints.rope_fusion)]
    sliding_window_description = _DOCUMENTED_SLIDING_WINDOWS[constraints.sliding_windows]
    embeddings = _format_documented_values(
        constraints.tied_word_embeddings,
        labels={False: "untied", True: "tied"},
    )
    return (
        f"Single-node {model_description}, {quantization}, "
        f"{attention_backends} attention, {moe_backend_description}{rope_description}"
        f"{sliding_window_description}, {embeddings} embeddings, {partition_description}, "
        f"PP/CP={pp_cp_sizes}, no LoRA, sparse attention, attention DP, {moe_exclusions}"
        "speculative mode, or separately loaded draft model"
    )


@pytest.mark.cpu_only
def test_public_support_table_matches_qualified_profile_registry() -> None:
    profiles = mx_loading_mod.MX_POST_TRANSFORM_PROFILE_REGISTRY.profiles
    documentation = (Path(__file__).parents[4] / "docs/source/features/model-express.md").read_text(
        encoding="utf-8"
    )
    lines = documentation.splitlines()
    table_header = (
        "| Profile | Root class | Config identity | Scope | Protocol | "
        "Transform-layout ABI | Constraints |"
    )
    table_header_index = lines.index(table_header)
    table_rows = []
    for line in lines[table_header_index + 2 :]:
        if not line.startswith("|"):
            break
        table_rows.append(line)

    assert len(table_rows) == len(profiles)
    for profile in profiles:
        scope = profile.transfer_scope.value.replace("_", " ").capitalize()
        expected_row = (
            f"| `{profile.profile_id}` | `{profile.root_model_class.class_name}` | "
            f"`{profile.architecture}` / `{profile.model_type}` | {scope} | "
            f"{profile.protocol_version} | `{profile.transform_abi_id}` | "
            f"{_documented_constraints(profile)} |"
        )
        assert expected_row in table_rows


def _repository_text(relative_path: str) -> str:
    """Read a text file given its path relative to the repository root."""
    return (Path(__file__).parents[4] / relative_path).read_text(encoding="utf-8")


def _mx_e2e_cases() -> dict[str, dict[str, object]]:
    """Read the donor/receiver `_MX_CASES` rows without importing the integration harness."""
    module = ast.parse(
        _repository_text("tests/integration/defs/model_express/test_model_express.py")
    )
    (cases,) = (
        statement.value
        for statement in module.body
        if isinstance(statement, ast.Assign)
        and [ast.unparse(target) for target in statement.targets] == ["_MX_CASES"]
    )
    rows: dict[str, dict[str, object]] = {}
    for param in cases.elts:
        (case,) = param.args
        options = {keyword.arg: keyword.value for keyword in param.keywords}
        row = {keyword.arg: ast.literal_eval(keyword.value) for keyword in case.keywords}
        marks = options["marks"]
        marks = marks.elts if isinstance(marks, (ast.List, ast.Tuple)) else [marks]
        # A bare mark such as `skip_no_hopper` is a name with no arguments.
        mark_args = {
            ast.unparse(mark.func if isinstance(mark, ast.Call) else mark): (
                mark.args if isinstance(mark, ast.Call) else []
            )
            for mark in marks
        }
        # Every row names its GPU count; a row may also require device memory or SM90.
        assert len(mark_args) == len(marks)
        assert set(mark_args) <= {
            "pytest.mark.skip_less_device",
            "pytest.mark.skip_less_device_memory",
            "skip_no_hopper",
        }
        row["min_devices"] = ast.literal_eval(mark_args["pytest.mark.skip_less_device"][0])
        case_id = ast.literal_eval(options["id"])
        assert case_id not in rows
        rows[case_id] = row
    return rows


def _l0_model_express_case_ids() -> dict[int, list[str]]:
    """Map each `l0_model_express.yml` GPU count to its donor/receiver case IDs.

    Blocks that share a GPU count, such as the pre-merge canary and the
    post-merge TP=1 rows, are combined.
    """
    test_db = yaml.safe_load(
        _repository_text("tests/integration/test_lists/test-db/l0_model_express.yml")
    )
    prefix = "model_express/test_model_express.py::test_mx_donor_receiver["
    case_ids: dict[int, list[str]] = {}
    for block in test_db["l0_model_express"]:
        gpu_count = block["condition"]["ranges"]["system_gpu_count"]
        assert gpu_count["gte"] == gpu_count["lte"]
        case_ids.setdefault(gpu_count["gte"], []).extend(
            test.removeprefix(prefix).removesuffix("]")
            for test in block["tests"]
            if test.startswith(prefix)
        )
    return case_ids


def _documented_names(text: str, start: str, end: str, known_names: set[str]) -> set[str]:
    """Return the family name that starts each item enumerated between `start` and `end`.

    Each item starts with one of `known_names`, and the longest match wins, so
    `Qwen3 MoE` is not read as `Qwen3`. A name matches only when no word
    character or dot follows it, so `Qwen3.5` is never read as `Qwen3`. A
    missing anchor fails with its text, so whoever rewords
    `docs/source/features/model-express.md` knows which anchor to update here.
    """
    doc = "docs/source/features/model-express.md"
    stop = text.find(end)
    if stop < 0:
        pytest.fail(f"{doc} no longer contains the anchor {end!r}; update this test's anchors")
    begin = text.rfind(start, 0, stop)
    if begin < 0:
        pytest.fail(f"{doc} has no anchor {start!r} before {end!r}; update this test's anchors")
    enumeration = text[begin + len(start) : stop]
    documented = set()
    for item in re.split(r",? and |, ", enumeration):
        matches = [name for name in known_names if re.match(re.escape(name) + r"(?![\w.])", item)]
        assert matches, f"{doc} names {item!r}, which no qualified profile covers"
        documented.add(max(matches, key=len))
    return documented


# The documentation spells these families differently from their architecture names.
_MX_FAMILY_DISPLAY_NAMES = {"Qwen3MoeForCausalLM": "Qwen3 MoE"}


def _documented_family(profile: PostTransformProfile) -> str:
    """Return the family name by which the doc refers to a profile, such as `Qwen3.5`."""
    if profile.architecture in _MX_FAMILY_DISPLAY_NAMES:
        return _MX_FAMILY_DISPLAY_NAMES[profile.architecture]
    family = re.sub(r"For(?:CausalLM|ConditionalGeneration)$", "", profile.architecture)
    return family.replace("_", ".")


def _moe_ep_row_partitions(profile: PostTransformProfile) -> list[tuple[int, int]]:
    """Return the `(TP size, MoE EP size)` pairs with MoE EP above 1 that a profile admits.

    MoE TP takes the rest of TP, so a pair is admitted when the profile admits that MoE TP size.
    """
    constraints = profile.runtime_constraints
    if constraints.moe_backends is None:
        return []
    return [
        (tp_size, moe_ep_size)
        for tp_size in sorted(constraints.tp_sizes)
        for moe_ep_size in sorted(constraints.moe_ep_sizes)
        if moe_ep_size > 1
        and tp_size % moe_ep_size == 0
        and tp_size // moe_ep_size in constraints.moe_tp_sizes
    ]


# Donor/receiver rows of each qualified profile, as `(case-ID stem, repository
# cache stem, model path override variable)`. The rows are `<case-ID stem>-tp1`
# and `<case-ID stem>-tp2`, plus `<case-ID stem>-tp<N>-ep<M>` for each MoE EP
# size above 1 that a MoE profile admits. A family can have several profiles, so
# a profile ID does not name its rows.
_MX_E2E_ROWS = {
    "llama-for-causal-lm-target-v1": ("llama-bf16", "llama", "TRTLLM_MX_LLAMA_MODEL"),
    "qwen2-for-causal-lm-bf16-target-v1": ("qwen2-bf16", "qwen2", "TRTLLM_MX_QWEN2_MODEL"),
    "qwen3-for-causal-lm-bf16-target-v1": ("qwen3-bf16", "qwen3", "TRTLLM_MX_QWEN3_MODEL"),
    "mistral-for-causal-lm-bf16-target-v1": ("mistral-bf16", "mistral", "TRTLLM_MX_MISTRAL_MODEL"),
    "phi3-for-causal-lm-bf16-target-v1": ("phi3-bf16", "phi3", "TRTLLM_MX_PHI3_MODEL"),
    "qwen3_5-for-conditional-generation-bf16-complete-v1": (
        "qwen3_5-bf16",
        "qwen3_5",
        "TRTLLM_MX_QWEN3_5_MODEL",
    ),
    "qwen3-moe-for-causal-lm-bf16-cutlass-target-v1": (
        "qwen3-moe-bf16",
        "qwen3-moe",
        "TRTLLM_MX_QWEN3_MOE_MODEL",
    ),
    "qwen3-for-causal-lm-fp8-block-sm90-target-v1": (
        "qwen3-fp8-block",
        "qwen3-fp8-block",
        "TRTLLM_MX_QWEN3_FP8_MODEL",
    ),
    "qwen3-for-causal-lm-bf16-tied-target-v1": (
        "qwen3-tied-bf16",
        "qwen3-tied",
        "TRTLLM_MX_QWEN3_TIED_MODEL",
    ),
}


@pytest.mark.cpu_only
def test_mx_e2e_rows_and_ci_list_match_qualified_profile_registry() -> None:
    """Every qualified profile has TP1 and TP2 donor/receiver rows that CI schedules.

    `_MX_E2E_ROWS` names the rows of each profile and needs one entry per
    qualified profile. The `_MX_CASES` rows must follow the shared naming
    scheme, and `l0_model_express.yml` must list each row exactly once, in a
    block whose GPU count matches the row's two TP groups.
    """
    profiles = mx_loading_mod.MX_POST_TRANSFORM_PROFILE_REGISTRY.profiles
    assert set(_MX_E2E_ROWS) == {profile.profile_id for profile in profiles}, (
        "Every qualified profile needs exactly one `_MX_E2E_ROWS` entry"
    )
    for column in zip(*_MX_E2E_ROWS.values(), strict=True):
        assert len(set(column)) == len(column), (
            f"Two profiles share the `_MX_E2E_ROWS` value of one column: {column}"
        )
    cases = _mx_e2e_cases()
    ci_case_ids = _l0_model_express_case_ids()
    expected_case_ids = set()
    for profile in profiles:
        case_stem, cache_stem, model_env = _MX_E2E_ROWS[profile.profile_id]
        for tp_size in (1, 2):
            case_id = f"{case_stem}-tp{tp_size}"
            expected_case_ids.add(case_id)
            expected_row = {
                "model_env": model_env,
                "default_model_subdir": cases[f"{case_stem}-tp1"]["default_model_subdir"],
                "repository_cache_prefix": f"models--trtllm-mx-e2e--{cache_stem}-tp{tp_size}",
                "tp_size": tp_size,
                # The donor and the receiver each hold one TP group.
                "min_devices": 2 * tp_size,
            }
            if profile.transfer_scope is PostTransformTransferScope.COMPLETE_MODEL:
                # A complete-model profile carries a vision encoder, so its rows probe with images.
                expected_row["probe"] = "image"
            if profile.runtime_constraints.moe_backends is not None:
                # MoE rows disable the CUTLASS FC2+finalize fusion, which is not
                # deterministic for top-k greater than 2.
                expected_row["moe_disable_finalize_fusion"] = True
            assert cases[case_id] == expected_row
            assert case_id in ci_case_ids[2 * tp_size]
        # A MoE profile has one more row for each admitted MoE EP size above 1. It
        # repeats the row of its TP size with its own repository cache and MoE EP size.
        for tp_size, moe_ep_size in _moe_ep_row_partitions(profile):
            case_id = f"{case_stem}-tp{tp_size}-ep{moe_ep_size}"
            expected_case_ids.add(case_id)
            assert cases[case_id] == {
                **cases[f"{case_stem}-tp{tp_size}"],
                "repository_cache_prefix": (
                    f"models--trtllm-mx-e2e--{cache_stem}-tp{tp_size}-ep{moe_ep_size}"
                ),
                "moe_ep_size": moe_ep_size,
            }
            assert case_id in ci_case_ids[2 * tp_size]

    assert set(cases) == expected_case_ids
    assert set(ci_case_ids) == {2, 4}
    assert sorted(ci_case_ids[2] + ci_case_ids[4]) == sorted(expected_case_ids)


@pytest.mark.cpu_only
def test_documentation_prose_matches_qualified_profile_registry() -> None:
    """The ModelExpress doc names exactly the families, cases, and overrides in the registry.

    Checks the text-only family list, the fused-RoPE family list, the
    limitations list, and the case IDs and `TRTLLM_MX_*_MODEL` variables named
    in the "Qualification Test" section.
    """
    profiles = mx_loading_mod.MX_POST_TRANSFORM_PROFILE_REGISTRY.profiles
    families = {_documented_family(profile) for profile in profiles}
    text_only_families = {
        _documented_family(profile)
        for profile in profiles
        if profile.transfer_scope is PostTransformTransferScope.TARGET_MODEL
    }
    fused_rope_families = {
        _documented_family(profile)
        for profile in profiles
        if profile.runtime_constraints.rope_fusion == frozenset({True})
    }
    documentation = _repository_text("docs/source/features/model-express.md")
    text = " ".join(documentation.split())

    assert (
        _documented_names(text, "The ", " profiles are text-only", families) == text_only_families
    )
    assert (
        _documented_names(
            text,
            "pins its qualified RoPE realization: ",
            " require the default fused RoPE path",
            families,
        )
        == fused_rope_families
    )
    assert (
        _documented_names(text, "currently limited to the exact ", " profiles above", families)
        == families
    )

    qualification_test = documentation[
        documentation.index("### Qualification Test") : documentation.index(
            "### Transform-Layout ABI Rules"
        )
    ]
    cases = _mx_e2e_cases()
    case_id_pattern = r"\b[a-z][a-z0-9_]*(?:-[a-z0-9_]+)*?-(?:bf16|fp8-block)-tp\d+(?:-ep\d+)?\b"
    assert set(re.findall(case_id_pattern, qualification_test)) == set(cases)
    assert set(re.findall(r"\bTRTLLM_MX_[A-Z0-9_]+_MODEL\b", qualification_test)) == {
        row["model_env"] for row in cases.values()
    }


@pytest.mark.cpu_only
def test_mx_success_initializes_mapper_skips_weight_mapping_and_reload_works(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = []
    loader = _make_loader(monkeypatch, events=events)
    monkeypatch.setattr(
        mx_loading_mod,
        "MX_POST_TRANSFORM_PROFILE_REGISTRY",
        _tiny_profile_registry(),
    )
    checkpoint_loader = _mock_mx_checkpoint_loader()
    checkpoint_loader.is_weights_preloaded.return_value = True
    checkpoint_loader.is_post_transform_weights_preloaded.return_value = False
    checkpoint_loader.load_weights.return_value = {}

    model, _ = loader.load("/ckpt", checkpoint_loader)

    checkpoint_loader.load_weights.assert_called_once()
    _args, kwargs = checkpoint_loader.load_weights.call_args
    assert kwargs["mapping"] is loader.mapping
    assert kwargs["model"] is model
    source_identity = kwargs["source_identity"]
    assert (
        kwargs["model_config"] is memory_init_mod.AutoModelForCausalLM.from_config.call_args.args[0]
    )
    assert kwargs["load_config"] is loader.llm_args
    assert (
        kwargs["post_transform_protocol_version"]
        == mx_loading_mod.MX_STAGED_RECEIVER_TRANSFORM_PROTOCOL_VERSION
    )
    assert kwargs["allow_post_transform_weights"] is True
    assert source_identity.transform_abi_id == LLAMA_POST_TRANSFORM_LAYOUT_ABI_V1
    assert model_hooks_mod.call_load_weights.call_count == 0
    checkpoint_loader.get_initialized_weight_mapper.assert_called_once()
    assert loader.weight_mapper is checkpoint_loader.get_initialized_weight_mapper.return_value
    # The publish carries the same identity that the read got.
    checkpoint_loader.publish_as_source.assert_called_once_with(
        model, source_identity=source_identity
    )

    # reload() uses self.weight_mapper unconditionally; MX success must
    # initialize it even though the initial load skipped call_load_weights.
    model._weights_transformed = True
    model.linear._weights_transformed = True
    loader.reload(model, {"reloaded": MagicMock()})
    assert model_hooks_mod.call_load_weights.call_count == 1
    assert model._weights_transformed is False
    assert model.linear._weights_transformed is False
    assert events == ["post_load_weights", "load_weights"]


def test_cleanup_releases_active_checkpoint_loader(monkeypatch):
    loader = _make_loader(monkeypatch, events=[])
    checkpoint_loader = _mock_mx_checkpoint_loader()
    checkpoint_loader.is_weights_preloaded.return_value = False
    checkpoint_loader.load_weights.return_value = {"weight": MagicMock()}

    loader.load("/ckpt", checkpoint_loader)
    loader.cleanup()

    checkpoint_loader.cleanup.assert_called_once_with()
    assert loader._checkpoint_loader is None


def test_cleanup_swallows_checkpoint_loader_failure(monkeypatch):
    loader = _make_loader(monkeypatch, events=[])
    checkpoint_loader = MagicMock(name="checkpoint_loader")
    cleanup_error = RuntimeError("cleanup failed")
    checkpoint_loader.cleanup.side_effect = cleanup_error
    loader._checkpoint_loader = checkpoint_loader
    warning = MagicMock()
    monkeypatch.setattr(model_loader_mod.logger, "warning", warning)

    loader.cleanup()

    checkpoint_loader.cleanup.assert_called_once_with()
    assert loader._checkpoint_loader is None
    warning.assert_called_once_with(
        f"Failed to clean up checkpoint loader {checkpoint_loader!r}: {cleanup_error!r}"
    )


def test_cleanup_releases_gms_backend_and_checkpoint_loader(monkeypatch):
    loader = _make_loader(monkeypatch, events=[])
    gms_backend = MagicMock(name="gms_backend")
    checkpoint_loader = MagicMock(name="checkpoint_loader")
    loader._gms_backend = gms_backend
    loader._checkpoint_loader = checkpoint_loader

    loader.cleanup()

    gms_backend.cleanup.assert_called_once_with()
    checkpoint_loader.cleanup.assert_called_once_with()
    assert loader._gms_backend is None
    assert loader._checkpoint_loader is None


def test_config_failure_retains_checkpoint_loader_for_cleanup(monkeypatch):
    loader = _make_loader(monkeypatch, events=[])
    checkpoint_loader = MagicMock(name="checkpoint_loader")
    loader._load_and_validate_config.side_effect = RuntimeError("config failed")

    with pytest.raises(RuntimeError, match="config failed"):
        loader.load("/ckpt", checkpoint_loader)

    assert loader._checkpoint_loader is checkpoint_loader
    loader.cleanup()
    checkpoint_loader.cleanup.assert_called_once_with()


@pytest.mark.cpu_only
def test_reload_partial_loading_preserves_weights_transformed_flags(monkeypatch):
    events = []
    loader = _make_loader(monkeypatch, events=events)
    loader.weight_mapper = MagicMock(name="weight_mapper")
    model = _TinyModel(events)
    model._weights_transformed = True
    model.linear._weights_transformed = True

    loader.reload(model, {"reloaded": MagicMock()}, allow_partial_loading=True)

    assert model_hooks_mod.call_load_weights.call_count == 1
    assert model_hooks_mod.call_load_weights.call_args.kwargs["allow_partial_loading"] is True
    assert model._weights_transformed is True
    assert model.linear._weights_transformed is True
    assert events == ["load_weights"]


@pytest.mark.cpu_only
def test_mx_partial_fallback_merges_returned_weights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = []
    loader = _make_loader(monkeypatch, events=events)
    monkeypatch.setattr(
        mx_loading_mod,
        "MX_POST_TRANSFORM_PROFILE_REGISTRY",
        _tiny_profile_registry(),
    )
    checkpoint_loader = _mock_mx_checkpoint_loader()
    checkpoint_loader.is_weights_preloaded.return_value = True
    checkpoint_loader.is_post_transform_weights_preloaded.return_value = False
    fallback_weights = {"mismatched.weight": MagicMock()}
    checkpoint_loader.load_weights.return_value = fallback_weights

    model, _ = loader.load("/ckpt", checkpoint_loader)

    assert model_hooks_mod.call_load_weights.call_count == 1
    load_fn, weights, mapper = model_hooks_mod.call_load_weights.call_args.args
    assert load_fn == model.load_weights
    assert weights is fallback_weights
    assert mapper is loader.weight_mapper
    checkpoint_loader.publish_as_source.assert_called_once_with(
        model,
        source_identity=checkpoint_loader.load_weights.call_args.kwargs["source_identity"],
    )


class _PostTransformMxLoader(MXCheckpointLoader):
    """MX loader whose P2P read delivers post-transform weights.

    When the load is not allowed to take post-transform weights, the read
    falls back to disk weights instead.
    """

    def __init__(self, *, post_transform: bool) -> None:
        super().__init__()
        self._post_transform = post_transform
        self._weights_preloaded = True
        self._disk_weight = MagicMock()
        self.load_weights = MagicMock(side_effect=self._load_weights)
        self.is_weights_preloaded = MagicMock(side_effect=lambda: self._weights_preloaded)
        self.get_initialized_weight_mapper = MagicMock(return_value=MagicMock())
        self.publish_as_source = MagicMock()

    def _load_weights(self, *_args: object, **kwargs: object) -> dict[str, object]:
        if self._post_transform and kwargs.get("allow_post_transform_weights") is False:
            self._post_transform = False
            self._weights_preloaded = False
            return {"disk.weight": self._disk_weight}
        if self._post_transform:
            prepare_receiver = cast(
                Callable[[nn.Module], None], kwargs["prepare_post_transform_receiver"]
            )
            prepare_receiver(cast(nn.Module, kwargs["model"]))
        return {}

    def is_post_transform_weights_preloaded(self) -> bool:
        return self._post_transform


class _UnsafePostTransformMxLoader(_PostTransformMxLoader):
    def _load_weights(self, *_args: object, **_kwargs: object) -> dict[str, object]:
        return {}


@pytest.mark.cpu_only
def test_mx_post_transform_receiver_uses_staged_path_when_qualified(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = []
    loader = _make_loader(monkeypatch, events=events)
    monkeypatch.setattr(
        mx_loading_mod,
        "MX_POST_TRANSFORM_PROFILE_REGISTRY",
        _tiny_profile_registry(),
    )
    checkpoint_loader = _PostTransformMxLoader(post_transform=True)

    model, _ = loader.load("/ckpt", checkpoint_loader)

    model_hooks_mod.call_load_weights.assert_not_called()
    _args, kwargs = checkpoint_loader.load_weights.call_args
    assert kwargs["allow_post_transform_weights"] is True
    assert (
        kwargs["model_config"] is memory_init_mod.AutoModelForCausalLM.from_config.call_args.args[0]
    )
    assert kwargs["load_config"] is loader.llm_args
    assert (
        kwargs["post_transform_protocol_version"]
        == mx_loading_mod.MX_STAGED_RECEIVER_TRANSFORM_PROTOCOL_VERSION
    )
    assert callable(kwargs["prepare_post_transform_receiver"])
    checkpoint_loader.publish_as_source.assert_called_once_with(
        model, source_identity=kwargs["source_identity"]
    )
    # Post-transform receivers skip transform_weights(), but the accepted
    # tensors are already in final layout. Keep the transform guard in sync so
    # future reload/refactor paths do not accidentally treat them as raw bytes.
    assert model._weights_transformed is True
    assert model.linear._weights_transformed is True
    assert model.draft_model.linear._weights_transformed is True
    assert events == ["setup_aliases", "setup_aliases", "cache_derived_state"]


class _TinyParameterModule(nn.Module):
    """One-parameter module standing in for a language model or an encoder."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(2))


class _TinyMultimodalModel(MultimodalModelMixin, _TinyModel):
    """`_TinyModel` with a language model and an optional local encoder."""

    def __init__(self, events: list[str], *, with_encoder: bool) -> None:
        super().__init__(events)
        self.llm = _TinyParameterModule()
        self.mm_encoder = _TinyParameterModule() if with_encoder else None

    @property
    def language_model(self) -> nn.Module:
        """Return the inner language model."""
        return self.llm


class _UnknownLanguageModelMultimodalModel(_TinyMultimodalModel):
    """Registered multimodal root that keeps the mixin's raising `language_model`."""

    language_model = MultimodalModelMixin.language_model


class _UnregisteredMultimodalModel(_TinyMultimodalModel):
    """Multimodal root without a profile; its language model must not be queried."""

    @property
    def language_model(self) -> nn.Module:
        """Fail if scope derivation inspects an unregistered root."""
        raise AssertionError("language_model queried for an unregistered root")


def _tiny_multimodal_profile(
    root_model_class: type[nn.Module],
) -> PostTransformProfile:
    """Return a complete-model test profile for `root_model_class`."""
    return PostTransformProfile(
        profile_id=f"{root_model_class.__name__}-complete-v1",
        root_model_class=root_model_class,
        architecture="TinyForCausalLM",
        model_type="tiny",
        speculative_mode=None,
        protocol_version=mx_loading_mod.MX_STAGED_RECEIVER_TRANSFORM_PROTOCOL_VERSION,
        transform_abi_id=LLAMA_POST_TRANSFORM_LAYOUT_ABI_V1,
        transfer_scope=PostTransformTransferScope.COMPLETE_MODEL,
        runtime_constraints=PostTransformRuntimeConstraints(),
    )


@pytest.mark.cpu_only
def test_post_transform_transfer_scope_is_derived_from_the_root(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A registered multimodal root is the complete model only with a local encoder.

    Text-only and unregistered roots keep the target-model scope without
    querying `language_model`, and a registered root whose language model is
    unknown keeps it too instead of raising, so qualification never breaks a
    plain load.
    """
    monkeypatch.setattr(
        mx_loading_mod,
        "MX_POST_TRANSFORM_PROFILE_REGISTRY",
        PostTransformProfileRegistry(
            profiles=(
                _tiny_multimodal_profile(_TinyMultimodalModel),
                _tiny_multimodal_profile(_UnknownLanguageModelMultimodalModel),
            )
        ),
    )
    derive_scope = mx_loading_mod.get_post_transform_transfer_scope

    assert derive_scope(_TinyModel([])) is PostTransformTransferScope.TARGET_MODEL
    assert (
        derive_scope(_TinyMultimodalModel([], with_encoder=True))
        is PostTransformTransferScope.COMPLETE_MODEL
    )
    assert (
        derive_scope(_TinyMultimodalModel([], with_encoder=False))
        is PostTransformTransferScope.LANGUAGE_MODEL
    )
    assert (
        derive_scope(_UnknownLanguageModelMultimodalModel([], with_encoder=True))
        is PostTransformTransferScope.TARGET_MODEL
    )
    assert (
        derive_scope(_UnregisteredMultimodalModel([], with_encoder=True))
        is PostTransformTransferScope.TARGET_MODEL
    )


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "with_encoder",
    [True, False],
    ids=["complete-model", "language-model"],
)
def test_mx_load_offers_post_transform_weights_only_for_the_registered_scope(
    monkeypatch: pytest.MonkeyPatch,
    with_encoder: bool,
) -> None:
    """An MX load accepts post-transform weights only when the derived scope matches.

    The profile registers the complete model, so a multimodal root without a
    local encoder falls back to native loading before any transfer.
    """
    events: list[str] = []
    loader = _make_loader(monkeypatch, events=events)
    monkeypatch.setattr(
        mx_loading_mod,
        "MX_POST_TRANSFORM_PROFILE_REGISTRY",
        PostTransformProfileRegistry(profiles=(_tiny_multimodal_profile(_TinyMultimodalModel),)),
    )
    monkeypatch.setattr(
        memory_init_mod.AutoModelForCausalLM,
        "from_config",
        MagicMock(return_value=_TinyMultimodalModel(events, with_encoder=with_encoder)),
    )
    checkpoint_loader = _PostTransformMxLoader(post_transform=True)

    model, _ = loader.load("/ckpt", checkpoint_loader)

    _args, kwargs = checkpoint_loader.load_weights.call_args
    assert kwargs["allow_post_transform_weights"] is with_encoder
    source_identity = kwargs["source_identity"]
    if with_encoder:
        assert source_identity.transform_abi_id == LLAMA_POST_TRANSFORM_LAYOUT_ABI_V1
        model_hooks_mod.call_load_weights.assert_not_called()
        checkpoint_loader.publish_as_source.assert_called_once_with(
            model, source_identity=source_identity
        )
    else:
        assert source_identity.transform_abi_id is None
        assert model_hooks_mod.call_load_weights.call_count == 1
        checkpoint_loader.publish_as_source.assert_not_called()


@pytest.mark.cpu_only
def test_qualified_profiles_bind_published_transform_abis() -> None:
    """Every ABI that a registered profile binds is exported by `weight_sharing`.

    The `SourceIdentity` ABI tests iterate the exported constants, so an ABI
    missing from `__all__` would otherwise escape them.
    """
    published = {
        getattr(weight_sharing_mod, name)
        for name in weight_sharing_mod.__all__
        if re.fullmatch(r".*_POST_TRANSFORM_LAYOUT_ABI_V\d+", name)
    }
    profiles = mx_loading_mod.MX_POST_TRANSFORM_PROFILE_REGISTRY.profiles

    assert {profile.transform_abi_id for profile in profiles} <= published


def test_default_profile_qualifies_real_tiny_llama_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = PostTransformQualificationCase(
        profile_id="llama-for-causal-lm-target-v1",
        model_factory=lambda: _tiny_llama_model(monkeypatch),
        unqualified_model_factory=lambda: _tiny_llama_model(
            monkeypatch,
            model_class=_UnqualifiedLlamaForCausalLM,
        ),
        qualify_model=lambda model: mx_loading_mod.qualify_post_transform_profile(
            model,
            speculative_mode=None,
            loads_draft_weights=False,
        ),
        state_probes=(("aliases", _llama_alias_state),),
        output_probes=(("embedding-logits", _llama_embedding_logits),),
    )

    producer, _receiver = assert_post_transform_lifecycle_equivalent(case)

    assert (
        PostTransformRuntimeConfig.from_model_config(producer.model_config, model=producer)
        == _bf16_dense_runtime_config()
    )


@pytest.mark.parametrize("rank", [0, 1])
def test_default_profile_qualifies_real_tiny_llama_tp2_rank_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
    rank: int,
) -> None:
    monkeypatch.setattr(mapping_mod, "mpi_disabled", lambda: False)
    monkeypatch.setattr(distributed_mod, "AllReduce", _AllReduceStub)
    monkeypatch.setattr(modeling_llama_mod, "AllReduce", _AllReduceStub)
    case = PostTransformQualificationCase(
        profile_id="llama-for-causal-lm-target-v1",
        model_factory=lambda: _tiny_llama_model(
            monkeypatch,
            tp_size=2,
            rank=rank,
        ),
        unqualified_model_factory=lambda: _tiny_llama_model(
            monkeypatch,
            model_class=_UnqualifiedLlamaForCausalLM,
            tp_size=2,
            rank=rank,
        ),
        qualify_model=lambda model: mx_loading_mod.qualify_post_transform_profile(
            model,
            speculative_mode=None,
            loads_draft_weights=False,
        ),
        state_probes=(("aliases", _llama_alias_state),),
        output_probes=(("input-embeddings", _llama_input_embeddings),),
    )

    producer, _receiver = assert_post_transform_lifecycle_equivalent(case)

    assert PostTransformRuntimeConfig.from_model_config(
        producer.model_config, model=producer
    ) == _bf16_dense_runtime_config(
        tp_size=2,
        moe_tp_size=2,
        attention_tp_size=2,
    )


def test_qwen2_dense_profile_qualifies_full_staged_lifecycle() -> None:
    case = PostTransformQualificationCase(
        profile_id="qwen2-for-causal-lm-bf16-target-v1",
        model_factory=_tiny_qwen2_model,
        unqualified_model_factory=lambda: _tiny_qwen2_model(
            model_class=_UnqualifiedQwen2ForCausalLM
        ),
        qualify_model=lambda model: mx_loading_mod.qualify_post_transform_profile(
            model,
            speculative_mode=None,
            loads_draft_weights=False,
        ),
        state_probes=(("layout", _qwen2_layout_state),),
        output_probes=(
            ("embedding-logits", _dense_embedding_logits),
            ("fused-qkv", _dense_fused_qkv_output),
            ("fused-gate-up", _dense_fused_gate_up_output),
        ),
    )

    producer, _receiver = assert_post_transform_lifecycle_equivalent(case)

    assert (
        PostTransformRuntimeConfig.from_model_config(producer.model_config, model=producer)
        == _bf16_dense_runtime_config()
    )
    assert _qwen2_layout_state(producer) == {
        "attention_type": modeling_qwen_mod.QwenAttention.__name__,
        "qkv_weight_mode": WeightMode.FUSED_QKV_LINEAR,
        "qkv_weight_shape": (32, 16),
        "gate_up_weight_mode": WeightMode.FUSED_GATE_UP_LINEAR,
        "gate_up_weight_shape": (64, 16),
        "qkv_bias": True,
        "rope_fusion": True,
        "rotary_embedding": None,
        "tied_lm_head": False,
    }


def test_qwen2_dense_profile_rejects_effective_unfused_rope() -> None:
    model = _tiny_qwen2_model(
        kv_cache_compression_config=SimpleNamespace(changes_physical_kv_length=True)
    )

    decision = mx_loading_mod.qualify_post_transform_profile(
        model,
        speculative_mode=None,
        loads_draft_weights=False,
    )

    assert model.model.layers[0].self_attn.rope_fusion is False
    assert not decision.qualified
    assert decision.reason is PostTransformQualificationReason.RUNTIME_CONFIG_NOT_SUPPORTED
    assert decision.unsupported_runtime_dimensions == frozenset({"rope_fusion"})


@pytest.mark.parametrize("rank", [0, 1])
def test_qwen2_dense_profile_qualifies_tp2_rank_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
    rank: int,
) -> None:
    monkeypatch.setattr(mapping_mod, "mpi_disabled", lambda: False)
    monkeypatch.setattr(distributed_mod, "AllReduce", _AllReduceStub)
    case = PostTransformQualificationCase(
        profile_id="qwen2-for-causal-lm-bf16-target-v1",
        model_factory=lambda: _tiny_qwen2_model(tp_size=2, rank=rank),
        unqualified_model_factory=lambda: _tiny_qwen2_model(
            model_class=_UnqualifiedQwen2ForCausalLM,
            tp_size=2,
            rank=rank,
        ),
        qualify_model=lambda model: mx_loading_mod.qualify_post_transform_profile(
            model,
            speculative_mode=None,
            loads_draft_weights=False,
        ),
        state_probes=(("layout", _qwen2_layout_state),),
        output_probes=(
            ("fused-qkv", _dense_fused_qkv_output),
            ("fused-gate-up", _dense_fused_gate_up_output),
        ),
    )

    producer, _receiver = assert_post_transform_lifecycle_equivalent(case)

    assert PostTransformRuntimeConfig.from_model_config(
        producer.model_config, model=producer
    ) == _bf16_dense_runtime_config(
        tp_size=2,
        moe_tp_size=2,
        attention_tp_size=2,
    )
    assert _qwen2_layout_state(producer)["qkv_weight_shape"] == (16, 16)
    assert _qwen2_layout_state(producer)["gate_up_weight_shape"] == (32, 16)


def test_qwen3_dense_profile_qualifies_full_staged_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = PostTransformQualificationCase(
        profile_id="qwen3-for-causal-lm-bf16-target-v1",
        model_factory=lambda: _tiny_qwen3_model(monkeypatch),
        unqualified_model_factory=lambda: _tiny_qwen3_model(
            monkeypatch,
            model_class=_UnqualifiedQwen3ForCausalLM,
        ),
        qualify_model=lambda model: mx_loading_mod.qualify_post_transform_profile(
            model,
            speculative_mode=None,
            loads_draft_weights=False,
        ),
        state_probes=(("layout", _qwen3_layout_state),),
        output_probes=(
            ("embedding-logits", _dense_embedding_logits),
            ("fused-qkv", _dense_fused_qkv_output),
            ("fused-gate-up", _dense_fused_gate_up_output),
        ),
    )

    producer, _receiver = assert_post_transform_lifecycle_equivalent(case)

    assert PostTransformRuntimeConfig.from_model_config(
        producer.model_config, model=producer
    ) == _bf16_dense_runtime_config(rope_fusion=False)
    assert _qwen3_layout_state(producer) == {
        "attention_type": modeling_qwen3_mod.Qwen3Attention.__name__,
        "qkv_weight_mode": WeightMode.FUSED_QKV_LINEAR,
        "qkv_weight_shape": (32, 16),
        "gate_up_weight_mode": WeightMode.FUSED_GATE_UP_LINEAR,
        "gate_up_weight_shape": (64, 16),
        "qkv_bias": False,
        "q_norm_weight_shape": (4,),
        "k_norm_weight_shape": (4,),
        "fuse_qk_norm_rope": True,
        "rope_fusion": False,
        "rotary_embedding_present": True,
        "tied_lm_head": False,
    }


@pytest.mark.parametrize("rank", [0, 1])
def test_qwen3_dense_profile_qualifies_tp2_rank_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
    rank: int,
) -> None:
    monkeypatch.setattr(mapping_mod, "mpi_disabled", lambda: False)
    monkeypatch.setattr(distributed_mod, "AllReduce", _AllReduceStub)
    case = PostTransformQualificationCase(
        profile_id="qwen3-for-causal-lm-bf16-target-v1",
        model_factory=lambda: _tiny_qwen3_model(
            monkeypatch,
            tp_size=2,
            rank=rank,
        ),
        unqualified_model_factory=lambda: _tiny_qwen3_model(
            monkeypatch,
            model_class=_UnqualifiedQwen3ForCausalLM,
            tp_size=2,
            rank=rank,
        ),
        qualify_model=lambda model: mx_loading_mod.qualify_post_transform_profile(
            model,
            speculative_mode=None,
            loads_draft_weights=False,
        ),
        state_probes=(("layout", _qwen3_layout_state),),
        output_probes=(
            ("fused-qkv", _dense_fused_qkv_output),
            ("fused-gate-up", _dense_fused_gate_up_output),
        ),
    )

    producer, _receiver = assert_post_transform_lifecycle_equivalent(case)

    assert PostTransformRuntimeConfig.from_model_config(
        producer.model_config, model=producer
    ) == _bf16_dense_runtime_config(
        tp_size=2,
        moe_tp_size=2,
        attention_tp_size=2,
        rope_fusion=False,
    )
    assert _qwen3_layout_state(producer)["qkv_weight_shape"] == (16, 16)
    assert _qwen3_layout_state(producer)["gate_up_weight_shape"] == (32, 16)


def _qwen3_tied_embedding_state(model: nn.Module) -> dict[str, object]:
    """Describe the tie through memory, which identifies it after meta-init materialization."""
    head = model.lm_head.weight
    table = model.model.embed_tokens.weight
    return {
        "same_view": (head.device, head.data_ptr(), head.shape, head.stride())
        == (table.device, table.data_ptr(), table.shape, table.stride()),
        "table_shape": tuple(table.shape),
        "mx_catalog_names_lm_head": "lm_head.weight" in _mx_canonical_parameter_catalog(model),
    }


def _rank_local_lm_head_logits(model: nn.Module) -> torch.Tensor:
    return nn.functional.linear(_dense_hidden_states(model), model.lm_head.weight)


def _copy_mx_catalog(producer: nn.Module, receiver: nn.Module) -> None:
    """Copy each tensor of the producer's MX catalog once, by name, as MX transfers it."""
    catalog = _mx_canonical_parameter_catalog(producer)
    assert _mx_canonical_parameter_catalog(receiver).keys() == catalog.keys()
    producer_parameters = dict(producer.named_parameters(remove_duplicate=False))
    receiver_parameters = dict(receiver.named_parameters(remove_duplicate=False))
    with torch.no_grad():
        for name in catalog:
            receiver_parameters[name].copy_(producer_parameters[name])


def _receive_mx_catalog_into_blank_receiver(producer: nn.Module, receiver: nn.Module) -> None:
    """Zero every receiver parameter, then transfer only the MX catalog.

    A parameter that the catalog does not cover, such as an untied LM head,
    stays zero and fails the lifecycle comparison.
    """
    with torch.no_grad():
        for _name, parameter in receiver.named_parameters(remove_duplicate=False):
            parameter.zero_()
    _copy_mx_catalog(producer, receiver)


def test_qwen3_dense_tied_profile_qualifies_full_staged_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = PostTransformQualificationCase(
        profile_id="qwen3-for-causal-lm-bf16-tied-target-v1",
        model_factory=lambda: _tiny_qwen3_model(monkeypatch, tie_word_embeddings=True),
        unqualified_model_factory=lambda: _tiny_qwen3_model(
            monkeypatch,
            model_class=_UnqualifiedQwen3ForCausalLM,
            tie_word_embeddings=True,
        ),
        qualify_model=lambda model: mx_loading_mod.qualify_post_transform_profile(
            model,
            speculative_mode=None,
            loads_draft_weights=False,
        ),
        state_probes=(
            ("layout", _qwen3_layout_state),
            ("tied-embedding", _qwen3_tied_embedding_state),
        ),
        output_probes=(
            ("embedding-logits", _dense_embedding_logits),
            ("fused-qkv", _dense_fused_qkv_output),
            ("fused-gate-up", _dense_fused_gate_up_output),
        ),
        prepare_receiver=_receive_mx_catalog_into_blank_receiver,
    )

    producer, receiver = assert_post_transform_lifecycle_equivalent(case)

    decision = mx_loading_mod.qualify_post_transform_profile(
        receiver,
        speculative_mode=None,
        loads_draft_weights=False,
    )
    assert decision.transform_abi_id == QWEN3_DENSE_TIED_POST_TRANSFORM_LAYOUT_ABI_V1
    assert PostTransformRuntimeConfig.from_model_config(
        producer.model_config, model=producer
    ) == _bf16_dense_runtime_config(rope_fusion=False, tied_word_embeddings=True)
    assert _qwen3_layout_state(producer)["tied_lm_head"] is True
    assert _qwen3_tied_embedding_state(receiver) == {
        "same_view": True,
        "table_shape": (32, 16),
        "mx_catalog_names_lm_head": False,
    }


@pytest.mark.parametrize("rank", [0, 1])
def test_qwen3_dense_tied_profile_qualifies_tp2_rank_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
    rank: int,
) -> None:
    monkeypatch.setattr(mapping_mod, "mpi_disabled", lambda: False)
    monkeypatch.setattr(distributed_mod, "AllReduce", _AllReduceStub)
    case = PostTransformQualificationCase(
        profile_id="qwen3-for-causal-lm-bf16-tied-target-v1",
        model_factory=lambda: _tiny_qwen3_model(
            monkeypatch,
            tp_size=2,
            rank=rank,
            tie_word_embeddings=True,
        ),
        unqualified_model_factory=lambda: _tiny_qwen3_model(
            monkeypatch,
            model_class=_UnqualifiedQwen3ForCausalLM,
            tp_size=2,
            rank=rank,
            tie_word_embeddings=True,
        ),
        qualify_model=lambda model: mx_loading_mod.qualify_post_transform_profile(
            model,
            speculative_mode=None,
            loads_draft_weights=False,
        ),
        state_probes=(
            ("layout", _qwen3_layout_state),
            ("tied-embedding", _qwen3_tied_embedding_state),
        ),
        output_probes=(
            ("rank-local-lm-head-logits", _rank_local_lm_head_logits),
            ("fused-qkv", _dense_fused_qkv_output),
            ("fused-gate-up", _dense_fused_gate_up_output),
        ),
        prepare_receiver=_receive_mx_catalog_into_blank_receiver,
    )

    producer, receiver = assert_post_transform_lifecycle_equivalent(case)

    assert PostTransformRuntimeConfig.from_model_config(
        producer.model_config, model=producer
    ) == _bf16_dense_runtime_config(
        tp_size=2,
        moe_tp_size=2,
        attention_tp_size=2,
        rope_fusion=False,
        tied_word_embeddings=True,
    )
    # Each rank holds its vocab shard of the shared table once.
    assert _qwen3_tied_embedding_state(receiver) == {
        "same_view": True,
        "table_shape": (16, 16),
        "mx_catalog_names_lm_head": False,
    }


@pytest.mark.parametrize(
    "tp_size, rank",
    [
        pytest.param(1, 0, id="tp1"),
        pytest.param(2, 0, id="tp2-rank0"),
        pytest.param(2, 1, id="tp2-rank1"),
    ],
)
def test_mx_canonical_catalog_holds_tied_embedding_once(
    monkeypatch: pytest.MonkeyPatch,
    tp_size: int,
    rank: int,
) -> None:
    if tp_size > 1:
        monkeypatch.setattr(mapping_mod, "mpi_disabled", lambda: False)
        monkeypatch.setattr(distributed_mod, "AllReduce", _AllReduceStub)
    untied = _tiny_qwen3_model(monkeypatch, tp_size=tp_size, rank=rank)
    tied = _tiny_qwen3_model(
        monkeypatch,
        tp_size=tp_size,
        rank=rank,
        tie_word_embeddings=True,
    )

    untied_catalog = _mx_canonical_parameter_catalog(untied)
    tied_catalog = _mx_canonical_parameter_catalog(tied)

    # The shared table is registered once, under its first name; the tied
    # catalog is the untied one without the LM head, which is why tied
    # checkpoints carry their own layout ABI.
    assert set(untied_catalog) - set(tied_catalog) == {"lm_head.weight"}
    assert set(tied_catalog) < set(untied_catalog)
    assert tied_catalog["model.embed_tokens.weight"] == (
        tied.lm_head.weight.data_ptr(),
        (32 // tp_size, 16),
        torch.bfloat16,
    )


def test_staged_qwen3_tied_finalization_preserves_mx_tensor_catalog(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _tiny_qwen3_model(monkeypatch, tie_word_embeddings=True)

    # MX discovers and registers tensors after receiver alias preparation.
    model_hooks_mod.setup_aliases(model)
    registered_catalog = _mx_canonical_parameter_catalog(model)

    # This is the receiver-side finalization sequence after RDMA completes.
    model_hooks_mod.setup_aliases(model)
    model_hooks_mod.mark_weights_transformed(model)
    model_hooks_mod.walk_cache_state(model)
    published_catalog = _mx_canonical_parameter_catalog(model)

    assert published_catalog == registered_catalog
    assert "lm_head.weight" not in published_catalog
    assert published_catalog["model.embed_tokens.weight"][0] == model.lm_head.weight.data_ptr()


@pytest.mark.parametrize(
    "declared_tie",
    [
        pytest.param(True, id="declared-tie-not-realized"),
        pytest.param(False, id="realized-tie-not-declared"),
    ],
)
def test_qwen3_dense_profiles_reject_tie_the_model_does_not_realize(
    monkeypatch: pytest.MonkeyPatch,
    declared_tie: bool,
) -> None:
    model = _tiny_qwen3_model(monkeypatch, tie_word_embeddings=declared_tie)
    if declared_tie:
        model.lm_head.weight = nn.Parameter(model.model.embed_tokens.weight.detach().clone())
    else:
        model.lm_head.weight = model.model.embed_tokens.weight

    decision = mx_loading_mod.qualify_post_transform_profile(
        model,
        speculative_mode=None,
        loads_draft_weights=False,
    )

    assert not decision.qualified
    assert decision.reason is PostTransformQualificationReason.RUNTIME_CONFIG_NOT_SUPPORTED
    assert decision.unsupported_runtime_dimensions == frozenset({"tied_word_embeddings"})


def test_qwen3_tied_and_untied_source_identities_never_pair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identities = {}
    for tie_word_embeddings, transform_abi_id in (
        (False, QWEN3_DENSE_POST_TRANSFORM_LAYOUT_ABI_V1),
        (True, QWEN3_DENSE_TIED_POST_TRANSFORM_LAYOUT_ABI_V1),
    ):
        model = _tiny_qwen3_model(monkeypatch, tie_word_embeddings=tie_word_embeddings)
        decision = mx_loading_mod.qualify_post_transform_profile(
            model,
            speculative_mode=None,
            loads_draft_weights=False,
        )
        assert decision.transform_abi_id == transform_abi_id
        identities[tie_word_embeddings] = SourceIdentity.from_model_config(
            model.model_config,
            model,
            artifact_identity=_SOURCE_IDENTITY.artifact_identity,
            transform_abi_id=decision.transform_abi_id,
        )

    result = identities[True].matches(identities[False])
    assert {"model_fingerprint", "shard_fingerprint", "transform_abi_id"} <= set(
        result.mismatched_fields
    )
    # ENFORCE skips the config and shard fingerprints, so the layout ABI alone
    # keeps a tied source and an untied receiver apart.
    enforce = check_weight_sharing_compatibility(
        identities[False], identities[True], IdentityCheckPolicy.ENFORCE
    )
    assert not enforce.should_share
    assert enforce.match_result.mismatched_fields == ["transform_abi_id"]


class _CatalogCopyingMxLoader(_PostTransformMxLoader):
    """An MX receiver that copies a producer's catalog into the model it prepared."""

    def __init__(self, producer: nn.Module) -> None:
        super().__init__(post_transform=True)
        self._producer = producer

    def _load_weights(self, *args: object, **kwargs: object) -> dict[str, object]:
        weights = super()._load_weights(*args, **kwargs)
        if self._post_transform:
            receiver = cast(nn.Module, kwargs["model"])
            assert "lm_head.weight" not in _mx_canonical_parameter_catalog(receiver)
            _copy_mx_catalog(self._producer, receiver)
        return weights


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Materializes the model on CUDA")
def test_mx_staged_receiver_load_keeps_tied_qwen3_table_in_one_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    producer = _tiny_qwen3_model(monkeypatch, tie_word_embeddings=True).cuda()
    model_hooks_mod.walk_full_post_load(producer)

    real_source_identity = model_loader_mod.SourceIdentity.__dict__["from_model_config"]
    loader = _make_loader(monkeypatch, events=[])
    # Build the receiver's real SourceIdentity from the meta-initialized model.
    monkeypatch.setattr(model_loader_mod.SourceIdentity, "from_model_config", real_source_identity)
    loader._load_and_validate_config = MagicMock(
        return_value=ModelConfig(
            pretrained_config=_tiny_qwen3_config(tie_word_embeddings=True),
            mapping=mapping_mod.Mapping(world_size=1, rank=0, tp_size=1),
            max_num_tokens=16,
            max_seq_len=16,
        )
    )
    construction_devices = []

    def _construct(model_config: ModelConfig) -> nn.Module:
        model = modeling_qwen3_mod.Qwen3ForCausalLM(model_config)
        construction_devices.append(model.model.embed_tokens.weight.device.type)
        return model

    # Run the loader's real meta-init construction and memoized materialization.
    monkeypatch.setattr(memory_init_mod, "MetaInitMode", MetaInitMode)
    monkeypatch.setattr(memory_init_mod.AutoModelForCausalLM, "from_config", _construct)
    checkpoint_loader = _CatalogCopyingMxLoader(producer)

    model, _ = loader.load("/ckpt", checkpoint_loader)

    assert construction_devices == ["meta"]
    model_hooks_mod.call_load_weights.assert_not_called()
    assert checkpoint_loader.load_weights.call_args.kwargs["allow_post_transform_weights"] is True
    source_identity = checkpoint_loader.load_weights.call_args.kwargs["source_identity"]
    assert source_identity.transform_abi_id == QWEN3_DENSE_TIED_POST_TRANSFORM_LAYOUT_ABI_V1

    # Materialization may give each module its own Parameter object, so the tie
    # is checked through memory: one view of one allocation.
    head = model.lm_head.weight
    table = model.model.embed_tokens.weight
    assert head.is_cuda
    assert (head.data_ptr(), head.shape, head.stride()) == (
        table.data_ptr(),
        table.shape,
        table.stride(),
    )
    assert PostTransformRuntimeConfig.from_model_config(
        model.model_config, model=model
    ) == _bf16_dense_runtime_config(rope_fusion=False, tied_word_embeddings=True)

    parameters = dict(model.named_parameters(remove_duplicate=False))
    catalog = _mx_canonical_parameter_catalog(model)
    assert "lm_head.weight" not in catalog
    # Every parameter lives in an allocation that the catalog transfers, so the
    # head has no second copy of the table.
    assert {parameter.untyped_storage().data_ptr() for parameter in parameters.values()} == {
        parameters[name].untyped_storage().data_ptr() for name in catalog
    }

    with torch.no_grad():
        torch.testing.assert_close(
            _dense_embedding_logits(model),
            _dense_embedding_logits(producer),
            rtol=0,
            atol=0,
        )


def _qwen3_fp8_block_case(
    monkeypatch: pytest.MonkeyPatch,
    *,
    tp_size: int = 1,
    rank: int = 0,
) -> PostTransformQualificationCase:
    output_probes = _fp8_block_output_probes()
    if tp_size == 1:
        output_probes = (("embedding-logits", _dense_embedding_logits), *output_probes)
    return PostTransformQualificationCase(
        profile_id=_QWEN3_FP8_BLOCK_PROFILE_ID,
        model_factory=lambda: _tiny_qwen3_fp8_block_model(monkeypatch, tp_size=tp_size, rank=rank),
        unqualified_model_factory=lambda: _tiny_qwen3_fp8_block_model(
            monkeypatch,
            model_class=_UnqualifiedQwen3ForCausalLM,
            tp_size=tp_size,
            rank=rank,
        ),
        qualify_model=lambda model: mx_loading_mod.qualify_post_transform_profile(
            model,
            speculative_mode=None,
            loads_draft_weights=False,
        ),
        state_probes=(("fp8-block-layout", _qwen3_fp8_block_layout_state),),
        output_probes=output_probes,
    )


def _expected_qwen3_fp8_block_layout(
    *,
    qkv_rows: int,
    o_columns: int,
    gate_up_rows: int,
    down_columns: int,
) -> dict[str, object]:
    def _blocks(features: int) -> int:
        return -(-features // _FP8_BLOCK_SIZE)

    expected: dict[str, object] = {
        "attention_type": modeling_qwen3_mod.Qwen3Attention.__name__,
        "qkv_weight_mode": WeightMode.FUSED_QKV_LINEAR,
        "qkv_weight_shape": (qkv_rows, 128),
        "gate_up_weight_mode": WeightMode.FUSED_GATE_UP_LINEAR,
        "gate_up_weight_shape": (gate_up_rows, 128),
        "qkv_bias": False,
        "q_norm_weight_shape": (128,),
        "k_norm_weight_shape": (128,),
        "fuse_qk_norm_rope": True,
        "rope_fusion": False,
        "rotary_embedding_present": True,
        "tied_lm_head": False,
        "attention_has_quant_scale": True,
        "lm_head_weight_dtype": torch.bfloat16,
        "lm_head_has_weight_scale": False,
        "share_gate_up_quantization": (False, False),
    }
    shapes = {
        "qkv": (qkv_rows, 128),
        "o": (128, o_columns),
        "gate_up": (gate_up_rows, 128),
        "down": (128, down_columns),
    }
    for name, (out_features, in_features) in shapes.items():
        expected[f"{name}_quant_method"] = linear_mod.FP8BlockScalesLinearMethod.__name__
        expected[f"{name}_weight"] = (torch.float8_e4m3fn, (out_features, in_features))
        expected[f"{name}_weight_scale"] = (
            torch.float32,
            (_blocks(out_features), _blocks(in_features)),
        )
        expected[f"{name}_input_scales"] = ((torch.float32, ()), (torch.float32, ()))
        expected[f"{name}_relayout_parameters"] = ()
        expected[f"{name}_low_m_gemm"] = linear_mod.Fp8BlockScaleLowMGemmRange()
    return expected


def test_qwen3_fp8_block_profile_qualifies_full_staged_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _simulate_sm(monkeypatch, 90)

    producer, receiver = assert_post_transform_lifecycle_equivalent(
        _qwen3_fp8_block_case(monkeypatch)
    )

    decision = mx_loading_mod.qualify_post_transform_profile(
        receiver,
        speculative_mode=None,
        loads_draft_weights=False,
    )
    assert (
        decision.transform_abi_id == QWEN3_DENSE_FP8_BLOCK128_FP32SCALE_POST_TRANSFORM_LAYOUT_ABI_V1
    )
    assert (
        PostTransformRuntimeConfig.from_model_config(
            producer.model_config, model=producer, sm_version=90
        )
        == _fp8_block_runtime_config()
    )
    assert _qwen3_fp8_block_layout_state(producer) == _expected_qwen3_fp8_block_layout(
        qkv_rows=1024,
        o_columns=512,
        gate_up_rows=512,
        down_columns=256,
    )


@pytest.mark.parametrize("rank", [0, 1])
def test_qwen3_fp8_block_profile_qualifies_tp2_rank_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
    rank: int,
) -> None:
    _simulate_sm(monkeypatch, 90)
    monkeypatch.setattr(mapping_mod, "mpi_disabled", lambda: False)
    monkeypatch.setattr(distributed_mod, "AllReduce", _AllReduceStub)

    producer, _receiver = assert_post_transform_lifecycle_equivalent(
        _qwen3_fp8_block_case(monkeypatch, tp_size=2, rank=rank)
    )

    assert PostTransformRuntimeConfig.from_model_config(
        producer.model_config, model=producer, sm_version=90
    ) == _fp8_block_runtime_config(tp_size=2, moe_tp_size=2, attention_tp_size=2)
    # Per rank: q, k, and v hold 256, 128, and 128 rows, and gate and up 128
    # rows each, so every fused shard starts on a scale-block boundary.
    assert _qwen3_fp8_block_layout_state(producer) == _expected_qwen3_fp8_block_layout(
        qkv_rows=512,
        o_columns=256,
        gate_up_rows=256,
        down_columns=128,
    )


def test_mistral_dense_profile_qualifies_full_staged_lifecycle() -> None:
    case = PostTransformQualificationCase(
        profile_id="mistral-for-causal-lm-bf16-target-v1",
        model_factory=_tiny_mistral_model,
        unqualified_model_factory=lambda: _tiny_mistral_model(
            model_class=_UnqualifiedMistralForCausalLM
        ),
        qualify_model=lambda model: mx_loading_mod.qualify_post_transform_profile(
            model,
            speculative_mode=None,
            loads_draft_weights=False,
        ),
        state_probes=(("layout", _mistral_layout_state),),
        output_probes=(
            ("embedding-logits", _dense_embedding_logits),
            ("fused-qkv", _dense_fused_qkv_output),
            ("fused-gate-up", _dense_fused_gate_up_output),
        ),
    )

    producer, _receiver = assert_post_transform_lifecycle_equivalent(case)

    assert PostTransformRuntimeConfig.from_model_config(
        producer.model_config, model=producer
    ) == _bf16_dense_runtime_config(sliding_window="none")
    assert _mistral_layout_state(producer) == {
        "attention_type": modeling_mistral_mod.MistralAttention.__name__,
        "qkv_weight_mode": WeightMode.FUSED_QKV_LINEAR,
        "qkv_weight_shape": (32, 16),
        "gate_up_weight_mode": WeightMode.FUSED_GATE_UP_LINEAR,
        "gate_up_weight_shape": (64, 16),
        "qkv_bias": False,
        "rope_fusion": True,
        "rotary_embedding_present": False,
        "attention_window_size": None,
        "tied_lm_head": False,
    }


@pytest.mark.parametrize("rank", [0, 1])
def test_mistral_dense_profile_qualifies_tp2_rank_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
    rank: int,
) -> None:
    monkeypatch.setattr(mapping_mod, "mpi_disabled", lambda: False)
    monkeypatch.setattr(distributed_mod, "AllReduce", _AllReduceStub)
    case = PostTransformQualificationCase(
        profile_id="mistral-for-causal-lm-bf16-target-v1",
        model_factory=lambda: _tiny_mistral_model(tp_size=2, rank=rank),
        unqualified_model_factory=lambda: _tiny_mistral_model(
            model_class=_UnqualifiedMistralForCausalLM,
            tp_size=2,
            rank=rank,
        ),
        qualify_model=lambda model: mx_loading_mod.qualify_post_transform_profile(
            model,
            speculative_mode=None,
            loads_draft_weights=False,
        ),
        state_probes=(("layout", _mistral_layout_state),),
        output_probes=(
            ("fused-qkv", _dense_fused_qkv_output),
            ("fused-gate-up", _dense_fused_gate_up_output),
        ),
    )

    producer, _receiver = assert_post_transform_lifecycle_equivalent(case)

    assert PostTransformRuntimeConfig.from_model_config(
        producer.model_config, model=producer
    ) == _bf16_dense_runtime_config(
        tp_size=2,
        moe_tp_size=2,
        attention_tp_size=2,
        sliding_window="none",
    )
    assert _mistral_layout_state(producer)["qkv_weight_shape"] == (16, 16)
    assert _mistral_layout_state(producer)["gate_up_weight_shape"] == (32, 16)


@pytest.mark.parametrize(
    "sliding_window, layer_types, expected_window_sizes",
    [
        pytest.param(4096, None, (4096, 4096), id="uniform-window"),
        pytest.param(
            4096,
            ("sliding_attention", "full_attention"),
            (4096, None),
            id="layer-types-window",
        ),
    ],
)
def test_mistral_dense_profile_rejects_sliding_window_models(
    sliding_window: int,
    layer_types: tuple[str, ...] | None,
    expected_window_sizes: tuple[int | None, ...],
) -> None:
    model = _tiny_mistral_model(sliding_window=sliding_window, layer_types=layer_types)

    decision = mx_loading_mod.qualify_post_transform_profile(
        model,
        speculative_mode=None,
        loads_draft_weights=False,
    )

    assert (
        tuple(layer.self_attn.attention_window_size for layer in model.model.layers)
        == expected_window_sizes
    )
    assert not decision.qualified
    assert decision.reason is PostTransformQualificationReason.RUNTIME_CONFIG_NOT_SUPPORTED
    assert decision.unsupported_runtime_dimensions == frozenset({"sliding_window"})


def test_mistral_dense_profile_qualifies_full_attention_layer_types() -> None:
    model = _tiny_mistral_model(
        sliding_window=4096,
        layer_types=("full_attention", "full_attention"),
    )

    decision = mx_loading_mod.qualify_post_transform_profile(
        model,
        speculative_mode=None,
        loads_draft_weights=False,
    )

    assert all(layer.self_attn.attention_window_size is None for layer in model.model.layers)
    assert decision.qualified
    assert decision.profile is not None
    assert decision.profile.profile_id == "mistral-for-causal-lm-bf16-target-v1"


def _qwen3_moe_profile() -> PostTransformProfile:
    (profile,) = (
        candidate
        for candidate in mx_loading_mod.MX_POST_TRANSFORM_PROFILE_REGISTRY.profiles
        if candidate.profile_id == _QWEN3_MOE_PROFILE_ID
    )
    return profile


def _qualify_qwen3_moe_model(model: nn.Module) -> PostTransformQualificationDecision:
    return mx_loading_mod.qualify_post_transform_profile(
        model,
        speculative_mode=None,
        loads_draft_weights=False,
    )


@_requires_cutlass_bf16_moe
def test_qwen3_moe_profile_qualifies_full_staged_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A staged receiver matches a producer that ran the real Hugging Face load path.

    The producer loads a checkpoint-format state dict through the production
    weight mapper and then runs every post-load hook; the receiver only gets
    the producer's parameters and the staged finalization walks. The
    routed-expert output has its own SM90+ case,
    `test_qwen3_moe_staged_receiver_matches_routed_expert_output`.
    """
    state_dict = _qwen3_moe_hf_state_dict()
    prepare_producer, factory_parameters = keep_factory_parameters(
        _qwen3_moe_hf_checkpoint_loader(state_dict)
    )
    case = PostTransformQualificationCase(
        profile_id=_QWEN3_MOE_PROFILE_ID,
        model_factory=lambda: _tiny_qwen3_moe_model(monkeypatch),
        unqualified_model_factory=lambda: _tiny_qwen3_moe_model(
            monkeypatch,
            model_class=_UnqualifiedQwen3MoeForCausalLM,
        ),
        qualify_model=_qualify_qwen3_moe_model,
        state_probes=(
            ("layout", _qwen3_moe_layout_state),
            ("module-state", module_state_snapshot),
        ),
        output_probes=(
            ("embedding-logits", _dense_embedding_logits),
            ("fused-qkv", _dense_fused_qkv_output),
            ("router-logits", _qwen3_moe_router_logits),
        ),
        prepare_producer=prepare_producer,
    )

    producer, receiver = assert_post_transform_lifecycle_equivalent(case)

    assert_every_parameter_was_loaded(producer, factory_parameters)
    # The producer's bytes come from the checkpoint through the weight mapper.
    assert torch.equal(
        producer.model.embed_tokens.weight.cpu(), state_dict["model.embed_tokens.weight"]
    )
    assert torch.equal(producer.lm_head.weight.cpu(), state_dict["lm_head.weight"])
    producer_backend = producer.model.layers[0].mlp.experts.backend
    for expert in range(_QWEN3_MOE_NUM_EXPERTS):
        prefix = f"model.layers.0.mlp.experts.{expert}."
        # `w3_w1_weight` stacks the up projection before the gate projection.
        assert torch.equal(
            producer_backend.w3_w1_weight[expert].cpu(),
            torch.cat(
                [state_dict[prefix + "up_proj.weight"], state_dict[prefix + "gate_proj.weight"]]
            ),
        )
        assert torch.equal(
            producer_backend.w2_weight[expert].cpu(), state_dict[prefix + "down_proj.weight"]
        )

    decision = _qualify_qwen3_moe_model(receiver)
    assert decision.transform_abi_id == QWEN3_MOE_CUTLASS_POST_TRANSFORM_LAYOUT_ABI_V1
    assert (
        PostTransformRuntimeConfig.from_model_config(producer.model_config, model=producer)
        == _qwen3_moe_runtime_config()
    )
    # The receiver marks the backend behind the `_weights_removed` wrapper
    # without loading, which a later transform walk must not undo.
    backend = receiver.model.layers[0].mlp.experts.backend
    assert backend._weights_transformed is True
    config = _qwen3_moe_hf_config()
    hidden_size, head_dim = config.hidden_size, config.head_dim
    intermediate_size = config.moe_intermediate_size
    qkv_rows = (config.num_attention_heads + 2 * config.num_key_value_heads) * head_dim
    assert _qwen3_moe_layout_state(producer) == {
        "attention_type": modeling_qwen3_mod.Qwen3Attention.__name__,
        "qkv_weight_mode": WeightMode.FUSED_QKV_LINEAR,
        "qkv_weight_shape": (qkv_rows, hidden_size),
        "q_norm_weight_shape": (head_dim,),
        "k_norm_weight_shape": (head_dim,),
        "fuse_qk_norm_rope": True,
        "rope_fusion": False,
        "router_weight_shape": (_QWEN3_MOE_NUM_EXPERTS, hidden_size),
        "router_weight_dtype": torch.bfloat16,
        "experts_type": ConfigurableMoE.__name__,
        "backend_type": TrtllmCutlassUnquantizedImpl.__name__,
        "backend_id": "trtllm.cutlass.grouped_gemm.none",
        "quant_method_type": UnquantizedFusedMoEMethod.__name__,
        "quant_scales": (),
        "weight_loading_mode": MoEWeightLoadingMode.VANILLA,
        "routing_method_type": "RenormalizeMoeRoutingMethod",
        "routing_scales_dtype": torch.float32,
        "w3_w1_weight_shape": (_QWEN3_MOE_NUM_EXPERTS, 2 * intermediate_size, hidden_size),
        "w2_weight_shape": (_QWEN3_MOE_NUM_EXPERTS, hidden_size, intermediate_size),
        "expert_size_per_partition": _QWEN3_MOE_NUM_EXPERTS,
        "initial_local_expert_ids": tuple(range(_QWEN3_MOE_NUM_EXPERTS)),
        "layer_load_balancer": None,
        "layer0_next_norm": True,
        "layer1_next_norm": True,
        "tied_lm_head": False,
    }


# The admitted TP2 partitions as `(MoE EP size, MoE TP size)`; `moe_ep_size=-1`
# leaves MoE TP equal to TP.
_QWEN3_MOE_TP2_PARTITIONS = (
    pytest.param(-1, 2, id="moe-tp2"),
    pytest.param(2, 1, id="moe-ep2"),
)


@_requires_cutlass_bf16_moe
@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("moe_ep_size, moe_tp_size", _QWEN3_MOE_TP2_PARTITIONS)
def test_qwen3_moe_profile_qualifies_tp2_rank_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
    rank: int,
    moe_ep_size: int,
    moe_tp_size: int,
) -> None:
    """Each simulated TP2 rank's staged receiver matches its producer that ran the real load."""
    prepare_producer, factory_parameters = keep_factory_parameters(
        _qwen3_moe_hf_checkpoint_loader(_qwen3_moe_hf_state_dict())
    )
    case = PostTransformQualificationCase(
        profile_id=_QWEN3_MOE_PROFILE_ID,
        model_factory=lambda: _tiny_qwen3_moe_model(
            monkeypatch,
            tp_size=2,
            rank=rank,
            moe_ep_size=moe_ep_size,
        ),
        unqualified_model_factory=lambda: _tiny_qwen3_moe_model(
            monkeypatch,
            model_class=_UnqualifiedQwen3MoeForCausalLM,
            tp_size=2,
            rank=rank,
            moe_ep_size=moe_ep_size,
        ),
        qualify_model=_qualify_qwen3_moe_model,
        state_probes=(
            ("layout", _qwen3_moe_layout_state),
            ("module-state", module_state_snapshot),
        ),
        output_probes=(
            ("fused-qkv", _dense_fused_qkv_output),
            ("router-logits", _qwen3_moe_router_logits),
        ),
        prepare_producer=prepare_producer,
    )

    producer, _receiver = assert_post_transform_lifecycle_equivalent(case)

    assert_every_parameter_was_loaded(producer, factory_parameters)
    resolved_moe_ep_size = 2 // moe_tp_size
    assert PostTransformRuntimeConfig.from_model_config(
        producer.model_config, model=producer
    ) == _qwen3_moe_runtime_config(
        tp_size=2,
        moe_tp_size=moe_tp_size,
        moe_ep_size=resolved_moe_ep_size,
        attention_tp_size=2,
    )
    config = _qwen3_moe_hf_config()
    hidden_size, head_dim = config.hidden_size, config.head_dim
    local_experts = _QWEN3_MOE_NUM_EXPERTS // resolved_moe_ep_size
    local_intermediate_size = config.moe_intermediate_size // moe_tp_size
    # Attention TP splits the query and key/value heads over the two ranks.
    local_qkv_rows = (config.num_attention_heads + 2 * config.num_key_value_heads) * head_dim // 2
    layout = _qwen3_moe_layout_state(producer)
    assert layout["qkv_weight_shape"] == (local_qkv_rows, hidden_size)
    assert layout["router_weight_shape"] == (_QWEN3_MOE_NUM_EXPERTS, hidden_size)
    assert layout["w3_w1_weight_shape"] == (
        local_experts,
        2 * local_intermediate_size,
        hidden_size,
    )
    assert layout["w2_weight_shape"] == (local_experts, hidden_size, local_intermediate_size)
    expert_rank = rank if resolved_moe_ep_size > 1 else 0
    assert layout["initial_local_expert_ids"] == tuple(
        range(expert_rank * local_experts, (expert_rank + 1) * local_experts)
    )


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
def test_qwen3_moe_staged_receiver_matches_routed_expert_output(
    monkeypatch: pytest.MonkeyPatch,
    tp_size: int,
    rank: int,
    moe_ep_size: int,
) -> None:
    """A staged receiver's routed experts compute exactly what its producer's compute.

    The other Qwen3 MoE lifecycle cases compare tensors, state, guards, and
    the dense outputs on every SM80+ GPU. This case adds the routed-expert
    output, which this fixture validates only on SM90 and newer.
    """

    def factory(
        model_class: type[nn.Module] = modeling_qwen3_moe_mod.Qwen3MoeForCausalLM,
    ) -> nn.Module:
        return _tiny_qwen3_moe_model(
            monkeypatch,
            model_class=model_class,
            tp_size=tp_size,
            rank=rank,
            moe_ep_size=moe_ep_size,
        )

    case = PostTransformQualificationCase(
        profile_id=_QWEN3_MOE_PROFILE_ID,
        model_factory=factory,
        unqualified_model_factory=lambda: factory(_UnqualifiedQwen3MoeForCausalLM),
        qualify_model=_qualify_qwen3_moe_model,
        state_probes=(("layout", _qwen3_moe_layout_state),),
        output_probes=(("routed-experts", _qwen3_moe_routed_expert_output),),
        prepare_producer=_qwen3_moe_hf_checkpoint_loader(_qwen3_moe_hf_state_dict()),
    )

    assert_post_transform_lifecycle_equivalent(case)


@_requires_cutlass_bf16_moe
def test_staged_qwen3_moe_finalization_preserves_mx_tensor_catalog(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Routed-expert tensors behind the removed wrapper stay in the MX catalog.

    `ConfigurableMoE` is `_weights_removed`, so loader walks skip it, but the
    expert weights live in its backend child and must be registered for
    transfer and keep their storage through staged finalization.
    """
    model = _tiny_qwen3_moe_model(monkeypatch)

    model_hooks_mod.setup_aliases(model)
    registered_catalog = _mx_canonical_parameter_catalog(model)

    model_hooks_mod.setup_aliases(model)
    model_hooks_mod.mark_weights_transformed(model)
    model_hooks_mod.walk_cache_state(model)
    published_catalog = _mx_canonical_parameter_catalog(model)

    assert published_catalog == registered_catalog
    for layer_index in range(2):
        prefix = f"model.layers.{layer_index}.mlp."
        assert {
            f"{prefix}gate.weight",
            f"{prefix}experts.backend.w3_w1_weight",
            f"{prefix}experts.backend.w2_weight",
        } <= registered_catalog.keys()


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "overrides, expected_dimensions",
    [
        pytest.param({"moe_backend": "TRTLLM"}, {"moe_backend"}, id="trtllm-moe-backend"),
        pytest.param({"moe_backend": "CUTEDSL"}, {"moe_backend"}, id="cutedsl-moe-backend"),
        pytest.param({"moe_backend": "DEEPGEMM"}, {"moe_backend"}, id="deepgemm-moe-backend"),
        pytest.param({"moe_backend": "TRITON"}, {"moe_backend"}, id="triton-moe-backend"),
        pytest.param({"moe_backend": "MARLIN"}, {"moe_backend"}, id="marlin-moe-backend"),
        pytest.param({"moe_backend": "VANILLA"}, {"moe_backend"}, id="vanilla-moe-backend"),
        pytest.param(
            {"tp_size": 4, "moe_tp_size": 4, "attention_tp_size": 4},
            {"tp_size", "moe_tp_size", "attention_tp_size"},
            id="moe-tp4",
        ),
        pytest.param(
            {"tp_size": 4, "moe_tp_size": 1, "moe_ep_size": 4, "attention_tp_size": 4},
            {"tp_size", "moe_ep_size", "attention_tp_size"},
            id="moe-ep4",
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
        pytest.param(
            {"quant_algorithm": "FP8_BLOCK_SCALES"},
            {"quant_algorithm"},
            id="fp8-block-scales-weights",
        ),
    ],
)
def test_qwen3_moe_profile_rejects_unqualified_moe_variants(
    overrides: dict[str, object],
    expected_dimensions: set[str],
) -> None:
    """The Qwen3 MoE profile rejects other MoE backends, wider partitions, and FP8 block scales.

    `test_bf16_profiles_reject_unqualified_runtime_variants` covers the
    variants that every BF16 profile rejects.
    """
    decision = mx_loading_mod.MX_POST_TRANSFORM_PROFILE_REGISTRY.qualify(
        root_model_class=modeling_qwen3_moe_mod.Qwen3MoeForCausalLM,
        architecture="Qwen3MoeForCausalLM",
        model_type="qwen3_moe",
        speculative_mode=None,
        protocol_version=mx_loading_mod.MX_STAGED_RECEIVER_TRANSFORM_PROTOCOL_VERSION,
        transfer_scope=PostTransformTransferScope.TARGET_MODEL,
        runtime_config=_qwen3_moe_runtime_config(**overrides),
    )

    assert not decision.qualified
    assert decision.reason is PostTransformQualificationReason.RUNTIME_CONFIG_NOT_SUPPORTED
    assert decision.unsupported_runtime_dimensions == frozenset(expected_dimensions)


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
@pytest.mark.parametrize(
    "feature",
    [PostTransformFeature.MOE_LOAD_BALANCER, PostTransformFeature.DWDP],
)
def test_qwen3_moe_profile_rejects_unsupported_moe_features(
    overrides: dict[str, object],
    feature: PostTransformFeature,
) -> None:
    def qualify(
        enabled_features: frozenset[PostTransformFeature],
    ) -> PostTransformQualificationDecision:
        return mx_loading_mod.MX_POST_TRANSFORM_PROFILE_REGISTRY.qualify(
            root_model_class=modeling_qwen3_moe_mod.Qwen3MoeForCausalLM,
            architecture="Qwen3MoeForCausalLM",
            model_type="qwen3_moe",
            speculative_mode=None,
            protocol_version=mx_loading_mod.MX_STAGED_RECEIVER_TRANSFORM_PROTOCOL_VERSION,
            transfer_scope=PostTransformTransferScope.TARGET_MODEL,
            enabled_features=enabled_features,
            runtime_config=_qwen3_moe_runtime_config(**overrides),
        )

    qualified = qualify(frozenset())
    assert qualified.qualified
    assert qualified.profile is _qwen3_moe_profile()

    decision = qualify(frozenset({feature}))
    assert not decision.qualified
    assert decision.reason is PostTransformQualificationReason.FEATURE_NOT_SUPPORTED
    assert decision.unsupported_features == frozenset({feature})


@pytest.mark.cpu_only
@pytest.mark.parametrize("sm", [80, 89, 90, 100, 103, 107, 120, 121])
def test_qwen3_moe_auto_backend_resolves_to_qualified_cutlass(
    monkeypatch: pytest.MonkeyPatch, sm: int
) -> None:
    """`AUTO` resolves to the pinned literal for BF16 Qwen3 MoE on every SM."""

    def in_sm_family(low: int, high: int) -> Callable[[int | None], bool]:
        return lambda sm_version=None: low <= (sm if sm_version is None else sm_version) < high

    # `resolve_moe_backend` reads the SM through these names of its module.
    monkeypatch.setattr(model_config_mod, "get_sm_version", lambda: sm)
    monkeypatch.setattr(model_config_mod, "is_sm_100f", in_sm_family(100, 110))
    monkeypatch.setattr(model_config_mod, "is_sm_120f", in_sm_family(120, 130))
    # `OVERRIDE_QUANT_ALGO=NVFP4_PLUS` would resolve AUTO to CUTEDSL on every SM.
    monkeypatch.delenv("OVERRIDE_QUANT_ALGO", raising=False)

    resolved = ModelConfig.resolve_moe_backend("AUTO", "Qwen3MoeForCausalLM", QuantConfig())

    assert resolved == "CUTLASS"
    assert _qwen3_moe_profile().runtime_constraints.moe_backends == frozenset({resolved})


@pytest.mark.cpu_only
@pytest.mark.parametrize("sm", [80, 89, 90, 100, 103, 107, 120, 121])
@pytest.mark.parametrize(
    "mapping_kwargs",
    [
        pytest.param({}, id="tp1"),
        pytest.param({"world_size": 2, "tp_size": 2}, id="moe-tp2"),
        pytest.param({"world_size": 2, "tp_size": 2, "moe_ep_size": 2}, id="moe-ep2"),
    ],
)
def test_moe_profile_backend_literals_realize_one_implementation(
    mapping_kwargs: dict[str, int], sm: int
) -> None:
    """Each pinned MoE backend literal resolves to exactly one implementation.

    MoE profiles pin the configured literal and the model quantization, not the
    realized implementation. That is exact only while one implementation can
    serve the pinned format under the literal, on every SM and admitted
    partition, so neither a degraded resolution nor a newer architecture
    realizes a different expert layout. A family that gains a second
    implementation for that format needs a realized-implementation dimension
    first.
    """
    moe_profiles = [
        profile
        for profile in mx_loading_mod.MX_POST_TRANSFORM_PROFILE_REGISTRY.profiles
        if profile.runtime_constraints.moe_backends is not None
    ]
    assert [profile.profile_id for profile in moe_profiles] == [_QWEN3_MOE_PROFILE_ID], (
        "extend this test with the new MoE profile's model configuration"
    )
    constraints = moe_profiles[0].runtime_constraints
    assert constraints.quant_algorithms == frozenset({"none"})
    for moe_backend in constraints.moe_backends:
        model_config = ModelConfig(
            # The Qwen3-30B-A3B expert shapes: 128 experts, top-8, MoE intermediate size 768.
            pretrained_config=_qwen3_moe_hf_config(
                hidden_size=2048,
                moe_intermediate_size=768,
                num_experts=128,
                num_experts_per_tok=8,
            ),
            mapping=mapping_mod.Mapping(**mapping_kwargs),
            moe_backend=moe_backend,
        )
        # Every optional dependency is present, so a candidate that only a missing
        # wheel keeps out of the running still counts.
        environment = MoEEnvironment(sm=sm, available_deps=tuple(dep.value for dep in MoEDep))
        with override_moe_environment(environment):
            report = resolve_moe_impl(model_config)

        assert report.eligible == (TrtllmCutlassUnquantizedImpl.__name__,)
        assert report.winner == TrtllmCutlassUnquantizedImpl.__name__
        assert not report.degraded


@pytest.mark.cpu_only
def test_qwen3_vl_moe_root_does_not_inherit_qwen3_moe_profile() -> None:
    vl_root = get_registered_model_class("Qwen3VLMoeForConditionalGeneration")
    assert vl_root is not None
    assert vl_root is not modeling_qwen3_moe_mod.Qwen3MoeForCausalLM
    assert (
        get_registered_model_class("Qwen3MoeForCausalLM")
        is modeling_qwen3_moe_mod.Qwen3MoeForCausalLM
    )

    decision = mx_loading_mod.MX_POST_TRANSFORM_PROFILE_REGISTRY.qualify(
        root_model_class=vl_root,
        architecture="Qwen3MoeForCausalLM",
        model_type="qwen3_moe",
        speculative_mode=None,
        protocol_version=mx_loading_mod.MX_STAGED_RECEIVER_TRANSFORM_PROTOCOL_VERSION,
        transfer_scope=PostTransformTransferScope.TARGET_MODEL,
        runtime_config=_qwen3_moe_runtime_config(),
    )

    assert not decision.qualified
    assert decision.reason is PostTransformQualificationReason.ROOT_MODEL_CLASS_NOT_REGISTERED


@pytest.mark.cpu_only
def test_moe_profiles_pin_partition_and_one_backend_per_abi() -> None:
    """Registry invariants that keep every MoE profile fail-closed.

    A MoE profile pins exactly one MoE backend (routed-expert layouts differ per
    backend), the MoE TP and EP sizes, and `moe_cluster_size=1`, and supports
    neither EPLB nor DWDP; each transform-layout ABI belongs to one root and MoE
    backend. Its TP stays at most 2, where independent MoE TP and EP sets admit
    only the pure MoE-TP and MoE-EP partitions. Dense profiles leave every
    MoE-only dimension unconstrained. A profile counts as MoE here when it pins
    `moe_backends`; `assert_post_transform_lifecycle_equivalent` requires those
    pins for every profile whose model holds routed experts, so a MoE profile
    cannot pass as dense.
    """
    abi_owners: dict[str, tuple[str, str]] = {}
    moe_profiles = 0
    for profile in mx_loading_mod.MX_POST_TRANSFORM_PROFILE_REGISTRY.profiles:
        constraints = profile.runtime_constraints
        if constraints.moe_backends is None:
            assert constraints.moe_tp_sizes is None
            assert constraints.moe_ep_sizes is None
            assert constraints.moe_cluster_sizes is None
            continue
        moe_profiles += 1
        (moe_backend,) = constraints.moe_backends
        assert constraints.moe_tp_sizes is not None
        assert constraints.moe_ep_sizes is not None
        assert constraints.moe_cluster_sizes == frozenset({1})
        assert constraints.tp_sizes is not None and max(constraints.tp_sizes) <= 2
        assert PostTransformFeature.MOE_LOAD_BALANCER not in profile.supported_features
        assert PostTransformFeature.DWDP not in profile.supported_features
        owner = (profile.root_model_class.class_name, moe_backend)
        assert abi_owners.setdefault(profile.transform_abi_id, owner) == owner
    assert moe_profiles


@pytest.mark.cpu_only
def test_legacy_llama_file_mistral_root_is_not_qualified() -> None:
    legacy_root = modeling_llama_mod.MistralForCausalLM
    assert legacy_root is not modeling_mistral_mod.MistralForCausalLM
    assert (
        get_registered_model_class("MistralForCausalLM") is modeling_mistral_mod.MistralForCausalLM
    )

    decision = mx_loading_mod.MX_POST_TRANSFORM_PROFILE_REGISTRY.qualify(
        root_model_class=legacy_root,
        architecture="MistralForCausalLM",
        model_type="mistral",
        speculative_mode=None,
        protocol_version=mx_loading_mod.MX_STAGED_RECEIVER_TRANSFORM_PROTOCOL_VERSION,
        transfer_scope=PostTransformTransferScope.TARGET_MODEL,
        runtime_config=_bf16_dense_runtime_config(sliding_window="none"),
    )

    assert not decision.qualified
    assert decision.reason is PostTransformQualificationReason.ROOT_MODEL_CLASS_NOT_REGISTERED


@pytest.mark.cpu_only
def test_mistral_dense_profile_rejects_native_format_model_type() -> None:
    decision = mx_loading_mod.MX_POST_TRANSFORM_PROFILE_REGISTRY.qualify(
        root_model_class=modeling_mistral_mod.MistralForCausalLM,
        architecture="MistralForCausalLM",
        model_type="mistral_common",
        speculative_mode=None,
        protocol_version=mx_loading_mod.MX_STAGED_RECEIVER_TRANSFORM_PROTOCOL_VERSION,
        transfer_scope=PostTransformTransferScope.TARGET_MODEL,
        runtime_config=_bf16_dense_runtime_config(sliding_window="none"),
    )

    assert not decision.qualified
    assert decision.reason is PostTransformQualificationReason.MODEL_TYPE_NOT_REGISTERED


def test_phi3_dense_profile_qualifies_full_staged_lifecycle() -> None:
    """A TP1 Phi3 staged receiver matches the full post-load producer and qualifies."""
    case = PostTransformQualificationCase(
        profile_id="phi3-for-causal-lm-bf16-target-v1",
        model_factory=_tiny_phi3_model,
        unqualified_model_factory=lambda: _tiny_phi3_model(model_class=_UnqualifiedPhi3ForCausalLM),
        qualify_model=lambda model: mx_loading_mod.qualify_post_transform_profile(
            model,
            speculative_mode=None,
            loads_draft_weights=False,
        ),
        state_probes=(("layout", _phi3_layout_state),),
        output_probes=(
            ("embedding-logits", _dense_embedding_logits),
            ("fused-qkv", _dense_fused_qkv_output),
            ("fused-gate-up", _dense_fused_gate_up_output),
        ),
    )

    producer, _receiver = assert_post_transform_lifecycle_equivalent(case)

    decision = mx_loading_mod.qualify_post_transform_profile(
        producer,
        speculative_mode=None,
        loads_draft_weights=False,
    )
    assert decision.transform_abi_id == PHI3_DENSE_POST_TRANSFORM_LAYOUT_ABI_V1
    assert (
        PostTransformRuntimeConfig.from_model_config(producer.model_config, model=producer)
        == _bf16_dense_runtime_config()
    )
    assert _phi3_layout_state(producer) == {
        "attention_type": modeling_phi3_mod.Phi3Attention.__name__,
        "qkv_weight_mode": WeightMode.FUSED_QKV_LINEAR,
        "qkv_weight_shape": (32, 16),
        "gate_up_weight_mode": WeightMode.FUSED_GATE_UP_LINEAR,
        "gate_up_weight_shape": (64, 16),
        "qkv_bias": False,
        "rope_fusion": True,
        "rotary_embedding_present": False,
        "share_gate_up_quantize": (False, False),
        "tied_lm_head": False,
    }


@pytest.mark.parametrize("rank", [0, 1])
def test_phi3_dense_profile_qualifies_tp2_rank_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
    rank: int,
) -> None:
    """Each TP2 rank's Phi3 staged receiver matches its full post-load producer."""
    monkeypatch.setattr(mapping_mod, "mpi_disabled", lambda: False)
    monkeypatch.setattr(distributed_mod, "AllReduce", _AllReduceStub)
    case = PostTransformQualificationCase(
        profile_id="phi3-for-causal-lm-bf16-target-v1",
        model_factory=lambda: _tiny_phi3_model(tp_size=2, rank=rank),
        unqualified_model_factory=lambda: _tiny_phi3_model(
            model_class=_UnqualifiedPhi3ForCausalLM,
            tp_size=2,
            rank=rank,
        ),
        qualify_model=lambda model: mx_loading_mod.qualify_post_transform_profile(
            model,
            speculative_mode=None,
            loads_draft_weights=False,
        ),
        state_probes=(("layout", _phi3_layout_state),),
        output_probes=(
            ("fused-qkv", _dense_fused_qkv_output),
            ("fused-gate-up", _dense_fused_gate_up_output),
        ),
    )

    producer, _receiver = assert_post_transform_lifecycle_equivalent(case)

    assert producer.model_config.mapping.tp_rank == rank
    assert PostTransformRuntimeConfig.from_model_config(
        producer.model_config, model=producer
    ) == _bf16_dense_runtime_config(
        tp_size=2,
        moe_tp_size=2,
        attention_tp_size=2,
    )
    assert _phi3_layout_state(producer)["qkv_weight_shape"] == (16, 16)
    assert _phi3_layout_state(producer)["gate_up_weight_shape"] == (32, 16)


def test_phi3_dense_profile_rejects_unregistered_subclass_root() -> None:
    """An otherwise identical subclass must not inherit the exact-root profile."""
    decision = mx_loading_mod.qualify_post_transform_profile(
        _tiny_phi3_model(model_class=_UnqualifiedPhi3ForCausalLM),
        speculative_mode=None,
        loads_draft_weights=False,
    )

    assert not decision.qualified
    assert decision.reason is PostTransformQualificationReason.ROOT_MODEL_CLASS_NOT_REGISTERED


def test_phi3_dense_profile_realizes_configured_window_as_full_attention() -> None:
    """A configured `sliding_window` still realizes full attention and qualifies.

    Phi-3 4k checkpoints configure `sliding_window`, but the Phi3 root runs full
    attention, so their realized runtime matches the qualified profile.
    """
    model = _tiny_phi3_model(sliding_window=8)

    decision = mx_loading_mod.qualify_post_transform_profile(
        model,
        speculative_mode=None,
        loads_draft_weights=False,
    )

    assert model.config.sliding_window == 8
    assert (
        PostTransformRuntimeConfig.from_model_config(model.model_config, model=model)
        == _bf16_dense_runtime_config()
    )
    assert decision.qualified
    assert decision.profile is not None
    assert decision.profile.profile_id == "phi3-for-causal-lm-bf16-target-v1"


@pytest.mark.parametrize(
    "variant, expected_dimension",
    [
        pytest.param(
            # Phi-3 128k, Phi-3.5-mini, and Phi-4-mini checkpoints use LongRoPE.
            {
                "rope_scaling": {
                    "rope_type": "longrope",
                    "short_factor": [1.0, 1.0],
                    "long_factor": [2.0, 2.0],
                }
            },
            "rope_type",
            id="longrope",
        ),
        pytest.param({"tie_word_embeddings": True}, "tied_word_embeddings", id="tied-embeddings"),
    ],
)
def test_phi3_dense_profile_rejects_longrope_and_tied_embedding_variants(
    variant: dict[str, object],
    expected_dimension: str,
) -> None:
    """LongRoPE and tied-embedding Phi3 variants fall outside the profile's constraints."""
    model = _tiny_phi3_model(**variant)

    decision = mx_loading_mod.qualify_post_transform_profile(
        model,
        speculative_mode=None,
        loads_draft_weights=False,
    )

    assert not decision.qualified
    assert decision.reason is PostTransformQualificationReason.RUNTIME_CONFIG_NOT_SUPPORTED
    assert decision.unsupported_runtime_dimensions == frozenset({expected_dimension})


def test_phi3_runtime_max_seq_len_reaches_model_fingerprint() -> None:
    """Known limitation: Phi3's runtime `max_seq_len` reaches the model fingerprint.

    `Phi3Attention.__init__` in `tensorrt_llm/_torch/models/modeling_phi3.py`
    writes the runtime `max_seq_len` into the shared pretrained config for
    LongRoPE, and `SourceIdentity` hashes that config. Donors and receivers with
    different `max_seq_len` values therefore differ in `model_fingerprint`, even
    though no default-RoPE weight depends on the value. This test pins current
    behavior, not a contract: once the model code stops recording the value,
    update it together with the limitation note in
    `docs/source/features/model-express.md`.
    """
    identities = []
    for max_seq_len in (16, 32):
        model = _tiny_phi3_model(max_seq_len=max_seq_len)
        assert model.config.max_seq_len == max_seq_len
        identities.append(
            SourceIdentity.from_model_config(
                model.model_config,
                model,
                artifact_identity=_SOURCE_IDENTITY.artifact_identity,
                transform_abi_id=PHI3_DENSE_POST_TRANSFORM_LAYOUT_ABI_V1,
            )
        )

    decision = check_weight_sharing_compatibility(*identities, IdentityCheckPolicy.WARN_FALLBACK)

    assert decision.match_result.mismatched_fields == ["model_fingerprint"]


@pytest.mark.cpu_only
def test_phi4mm_root_does_not_inherit_phi3_profile() -> None:
    """The Phi4MM root does not qualify through its nested Phi3 language model.

    Phi4MM builds a nested `Phi3ForCausalLM` language model under its own root,
    and qualification matches the outer root class.
    """
    phi4mm_root = get_registered_model_class("Phi4MMForCausalLM")
    assert phi4mm_root is not None

    decision = mx_loading_mod.MX_POST_TRANSFORM_PROFILE_REGISTRY.qualify(
        root_model_class=phi4mm_root,
        architecture="Phi3ForCausalLM",
        model_type="phi3",
        speculative_mode=None,
        protocol_version=mx_loading_mod.MX_STAGED_RECEIVER_TRANSFORM_PROTOCOL_VERSION,
        transfer_scope=PostTransformTransferScope.TARGET_MODEL,
        runtime_config=_bf16_dense_runtime_config(),
    )

    assert not decision.qualified
    assert decision.reason is PostTransformQualificationReason.ROOT_MODEL_CLASS_NOT_REGISTERED


# Runtime variants that no BF16 dense profile qualifies, with the dimension each
# one is rejected on.
_UNQUALIFIED_RUNTIME_VARIANTS = [
    pytest.param({"dtype": "float16"}, {"dtype"}, id="fp16"),
    pytest.param(
        {"quant_algorithm": "FP8"},
        {"quant_algorithm"},
        id="weight-quantization",
    ),
    pytest.param(
        {"kv_cache_quant_algorithm": "FP8"},
        {"kv_cache_quant_algorithm"},
        id="kv-cache-quantization",
    ),
    pytest.param(
        {"layerwise_quantization": True},
        {"layerwise_quantization"},
        id="layerwise-quantization",
    ),
    pytest.param(
        {"force_dynamic_quantization": True},
        {"force_dynamic_quantization"},
        id="dynamic-quantization",
    ),
    pytest.param({"lora_enabled": True}, {"lora_enabled"}, id="lora"),
    pytest.param(
        {"sparse_attention_enabled": True},
        {"sparse_attention_enabled"},
        id="sparse-attention",
    ),
    pytest.param(
        {"attention_backend": "FLASHINFER"},
        {"attention_backend"},
        id="attention-backend",
    ),
    pytest.param({"tp_size": 4}, {"tp_size"}, id="tp4"),
    pytest.param({"pp_size": 2}, {"pp_size"}, id="pipeline-parallel"),
    pytest.param({"cp_size": 2}, {"cp_size"}, id="context-parallel"),
    pytest.param(
        {"attention_tp_size": 4},
        {"attention_tp_size"},
        id="attention-tensor-parallel",
    ),
    pytest.param(
        {"attention_cp_size": 2},
        {"attention_cp_size"},
        id="attention-context-parallel",
    ),
    pytest.param(
        {"attention_dp": True},
        {"attention_dp"},
        id="attention-dp",
    ),
    pytest.param({"multi_node": True}, {"multi_node"}, id="multi-node"),
    pytest.param({"rope_type": "yarn"}, {"rope_type"}, id="yarn"),
]


@pytest.mark.cpu_only
@pytest.mark.parametrize("overrides, expected_dimensions", _UNQUALIFIED_RUNTIME_VARIANTS)
@pytest.mark.parametrize(
    "root_model_class, architecture, model_type, supported_rope_fusion, supported_sliding_window",
    [
        pytest.param(
            modeling_llama_mod.LlamaForCausalLM,
            "LlamaForCausalLM",
            "llama",
            True,
            None,
            id="llama",
        ),
        pytest.param(
            modeling_qwen_mod.Qwen2ForCausalLM,
            "Qwen2ForCausalLM",
            "qwen2",
            True,
            None,
            id="qwen2",
        ),
        pytest.param(
            modeling_qwen3_mod.Qwen3ForCausalLM,
            "Qwen3ForCausalLM",
            "qwen3",
            False,
            None,
            id="qwen3",
        ),
        pytest.param(
            modeling_mistral_mod.MistralForCausalLM,
            "MistralForCausalLM",
            "mistral",
            True,
            "none",
            id="mistral",
        ),
        pytest.param(
            modeling_phi3_mod.Phi3ForCausalLM,
            "Phi3ForCausalLM",
            "phi3",
            True,
            None,
            id="phi3",
        ),
        pytest.param(
            modeling_qwen3_moe_mod.Qwen3MoeForCausalLM,
            "Qwen3MoeForCausalLM",
            "qwen3_moe",
            False,
            None,
            id="qwen3-moe",
        ),
    ],
)
def test_bf16_profiles_reject_unqualified_runtime_variants(
    overrides: dict[str, object],
    expected_dimensions: set[str],
    root_model_class: type[nn.Module],
    architecture: str,
    model_type: str,
    supported_rope_fusion: bool,
    supported_sliding_window: str | None,
) -> None:
    """Each BF16 profile, dense or MoE, rejects runtime variants outside its qualified constraints.

    The MoE-only variants of the Qwen3 MoE profile are in
    `test_qwen3_moe_profile_rejects_unqualified_moe_variants`.
    """
    decision = mx_loading_mod.MX_POST_TRANSFORM_PROFILE_REGISTRY.qualify(
        root_model_class=root_model_class,
        architecture=architecture,
        model_type=model_type,
        speculative_mode=None,
        protocol_version=mx_loading_mod.MX_STAGED_RECEIVER_TRANSFORM_PROTOCOL_VERSION,
        transfer_scope=PostTransformTransferScope.TARGET_MODEL,
        runtime_config=_bf16_dense_runtime_config(
            rope_fusion=supported_rope_fusion,
            sliding_window=supported_sliding_window,
            **overrides,
        ),
    )

    assert not decision.qualified
    assert decision.reason is PostTransformQualificationReason.RUNTIME_CONFIG_NOT_SUPPORTED
    assert decision.unsupported_runtime_dimensions == frozenset(expected_dimensions)


_BF16_DENSE_ROOTS = (
    pytest.param(
        modeling_llama_mod.LlamaForCausalLM,
        "LlamaForCausalLM",
        "llama",
        True,
        None,
        id="llama",
    ),
    pytest.param(
        modeling_qwen_mod.Qwen2ForCausalLM,
        "Qwen2ForCausalLM",
        "qwen2",
        True,
        None,
        id="qwen2",
    ),
    pytest.param(
        modeling_qwen3_mod.Qwen3ForCausalLM,
        "Qwen3ForCausalLM",
        "qwen3",
        False,
        None,
        id="qwen3",
    ),
    pytest.param(
        modeling_mistral_mod.MistralForCausalLM,
        "MistralForCausalLM",
        "mistral",
        True,
        "none",
        id="mistral",
    ),
    pytest.param(
        modeling_phi3_mod.Phi3ForCausalLM,
        "Phi3ForCausalLM",
        "phi3",
        True,
        None,
        id="phi3",
    ),
)

# Roots whose tied checkpoints qualify, mapped to the tied profile and its ABI.
_TIED_EMBEDDING_PROFILES = {
    "qwen3": (
        "qwen3-for-causal-lm-bf16-tied-target-v1",
        QWEN3_DENSE_TIED_POST_TRANSFORM_LAYOUT_ABI_V1,
    ),
}


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "root_model_class, architecture, model_type, supported_rope_fusion, supported_sliding_window",
    _BF16_DENSE_ROOTS,
)
def test_bf16_dense_profiles_scope_tied_embeddings(
    root_model_class: type[nn.Module],
    architecture: str,
    model_type: str,
    supported_rope_fusion: bool,
    supported_sliding_window: str | None,
) -> None:
    """Only roots with a tied sibling profile qualify a realized tie; no root accepts an unknown tie."""

    def qualify(tied_word_embeddings: bool | None):
        return mx_loading_mod.MX_POST_TRANSFORM_PROFILE_REGISTRY.qualify(
            root_model_class=root_model_class,
            architecture=architecture,
            model_type=model_type,
            speculative_mode=None,
            protocol_version=mx_loading_mod.MX_STAGED_RECEIVER_TRANSFORM_PROTOCOL_VERSION,
            transfer_scope=PostTransformTransferScope.TARGET_MODEL,
            runtime_config=_bf16_dense_runtime_config(
                rope_fusion=supported_rope_fusion,
                sliding_window=supported_sliding_window,
                tied_word_embeddings=tied_word_embeddings,
            ),
        )

    untied = qualify(False)
    assert untied.qualified
    assert untied.profile is not None
    assert "-tied-" not in untied.profile.profile_id
    assert "-tied-" not in untied.transform_abi_id

    tied = qualify(True)
    if model_type in _TIED_EMBEDDING_PROFILES:
        assert tied.qualified
        assert tied.profile is not None
        assert (tied.profile.profile_id, tied.transform_abi_id) == (
            _TIED_EMBEDDING_PROFILES[model_type]
        )
        assert tied.transform_abi_id != untied.transform_abi_id
    else:
        assert not tied.qualified
        assert tied.reason is PostTransformQualificationReason.RUNTIME_CONFIG_NOT_SUPPORTED
        assert tied.unsupported_runtime_dimensions == frozenset({"tied_word_embeddings"})

    # A tie that the constructed model does not realize as declared is unknown.
    unknown = qualify(None)
    assert not unknown.qualified
    assert unknown.reason is PostTransformQualificationReason.RUNTIME_CONFIG_NOT_SUPPORTED
    assert unknown.unsupported_runtime_dimensions == frozenset({"tied_word_embeddings"})
    assert unknown.profile is untied.profile


@pytest.mark.cpu_only
@pytest.mark.parametrize("overrides, expected_dimensions", _UNQUALIFIED_RUNTIME_VARIANTS)
def test_qwen3_tied_profile_rejects_unqualified_runtime_variants(
    overrides: dict[str, object],
    expected_dimensions: set[str],
) -> None:
    decision = mx_loading_mod.MX_POST_TRANSFORM_PROFILE_REGISTRY.qualify(
        root_model_class=modeling_qwen3_mod.Qwen3ForCausalLM,
        architecture="Qwen3ForCausalLM",
        model_type="qwen3",
        speculative_mode=None,
        protocol_version=mx_loading_mod.MX_STAGED_RECEIVER_TRANSFORM_PROTOCOL_VERSION,
        transfer_scope=PostTransformTransferScope.TARGET_MODEL,
        runtime_config=_bf16_dense_runtime_config(
            rope_fusion=False,
            tied_word_embeddings=True,
            **overrides,
        ),
    )

    assert not decision.qualified
    assert decision.reason is PostTransformQualificationReason.RUNTIME_CONFIG_NOT_SUPPORTED
    assert decision.unsupported_runtime_dimensions == frozenset(expected_dimensions)
    assert decision.profile is not None
    assert decision.profile.profile_id == "qwen3-for-causal-lm-bf16-tied-target-v1"


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "root_model_class, architecture, model_type, realized_overrides, expected_dimension",
    [
        pytest.param(
            modeling_llama_mod.LlamaForCausalLM,
            "LlamaForCausalLM",
            "llama",
            {"rope_fusion": False},
            "rope_fusion",
            id="llama-unfused-rope",
        ),
        pytest.param(
            modeling_qwen_mod.Qwen2ForCausalLM,
            "Qwen2ForCausalLM",
            "qwen2",
            {"rope_fusion": False},
            "rope_fusion",
            id="qwen2-unfused-rope",
        ),
        pytest.param(
            modeling_qwen3_mod.Qwen3ForCausalLM,
            "Qwen3ForCausalLM",
            "qwen3",
            {"rope_fusion": True},
            "rope_fusion",
            id="qwen3-fused-rope",
        ),
        pytest.param(
            modeling_qwen3_moe_mod.Qwen3MoeForCausalLM,
            "Qwen3MoeForCausalLM",
            "qwen3_moe",
            {"rope_fusion": True},
            "rope_fusion",
            id="qwen3-moe-fused-rope",
        ),
        pytest.param(
            modeling_qwen3_mod.Qwen3ForCausalLM,
            "Qwen3ForCausalLM",
            "qwen3",
            {"rope_fusion": True, "tied_word_embeddings": True},
            "rope_fusion",
            id="qwen3-tied-fused-rope",
        ),
        pytest.param(
            modeling_mistral_mod.MistralForCausalLM,
            "MistralForCausalLM",
            "mistral",
            {"rope_fusion": False, "sliding_window": "none"},
            "rope_fusion",
            id="mistral-unfused-rope",
        ),
        pytest.param(
            modeling_mistral_mod.MistralForCausalLM,
            "MistralForCausalLM",
            "mistral",
            {"rope_fusion": True, "sliding_window": "uniform"},
            "sliding_window",
            id="mistral-sliding-window",
        ),
        pytest.param(
            modeling_phi3_mod.Phi3ForCausalLM,
            "Phi3ForCausalLM",
            "phi3",
            {"rope_fusion": False},
            "rope_fusion",
            id="phi3-unfused-rope",
        ),
    ],
)
def test_bf16_profiles_reject_wrong_realized_dimension(
    root_model_class: type[nn.Module],
    architecture: str,
    model_type: str,
    realized_overrides: dict[str, object],
    expected_dimension: str,
) -> None:
    """Each BF16 profile rejects a realized RoPE fusion or window it was not qualified for."""
    decision = mx_loading_mod.MX_POST_TRANSFORM_PROFILE_REGISTRY.qualify(
        root_model_class=root_model_class,
        architecture=architecture,
        model_type=model_type,
        speculative_mode=None,
        protocol_version=mx_loading_mod.MX_STAGED_RECEIVER_TRANSFORM_PROTOCOL_VERSION,
        transfer_scope=PostTransformTransferScope.TARGET_MODEL,
        runtime_config=_bf16_dense_runtime_config(**realized_overrides),
    )

    assert not decision.qualified
    assert decision.reason is PostTransformQualificationReason.RUNTIME_CONFIG_NOT_SUPPORTED
    assert decision.unsupported_runtime_dimensions == frozenset({expected_dimension})


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "overrides",
    [
        pytest.param({"moe_backend": "TRTLLM"}, id="alternate-moe-backend"),
        pytest.param(
            {
                "tp_size": 2,
                "moe_tp_size": 1,
                "moe_ep_size": 2,
                "attention_tp_size": 2,
            },
            id="alternate-moe-partition",
        ),
        pytest.param(
            {
                "tp_size": 2,
                "moe_tp_size": 1,
                "moe_ep_size": 1,
                "moe_cluster_size": 2,
                "attention_tp_size": 2,
            },
            id="moe-cluster-partition",
        ),
    ],
)
@pytest.mark.parametrize(
    "root_model_class, architecture, model_type, supported_rope_fusion, supported_sliding_window",
    [
        pytest.param(
            modeling_llama_mod.LlamaForCausalLM,
            "LlamaForCausalLM",
            "llama",
            True,
            None,
            id="llama",
        ),
        pytest.param(
            modeling_qwen_mod.Qwen2ForCausalLM,
            "Qwen2ForCausalLM",
            "qwen2",
            True,
            None,
            id="qwen2",
        ),
        pytest.param(
            modeling_qwen3_mod.Qwen3ForCausalLM,
            "Qwen3ForCausalLM",
            "qwen3",
            False,
            None,
            id="qwen3",
        ),
        pytest.param(
            modeling_mistral_mod.MistralForCausalLM,
            "MistralForCausalLM",
            "mistral",
            True,
            "none",
            id="mistral",
        ),
        pytest.param(
            modeling_phi3_mod.Phi3ForCausalLM,
            "Phi3ForCausalLM",
            "phi3",
            True,
            None,
            id="phi3",
        ),
    ],
)
def test_bf16_dense_profiles_ignore_moe_only_runtime_dimensions(
    overrides: dict[str, object],
    root_model_class: type[nn.Module],
    architecture: str,
    model_type: str,
    supported_rope_fusion: bool,
    supported_sliding_window: str | None,
) -> None:
    """Dense profiles still qualify when only MoE-specific runtime dimensions differ."""
    decision = mx_loading_mod.MX_POST_TRANSFORM_PROFILE_REGISTRY.qualify(
        root_model_class=root_model_class,
        architecture=architecture,
        model_type=model_type,
        speculative_mode=None,
        protocol_version=mx_loading_mod.MX_STAGED_RECEIVER_TRANSFORM_PROTOCOL_VERSION,
        transfer_scope=PostTransformTransferScope.TARGET_MODEL,
        runtime_config=_bf16_dense_runtime_config(
            rope_fusion=supported_rope_fusion,
            sliding_window=supported_sliding_window,
            **overrides,
        ),
    )

    assert decision.qualified


@pytest.mark.cpu_only
@pytest.mark.parametrize("sm_version", [None, -1, 90, 100, 103, 120])
@pytest.mark.parametrize(
    "root_model_class, architecture, model_type, supported_rope_fusion, supported_sliding_window",
    [
        pytest.param(
            modeling_llama_mod.LlamaForCausalLM,
            "LlamaForCausalLM",
            "llama",
            True,
            None,
            id="llama",
        ),
        pytest.param(
            modeling_qwen_mod.Qwen2ForCausalLM,
            "Qwen2ForCausalLM",
            "qwen2",
            True,
            None,
            id="qwen2",
        ),
        pytest.param(
            modeling_qwen3_mod.Qwen3ForCausalLM,
            "Qwen3ForCausalLM",
            "qwen3",
            False,
            None,
            id="qwen3",
        ),
        pytest.param(
            modeling_mistral_mod.MistralForCausalLM,
            "MistralForCausalLM",
            "mistral",
            True,
            "none",
            id="mistral",
        ),
        pytest.param(
            modeling_phi3_mod.Phi3ForCausalLM,
            "Phi3ForCausalLM",
            "phi3",
            True,
            None,
            id="phi3",
        ),
    ],
)
def test_bf16_dense_profiles_qualify_on_any_sm_version(
    sm_version: int | None,
    root_model_class: type[nn.Module],
    architecture: str,
    model_type: str,
    supported_rope_fusion: bool,
    supported_sliding_window: str | None,
) -> None:
    decision = mx_loading_mod.MX_POST_TRANSFORM_PROFILE_REGISTRY.qualify(
        root_model_class=root_model_class,
        architecture=architecture,
        model_type=model_type,
        speculative_mode=None,
        protocol_version=mx_loading_mod.MX_STAGED_RECEIVER_TRANSFORM_PROTOCOL_VERSION,
        transfer_scope=PostTransformTransferScope.TARGET_MODEL,
        runtime_config=_bf16_dense_runtime_config(
            rope_fusion=supported_rope_fusion,
            sliding_window=supported_sliding_window,
            sm_version=sm_version,
        ),
    )

    assert decision.qualified
    assert decision.profile is not None
    assert decision.profile.runtime_constraints.sm_versions is None


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "device_sm_version, expected_dimensions",
    [
        pytest.param(90, frozenset(), id="sm90"),
        pytest.param(100, frozenset({"sm_version"}), id="sm100"),
        pytest.param(-1, frozenset({"sm_version"}), id="no-cuda-device"),
    ],
)
def test_qualify_post_transform_profile_uses_device_sm_version(
    monkeypatch: pytest.MonkeyPatch,
    device_sm_version: int,
    expected_dimensions: frozenset[str],
) -> None:
    (tiny_profile,) = _tiny_profile_registry().profiles
    monkeypatch.setattr(
        mx_loading_mod,
        "MX_POST_TRANSFORM_PROFILE_REGISTRY",
        PostTransformProfileRegistry(
            profiles=(
                replace(
                    tiny_profile,
                    runtime_constraints=PostTransformRuntimeConstraints(
                        sm_versions=frozenset({90})
                    ),
                ),
            )
        ),
    )
    monkeypatch.setattr(mx_loading_mod, "get_sm_version", lambda: device_sm_version)

    decision = mx_loading_mod.qualify_post_transform_profile(
        _TinyModel([]),
        speculative_mode=None,
        loads_draft_weights=False,
    )

    assert decision.qualified is (not expected_dimensions)
    assert decision.unsupported_runtime_dimensions == expected_dimensions


def _qualify_qwen3_runtime(runtime_config: PostTransformRuntimeConfig):
    return mx_loading_mod.MX_POST_TRANSFORM_PROFILE_REGISTRY.qualify(
        root_model_class=modeling_qwen3_mod.Qwen3ForCausalLM,
        architecture="Qwen3ForCausalLM",
        model_type="qwen3",
        speculative_mode=None,
        protocol_version=mx_loading_mod.MX_STAGED_RECEIVER_TRANSFORM_PROTOCOL_VERSION,
        transfer_scope=PostTransformTransferScope.TARGET_MODEL,
        runtime_config=runtime_config,
    )


@pytest.mark.cpu_only
def test_qwen3_fp8_block_profile_is_the_sm90_variant_of_the_bf16_row() -> None:
    profiles = {
        profile.profile_id: profile
        for profile in mx_loading_mod.MX_POST_TRANSFORM_PROFILE_REGISTRY.profiles
    }
    fp8_profile = profiles[_QWEN3_FP8_BLOCK_PROFILE_ID]
    bf16_profile = profiles["qwen3-for-causal-lm-bf16-target-v1"]

    assert (
        fp8_profile.transform_abi_id
        == QWEN3_DENSE_FP8_BLOCK128_FP32SCALE_POST_TRANSFORM_LAYOUT_ABI_V1
    )
    assert fp8_profile.transform_abi_id != bf16_profile.transform_abi_id
    assert fp8_profile.runtime_constraints == mx_loading_mod._with_fp8_block_scale_weights(
        bf16_profile.runtime_constraints, {90}
    )
    assert fp8_profile.runtime_constraints == replace(
        bf16_profile.runtime_constraints,
        quant_algorithms=frozenset({"FP8_BLOCK_SCALES"}),
        sm_versions=frozenset({90}),
        weight_block_sizes=frozenset({(128, 128)}),
    )
    assert not fp8_profile.runtime_constraints.overlaps(bf16_profile.runtime_constraints)
    assert fp8_profile.supported_features == bf16_profile.supported_features == frozenset()


@pytest.mark.cpu_only
def test_qwen3_fp8_block_row_precedes_the_bf16_qwen3_row() -> None:
    profile_ids = [
        profile.profile_id for profile in mx_loading_mod.MX_POST_TRANSFORM_PROFILE_REGISTRY.profiles
    ]

    assert profile_ids.index(_QWEN3_FP8_BLOCK_PROFILE_ID) + 1 == profile_ids.index(
        "qwen3-for-causal-lm-bf16-target-v1"
    )


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "overrides",
    [
        pytest.param({}, id="default"),
        pytest.param({"tp_size": 2, "moe_tp_size": 2, "attention_tp_size": 2}, id="tp2"),
        pytest.param({"moe_backend": "TRTLLM"}, id="alternate-moe-backend"),
        pytest.param(
            {"tp_size": 2, "moe_tp_size": 1, "moe_ep_size": 2, "attention_tp_size": 2},
            id="alternate-moe-partition",
        ),
    ],
)
def test_qwen3_fp8_block_profile_qualifies_sm90_runtime(overrides: dict[str, object]) -> None:
    decision = _qualify_qwen3_runtime(_fp8_block_runtime_config(**overrides))

    assert decision.qualified
    assert decision.profile is not None
    assert decision.profile.profile_id == _QWEN3_FP8_BLOCK_PROFILE_ID
    assert (
        decision.transform_abi_id == QWEN3_DENSE_FP8_BLOCK128_FP32SCALE_POST_TRANSFORM_LAYOUT_ABI_V1
    )


@pytest.mark.cpu_only
@pytest.mark.parametrize("sm_version", [89, 100, 103, 107, 120, 121, -1, None])
def test_qwen3_fp8_block_profile_rejects_other_sm_versions(sm_version: int | None) -> None:
    decision = _qualify_qwen3_runtime(_fp8_block_runtime_config(sm_version=sm_version))

    assert not decision.qualified
    assert decision.reason is PostTransformQualificationReason.RUNTIME_CONFIG_NOT_SUPPORTED
    # The FP8 row and the BF16 row each miss by one dimension; the registry
    # order makes the report name the SM rather than the quantization.
    assert decision.profile is not None
    assert decision.profile.profile_id == _QWEN3_FP8_BLOCK_PROFILE_ID
    assert decision.unsupported_runtime_dimensions == frozenset({"sm_version"})


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "overrides, expected_dimensions",
    [
        pytest.param({"dtype": "float16"}, {"dtype"}, id="fp16"),
        pytest.param({"quant_algorithm": "FP8"}, {"quant_algorithm"}, id="fp8-per-tensor"),
        pytest.param({"quant_algorithm": "NVFP4"}, {"quant_algorithm"}, id="nvfp4"),
        pytest.param(
            {"kv_cache_quant_algorithm": "FP8"},
            {"kv_cache_quant_algorithm"},
            id="kv-cache-quantization",
        ),
        pytest.param(
            {"layerwise_quantization": True},
            {"layerwise_quantization"},
            id="layerwise-quantization",
        ),
        pytest.param(
            {"force_dynamic_quantization": True},
            {"force_dynamic_quantization"},
            id="dynamic-quantization",
        ),
        pytest.param({"lora_enabled": True}, {"lora_enabled"}, id="lora"),
        pytest.param(
            {"sparse_attention_enabled": True},
            {"sparse_attention_enabled"},
            id="sparse-attention",
        ),
        pytest.param(
            {"attention_backend": "FLASHINFER"},
            {"attention_backend"},
            id="attention-backend",
        ),
        pytest.param({"tp_size": 4}, {"tp_size"}, id="tp4"),
        pytest.param({"pp_size": 2}, {"pp_size"}, id="pipeline-parallel"),
        pytest.param({"cp_size": 2}, {"cp_size"}, id="context-parallel"),
        pytest.param(
            {"attention_tp_size": 4},
            {"attention_tp_size"},
            id="attention-tensor-parallel",
        ),
        pytest.param(
            {"attention_cp_size": 2},
            {"attention_cp_size"},
            id="attention-context-parallel",
        ),
        pytest.param({"attention_dp": True}, {"attention_dp"}, id="attention-dp"),
        pytest.param({"multi_node": True}, {"multi_node"}, id="multi-node"),
        pytest.param(
            {"tied_word_embeddings": True},
            {"tied_word_embeddings"},
            id="tied-embeddings",
        ),
        pytest.param({"rope_type": "yarn"}, {"rope_type"}, id="yarn"),
        pytest.param({"rope_fusion": True}, {"rope_fusion"}, id="fused-rope"),
        pytest.param(
            {"weight_block_size": None},
            {"weight_block_size"},
            id="undeclared-weight-block-size",
        ),
        pytest.param(
            {"weight_block_size": (1, 128)},
            {"weight_block_size"},
            id="1x128-weight-blocks",
        ),
        pytest.param(
            {"weight_block_size": (64, 64)},
            {"weight_block_size"},
            id="64x64-weight-blocks",
        ),
    ],
)
def test_qwen3_fp8_block_profile_rejects_unqualified_runtime_variants(
    overrides: dict[str, object],
    expected_dimensions: set[str],
) -> None:
    decision = _qualify_qwen3_runtime(_fp8_block_runtime_config(**overrides))

    assert not decision.qualified
    assert decision.reason is PostTransformQualificationReason.RUNTIME_CONFIG_NOT_SUPPORTED
    assert decision.profile is not None
    assert decision.profile.profile_id == _QWEN3_FP8_BLOCK_PROFILE_ID
    assert decision.unsupported_runtime_dimensions == frozenset(expected_dimensions)


@pytest.mark.cpu_only
def test_fp8_block_scale_linear_keeps_checkpoint_layout_on_sm90(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _simulate_sm(monkeypatch, 90)
    linear = Linear(
        256, 384, bias=False, dtype=torch.bfloat16, quant_config=_fp8_block_quant_config()
    )
    with torch.no_grad():
        linear.weight.copy_(torch.ones(linear.weight.shape).to(torch.float8_e4m3fn))
        linear.weight_scale.fill_(0.5)
    parameters = dict(linear.named_parameters())
    before = {name: parameter.detach().clone() for name, parameter in parameters.items()}

    linear.transform_weights()

    assert type(linear.quant_method) is linear_mod.FP8BlockScalesLinearMethod
    assert dict(linear.named_parameters()) == parameters
    for name, parameter in linear.named_parameters():
        assert torch.equal(
            parameter.reshape(-1).view(torch.uint8), before[name].reshape(-1).view(torch.uint8)
        )
    assert linear.weight.dtype == torch.float8_e4m3fn
    assert tuple(linear.weight_scale.shape) == (3, 2)
    assert linear.weight_scale.dtype == torch.float32
    assert linear.rebuild_tensor_metadata == {}
    assert linear._weights_transformed is True


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "device_sm_version, publishes",
    [pytest.param(90, True, id="sm90"), pytest.param(100, False, id="sm100")],
)
def test_mx_donor_publishes_only_on_a_qualified_sm_version(
    monkeypatch: pytest.MonkeyPatch,
    device_sm_version: int,
    publishes: bool,
) -> None:
    loader = _make_loader(monkeypatch, events=[])
    (tiny_profile,) = _tiny_profile_registry().profiles
    monkeypatch.setattr(
        mx_loading_mod,
        "MX_POST_TRANSFORM_PROFILE_REGISTRY",
        PostTransformProfileRegistry(
            profiles=(
                replace(
                    tiny_profile,
                    runtime_constraints=PostTransformRuntimeConstraints(
                        sm_versions=frozenset({90})
                    ),
                ),
            )
        ),
    )
    monkeypatch.setattr(mx_loading_mod, "get_sm_version", lambda: device_sm_version)
    checkpoint_loader = _mock_mx_checkpoint_loader()
    checkpoint_loader.is_weights_preloaded.return_value = False
    checkpoint_loader.load_weights.return_value = {"weight": MagicMock()}
    checkpoint_loader.get_initialized_weight_mapper.return_value = MagicMock()

    loader.load("/ckpt", checkpoint_loader)

    _args, kwargs = checkpoint_loader.load_weights.call_args
    assert kwargs["allow_post_transform_weights"] is publishes
    assert checkpoint_loader.publish_as_source.called is publishes


def test_staged_llama_finalization_preserves_mx_tensor_catalog(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _tiny_llama_model(monkeypatch)

    # MX discovers and registers tensors after receiver alias preparation.
    model_hooks_mod.setup_aliases(model)
    registered_catalog = _mx_canonical_parameter_catalog(model)

    # This is the receiver-side finalization sequence after RDMA completes.
    model_hooks_mod.setup_aliases(model)
    model_hooks_mod.mark_weights_transformed(model)
    model_hooks_mod.walk_cache_state(model)
    published_catalog = _mx_canonical_parameter_catalog(model)

    assert registered_catalog
    assert published_catalog == registered_catalog


@pytest.mark.cpu_only
def test_separate_draft_model_is_not_qualified_by_target_only_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        mx_loading_mod,
        "MX_POST_TRANSFORM_PROFILE_REGISTRY",
        _tiny_profile_registry(speculative_mode="eagle3_one_model"),
    )

    decision = mx_loading_mod.qualify_post_transform_profile(
        _TinyModel([]),
        speculative_mode="eagle3_one_model",
        loads_draft_weights=True,
    )

    assert not decision.qualified
    assert decision.reason is PostTransformQualificationReason.FEATURE_NOT_SUPPORTED
    assert decision.unsupported_features == frozenset({PostTransformFeature.SEPARATE_DRAFT_MODEL})


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "uses_load_balancer, dwdp_size, expected_features",
    [
        pytest.param(False, 0, frozenset(), id="no-moe-feature"),
        pytest.param(
            True,
            0,
            frozenset({PostTransformFeature.MOE_LOAD_BALANCER}),
            id="realized-load-balancer",
        ),
        pytest.param(
            False,
            2,
            frozenset({PostTransformFeature.DWDP}),
            id="dwdp",
        ),
    ],
)
def test_qualification_reports_realized_moe_lifecycle_features(
    monkeypatch: pytest.MonkeyPatch,
    uses_load_balancer: bool,
    dwdp_size: int,
    expected_features: frozenset[PostTransformFeature],
) -> None:
    """A realized EPLB balancer or a DWDP mapping fails qualification closed.

    EPLB changes which experts a rank stores without changing the MoE TP or EP
    size, and DWDP forces both sizes to 1 while each rank stores its own expert
    range, so the size constraints cannot reject either; the loader reports
    them as features that no profile supports. MoE layers keep
    `layer_load_balancer` at `None` unless EPLB applies. The test uses a real
    MoE weight owner and a real `Mapping`, so renaming either attribute that
    the loader reads fails it.
    """
    monkeypatch.setattr(
        mx_loading_mod,
        "MX_POST_TRANSFORM_PROFILE_REGISTRY",
        _tiny_profile_registry(),
    )
    model = _TinyModel([])
    # `__new__`: the gate reads only the realized balancer, not kernels or weights.
    experts = TrtllmCutlassUnquantizedImpl.__new__(TrtllmCutlassUnquantizedImpl)
    nn.Module.__init__(experts)
    experts.layer_load_balancer = object() if uses_load_balancer else None
    model.experts = experts
    if dwdp_size:
        model.model_config.mapping = mapping_mod.Mapping(
            world_size=dwdp_size, tp_size=dwdp_size, dwdp_size=dwdp_size
        )

    decision = mx_loading_mod.qualify_post_transform_profile(
        model,
        speculative_mode=None,
        loads_draft_weights=False,
    )

    assert decision.qualified == (not expected_features)
    assert decision.unsupported_features == expected_features
    if expected_features:
        assert decision.reason is PostTransformQualificationReason.FEATURE_NOT_SUPPORTED


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "decision, expected_details",
    [
        pytest.param(
            PostTransformQualificationDecision(
                PostTransformQualificationReason.ROOT_MODEL_CLASS_NOT_REGISTERED
            ),
            "",
            id="reason-only",
        ),
        pytest.param(
            PostTransformQualificationDecision(
                PostTransformQualificationReason.RUNTIME_CONFIG_NOT_SUPPORTED,
                unsupported_runtime_dimensions=frozenset({"tp_size", "moe_backend"}),
            ),
            "; unsupported_runtime_dimensions=moe_backend,tp_size",
            id="runtime-dimensions",
        ),
        pytest.param(
            PostTransformQualificationDecision(
                PostTransformQualificationReason.FEATURE_NOT_SUPPORTED,
                unsupported_features=frozenset(
                    {PostTransformFeature.MOE_LOAD_BALANCER, PostTransformFeature.DWDP}
                ),
            ),
            "; unsupported_features=dwdp,moe_load_balancer",
            id="features",
        ),
    ],
)
def test_skipped_mx_publish_logs_only_reported_details(
    monkeypatch: pytest.MonkeyPatch,
    decision: PostTransformQualificationDecision,
    expected_details: str,
) -> None:
    """The skip-publish log names only the details the decision reports.

    `qualify()` stops at a runtime mismatch before it checks features, so an
    empty feature set there does not mean that no unsupported feature is active.
    """
    messages: list[str] = []
    monkeypatch.setattr(mx_loading_mod.logger, "info", messages.append)
    checkpoint_loader = _mock_mx_checkpoint_loader()

    mx_loading_mod.publish_mx_weights(
        checkpoint_loader,
        _TinyModel([]),
        weights_preloaded=False,
        qualification=decision,
        source_identity=None,
    )

    assert messages == [
        "Skipping MX post-transform publish for _TinyModel: qualification "
        f"reason={decision.reason.value}{expected_details}."
    ]
    checkpoint_loader.publish_as_source.assert_not_called()


@pytest.mark.cpu_only
def test_gms_writer_draft_load_preserves_mx_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mapping = MagicMock()
    checkpoint_loader = MagicMock()
    checkpoint_loader.checkpoint_format = "MX"
    draft_weights = {"draft": object()}
    checkpoint_loader.load_weights.return_value = draft_weights
    target_weight_mapper = MagicMock()
    call_load_weights = MagicMock()
    monkeypatch.setattr(model_hooks_mod, "call_load_weights", call_load_weights)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    model = SimpleNamespace(load_draft_weights=lambda _weights: None)

    gms_loading_mod._load_gms_writer_draft_weights(
        model,
        checkpoint_loader,
        speculative_model="draft-checkpoint",
        mapping=mapping,
        target_weight_mapper=target_weight_mapper,
        metrics={},
    )

    checkpoint_loader.load_weights.assert_called_once_with(
        "draft-checkpoint",
        mapping=mapping,
        _preserve_mx_session=True,
    )
    # The draft weights go through the draft mapper that the checkpoint loader builds.
    checkpoint_loader.get_initialized_draft_weight_mapper.assert_called_once_with(
        model, target_weight_mapper
    )
    call_load_weights.assert_called_once_with(
        model.load_draft_weights,
        draft_weights,
        checkpoint_loader.get_initialized_draft_weight_mapper.return_value,
    )


@pytest.mark.cpu_only
def test_one_engine_speculative_mode_is_not_qualified_by_target_only_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        mx_loading_mod,
        "MX_POST_TRANSFORM_PROFILE_REGISTRY",
        _tiny_profile_registry(),
    )

    decision = mx_loading_mod.qualify_post_transform_profile(
        _TinyModel([]),
        speculative_mode="mtp",
        loads_draft_weights=False,
    )

    assert not decision.qualified
    assert decision.reason is PostTransformQualificationReason.SPECULATIVE_MODE_NOT_REGISTERED
    assert decision.unsupported_features == frozenset()


@pytest.mark.cpu_only
def test_speculative_mode_name_is_canonical_and_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    warning = MagicMock()
    monkeypatch.setattr(mx_loading_mod.logger, "warning", warning)
    assert mx_loading_mod.get_speculative_mode_name(None) is None
    warning.assert_not_called()
    assert (
        mx_loading_mod.get_speculative_mode_name(
            SimpleNamespace(spec_dec_mode=SimpleNamespace(name="MTP"))
        )
        == "mtp"
    )
    warning.assert_not_called()
    assert (
        mx_loading_mod.get_speculative_mode_name(SimpleNamespace(spec_dec_mode=SimpleNamespace()))
        == "unknown"
    )
    warning.assert_called_once_with(
        "Unable to identify the speculative decoding mode from %s; "
        "post-transform sharing is disabled for this load.",
        "SimpleNamespace",
    )


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "qualification_case",
    ["unregistered-model", "locality_domain-feature"],
)
def test_mx_post_transform_receiver_falls_back_for_unqualified_model(
    monkeypatch,
    qualification_case,
):
    events = []
    loader = _make_loader(monkeypatch, events=events)
    if qualification_case == "locality_domain-feature":
        monkeypatch.setattr(
            mx_loading_mod,
            "MX_POST_TRANSFORM_PROFILE_REGISTRY",
            _tiny_profile_registry(),
        )
        model = _TinyModel(events)
        model.model_config.locality_domain_policy = LocalityDomainPolicy(enabled=True)
        monkeypatch.setattr(
            memory_init_mod.AutoModelForCausalLM,
            "from_config",
            MagicMock(return_value=model),
        )
    checkpoint_loader = _PostTransformMxLoader(post_transform=True)

    model, _ = loader.load("/ckpt", checkpoint_loader)

    _args, kwargs = checkpoint_loader.load_weights.call_args
    assert kwargs["allow_post_transform_weights"] is False
    assert "prepare_post_transform_receiver" not in kwargs
    assert checkpoint_loader.is_weights_preloaded() is False
    assert model_hooks_mod.call_load_weights.call_count == 1
    load_fn, weights, mapper = model_hooks_mod.call_load_weights.call_args.args
    assert load_fn == model.load_weights
    assert weights == {"disk.weight": checkpoint_loader._disk_weight}
    assert mapper is loader.weight_mapper
    assert kwargs["source_identity"].transform_abi_id is None
    assert events == ["load_weights", "post_load_weights"]
    checkpoint_loader.publish_as_source.assert_not_called()


def test_load_qualifies_with_preconstruction_identity_after_model_normalization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    loader = _make_loader(monkeypatch, events=events)
    registry = MagicMock(wraps=_tiny_profile_registry())
    monkeypatch.setattr(
        mx_loading_mod,
        "MX_POST_TRANSFORM_PROFILE_REGISTRY",
        registry,
    )
    normalized_model = _TinyModel(events)
    normalized_model.model_config.pretrained_config.architectures = ["NormalizedForCausalLM"]
    normalized_model.model_config.pretrained_config.model_type = "normalized"
    monkeypatch.setattr(
        memory_init_mod.AutoModelForCausalLM,
        "from_config",
        MagicMock(return_value=normalized_model),
    )
    checkpoint_loader = _mock_mx_checkpoint_loader()
    checkpoint_loader.load_weights.return_value = {"weight": MagicMock()}
    checkpoint_loader.is_weights_preloaded.return_value = False

    loader.load("/ckpt", checkpoint_loader)

    _args, kwargs = checkpoint_loader.load_weights.call_args
    registry.qualify.assert_called_once()
    assert kwargs["allow_post_transform_weights"] is True
    assert kwargs["source_identity"].transform_abi_id == LLAMA_POST_TRANSFORM_LAYOUT_ABI_V1


@pytest.mark.cpu_only
def test_mx_rejects_post_transform_preload_after_failed_qualification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loader = _make_loader(monkeypatch, events=[])
    checkpoint_loader = _UnsafePostTransformMxLoader(post_transform=True)

    with pytest.raises(
        RuntimeError,
        match="reason=root_model_class_not_registered",
    ):
        loader.load("/ckpt", checkpoint_loader)

    _args, kwargs = checkpoint_loader.load_weights.call_args
    assert kwargs["allow_post_transform_weights"] is False
    checkpoint_loader.publish_as_source.assert_not_called()


@pytest.mark.cpu_only
def test_mx_fallback_runs_standard_weight_mapping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = []
    loader = _make_loader(monkeypatch, events=events)
    monkeypatch.setattr(
        mx_loading_mod,
        "MX_POST_TRANSFORM_PROFILE_REGISTRY",
        _tiny_profile_registry(),
    )
    checkpoint_loader = _mock_mx_checkpoint_loader()
    checkpoint_loader.is_weights_preloaded.return_value = False
    checkpoint_loader.load_weights.return_value = {"weight": MagicMock()}
    checkpoint_loader.get_initialized_weight_mapper.return_value = MagicMock()

    model, _ = loader.load("/ckpt", checkpoint_loader)

    assert model_hooks_mod.call_load_weights.call_count == 1
    assert events[0] == "load_weights"
    assert "post_load_weights" in events
    checkpoint_loader.publish_as_source.assert_called_once_with(
        model,
        source_identity=checkpoint_loader.load_weights.call_args.kwargs["source_identity"],
    )


def test_mx_artifact_identity_failure_falls_back_to_disk(monkeypatch):
    events = []
    loader = _make_loader(monkeypatch, events=events)
    monkeypatch.setattr(
        mx_loading_mod,
        "MX_POST_TRANSFORM_PROFILE_REGISTRY",
        _tiny_profile_registry(),
    )
    artifact_error = ValueError(
        "Checkpoint manifests do not support nested symlinked directories: /ckpt/shards"
    )
    monkeypatch.setattr(
        model_loader_mod.ArtifactIdentity,
        "from_checkpoint",
        MagicMock(side_effect=artifact_error),
    )
    source_identity_factory = MagicMock()
    monkeypatch.setattr(
        model_loader_mod.SourceIdentity,
        "from_model_config",
        source_identity_factory,
    )
    warning = MagicMock()
    monkeypatch.setattr(model_loader_mod.logger, "warning", warning)

    checkpoint_loader = _mock_mx_checkpoint_loader()
    checkpoint_loader.is_weights_preloaded.return_value = False
    checkpoint_loader.load_weights.return_value = {"weight": MagicMock()}
    checkpoint_loader.get_initialized_weight_mapper.return_value = MagicMock()

    model, _ = loader.load("/ckpt", checkpoint_loader)

    source_identity_factory.assert_not_called()
    warning.assert_called_once()
    assert "falling back to regular checkpoint loading" in warning.call_args.args[0]
    _args, kwargs = checkpoint_loader.load_weights.call_args
    assert kwargs["source_identity"] is None
    checkpoint_loader.publish_as_source.assert_called_once_with(model, source_identity=None)


class _HookRecorder(nn.Module):
    def __init__(
        self,
        name: str,
        events: list[tuple[str, str]],
        *,
        removed: bool | None = None,
        transformed: bool | None = None,
    ) -> None:
        super().__init__()
        self.name = name
        self.events = events
        if removed is not None:
            self._weights_removed = removed
        if transformed is not None:
            self._weights_transformed = transformed

    def setup_aliases(self) -> None:
        self.events.append((self.name, "setup_aliases"))

    def cache_derived_state(self) -> None:
        self.events.append((self.name, "cache_derived_state"))

    def post_load_weights(self) -> None:
        self.events.append((self.name, "post_load_weights"))


class _HookModel(_HookRecorder):
    def __init__(self, events):
        super().__init__("model", events)
        self.child = _HookRecorder("child", events)
        self.transformed_child = _HookRecorder("transformed_child", events, transformed=True)
        self.removed_child = _HookRecorder("removed_child", events, removed=True)


@pytest.mark.cpu_only
def test_staged_hook_setup_aliases_walks_skip_removed_modules():
    events = []
    model = _HookModel(events)

    model_hooks_mod.setup_aliases(model)

    assert events == [
        ("model", "setup_aliases"),
        ("child", "setup_aliases"),
        ("transformed_child", "setup_aliases"),
    ]


@pytest.mark.cpu_only
def test_cache_state_and_full_post_load_walks_skip_removed_modules():
    events = []
    model = _HookModel(events)

    model_hooks_mod.walk_cache_state(model)
    model_hooks_mod.walk_full_post_load(model)

    assert events == [
        ("model", "cache_derived_state"),
        ("child", "cache_derived_state"),
        ("transformed_child", "cache_derived_state"),
        ("model", "post_load_weights"),
        ("child", "post_load_weights"),
        ("transformed_child", "post_load_weights"),
    ]


@pytest.mark.cpu_only
def test_reset_weights_transformed_only_resets_existing_flags():
    events = []
    model = _HookModel(events)
    model._weights_transformed = True
    model.child._weights_transformed = True

    model_hooks_mod.reset_weights_transformed(model)

    assert model._weights_transformed is False
    assert model.child._weights_transformed is False
    assert model.transformed_child._weights_transformed is False
    assert not hasattr(model.removed_child, "_weights_transformed")


@pytest.mark.cpu_only
def test_mark_weights_transformed_only_sets_existing_flags():
    events = []
    model = _HookModel(events)
    model._weights_transformed = False
    model.child._weights_transformed = False

    model_hooks_mod.mark_weights_transformed(model)

    assert model._weights_transformed is True
    assert model.child._weights_transformed is True
    assert model.transformed_child._weights_transformed is True
    assert not hasattr(model.removed_child, "_weights_transformed")


@pytest.mark.cpu_only
def test_linear_transform_weights_is_idempotent():
    linear = Linear(
        1,
        1,
        bias=False,
        reduce_output=False,
        skip_create_weights_in_init=True,
    )
    linear.quant_method = MagicMock()

    linear.transform_weights()
    linear.post_load_weights()

    linear.quant_method.transform_weights.assert_called_once_with(linear)
    assert linear._weights_transformed is True

    linear._weights_transformed = False
    linear.post_load_weights()
    assert linear.quant_method.transform_weights.call_count == 2

    linear._weights_transformed = False
    linear.cache_derived_state()
    assert linear._weights_transformed is True


@pytest.mark.cpu_only
def test_mla_transform_weights_is_idempotent(monkeypatch):
    monkeypatch.setattr(mla_mod, "is_sm_120f", lambda: True)
    quant_mode = SimpleNamespace(has_fp8_block_scales=lambda: True)
    mla = MLA.__new__(MLA)
    mla.mapping = mapping_mod.Mapping()
    mla.num_heads_tp = 1
    mla.num_context_heads = 1
    mla._weights_transformed = False
    mla.kv_b_proj = SimpleNamespace(quant_config=SimpleNamespace(quant_mode=quant_mode))
    mla.k_b_proj_trans = "k_weight"
    mla.k_b_proj_trans_scale = "k_scale"
    mla.v_b_proj = "v_weight"
    mla.v_b_proj_scale = "v_scale"
    calls = []

    def fake_resmooth(weight, scale, recipe):
        calls.append((weight, scale, recipe))
        return f"{weight}_transformed", f"{scale}_transformed"

    mla.resmooth_parameters = fake_resmooth

    MLA.transform_weights(mla)
    MLA.post_load_weights(mla)

    assert calls == [
        ("k_weight", "k_scale", (1, 128, 128)),
        ("v_weight", "v_scale", (1, 128, 128)),
    ]
    assert mla.k_b_proj_trans == "k_weight_transformed"
    assert mla.k_b_proj_trans_scale == "k_scale_transformed"
    assert mla.v_b_proj == "v_weight_transformed"
    assert mla.v_b_proj_scale == "v_scale_transformed"
    assert mla._weights_transformed is True

    mla._weights_transformed = False
    MLA.cache_derived_state(mla)
    assert mla._weights_transformed is True


def _make_manifest_loader(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    events: list[str],
    role: str | None,
    mapping: mapping_mod.Mapping | None = None,
) -> ModelLoader:
    loader = _make_loader(monkeypatch, events=events)
    loader.mapping = mapping or mapping_mod.Mapping(world_size=1, rank=0, tp_size=1)
    monkeypatch.setattr(
        memory_init_mod.AutoModelForCausalLM,
        "from_config",
        MagicMock(return_value=_ManifestTinyModel(events)),
    )
    if role is None:
        monkeypatch.delenv(WEIGHT_MANIFEST_DIR_ENV, raising=False)
        monkeypatch.delenv(WEIGHT_MANIFEST_ROLE_ENV, raising=False)
    else:
        monkeypatch.setenv(WEIGHT_MANIFEST_DIR_ENV, str(tmp_path))
        monkeypatch.setenv(WEIGHT_MANIFEST_ROLE_ENV, role)
    real_write = mx_loading_mod.maybe_write_weight_manifest

    def _recording_write(*args, **kwargs):
        events.append("manifest")
        return real_write(*args, **kwargs)

    monkeypatch.setattr(mx_loading_mod, "maybe_write_weight_manifest", _recording_write)
    return loader


def _hf_checkpoint_loader() -> MagicMock:
    checkpoint_loader = MagicMock(name="checkpoint_loader")
    checkpoint_loader.checkpoint_format = "HF"
    checkpoint_loader.load_weights.return_value = {"weight": MagicMock()}
    return checkpoint_loader


@pytest.mark.cpu_only
def test_load_writes_final_manifest_at_end_of_load(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    events = []
    loader = _make_manifest_loader(monkeypatch, tmp_path, events=events, role="baseline")
    from_checkpoint = MagicMock()
    monkeypatch.setattr(model_loader_mod.ArtifactIdentity, "from_checkpoint", from_checkpoint)

    model, _ = loader.load("/ckpt", _hf_checkpoint_loader())

    # HF/AUTO loading needs no source identity, so it never reads or hashes the checkpoint.
    from_checkpoint.assert_not_called()
    assert events == ["load_weights", "post_load_weights", "manifest"]
    files = sorted(path.name for path in tmp_path.iterdir())
    assert files == ["manifest.final.baseline.rank0.json"]
    manifest = load_weight_manifest(tmp_path / files[0])
    assert [entry.fqn for entry in manifest.entries] == ["scale", "weight"]
    assert manifest.context["boundary"] == "model_loader_load_end"
    assert manifest.context["checkpoint_format"] == "HF"
    assert manifest.context["weights_preloaded"] is False
    assert manifest.context["load_format"] == str(LoadFormat.AUTO)
    assert manifest.context["model_class"] == type(model).__name__
    assert manifest.context["tp_rank"] == 0 and manifest.context["world_size"] == 1
    assert model_loader_mod.ModelLoaderMetricNames.WEIGHT_MANIFEST_SECONDS.value in loader._metrics


@pytest.mark.cpu_only
def test_load_final_manifest_records_mx_receiver_facts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    events = []
    loader = _make_manifest_loader(monkeypatch, tmp_path, events=events, role="receiver")
    checkpoint_loader = _mock_mx_checkpoint_loader()
    checkpoint_loader.is_weights_preloaded.return_value = True
    # `_ManifestTinyModel` is not a registered root, so the post-transform
    # staged path must stay off; the full post-load path still ends in a manifest.
    checkpoint_loader.is_post_transform_weights_preloaded.return_value = False
    checkpoint_loader.load_weights.return_value = {}

    loader.load("/ckpt", checkpoint_loader)

    assert events[-1] == "manifest"
    manifest = load_weight_manifest(tmp_path / "manifest.final.receiver.rank0.json")
    assert manifest.context["checkpoint_format"] == "MX"
    assert manifest.context["weights_preloaded"] is True
    assert manifest.context["role"] == "receiver"


@pytest.mark.cpu_only
def test_load_writes_no_manifest_when_env_unset(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    events = []
    loader = _make_manifest_loader(monkeypatch, tmp_path, events=events, role=None)

    loader.load("/ckpt", _hf_checkpoint_loader())

    assert events == ["load_weights", "post_load_weights", "manifest"]
    assert list(tmp_path.iterdir()) == []
    assert (
        model_loader_mod.ModelLoaderMetricNames.WEIGHT_MANIFEST_SECONDS.value not in loader._metrics
    )


@pytest.mark.cpu_only
def test_load_final_manifest_uses_mapping_rank(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    events = []
    loader = _make_manifest_loader(
        monkeypatch,
        tmp_path,
        events=events,
        role="donor",
        mapping=mapping_mod.Mapping(world_size=2, rank=1, tp_size=2),
    )

    loader.load("/ckpt", _hf_checkpoint_loader())

    manifest = load_weight_manifest(tmp_path / "manifest.final.donor.rank1.json")
    assert manifest.context["rank"] == 1
    assert manifest.context["tp_rank"] == 1
    assert manifest.context["world_size"] == 2
