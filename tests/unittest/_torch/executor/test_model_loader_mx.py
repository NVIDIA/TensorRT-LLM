# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for MX-specific ModelLoader branches."""

from collections.abc import Callable
from contextlib import contextmanager, nullcontext
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock

import pytest
import torch
from torch import nn
from transformers import LlamaConfig, MistralConfig, Qwen2Config, Qwen3Config
from utils.post_transform_qualification import (
    PostTransformQualificationCase,
    assert_post_transform_lifecycle_equivalent,
)

import tensorrt_llm.mapping as mapping_mod
from tensorrt_llm._torch import distributed as distributed_mod
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models import modeling_llama as modeling_llama_mod
from tensorrt_llm._torch.models import modeling_mistral as modeling_mistral_mod
from tensorrt_llm._torch.models import modeling_qwen as modeling_qwen_mod
from tensorrt_llm._torch.models import modeling_qwen3 as modeling_qwen3_mod
from tensorrt_llm._torch.models.checkpoints.mx.checkpoint_loader import MXCheckpointLoader
from tensorrt_llm._torch.models.modeling_utils import get_registered_model_class
from tensorrt_llm._torch.modules import mla as mla_mod
from tensorrt_llm._torch.modules.linear import Linear, WeightMode
from tensorrt_llm._torch.modules.mla import MLA
from tensorrt_llm._torch.pyexecutor import model_loader as model_loader_mod
from tensorrt_llm._torch.pyexecutor.model_loader import ModelLoader
from tensorrt_llm._torch.weight_sharing import (
    ARTIFACT_IDENTITY_FORMAT_VERSION,
    LLAMA_POST_TRANSFORM_LAYOUT_ABI_V1,
    SOURCE_IDENTITY_FORMAT_VERSION,
    WEIGHT_MANIFEST_DIR_ENV,
    WEIGHT_MANIFEST_ROLE_ENV,
    ArtifactIdentity,
    PostTransformFeature,
    PostTransformProfile,
    PostTransformProfileRegistry,
    PostTransformQualificationReason,
    PostTransformRuntimeConfig,
    PostTransformRuntimeConstraints,
    PostTransformTransferScope,
    load_weight_manifest,
)
from tensorrt_llm.llmapi.llm_args import LoadFormat

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
            pretrained_config=SimpleNamespace(architectures=["TinyForCausalLM"], model_type="tiny")
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
) -> nn.Module:
    monkeypatch.setattr(torch.cuda, "Stream", lambda *_args, **_kwargs: MagicMock())
    monkeypatch.setattr(torch.cuda, "Event", lambda *_args, **_kwargs: MagicMock())
    qwen3_config = Qwen3Config(
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
        tie_word_embeddings=False,
        torch_dtype=torch.bfloat16,
        vocab_size=32,
    )
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


def _bf16_dense_runtime_config(**overrides: object) -> PostTransformRuntimeConfig:
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
        "attention_tp_size": 1,
        "attention_cp_size": 1,
        "attention_dp": False,
        "multi_node": False,
        "tied_word_embeddings": False,
        "rope_type": "default",
        "rope_fusion": True,
        # Llama and Qwen attention modules expose no window state, so their real
        # tiny models realize `None`; Mistral tests override this to `"none"`.
        "sliding_window": None,
    }
    values.update(overrides)
    return PostTransformRuntimeConfig(**values)


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
                protocol_version=(ModelLoader._MX_STAGED_RECEIVER_TRANSFORM_PROTOCOL_VERSION),
                transform_abi_id=LLAMA_POST_TRANSFORM_LAYOUT_ABI_V1,
                transfer_scope=PostTransformTransferScope.TARGET_MODEL,
                runtime_constraints=PostTransformRuntimeConstraints(),
            ),
        )
    )


def _make_loader(monkeypatch, *, events, spec_config=None):
    llm_args = SimpleNamespace(load_format=LoadFormat.AUTO)
    loader = ModelLoader(
        llm_args=llm_args,
        mapping=MagicMock(name="mapping"),
        spec_config=spec_config,
        sparse_attention_config=None,
        max_num_tokens=128,
        max_seq_len=128,
    )
    loader._call_load_weights = MagicMock(
        side_effect=lambda fn, weights, mapper, **kwargs: fn(weights, mapper)
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
    monkeypatch.setattr(model_loader_mod, "MetaInitMode", lambda: nullcontext())

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
        model_loader_mod.AutoModelForCausalLM,
        "from_config",
        MagicMock(return_value=_TinyModel(events)),
    )
    monkeypatch.setattr(model_loader_mod, "get_rank_model_storage", lambda _model: 0)
    return loader


@pytest.mark.cpu_only
def test_construct_checkpoint_loader_passes_mx_config():
    mx_config = SimpleNamespace(
        server_url="http://mx:8001",
        server_query_timeout_s=17,
    )

    checkpoint_loader = model_loader_mod._construct_checkpoint_loader(
        "pytorch",
        None,
        "MX",
        mx_config=mx_config,
        mx_model_name="Qwen/Qwen3-8B",
    )

    assert isinstance(checkpoint_loader, MXCheckpointLoader)
    assert checkpoint_loader.mx_server_url == "http://mx:8001"
    assert checkpoint_loader.query_timeout_s == 17
    assert checkpoint_loader.model_name == "Qwen/Qwen3-8B"


def _format_documented_values(
    values: frozenset[object] | None,
    *,
    labels: dict[object, str] | None = None,
) -> str:
    assert values is not None
    assert None not in values
    labels = labels or {}
    return " or ".join(labels.get(value, str(value)) for value in sorted(values, key=str))


def _documented_dense_constraints(profile: PostTransformProfile) -> str:
    constraints = profile.runtime_constraints
    assert constraints.quant_algorithms == frozenset({"none"})
    assert constraints.kv_cache_quant_algorithms == frozenset({"none"})
    assert constraints.layerwise_quantization == frozenset({False})
    assert constraints.force_dynamic_quantization == frozenset({False})
    assert constraints.lora_enabled == frozenset({False})
    assert constraints.sparse_attention_enabled == frozenset({False})
    assert constraints.attention_dp == frozenset({False})
    assert constraints.multi_node == frozenset({False})
    assert constraints.tied_word_embeddings == frozenset({False})
    assert constraints.rope_types == frozenset({"default"})
    qwen3_profile = profile.model_type == "qwen3"
    assert constraints.rope_fusion == frozenset({not qwen3_profile})
    mistral_profile = profile.model_type == "mistral"
    assert constraints.sliding_windows == (frozenset({"none"}) if mistral_profile else None)
    assert constraints.moe_backends is None
    assert constraints.moe_tp_sizes is None
    assert constraints.moe_ep_sizes is None
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
    rope_description = "default fused QK-norm/RoPE" if qwen3_profile else "default fused RoPE"
    sliding_window_description = ", no sliding window" if mistral_profile else ""
    return (
        f"Single-node dense {dtypes}, unquantized weights and KV cache, "
        f"{attention_backends} attention, {rope_description}{sliding_window_description}, "
        f"untied embeddings, TP={tp_sizes}, PP/CP={pp_cp_sizes}, no LoRA, sparse attention, "
        "attention DP, speculative mode, or separately loaded draft model"
    )


@pytest.mark.cpu_only
def test_public_support_table_matches_qualified_profile_registry() -> None:
    profiles = ModelLoader._post_transform_profile_registry().profiles
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
            f"{_documented_dense_constraints(profile)} |"
        )
        assert expected_row in table_rows


@pytest.mark.cpu_only
def test_mx_success_initializes_mapper_skips_weight_mapping_and_reload_works(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = []
    loader = _make_loader(monkeypatch, events=events)
    monkeypatch.setattr(
        ModelLoader,
        "_POST_TRANSFORM_PROFILE_REGISTRY",
        _tiny_profile_registry(),
    )
    checkpoint_loader = MagicMock(name="checkpoint_loader")
    checkpoint_loader.checkpoint_format = "MX"
    checkpoint_loader.is_weights_preloaded.return_value = True
    checkpoint_loader.load_weights.return_value = {}

    model, _ = loader.load("/ckpt", checkpoint_loader)

    checkpoint_loader.load_weights.assert_called_once()
    _args, kwargs = checkpoint_loader.load_weights.call_args
    assert kwargs["mapping"] is loader.mapping
    assert kwargs["model"] is model
    assert kwargs["source_identity"] is loader._source_identity
    assert kwargs["allow_post_transform_weights"] is True
    assert loader._source_identity.transform_abi_id == LLAMA_POST_TRANSFORM_LAYOUT_ABI_V1
    assert loader._call_load_weights.call_count == 0
    checkpoint_loader.get_initialized_weight_mapper.assert_called_once()
    assert loader.weight_mapper is checkpoint_loader.get_initialized_weight_mapper.return_value
    checkpoint_loader.post_load_publish.assert_called_once_with(
        model,
        checkpoint_dir="/ckpt",
        weights_preloaded=True,
        source_identity=loader._source_identity,
    )

    # reload() uses self.weight_mapper unconditionally; MX success must
    # initialize it even though the initial load skipped _call_load_weights.
    model._weights_transformed = True
    model.linear._weights_transformed = True
    loader.reload(model, {"reloaded": MagicMock()})
    assert loader._call_load_weights.call_count == 1
    assert model._weights_transformed is False
    assert model.linear._weights_transformed is False
    assert events == ["post_load_weights", "load_weights"]


@pytest.mark.cpu_only
def test_reload_partial_loading_preserves_weights_transformed_flags(monkeypatch):
    events = []
    loader = _make_loader(monkeypatch, events=events)
    loader.weight_mapper = MagicMock(name="weight_mapper")
    model = _TinyModel(events)
    model._weights_transformed = True
    model.linear._weights_transformed = True

    loader.reload(model, {"reloaded": MagicMock()}, allow_partial_loading=True)

    assert loader._call_load_weights.call_count == 1
    assert loader._call_load_weights.call_args.kwargs["allow_partial_loading"] is True
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
        ModelLoader,
        "_POST_TRANSFORM_PROFILE_REGISTRY",
        _tiny_profile_registry(),
    )
    checkpoint_loader = MagicMock(name="checkpoint_loader")
    checkpoint_loader.checkpoint_format = "MX"
    checkpoint_loader.is_weights_preloaded.return_value = True
    fallback_weights = {"mismatched.weight": MagicMock()}
    checkpoint_loader.load_weights.return_value = fallback_weights

    model, _ = loader.load("/ckpt", checkpoint_loader)

    assert loader._call_load_weights.call_count == 1
    load_fn, weights, mapper = loader._call_load_weights.call_args.args
    assert load_fn == model.load_weights
    assert weights is fallback_weights
    assert mapper is loader.weight_mapper
    checkpoint_loader.post_load_publish.assert_called_once_with(
        model,
        checkpoint_dir="/ckpt",
        weights_preloaded=True,
        source_identity=loader._source_identity,
    )


class _PostTransformMxLoader:
    checkpoint_format = "MX"

    def __init__(self, *, post_transform: bool) -> None:
        self._post_transform = post_transform
        self._weights_preloaded = True
        self._disk_weight = MagicMock()
        self.load_weights = MagicMock(side_effect=self._load_weights)
        self.is_weights_preloaded = MagicMock(side_effect=lambda: self._weights_preloaded)
        self.get_initialized_weight_mapper = MagicMock(return_value=MagicMock())
        self.post_load_apply = MagicMock()
        self.post_load_publish = MagicMock()

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
        ModelLoader,
        "_POST_TRANSFORM_PROFILE_REGISTRY",
        _tiny_profile_registry(),
    )
    checkpoint_loader = _PostTransformMxLoader(post_transform=True)

    model, _ = loader.load("/ckpt", checkpoint_loader)

    loader._call_load_weights.assert_not_called()
    _args, kwargs = checkpoint_loader.load_weights.call_args
    assert kwargs["allow_post_transform_weights"] is True
    assert callable(kwargs["prepare_post_transform_receiver"])
    checkpoint_loader.post_load_publish.assert_called_once_with(
        model,
        checkpoint_dir="/ckpt",
        weights_preloaded=True,
        source_identity=loader._source_identity,
    )
    # Post-transform receivers skip transform_weights(), but the accepted
    # tensors are already in final layout. Keep the transform guard in sync so
    # future reload/refactor paths do not accidentally treat them as raw bytes.
    assert model._weights_transformed is True
    assert model.linear._weights_transformed is True
    assert model.draft_model.linear._weights_transformed is True
    assert events == ["setup_aliases", "setup_aliases", "cache_derived_state"]


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
        qualify_model=lambda model: ModelLoader._qualify_post_transform_profile(
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
        qualify_model=lambda model: ModelLoader._qualify_post_transform_profile(
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
        qualify_model=lambda model: ModelLoader._qualify_post_transform_profile(
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

    decision = ModelLoader._qualify_post_transform_profile(
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
        qualify_model=lambda model: ModelLoader._qualify_post_transform_profile(
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
        qualify_model=lambda model: ModelLoader._qualify_post_transform_profile(
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
        qualify_model=lambda model: ModelLoader._qualify_post_transform_profile(
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


def test_mistral_dense_profile_qualifies_full_staged_lifecycle() -> None:
    case = PostTransformQualificationCase(
        profile_id="mistral-for-causal-lm-bf16-target-v1",
        model_factory=_tiny_mistral_model,
        unqualified_model_factory=lambda: _tiny_mistral_model(
            model_class=_UnqualifiedMistralForCausalLM
        ),
        qualify_model=lambda model: ModelLoader._qualify_post_transform_profile(
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
        qualify_model=lambda model: ModelLoader._qualify_post_transform_profile(
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

    decision = ModelLoader._qualify_post_transform_profile(
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

    decision = ModelLoader._qualify_post_transform_profile(
        model,
        speculative_mode=None,
        loads_draft_weights=False,
    )

    assert all(layer.self_attn.attention_window_size is None for layer in model.model.layers)
    assert decision.qualified
    assert decision.profile is not None
    assert decision.profile.profile_id == "mistral-for-causal-lm-bf16-target-v1"


@pytest.mark.cpu_only
def test_legacy_llama_file_mistral_root_is_not_qualified() -> None:
    legacy_root = modeling_llama_mod.MistralForCausalLM
    assert legacy_root is not modeling_mistral_mod.MistralForCausalLM
    assert (
        get_registered_model_class("MistralForCausalLM") is modeling_mistral_mod.MistralForCausalLM
    )

    decision = ModelLoader._post_transform_profile_registry().qualify(
        root_model_class=legacy_root,
        architecture="MistralForCausalLM",
        model_type="mistral",
        speculative_mode=None,
        protocol_version=ModelLoader._MX_STAGED_RECEIVER_TRANSFORM_PROTOCOL_VERSION,
        transfer_scope=PostTransformTransferScope.TARGET_MODEL,
        runtime_config=_bf16_dense_runtime_config(sliding_window="none"),
    )

    assert not decision.qualified
    assert decision.reason is PostTransformQualificationReason.ROOT_MODEL_CLASS_NOT_REGISTERED


@pytest.mark.cpu_only
def test_mistral_dense_profile_rejects_native_format_model_type() -> None:
    decision = ModelLoader._post_transform_profile_registry().qualify(
        root_model_class=modeling_mistral_mod.MistralForCausalLM,
        architecture="MistralForCausalLM",
        model_type="mistral_common",
        speculative_mode=None,
        protocol_version=ModelLoader._MX_STAGED_RECEIVER_TRANSFORM_PROTOCOL_VERSION,
        transfer_scope=PostTransformTransferScope.TARGET_MODEL,
        runtime_config=_bf16_dense_runtime_config(sliding_window="none"),
    )

    assert not decision.qualified
    assert decision.reason is PostTransformQualificationReason.MODEL_TYPE_NOT_REGISTERED


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "overrides, expected_dimensions",
    [
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
        pytest.param(
            {"tied_word_embeddings": True},
            {"tied_word_embeddings"},
            id="tied-embeddings",
        ),
        pytest.param({"rope_type": "yarn"}, {"rope_type"}, id="yarn"),
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
    ],
)
def test_bf16_dense_profiles_reject_unqualified_runtime_variants(
    overrides: dict[str, object],
    expected_dimensions: set[str],
    root_model_class: type[nn.Module],
    architecture: str,
    model_type: str,
    supported_rope_fusion: bool,
    supported_sliding_window: str | None,
) -> None:
    decision = ModelLoader._post_transform_profile_registry().qualify(
        root_model_class=root_model_class,
        architecture=architecture,
        model_type=model_type,
        speculative_mode=None,
        protocol_version=ModelLoader._MX_STAGED_RECEIVER_TRANSFORM_PROTOCOL_VERSION,
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
    ],
)
def test_bf16_dense_profiles_reject_wrong_realized_dimension(
    root_model_class: type[nn.Module],
    architecture: str,
    model_type: str,
    realized_overrides: dict[str, object],
    expected_dimension: str,
) -> None:
    decision = ModelLoader._post_transform_profile_registry().qualify(
        root_model_class=root_model_class,
        architecture=architecture,
        model_type=model_type,
        speculative_mode=None,
        protocol_version=ModelLoader._MX_STAGED_RECEIVER_TRANSFORM_PROTOCOL_VERSION,
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
    decision = ModelLoader._post_transform_profile_registry().qualify(
        root_model_class=root_model_class,
        architecture=architecture,
        model_type=model_type,
        speculative_mode=None,
        protocol_version=ModelLoader._MX_STAGED_RECEIVER_TRANSFORM_PROTOCOL_VERSION,
        transfer_scope=PostTransformTransferScope.TARGET_MODEL,
        runtime_config=_bf16_dense_runtime_config(
            rope_fusion=supported_rope_fusion,
            sliding_window=supported_sliding_window,
            **overrides,
        ),
    )

    assert decision.qualified


@pytest.mark.cpu_only
def test_separate_draft_model_is_not_qualified_by_target_only_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        ModelLoader,
        "_POST_TRANSFORM_PROFILE_REGISTRY",
        _tiny_profile_registry(speculative_mode="eagle3_one_model"),
    )

    decision = ModelLoader._qualify_post_transform_profile(
        _TinyModel([]),
        speculative_mode="eagle3_one_model",
        loads_draft_weights=True,
    )

    assert not decision.qualified
    assert decision.reason is PostTransformQualificationReason.FEATURE_NOT_SUPPORTED
    assert decision.unsupported_features == frozenset({PostTransformFeature.SEPARATE_DRAFT_MODEL})


@pytest.mark.cpu_only
def test_one_engine_speculative_mode_is_not_qualified_by_target_only_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        ModelLoader,
        "_POST_TRANSFORM_PROFILE_REGISTRY",
        _tiny_profile_registry(),
    )

    decision = ModelLoader._qualify_post_transform_profile(
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
    monkeypatch.setattr(model_loader_mod.logger, "warning", warning)
    assert ModelLoader._speculative_mode_name(None) is None
    warning.assert_not_called()
    assert (
        ModelLoader._speculative_mode_name(
            SimpleNamespace(spec_dec_mode=SimpleNamespace(name="MTP"))
        )
        == "mtp"
    )
    warning.assert_not_called()
    assert (
        ModelLoader._speculative_mode_name(SimpleNamespace(spec_dec_mode=SimpleNamespace()))
        == "unknown"
    )
    warning.assert_called_once_with(
        "Unable to identify the speculative decoding mode from %s; "
        "post-transform sharing is disabled for this load.",
        "SimpleNamespace",
    )


@pytest.mark.cpu_only
def test_mx_post_transform_receiver_falls_back_for_unqualified_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = []
    loader = _make_loader(monkeypatch, events=events)
    checkpoint_loader = _PostTransformMxLoader(post_transform=True)

    model, _ = loader.load("/ckpt", checkpoint_loader)

    _args, kwargs = checkpoint_loader.load_weights.call_args
    assert kwargs["allow_post_transform_weights"] is False
    assert "prepare_post_transform_receiver" not in kwargs
    assert checkpoint_loader.is_weights_preloaded() is False
    assert loader._call_load_weights.call_count == 1
    load_fn, weights, mapper = loader._call_load_weights.call_args.args
    assert load_fn == model.load_weights
    assert weights == {"disk.weight": checkpoint_loader._disk_weight}
    assert mapper is loader.weight_mapper
    assert loader._source_identity.transform_abi_id is None
    assert events == ["load_weights", "post_load_weights"]
    checkpoint_loader.post_load_publish.assert_not_called()


def test_load_qualifies_with_preconstruction_identity_after_model_normalization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    loader = _make_loader(monkeypatch, events=events)
    registry = MagicMock(wraps=_tiny_profile_registry())
    monkeypatch.setattr(
        ModelLoader,
        "_POST_TRANSFORM_PROFILE_REGISTRY",
        registry,
    )
    normalized_model = _TinyModel(events)
    normalized_model.model_config.pretrained_config.architectures = ["NormalizedForCausalLM"]
    normalized_model.model_config.pretrained_config.model_type = "normalized"
    monkeypatch.setattr(
        model_loader_mod.AutoModelForCausalLM,
        "from_config",
        MagicMock(return_value=normalized_model),
    )
    checkpoint_loader = MagicMock(name="checkpoint_loader")
    checkpoint_loader.checkpoint_format = "MX"
    checkpoint_loader.load_weights.return_value = {"weight": MagicMock()}
    checkpoint_loader.is_weights_preloaded.return_value = False

    loader.load("/ckpt", checkpoint_loader)

    _args, kwargs = checkpoint_loader.load_weights.call_args
    registry.qualify.assert_called_once()
    assert kwargs["allow_post_transform_weights"] is True
    assert loader._source_identity.transform_abi_id == LLAMA_POST_TRANSFORM_LAYOUT_ABI_V1


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
    checkpoint_loader.post_load_publish.assert_not_called()


@pytest.mark.cpu_only
def test_mx_fallback_runs_standard_weight_mapping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = []
    loader = _make_loader(monkeypatch, events=events)
    monkeypatch.setattr(
        ModelLoader,
        "_POST_TRANSFORM_PROFILE_REGISTRY",
        _tiny_profile_registry(),
    )
    checkpoint_loader = MagicMock(name="checkpoint_loader")
    checkpoint_loader.checkpoint_format = "MX"
    checkpoint_loader.is_weights_preloaded.return_value = False
    checkpoint_loader.load_weights.return_value = {"weight": MagicMock()}
    checkpoint_loader.get_initialized_weight_mapper.return_value = MagicMock()

    model, _ = loader.load("/ckpt", checkpoint_loader)

    assert loader._call_load_weights.call_count == 1
    assert events[0] == "load_weights"
    assert "post_load_weights" in events
    checkpoint_loader.post_load_publish.assert_called_once_with(
        model,
        checkpoint_dir="/ckpt",
        weights_preloaded=False,
        source_identity=loader._source_identity,
    )


def test_mx_artifact_identity_failure_falls_back_to_disk(monkeypatch):
    events = []
    loader = _make_loader(monkeypatch, events=events)
    monkeypatch.setattr(
        ModelLoader,
        "_POST_TRANSFORM_PROFILE_REGISTRY",
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

    checkpoint_loader = MagicMock(name="checkpoint_loader")
    checkpoint_loader.checkpoint_format = "MX"
    checkpoint_loader.is_weights_preloaded.return_value = False
    checkpoint_loader.load_weights.return_value = {"weight": MagicMock()}
    checkpoint_loader.get_initialized_weight_mapper.return_value = MagicMock()

    model, _ = loader.load("/ckpt", checkpoint_loader)

    assert loader._source_identity is None
    source_identity_factory.assert_not_called()
    warning.assert_called_once()
    assert "falling back to regular checkpoint loading" in warning.call_args.args[0]
    _args, kwargs = checkpoint_loader.load_weights.call_args
    assert kwargs["source_identity"] is None
    checkpoint_loader.post_load_publish.assert_called_once_with(
        model,
        checkpoint_dir="/ckpt",
        weights_preloaded=False,
        source_identity=None,
    )


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

    def transform_weights(self) -> None:
        self.events.append((self.name, "transform_weights"))
        self._weights_transformed = True

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

    ModelLoader._setup_aliases(model)

    assert events == [
        ("model", "setup_aliases"),
        ("child", "setup_aliases"),
        ("transformed_child", "setup_aliases"),
    ]


@pytest.mark.cpu_only
def test_staged_hook_walks_skip_removed_and_transformed_modules():
    events = []
    model = _HookModel(events)

    ModelLoader._walk_transform(model)
    ModelLoader._walk_cache_state(model)
    ModelLoader._walk_full_post_load(model)

    assert events == [
        ("model", "transform_weights"),
        ("child", "transform_weights"),
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

    ModelLoader._reset_weights_transformed(model)

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

    ModelLoader._mark_weights_transformed(model)

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
    monkeypatch.setattr(mla_mod, "get_sm_version", lambda: 120)
    quant_mode = SimpleNamespace(has_fp8_block_scales=lambda: True)
    mla = MLA.__new__(MLA)
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
        model_loader_mod.AutoModelForCausalLM,
        "from_config",
        MagicMock(return_value=_ManifestTinyModel(events)),
    )
    if role is None:
        monkeypatch.delenv(WEIGHT_MANIFEST_DIR_ENV, raising=False)
        monkeypatch.delenv(WEIGHT_MANIFEST_ROLE_ENV, raising=False)
    else:
        monkeypatch.setenv(WEIGHT_MANIFEST_DIR_ENV, str(tmp_path))
        monkeypatch.setenv(WEIGHT_MANIFEST_ROLE_ENV, role)
    real_write = model_loader_mod.maybe_write_weight_manifest

    def _recording_write(*args, **kwargs):
        events.append("manifest")
        return real_write(*args, **kwargs)

    monkeypatch.setattr(model_loader_mod, "maybe_write_weight_manifest", _recording_write)
    return loader


def _hf_checkpoint_loader() -> MagicMock:
    checkpoint_loader = MagicMock(name="checkpoint_loader")
    checkpoint_loader.checkpoint_format = "HF"
    checkpoint_loader.is_weights_preloaded.return_value = False
    checkpoint_loader.load_weights.return_value = {"weight": MagicMock()}
    return checkpoint_loader


@pytest.mark.cpu_only
def test_load_writes_final_manifest_at_end_of_load(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    events = []
    loader = _make_manifest_loader(monkeypatch, tmp_path, events=events, role="baseline")

    model, _ = loader.load("/ckpt", _hf_checkpoint_loader())

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
    checkpoint_loader = MagicMock(name="checkpoint_loader")
    checkpoint_loader.checkpoint_format = "MX"
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
