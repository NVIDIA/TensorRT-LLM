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
"""MX post-transform qualification tests for the Qwen3.5 dense vision-language root."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import transformers
from torch import nn
from utils.post_transform_qualification import (
    AllReduceStub,
    PostTransformQualificationCase,
    assert_post_transform_lifecycle_equivalent,
    module_state_snapshot,
    mx_canonical_parameter_catalog,
)

import tensorrt_llm.mapping as mapping_mod
from tensorrt_llm._torch import distributed as distributed_mod
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models import modeling_qwen3_5 as modeling_qwen3_5_mod
from tensorrt_llm._torch.models import modeling_qwen3_next as modeling_qwen3_next_mod
from tensorrt_llm._torch.models import modeling_qwen3vl as modeling_qwen3vl_mod
from tensorrt_llm._torch.models.checkpoints.hf.qwen3_5_weight_mapper import Qwen3_5MoeHfWeightMapper
from tensorrt_llm._torch.models.modeling_utils import MetaInitMode
from tensorrt_llm._torch.modules.linear import Linear
from tensorrt_llm._torch.pyexecutor.config_utils import load_pretrained_config
from tensorrt_llm._torch.pyexecutor.model_loader import ModelLoader
from tensorrt_llm._torch.weight_sharing import (
    ARTIFACT_IDENTITY_FORMAT_VERSION,
    QWEN3_5_DENSE_VL_POST_TRANSFORM_LAYOUT_ABI_V1,
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

_PROFILE_ID = "qwen3_5-for-conditional-generation-bf16-complete-v1"
# ModelLoader captures this identity before construction; the constructed root
# rewrites its own `pretrained_config` to the text config.
_CONFIG_IDENTITY = PostTransformConfigIdentity("Qwen3_5ForConditionalGeneration", "qwen3_5")
_ARTIFACT_IDENTITY = ArtifactIdentity(
    format_version=ARTIFACT_IDENTITY_FORMAT_VERSION,
    scheme="checkpoint_manifest_sha256",
    digest="0" * 64,
)
_MAX_NUM_TOKENS = 16


class _UnqualifiedQwen3_5VLModel(modeling_qwen3_5_mod.Qwen3_5VLModel):
    """Subclass that must not inherit the exact-root Qwen3.5 profile."""


def _tiny_hf_config_dict(*, tie_word_embeddings: bool = False) -> dict:
    """Tiny Qwen3.5 dense VLM config with the real checkpoints' M-RoPE layout.

    `head_dim` 256 with `partial_rotary_factor` 0.25 gives the rotary width
    that the real `[11, 11, 10]` M-RoPE sections require.
    """
    return {
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "image_token_id": 60,
        "model_type": "qwen3_5",
        "text_config": {
            "attention_bias": False,
            "attention_dropout": 0.0,
            "attn_output_gate": True,
            "dtype": "bfloat16",
            "eos_token_id": 59,
            "full_attention_interval": 2,
            "head_dim": 256,
            "hidden_act": "silu",
            "hidden_size": 64,
            "intermediate_size": 128,
            "linear_conv_kernel_dim": 4,
            "linear_key_head_dim": 16,
            "linear_num_key_heads": 2,
            "linear_num_value_heads": 4,
            "linear_value_head_dim": 16,
            "mamba_ssm_dtype": "float32",
            "max_position_embeddings": 64,
            "mlp_only_layers": [],
            "model_type": "qwen3_5_text",
            "num_attention_heads": 4,
            "num_hidden_layers": 2,
            "num_key_value_heads": 2,
            "rms_norm_eps": 1e-6,
            "rope_parameters": {
                "mrope_interleaved": True,
                "mrope_section": [11, 11, 10],
                "partial_rotary_factor": 0.25,
                "rope_theta": 10000000.0,
                "rope_type": "default",
            },
            "tie_word_embeddings": False,
            "use_cache": True,
            "vocab_size": 64,
        },
        "tie_word_embeddings": tie_word_embeddings,
        "video_token_id": 61,
        "vision_config": {
            "deepstack_visual_indexes": [],
            "depth": 1,
            "hidden_act": "gelu_pytorch_tanh",
            "hidden_size": 32,
            "in_channels": 3,
            "intermediate_size": 64,
            "model_type": "qwen3_5",
            "num_heads": 2,
            "num_position_embeddings": 16,
            "out_hidden_size": 64,
            "patch_size": 16,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
        },
        "vision_end_token_id": 63,
        "vision_start_token_id": 62,
    }


def _write_tiny_checkpoint_config(path: Path, **config_overrides: object) -> Path:
    """Write the tiny `config.json` into `path` and return `path`."""
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text(
        json.dumps(_tiny_hf_config_dict(**config_overrides)), encoding="utf-8"
    )
    return path


def _tiny_qwen3_5_vl_model(
    checkpoint_dir: Path,
    *,
    model_class: type[nn.Module] = modeling_qwen3_5_mod.Qwen3_5VLModel,
    tp_size: int = 1,
    rank: int = 0,
    max_num_tokens: int = _MAX_NUM_TOKENS,
    disable_mm_encoder: bool = False,
) -> nn.Module:
    """Build a tiny root through the production config loader and normalizer."""
    model_config = ModelConfig(
        pretrained_config=load_pretrained_config(str(checkpoint_dir)),
        mapping=mapping_mod.Mapping(world_size=tp_size, rank=rank, tp_size=tp_size),
        max_num_tokens=max_num_tokens,
        max_seq_len=max_num_tokens,
        disable_mm_encoder=disable_mm_encoder,
    )
    with torch.device("cuda"):
        model = model_class(model_config)
    # Distinct non-zero values, so a missed copy cannot pass by accident.
    with torch.no_grad():
        for index, parameter in enumerate(model.parameters()):
            parameter.fill_((index % 13 + 1) / 64)
    return model


def _tiny_hf_state_dict(*, tie_word_embeddings: bool = False) -> dict[str, torch.Tensor]:
    """Hugging Face-format checkpoint tensors for the tiny config."""
    hf_config = transformers.Qwen3_5Config.from_dict(
        _tiny_hf_config_dict(tie_word_embeddings=tie_word_embeddings)
    )
    hf_model = transformers.Qwen3_5ForConditionalGeneration(hf_config).to(torch.bfloat16)
    state_dict = {}
    for index, (name, tensor) in enumerate(hf_model.state_dict().items()):
        values = torch.arange(tensor.numel(), dtype=torch.float32).reshape(tensor.shape)
        state_dict[name] = (((values + index) % 17) / 17).to(tensor.dtype)
    return state_dict


def _hf_checkpoint_loader(state_dict: dict[str, torch.Tensor]):
    """Return a `prepare_producer` hook that loads `state_dict` like a real HF load."""

    def _load(model: nn.Module) -> None:
        mapper = Qwen3_5MoeHfWeightMapper()
        mapper.init_model_and_config(model, model.model_config)
        model.load_weights(dict(state_dict), mapper)

    return _load


def _qualify(model: nn.Module):
    """Qualify `model` with the identity that `ModelLoader` captures before construction."""
    return ModelLoader._qualify_post_transform_profile(
        model,
        config_identity=_CONFIG_IDENTITY,
        speculative_mode=None,
        loads_draft_weights=False,
    )


def _runtime_config(model: nn.Module) -> PostTransformRuntimeConfig:
    """Realize the runtime config of a constructed root."""
    return PostTransformRuntimeConfig.from_model_config(model.model_config, model=model)


def _qwen3_5_vl_runtime_config(**overrides: object) -> PostTransformRuntimeConfig:
    """The realized runtime config of the qualified TP1 envelope, with overrides."""
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
        "rope_type": "mrope",
        "rope_fusion": False,
        "sliding_window": None,
    }
    values.update(overrides)
    return PostTransformRuntimeConfig(**values)


def _qwen3_5_vl_layout_state(model: nn.Module) -> dict[str, object]:
    """Structural and realized facts that the profile's layout ABI depends on."""
    llm = model.llm
    layers = llm.model.layers
    full_attention = layers[1].self_attn
    gated_mlps = [module for module in model.modules() if hasattr(module, "gate_up_proj")]
    state = {
        "mm_encoder_present": model.mm_encoder is not None,
        "deepstack_num_level": model.deepstack_num_level,
        "layer_types": tuple(type(layer).__name__ for layer in layers),
        "next_layer_layernorm_aliases": tuple(
            layer.next_layer_layernorm
            is (layers[index + 1].input_layernorm if index + 1 < len(layers) else llm.model.norm)
            for index, layer in enumerate(layers)
        ),
        "gdn_norm_fp8_scale_is_none": layers[0].linear_attn.norm.fp8_scale is None,
        "share_gate_up_quantize": tuple(mlp._maybe_share_gate_up_quantize for mlp in gated_mlps),
        "fuse_qk_norm_rope": full_attention.fuse_qk_norm_rope,
        "rope_fusion": full_attention.rope_fusion,
        "tied_lm_head": llm.lm_head.weight.data_ptr() == llm.model.embed_tokens.weight.data_ptr(),
    }
    if model.mm_encoder is not None:
        vision_config = model.mm_encoder.visual.config
        state["vision_num_attention_heads"] = vision_config.num_attention_heads
    return state


def _derived_tensor_caches(model: nn.Module) -> dict[str, torch.Tensor]:
    """Plain tensor attributes, such as the Gemma-norm `_fused_norm_weight` caches."""
    caches = {}
    for module_name, module in model.named_modules():
        for attribute, value in vars(module).items():
            if isinstance(value, torch.Tensor) and not isinstance(value, nn.Parameter):
                caches[f"{module_name}.{attribute}"] = value
    return caches


def _vision_patch_embed_output(model: nn.Module) -> torch.Tensor:
    """Run the vision patch embedding on fixed flattened patches."""
    patch_embed = model.mm_encoder.visual.patch_embed
    weight = patch_embed.proj.weight
    patch_size = weight[0].numel()
    patches = torch.arange(4 * patch_size, dtype=torch.float32, device=weight.device)
    return patch_embed(((patches % 7) / 7).reshape(4, patch_size).to(weight.dtype))


def _linear_outputs(model: nn.Module) -> dict[str, torch.Tensor]:
    """Run every exact `Linear` module, fused projections included, on a fixed input."""
    outputs = {}
    for module_name, module in model.named_modules():
        if type(module) is not Linear:
            continue
        inputs = torch.arange(
            3 * module.in_features, dtype=torch.float32, device=module.weight.device
        ).reshape(3, module.in_features)
        outputs[module_name] = module(((inputs % 11) / 11).to(module.weight.dtype))
    return outputs


def _embedding_logits(model: nn.Module) -> torch.Tensor:
    """Run token embedding straight into the LM head."""
    llm = model.llm
    token_ids = torch.arange(8, device=llm.model.embed_tokens.weight.device)
    return llm.lm_head(llm.model.embed_tokens(token_ids))


def _assert_every_parameter_was_loaded(producer: nn.Module, factory_model: nn.Module) -> None:
    """Assert that `prepare_producer` overwrote every parameter of the factory fill.

    The lifecycle probes then compare bytes that came from the checkpoint
    through the weight mapper rather than the factory's placeholder values.
    """
    # Look parameters up by the factory's canonical names: after
    # `setup_aliases()` the producer also reaches some of them through
    # `next_layer_layernorm` alias names.
    unchanged = [
        name
        for name, factory_parameter in factory_model.named_parameters()
        if torch.equal(producer.get_parameter(name), factory_parameter)
    ]
    assert unchanged == [], f"Parameters not written by the producer load: {unchanged}"


def _lifecycle_case(
    checkpoint_dir: Path,
    state_dict: dict[str, torch.Tensor],
    *,
    tp_size: int = 1,
    rank: int = 0,
) -> PostTransformQualificationCase:
    """Build the lifecycle case whose producer runs the real HF load of `state_dict`."""
    output_probes = [
        ("linear-outputs", _linear_outputs),
        ("vision-patch-embed", _vision_patch_embed_output),
        ("derived-tensor-caches", _derived_tensor_caches),
    ]
    if tp_size == 1:
        output_probes.append(("embedding-logits", _embedding_logits))
    return PostTransformQualificationCase(
        profile_id=_PROFILE_ID,
        model_factory=lambda: _tiny_qwen3_5_vl_model(checkpoint_dir, tp_size=tp_size, rank=rank),
        unqualified_model_factory=lambda: _tiny_qwen3_5_vl_model(
            checkpoint_dir,
            model_class=_UnqualifiedQwen3_5VLModel,
            tp_size=tp_size,
            rank=rank,
        ),
        qualify_model=_qualify,
        state_probes=(
            ("layout", _qwen3_5_vl_layout_state),
            ("module-state", module_state_snapshot),
        ),
        output_probes=tuple(output_probes),
        prepare_producer=_hf_checkpoint_loader(state_dict),
    )


def test_qwen3_5_vl_profile_qualifies_full_staged_lifecycle(tmp_path: Path) -> None:
    """A staged complete-model receiver matches a producer that ran the real load path.

    The producer loads a Hugging Face-format state dict through the production
    weight mapper and then runs every post-load hook; the receiver only gets
    the producer's parameters and the staged finalization walks.
    """
    checkpoint_dir = _write_tiny_checkpoint_config(tmp_path / "ckpt")
    state_dict = _tiny_hf_state_dict()

    producer, receiver = assert_post_transform_lifecycle_equivalent(
        _lifecycle_case(checkpoint_dir, state_dict)
    )

    _assert_every_parameter_was_loaded(producer, _tiny_qwen3_5_vl_model(checkpoint_dir))

    # The producer's bytes come from the checkpoint through the weight mapper.
    for trtllm_name, hf_name in (
        ("llm.model.embed_tokens.weight", "model.language_model.embed_tokens.weight"),
        ("llm.lm_head.weight", "lm_head.weight"),
        ("mm_encoder.visual.patch_embed.proj.weight", "model.visual.patch_embed.proj.weight"),
    ):
        assert torch.equal(producer.get_parameter(trtllm_name).cpu(), state_dict[hf_name]), (
            trtllm_name
        )
    assert any(name.endswith("._fused_norm_weight") for name in _derived_tensor_caches(producer))
    decision = _qualify(receiver)
    assert decision.transform_abi_id == QWEN3_5_DENSE_VL_POST_TRANSFORM_LAYOUT_ABI_V1
    assert (
        ModelLoader._post_transform_transfer_scope(receiver)
        is PostTransformTransferScope.COMPLETE_MODEL
    )
    assert _runtime_config(producer) == _qwen3_5_vl_runtime_config()
    assert _qwen3_5_vl_layout_state(producer) == {
        "mm_encoder_present": True,
        "deepstack_num_level": 0,
        "layer_types": ("Qwen3NextLinearDecoderLayer", "Qwen3NextFullAttentionDecoderLayer"),
        "next_layer_layernorm_aliases": (True, True),
        "gdn_norm_fp8_scale_is_none": True,
        "share_gate_up_quantize": (False, False),
        "fuse_qk_norm_rope": True,
        "rope_fusion": False,
        "tied_lm_head": False,
        "vision_num_attention_heads": 2,
    }
    # The vision tower is part of the transfer: its parameters are not owned
    # by the language model.
    language_model_parameters = {id(parameter) for parameter in producer.llm.parameters()}
    assert any(
        id(parameter) not in language_model_parameters
        for parameter in producer.mm_encoder.parameters()
    )
    for model in (producer, receiver):
        workspace = model.mrope_rotary_cos_sin_workspace
        assert workspace.shape[-1] % _MAX_NUM_TOKENS == 0
        assert not workspace.any()


def test_qwen3_5_vl_mx_tensor_catalog_is_complete_and_stable(tmp_path: Path) -> None:
    """The MX tensor catalog holds vision and language tensors and survives staged finalization.

    MX registers tensors after the receiver's alias preparation and publishes
    them after finalization, so the catalog must not change in between and must
    match the producer's names, shapes, and dtypes.
    """
    checkpoint_dir = _write_tiny_checkpoint_config(tmp_path / "ckpt")
    producer = _tiny_qwen3_5_vl_model(checkpoint_dir)
    _hf_checkpoint_loader(_tiny_hf_state_dict())(producer)
    ModelLoader._walk_full_post_load(producer)
    receiver = _tiny_qwen3_5_vl_model(checkpoint_dir)

    ModelLoader._setup_aliases(receiver)
    registered_catalog = mx_canonical_parameter_catalog(receiver)
    ModelLoader._setup_aliases(receiver)
    ModelLoader._mark_weights_transformed(receiver)
    ModelLoader._walk_cache_state(receiver)
    published_catalog = mx_canonical_parameter_catalog(receiver)

    assert published_catalog == registered_catalog
    assert any(name.startswith("mm_encoder.") for name in registered_catalog)
    assert any(name.startswith("llm.") for name in registered_catalog)
    producer_layout = {
        name: entry[1:] for name, entry in mx_canonical_parameter_catalog(producer).items()
    }
    assert {name: entry[1:] for name, entry in registered_catalog.items()} == producer_layout


def test_qwen3_vl_mrope_workspace_is_zero_initialized_under_meta_init(tmp_path: Path) -> None:
    """Meta-device construction still allocates the M-RoPE workspace as CUDA zeros.

    `ModelLoader` constructs roots under `MetaInitMode`, which turns `empty`
    allocations into meta tensors that are later materialized uninitialized;
    weight manifests hash this buffer, so its bytes must be deterministic.
    """
    model_config = ModelConfig(
        pretrained_config=load_pretrained_config(
            str(_write_tiny_checkpoint_config(tmp_path / "ckpt"))
        ),
        max_num_tokens=_MAX_NUM_TOKENS,
        max_seq_len=_MAX_NUM_TOKENS,
    )
    with MetaInitMode():
        model = modeling_qwen3_5_mod.Qwen3_5VLModel(model_config)

    workspace = model.mrope_rotary_cos_sin_workspace
    assert workspace.device.type == "cuda"
    assert not workspace.any()


def test_qwen3_5_vl_qualification_requires_preconstruction_identity(tmp_path: Path) -> None:
    """The constructed root reports the inner text identity, which never qualifies.

    `ModelLoader` therefore qualifies with the identity it captured before
    construction.
    """
    model = _tiny_qwen3_5_vl_model(_write_tiny_checkpoint_config(tmp_path / "ckpt"))

    post_construction_identity = PostTransformConfigIdentity.from_model_config(model.model_config)
    decision = ModelLoader._qualify_post_transform_profile(
        model,
        config_identity=post_construction_identity,
        speculative_mode=None,
        loads_draft_weights=False,
    )

    assert post_construction_identity == PostTransformConfigIdentity(
        "Qwen3_5ForCausalLM", "qwen3_5_text"
    )
    assert decision.reason is PostTransformQualificationReason.ARCHITECTURE_NOT_REGISTERED
    assert _qualify(model).qualified


@pytest.mark.parametrize("rank", [0, 1])
def test_qwen3_5_vl_profile_qualifies_tp2_rank_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    rank: int,
) -> None:
    """Each simulated TP2 rank's staged receiver matches its fully loaded producer."""
    monkeypatch.setattr(mapping_mod, "mpi_disabled", lambda: False)
    monkeypatch.setattr(distributed_mod, "AllReduce", AllReduceStub)
    monkeypatch.setattr(modeling_qwen3_next_mod, "AllReduce", AllReduceStub)
    checkpoint_dir = _write_tiny_checkpoint_config(tmp_path / "ckpt")

    producer, _receiver = assert_post_transform_lifecycle_equivalent(
        _lifecycle_case(checkpoint_dir, _tiny_hf_state_dict(), tp_size=2, rank=rank)
    )

    _assert_every_parameter_was_loaded(
        producer, _tiny_qwen3_5_vl_model(checkpoint_dir, tp_size=2, rank=rank)
    )

    assert _runtime_config(producer) == _qwen3_5_vl_runtime_config(
        tp_size=2,
        moe_tp_size=2,
        attention_tp_size=2,
    )
    full_attention = producer.llm.model.layers[1].self_attn
    assert full_attention.qkv_proj.weight.shape[0] == (4 * 2 + 2 + 2) * 256 // 2


@pytest.mark.parametrize(
    ("top_level_tie", "text_tie", "qualified"),
    [(False, True, True), (True, False, False)],
    ids=["untied-top-level", "tied-top-level"],
)
def test_qwen3_5_vl_tied_dimension_follows_top_level_flag(
    tmp_path: Path,
    top_level_tie: bool,
    text_tie: bool,
    qualified: bool,
) -> None:
    """The realized tie flag mirrors the checkpoint's top-level `tie_word_embeddings`.

    The root copies the top-level flag onto the text config, so a stray text
    default cannot tie an untied checkpoint, and tied checkpoints fall back.
    """
    checkpoint_dir = _write_tiny_checkpoint_config(
        tmp_path / "ckpt", tie_word_embeddings=top_level_tie
    )
    config_path = checkpoint_dir / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["text_config"]["tie_word_embeddings"] = text_tie
    config_path.write_text(json.dumps(config), encoding="utf-8")

    decision = _qualify(_tiny_qwen3_5_vl_model(checkpoint_dir))

    assert decision.qualified is qualified
    if not qualified:
        assert decision.reason is PostTransformQualificationReason.RUNTIME_CONFIG_NOT_SUPPORTED
        assert decision.unsupported_runtime_dimensions == frozenset({"tied_word_embeddings"})


@pytest.mark.parametrize("encoder_mode", ["disable_mm_encoder", "mm_disaggregated"])
def test_qwen3_5_vl_without_local_encoder_falls_back(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    encoder_mode: str,
) -> None:
    """Without its local encoder the root carries only the language model and falls back."""
    if encoder_mode == "mm_disaggregated":
        monkeypatch.setenv("TLLM_MULTIMODAL_DISAGGREGATED", "1")
    model = _tiny_qwen3_5_vl_model(
        _write_tiny_checkpoint_config(tmp_path / "ckpt"),
        disable_mm_encoder=encoder_mode == "disable_mm_encoder",
    )

    assert model.mm_encoder is None
    assert (
        ModelLoader._post_transform_transfer_scope(model)
        is PostTransformTransferScope.LANGUAGE_MODEL
    )
    assert _qualify(model).reason is PostTransformQualificationReason.TRANSFER_SCOPE_NOT_REGISTERED


def test_qwen3_5_vl_source_identity_binds_runtime_max_num_tokens(tmp_path: Path) -> None:
    """Donors and receivers with different `max_num_tokens` do not share.

    The root's M-RoPE buffers are sized by `max_num_tokens`, and `SourceIdentity`
    hashes every realized buffer shape, so all roles must use the same value.
    """
    checkpoint_dir = _write_tiny_checkpoint_config(tmp_path / "ckpt")
    identities = []
    for max_num_tokens in (16, 32):
        model = _tiny_qwen3_5_vl_model(checkpoint_dir, max_num_tokens=max_num_tokens)
        identities.append(
            SourceIdentity.from_model_config(
                model.model_config,
                model,
                artifact_identity=_ARTIFACT_IDENTITY,
                transform_abi_id=QWEN3_5_DENSE_VL_POST_TRANSFORM_LAYOUT_ABI_V1,
            )
        )

    decision = check_weight_sharing_compatibility(*identities, IdentityCheckPolicy.WARN_FALLBACK)

    assert not decision.should_share
    assert decision.match_result.mismatched_fields == ["shard_fingerprint"]


def _qualify_registry(**overrides: object):
    """Qualify the registered Qwen3.5 identity directly, with `overrides` applied."""
    arguments = {
        "root_model_class": modeling_qwen3_5_mod.Qwen3_5VLModel,
        "architecture": "Qwen3_5ForConditionalGeneration",
        "model_type": "qwen3_5",
        "speculative_mode": None,
        "protocol_version": ModelLoader._MX_STAGED_RECEIVER_TRANSFORM_PROTOCOL_VERSION,
        "transfer_scope": PostTransformTransferScope.COMPLETE_MODEL,
        "runtime_config": _qwen3_5_vl_runtime_config(),
    }
    arguments.update(overrides)
    return ModelLoader._post_transform_profile_registry().qualify(**arguments)


@pytest.mark.cpu_only
def test_qwen3_5_vl_profile_matches_its_exact_identity() -> None:
    """The registered identity qualifies with the Qwen3.5 dense VL ABI."""
    decision = _qualify_registry()

    assert decision.qualified
    assert decision.profile.profile_id == _PROFILE_ID
    assert decision.transform_abi_id == QWEN3_5_DENSE_VL_POST_TRANSFORM_LAYOUT_ABI_V1


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        pytest.param(
            {"root_model_class": modeling_qwen3_5_mod.Qwen3_5ForCausalLM},
            PostTransformQualificationReason.ROOT_MODEL_CLASS_NOT_REGISTERED,
            id="text-only-root",
        ),
        pytest.param(
            {"root_model_class": modeling_qwen3_5_mod.Qwen3_5MoeVLModel},
            PostTransformQualificationReason.ROOT_MODEL_CLASS_NOT_REGISTERED,
            id="moe-vlm-root",
        ),
        pytest.param(
            {"root_model_class": modeling_qwen3vl_mod.Qwen3VLModel},
            PostTransformQualificationReason.ROOT_MODEL_CLASS_NOT_REGISTERED,
            id="qwen3-vl-root",
        ),
        pytest.param(
            {"root_model_class": modeling_qwen3vl_mod.Qwen3VisionModelBase},
            PostTransformQualificationReason.ROOT_MODEL_CLASS_NOT_REGISTERED,
            id="encoder-only-root",
        ),
        pytest.param(
            {"architecture": "Qwen3_5MoeForConditionalGeneration"},
            PostTransformQualificationReason.ARCHITECTURE_NOT_REGISTERED,
            id="moe-architecture",
        ),
        pytest.param(
            {"model_type": "qwen3_5_moe"},
            PostTransformQualificationReason.MODEL_TYPE_NOT_REGISTERED,
            id="moe-model-type",
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
    ],
)
def test_qwen3_5_vl_profile_is_exact(
    overrides: dict[str, object],
    reason: PostTransformQualificationReason,
) -> None:
    """Neighboring roots, identities, speculative modes, and scopes do not qualify.

    `Qwen3VisionModelBase` is the root that `mm_encoder_only` workers build.
    """
    decision = _qualify_registry(**overrides)

    assert not decision.qualified
    assert decision.reason is reason


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        pytest.param(
            {"speculative_mode": "mtp"},
            PostTransformQualificationReason.SPECULATIVE_MODE_NOT_REGISTERED,
            id="mtp",
        ),
        pytest.param(
            {"speculative_mode": "dflash"},
            PostTransformQualificationReason.SPECULATIVE_MODE_NOT_REGISTERED,
            id="dflash",
        ),
        pytest.param(
            {"enabled_features": frozenset({PostTransformFeature.SEPARATE_DRAFT_MODEL})},
            PostTransformQualificationReason.FEATURE_NOT_SUPPORTED,
            id="synthetic-separate-draft-model",
        ),
        pytest.param(
            {
                "enabled_features": frozenset(
                    {PostTransformFeature.LOCALITY_DOMAIN_LOCALIZED_WEIGHTS}
                )
            },
            PostTransformQualificationReason.FEATURE_NOT_SUPPORTED,
            id="synthetic-localized-weights",
        ),
    ],
)
def test_qwen3_5_vl_profile_rejects_speculative_and_draft_loads(
    overrides: dict[str, object],
    reason: PostTransformQualificationReason,
) -> None:
    """Speculative modes and lifecycle features fall back.

    The checkpoints ship MTP layers, and DFlash loads report their own mode, so
    both are rejected by speculative mode. The feature cases call the registry
    directly with `speculative_mode=None`.
    """
    decision = _qualify_registry(**overrides)

    assert not decision.qualified
    assert decision.reason is reason
    if reason is PostTransformQualificationReason.FEATURE_NOT_SUPPORTED:
        assert decision.unsupported_features == overrides["enabled_features"]


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    ("runtime_overrides", "dimension"),
    [
        pytest.param({"dtype": "float16"}, "dtype", id="float16"),
        pytest.param({"quant_algorithm": "FP8"}, "quant_algorithm", id="fp8"),
        pytest.param(
            {"quant_algorithm": "FP8_BLOCK_SCALES"}, "quant_algorithm", id="fp8-block-scales"
        ),
        pytest.param({"kv_cache_quant_algorithm": "FP8"}, "kv_cache_quant_algorithm", id="fp8-kv"),
        pytest.param({"layerwise_quantization": True}, "layerwise_quantization", id="layerwise"),
        pytest.param(
            {"force_dynamic_quantization": True},
            "force_dynamic_quantization",
            id="dynamic-quantization",
        ),
        pytest.param({"lora_enabled": True}, "lora_enabled", id="lora"),
        pytest.param({"sparse_attention_enabled": True}, "sparse_attention_enabled", id="sparse"),
        pytest.param({"attention_backend": "FLASHINFER"}, "attention_backend", id="flashinfer"),
        pytest.param({"tp_size": 4}, "tp_size", id="tp4"),
        pytest.param({"pp_size": 2}, "pp_size", id="pp2"),
        pytest.param({"cp_size": 2}, "cp_size", id="cp2"),
        pytest.param({"attention_tp_size": 4}, "attention_tp_size", id="attention-tp4"),
        pytest.param({"attention_cp_size": 2}, "attention_cp_size", id="attention-cp2"),
        pytest.param({"attention_dp": True}, "attention_dp", id="attention-dp"),
        pytest.param({"multi_node": True}, "multi_node", id="multi-node"),
        pytest.param({"rope_type": "default"}, "rope_type", id="rope-default"),
        pytest.param({"rope_type": "yarn"}, "rope_type", id="rope-yarn"),
        pytest.param({"rope_fusion": True}, "rope_fusion", id="rope-fusion"),
        pytest.param({"tied_word_embeddings": True}, "tied_word_embeddings", id="tied"),
    ],
)
def test_qwen3_5_vl_profile_rejects_unqualified_runtime_variants(
    runtime_overrides: dict[str, object],
    dimension: str,
) -> None:
    """Each unqualified runtime variant reports exactly the dimension that blocks it."""
    decision = _qualify_registry(runtime_config=_qwen3_5_vl_runtime_config(**runtime_overrides))

    assert decision.reason is PostTransformQualificationReason.RUNTIME_CONFIG_NOT_SUPPORTED
    assert decision.unsupported_runtime_dimensions == frozenset({dimension})


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "runtime_overrides",
    [
        {"moe_backend": "TRTLLM"},
        {"moe_backend": None},
        {"moe_tp_size": 2},
        {"moe_ep_size": 2},
    ],
)
def test_qwen3_5_vl_profile_ignores_moe_only_runtime_dimensions(
    runtime_overrides: dict[str, object],
) -> None:
    """The dense profile does not constrain MoE-only runtime dimensions."""
    decision = _qualify_registry(runtime_config=_qwen3_5_vl_runtime_config(**runtime_overrides))

    assert decision.qualified


@pytest.mark.cpu_only
def test_qwen3_5_vl_config_realizes_mrope_rope_type(tmp_path: Path) -> None:
    """The production config normalizer leaves a text config that realizes `mrope`.

    The checkpoints declare `rope_type: default` with M-RoPE sections; the
    normalizer flattens them into `{type: mrope}` without `rope_type`, which is
    the value the profile pins. A bare `{type: mrope}` dict realizes the same.
    """
    config = load_pretrained_config(str(_write_tiny_checkpoint_config(tmp_path / "ckpt")))

    realized = PostTransformRuntimeConfig.from_model_config(
        SimpleNamespace(pretrained_config=config.text_config)
    )
    flattened = PostTransformRuntimeConfig.from_model_config(
        SimpleNamespace(pretrained_config=SimpleNamespace(rope_scaling={"type": "mrope"}))
    )

    assert realized.rope_type == "mrope"
    assert flattened.rope_type == "mrope"
