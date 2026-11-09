# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Open-export ("muse_spark") support: config translation and key renaming.

The open export re-publishes the identical model with an HF composite config
and HF-conventional tensor names. Support is two pure functions -- a config
translation that must reproduce the flat config the runtime was written
against, and a key rename that must be a bijection onto the source spelling --
and both are checkable without a checkpoint. The key inventory here is
synthesized from the documented export structure (2360 tensors: 1479 text +
809 vision + 72 audio); that the released checkpoint actually carries these
names is a separate, measured fact (the offline key-map audit ran against both
real checkpoints: 2360/2360 mapped, image == the source export's canonical
set, shapes identical except the 62 loader-handled router transposes).
"""

import pytest
import torch

from tensorrt_llm._torch.configs.musespark_open import MuseSparkOpenConfig
from tensorrt_llm._torch.models.checkpoints.base_weight_loader import ConsumableWeightsDict
from tensorrt_llm._torch.models.checkpoints.hf.musespark_open_format import (
    OPEN_FORMAT_SENTINEL_KEY,
    build_open_format_rename,
    is_open_format,
    open_format_to_source_key,
)
from tensorrt_llm._torch.models.checkpoints.hf.musespark_open_weight_mapper import (
    VISION_PROJECTION_ALIAS,
    _load_router,
    interleave_rope_rows,
    missing_in_range_keys,
    missing_vision_keys,
    rename_open_format_weights,
)

pytestmark = pytest.mark.cpu_only

NUM_LAYERS = 62
NUM_VISION_LAYERS = 50
NUM_AUDIO_STAGES = 4
BF16_EXPERT_LAYERS = (0, 61)

AUDIO_ENCODER = "modalities.soft_audio_perception.interaction_encoder"


# ---------------------------------------------------------------------------
# Synthetic inventories, from the documented export structure.
# ---------------------------------------------------------------------------


def open_format_keys():
    keys = [
        "lm_head.weight",
        "model.language_model.embed_tokens.weight",
        "model.language_model.norm.weight",
    ]
    for i in range(NUM_LAYERS):
        prefix = f"model.language_model.layers.{i}"
        keys += [
            f"{prefix}.input_layernorm.weight",
            f"{prefix}.self_attn.q_proj.weight",
            f"{prefix}.self_attn.k_proj.weight",
            f"{prefix}.self_attn.v_proj.weight",
            f"{prefix}.self_attn.o_proj.weight",
            f"{prefix}.self_attn.gate_proj.weight",
            f"{prefix}.post_attention_residual_gate.gate",
            f"{prefix}.post_feedforward_residual_gate.gate",
            f"{prefix}.pre_feedforward_layernorm.weight",
            f"{prefix}.post_feedforward_layernorm.weight",
            f"{prefix}.mlp.gate.weight",
            f"{prefix}.mlp.gate.e_score_correction_bias",
            f"{prefix}.mlp.pre_expert_proj.weight",
            f"{prefix}.mlp.post_expert_proj.weight",
            f"{prefix}.mlp.pre_expert_norm.weight",
            f"{prefix}.mlp.experts.post_expert_norm.weight",
            f"{prefix}.mlp.experts.gate_up_proj",
            f"{prefix}.mlp.experts.down_proj",
        ]
        if i not in BF16_EXPERT_LAYERS:
            keys += [
                f"{prefix}.mlp.experts.gate_up_proj_weight_scale",
                f"{prefix}.mlp.experts.gate_up_proj_weight_scale_2",
                f"{prefix}.mlp.experts.gate_up_proj_input_scale",
                f"{prefix}.mlp.experts.down_proj_weight_scale",
                f"{prefix}.mlp.experts.down_proj_weight_scale_2",
                f"{prefix}.mlp.experts.down_proj_input_scale",
            ]
    keys += [
        "model.vision_tower.patch_embedder.patch_embedding.weight",
        "model.vision_tower.patch_embedder.position_embedding_table.weight",
        "model.vision_tower.ln_pre.weight",
        "model.vision_tower.ln_pre.bias",
        "model.vision_tower.ln_post.weight",
        "model.vision_tower.ln_post.bias",
        "model.vision_adapter.fc1.weight",
        "model.vision_adapter.fc2.weight",
        "model.vision_projection.weight",
    ]
    for i in range(NUM_VISION_LAYERS):
        prefix = f"model.vision_tower.layers.{i}"
        for name in ("norm1", "norm2"):
            keys += [f"{prefix}.{name}.weight", f"{prefix}.{name}.bias"]
        for name in ("q_proj", "k_proj", "v_proj", "proj"):
            keys += [f"{prefix}.attn.{name}.weight", f"{prefix}.attn.{name}.bias"]
        for name in ("fc1", "fc2"):
            keys += [f"{prefix}.mlp.{name}.weight", f"{prefix}.mlp.{name}.bias"]
    keys += [
        "model.audio_tower.norm.weight",
        "model.audio_tower.post_proj_norm.weight",
        "model.audio_tower.proj.weight",
        "model.audio_tower.proj.bias",
    ]
    for i in range(NUM_AUDIO_STAGES):
        prefix = f"model.audio_tower.layers.{i}"
        keys += [
            f"{prefix}.patch_embedder.norm.weight",
            f"{prefix}.patch_embedder.proj.weight",
            f"{prefix}.patch_embedder.proj.bias",
        ]
        for block in range(2):
            block_prefix = f"{prefix}.conv_blocks.{block}"
            keys += [
                f"{block_prefix}.conv.weight",
                f"{block_prefix}.conv.bias",
                f"{block_prefix}.mlp.fc1.weight",
                f"{block_prefix}.mlp.fc1.bias",
                f"{block_prefix}.mlp.fc2.weight",
                f"{block_prefix}.mlp.fc2.bias",
                f"{block_prefix}.norm.weight",
            ]
    return keys


def source_format_keys():
    """The source spelling of the same 2360 tensors.

    Text and vision come from the loader's own completeness inventories (the
    functions the load-accounting test uses), so this test cannot drift from
    what the loader actually consumes; the audio stage walk is spelled out
    (the loader derives it from the tower's module tree, which needs a built
    tower).
    """
    keys = missing_in_range_keys([], NUM_LAYERS, list(BF16_EXPERT_LAYERS))
    vision = missing_vision_keys([], NUM_VISION_LAYERS)
    # The open export carries the projection once, under the bare alias.
    keys += [VISION_PROJECTION_ALIAS if ".vision_projection." in key else key for key in vision]
    keys += [
        f"{AUDIO_ENCODER}.norm.weight",
        f"{AUDIO_ENCODER}.dual_norm.weight",
        f"{AUDIO_ENCODER}.o_proj.weight",
        f"{AUDIO_ENCODER}.o_proj.bias",
    ]
    for i in range(NUM_AUDIO_STAGES):
        prefix = f"{AUDIO_ENCODER}.stages.{i}"
        keys += [
            f"{prefix}.0.norm.weight",
            f"{prefix}.0.linear.weight",
            f"{prefix}.0.linear.bias",
        ]
        for block in (1, 2):
            keys += [
                f"{prefix}.{block}.conv.conv.weight",
                f"{prefix}.{block}.conv.conv.bias",
                f"{prefix}.{block}.linear1.weight",
                f"{prefix}.{block}.linear1.bias",
                f"{prefix}.{block}.linear2.weight",
                f"{prefix}.{block}.linear2.bias",
                f"{prefix}.{block}.norm.weight",
            ]
    return keys


def open_format_config_dict():
    """The release export's config, reduced to the fields the translation reads."""
    layer_types = []
    layer_rope_theta = []
    for i in range(NUM_LAYERS):
        nope = (NUM_LAYERS - i - 1) % 4 == 0
        layer_types.append("full_attention" if nope else "sliding_attention")
        layer_rope_theta.append(0 if nope else 500000.0)
    return {
        "architectures": ["MuseSparkForConditionalGeneration"],
        "model_type": "muse_spark",
        "dtype": "bfloat16",
        "image_token_id": 200092,
        "video_token_id": 200091,
        "audio_token_id": 200038,
        "out_hidden_size": 6144,
        "projector_hidden_size": 4096,
        "projector_hidden_act": "gelu",
        "text_config": {
            "model_type": "muse_spark_text",
            "vocab_size": 202048,
            "hidden_size": 8192,
            "num_hidden_layers": NUM_LAYERS,
            "num_attention_heads": 128,
            "num_key_value_heads": 16,
            "head_dim": 64,
            "max_position_embeddings": 1048576,
            "rms_norm_eps": 1e-5,
            "post_norm_eps": 1e-8,
            "rope_parameters": {"rope_theta": 500000.0, "rope_type": "default"},
            "layer_types": layer_types,
            "layer_rope_theta": layer_rope_theta,
            "sliding_window": 2048,
            "mlp_layer_types": ["sparse"] * NUM_LAYERS,
            "hidden_act": "silu",
            "attention_bias": False,
            "attention_dropout": 0.0,
            "intermediate_size": 16384,
            "moe_hidden_size": 4096,
            "moe_intermediate_size": 4096,
            "num_local_experts": 256,
            "num_experts_per_tok": 8,
            "num_shared_experts": 0,
            "routed_scaling_factor": 1.0,
            "router_score_division_eps": 1e-15,
            "final_logit_softcapping": 20.0,
            "output_multiplier": 0.17677669529663687,
            "qk_scale_factor": 5.4730064863875,
            "residual_gate_temperature": 0.3,
            "tie_word_embeddings": False,
            "bos_token_id": 200000,
            "eos_token_id": 200001,
            "pad_token_id": 200018,
        },
        "vision_config": {
            "model_type": "muse_spark_vision",
            "hidden_size": 1536,
            "num_hidden_layers": NUM_VISION_LAYERS,
            "num_attention_heads": 16,
            "intermediate_size": 8960,
            "layer_norm_eps": 1e-5,
            "patch_size": 14,
            "patch_temporal": 2,
            "merge_size": 2,
            "pos_emb_height": 32,
            "pos_emb_width": 32,
            "rope_parameters": {"rope_theta": 10000, "rope_type": "axial"},
            "hidden_act": "gelu",
            "layer_types": [
                "full_attention"
                if (i + 1) % 4 == 0 or i == NUM_VISION_LAYERS - 1
                else "window_attention"
                for i in range(NUM_VISION_LAYERS)
            ],
        },
        "audio_config": {
            "model_type": "muse_spark_audio",
            "patch_sizes": [16, 6, 5, 4],
            "hidden_sizes": [256, 512, 1024, 2048],
            "depths": [2, 2, 2, 2],
            "num_hidden_layers": 4,
            "conv_kernel_size": 7,
            "hidden_act": "gelu",
            "mlp_ratio": 4.0,
            "out_hidden_size": 8192,
            "rms_norm_eps": 1e-6,
        },
        "quantization_config": {"quant_algo": "NVFP4", "group_size": 16},
    }


# ---------------------------------------------------------------------------
# Config translation
# ---------------------------------------------------------------------------


def test_from_open_format_reproduces_flat_config():
    flat = MuseSparkOpenConfig.from_open_format(open_format_config_dict())
    reference = MuseSparkOpenConfig(
        moe={"topk": 8, "num_experts": 256, "override_routed_hidden_dim": 4096},
        metap={"base_width": 256.0},
        vision_config={
            "navit_config": {"vision_output_dim": 6144},
            "projector_input_dim": 4096,
        },
    )
    for field in (
        "vocab_size",
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "max_position_embeddings",
        "rms_norm_eps",
        "rms_post_norm_eps",
        "rope_theta",
        "every_n_layers_nope",
        "global_attn_cfg",
        "output_soft_cap_temp",
        "num_experts",
        "moe_topk",
        "compression_factor",
        "expert_hidden_size",
        "intermediate_size",
        "sliding_window",
        "layer_types",
        "use_rope_per_layer",
        "scale_query_by",
        "metap_multiplier",
        "patch_token_id",
        "video_token_id",
        "audio_token_id",
    ):
        assert getattr(flat, field) == getattr(reference, field), field
    for field in (
        "hidden_size",
        "num_hidden_layers",
        "intermediate_size",
        "downsample_factor",
        "sparse_attention_factor",
        "image_size",
        "vision_output_dim",
        "projector_input_dim",
        "full_attention_layers",
    ):
        assert getattr(flat.vision_config, field) == getattr(reference.vision_config, field), field
    # The audio tower's config gate, synthesized because the open export
    # publishes the tower as audio_config instead.
    assert flat.audio_perception_modality_args["input_mode"] == "wav"
    assert flat.model_type == "muse_spark"
    assert flat.torch_dtype == reference.torch_dtype


@pytest.mark.parametrize(
    "mutate,match",
    [
        (lambda c: c["text_config"].__setitem__("hidden_act", "gelu"), "hidden_act"),
        (
            lambda c: c["text_config"].__setitem__("residual_gate_temperature", 0.5),
            "residual_gate_temperature",
        ),
        (lambda c: c["text_config"].__setitem__("qk_scale_factor", 1.0), "qk_scale_factor"),
        (lambda c: c["text_config"].__setitem__("output_multiplier", 0.3), "output_multiplier"),
        (lambda c: c["text_config"]["layer_rope_theta"].__setitem__(3, 123.0), "rope"),
        (
            lambda c: c["text_config"]["layer_types"].__setitem__(0, "full_attention"),
            "layer_types|residue",
        ),
        (
            lambda c: c["text_config"].__setitem__("mlp_layer_types", ["dense"] * NUM_LAYERS),
            "sparse",
        ),
        (lambda c: c["audio_config"].__setitem__("patch_sizes", [8, 8, 6, 5]), "patch"),
        (
            lambda c: c["vision_config"]["layer_types"].__setitem__(0, "full_attention"),
            "sparse_attention_factor",
        ),
        (lambda c: c.__setitem__("image_token_id", 12345), "image_token_id"),
    ],
)
def test_from_open_format_rejects_semantic_changes(mutate, match):
    config = open_format_config_dict()
    mutate(config)
    with pytest.raises(ValueError, match=match):
        MuseSparkOpenConfig.from_open_format(config)


# ---------------------------------------------------------------------------
# Key renaming
# ---------------------------------------------------------------------------


def test_rename_is_a_bijection_onto_the_source_inventory():
    open_keys = open_format_keys()
    assert len(open_keys) == 2360
    rename = build_open_format_rename(open_keys)
    assert len(rename) == len(open_keys)
    assert sorted(rename.values()) == sorted(source_format_keys())


def test_unknown_key_fails_loudly():
    with pytest.raises(ValueError, match="no source-spelling mapping"):
        open_format_to_source_key("model.language_model.layers.0.mlp.new_tensor")
    with pytest.raises(ValueError, match="no source-spelling mapping"):
        open_format_to_source_key("model.novel_tower.weight")


def test_rename_open_format_weights_in_place():
    weights = ConsumableWeightsDict({key: key for key in open_format_keys()})
    assert is_open_format(weights.keys())
    assert rename_open_format_weights(weights)
    assert sorted(weights.keys()) == sorted(source_format_keys())
    # Values traveled with their keys.
    assert weights["model.embed_tokens.weight"] == OPEN_FORMAT_SENTINEL_KEY
    assert (
        weights["model.layers.5.self_attn.o_gate_proj.weight"]
        == "model.language_model.layers.5.self_attn.gate_proj.weight"
    )
    assert (
        weights["model.layers.7.feed_forward.experts.w13_weight_scale_2"]
        == "model.language_model.layers.7.mlp.experts.gate_up_proj_weight_scale_2"
    )
    assert (
        weights["vision_model.encoder.layers.3.attn.out_proj.bias"]
        == "model.vision_tower.layers.3.attn.proj.bias"
    )
    assert (
        weights[f"{AUDIO_ENCODER}.stages.2.1.linear1.weight"]
        == "model.audio_tower.layers.2.conv_blocks.0.mlp.fc1.weight"
    )


def test_rename_is_a_no_op_on_source_format():
    source = {key: key for key in source_format_keys()}
    weights = ConsumableWeightsDict(dict(source))
    assert not is_open_format(weights.keys())
    assert not rename_open_format_weights(weights)
    assert sorted(weights.keys()) == sorted(source.keys())


# ---------------------------------------------------------------------------
# q/k rotary row layout
# ---------------------------------------------------------------------------


def test_interleave_rope_rows_inverts_the_half_layout():
    """Ground truth: the open export stores row `j` of a head where the
    source stored row `2j` (first half) / `2(j - half) + 1` (second half).
    Restoring must put every source row back, for weights and biases alike."""
    heads, head_dim, in_dim = 3, 8, 5
    half = head_dim // 2

    def to_half_layout(source, rows_per_head):
        exported = torch.empty_like(source)
        for h in range(heads):
            for j in range(rows_per_head):
                src = 2 * j if j < half else 2 * (j - half) + 1
                exported[h * rows_per_head + j] = source[h * rows_per_head + src]
        return exported

    weight = torch.arange(heads * head_dim * in_dim, dtype=torch.float32).reshape(
        heads * head_dim, in_dim
    )
    assert torch.equal(interleave_rope_rows(to_half_layout(weight, head_dim), head_dim), weight)

    bias = torch.arange(heads * head_dim, dtype=torch.float32)
    assert torch.equal(interleave_rope_rows(to_half_layout(bias, head_dim), head_dim), bias)

    with pytest.raises(ValueError, match="re-interleave"):
        interleave_rope_rows(torch.zeros(10, 4), 8)


# ---------------------------------------------------------------------------
# Router orientation
# ---------------------------------------------------------------------------


class _RouterStub:
    def __init__(self):
        self.loaded = None

    def load_weights(self, entries):
        self.loaded = entries[0]["weight"]


class _MoeStub:
    def __init__(self):
        self.router = _RouterStub()


def test_load_router_handles_both_orientations():
    num_experts, hidden = 4, 8
    canonical = torch.arange(num_experts * hidden, dtype=torch.float32).reshape(num_experts, hidden)

    moe = _MoeStub()
    _load_router(moe, canonical.clone(), num_experts, hidden)
    assert torch.equal(moe.router.loaded, canonical)

    moe = _MoeStub()
    _load_router(moe, canonical.t().contiguous(), num_experts, hidden)
    assert torch.equal(moe.router.loaded, canonical)

    with pytest.raises(ValueError, match="neither"):
        _load_router(_MoeStub(), torch.zeros(3, 5), num_experts, hidden)
    with pytest.raises(ValueError, match="orientation"):
        _load_router(_MoeStub(), torch.zeros(4, 4), 4, 4)
