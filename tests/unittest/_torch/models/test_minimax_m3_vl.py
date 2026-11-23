# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MiniMax-M3 VL model surface weight accounting.

This test file covers / acceptance item #1: the real MiniMax-M3
VL checkpoint loads with nonempty ``vision_tower.*``,
``multi_modal_projector.*``, and ``patch_merge_mlp.*`` weights, and the
previously closed text-only generation path keeps working.

Helper tests exercise the vision config + module construction and the
checkpoint-key re-anchoring / split helpers. CUDA tests validate that
the real-checkpoint vision branch loads into the new vision tower's
parameter slots with matching shapes and dtypes.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import os
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple

import numpy as np
import pytest
import torch
from PIL import Image
from safetensors import safe_open
from transformers import AutoConfig, AutoProcessor, AutoTokenizer
from utils.llm_data import llm_models_root

from tensorrt_llm._torch.models.modeling_minimaxm3 import get_text_config
from tensorrt_llm._torch.models.modeling_minimaxm3_vl import (
    MINIMAX_M3_VL_VISION_END_TOKEN,
    MINIMAX_M3_VL_VISION_START_TOKEN,
    CLIPVisionConfig,
    MiniMaxVLPatchEmbedding,
    MiniMaxVLPatchMerger,
    MiniMaxVLVisionModel,
    apply_multimodal_pad_values,
    build_multimodal_input_ids,
    compute_visual_token_count,
    compute_visual_token_counts,
    expand_multimodal_placeholders,
    get_minimax_m3_vl_input_processor_cls,
    load_minimax_m3_vl_state_dict,
    merge_multimodal_embeddings,
    merge_multimodal_embeddings_inclusive,
    pad_multimodal_input_tokens,
    reanchor_multimodal_checkpoint_keys,
    split_multimodal_weights,
)
from tensorrt_llm._torch.models.modeling_utils import MetaInitMode

# ---------------------------------------------------------------------------
# Shared helpers (mirror the conventions used by test_minimax_m3.py).
# ---------------------------------------------------------------------------

_DEFAULT_CHECKPOINT_PATH = f"{llm_models_root()}/MiniMax-M3"


def _checkpoint_path() -> str:
    return _DEFAULT_CHECKPOINT_PATH


def _has_cuda() -> bool:
    try:
        return torch.cuda.is_available()
    except Exception:
        return False


def _make_vision_config_dict() -> dict:
    """A minimal but faithful copy of the M3 VL checkpoint vision_config."""
    return {
        "hidden_size": 1280,
        "num_attention_heads": 16,
        "num_hidden_layers": 32,
        "intermediate_size": 5120,
        "patch_size": 14,
        "image_size": 672,
        "projection_dim": 6144,
        "position_embedding_type": "rope",
        "rope_mode": "3d",
        "rope_theta": 10000.0,
        "attention_dropout": 0.0,
        "hidden_act": "gelu",
        "layer_norm_eps": 1e-5,
        "num_channels": 3,
        "img_token_compression_config": {
            "image_token_compression_method": "patch_merge",
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
        },
        "vision_segment_max_frames": 4,
    }


# ---------------------------------------------------------------------------
# CPU unit tests for CLIPVisionConfig and module construction.
# ---------------------------------------------------------------------------


@pytest.mark.cpu_only
def test_clip_vision_config_from_dict_uses_checkpoint_fields():
    cfg = CLIPVisionConfig.from_dict_or_obj(_make_vision_config_dict())
    assert cfg.hidden_size == 1280
    assert cfg.num_attention_heads == 16
    assert cfg.num_hidden_layers == 32
    assert cfg.intermediate_size == 5120
    assert cfg.patch_size == 14
    assert cfg.image_size == 672
    assert cfg.num_channels == 3
    assert cfg.position_embedding_type == "rope"
    assert cfg.rope_mode == "3d"
    assert cfg.img_token_compression_config["spatial_merge_size"] == 2
    assert cfg.img_token_compression_config["temporal_patch_size"] == 2


@pytest.mark.cpu_only
def test_clip_vision_config_from_dict_ignores_unknown_keys():
    data = _make_vision_config_dict()
    data["projection_dim"] = 6144  # unknown to the dataclass; must be ignored
    data["model_type"] = "clip_vision_model"
    cfg = CLIPVisionConfig.from_dict_or_obj(data)
    assert cfg.hidden_size == 1280


@pytest.mark.cpu_only
def test_clip_vision_config_from_dict_accepts_simplenamespace():
    ns = SimpleNamespace(
        hidden_size=128,
        num_attention_heads=4,
        num_hidden_layers=2,
        img_token_compression_config={"spatial_merge_size": 2, "temporal_patch_size": 2},
        position_embedding_type="rope",
        rope_mode="3d",
    )
    cfg = CLIPVisionConfig.from_dict_or_obj(ns)
    assert cfg.hidden_size == 128
    assert cfg.num_hidden_layers == 2


@pytest.mark.cpu_only
def test_clip_vision_config_from_none_yields_defaults():
    cfg = CLIPVisionConfig.from_dict_or_obj(None)
    # Defaults match the M3 VL checkpoint vision_config dimensions.
    assert cfg.hidden_size == 1280
    assert cfg.num_hidden_layers == 32


@pytest.mark.cpu_only
def test_minimax_vl_vision_model_state_dict_keys_match_checkpoint_pattern():
    cfg = CLIPVisionConfig.from_dict_or_obj(_make_vision_config_dict())
    model = MiniMaxVLVisionModel(
        config=cfg,
        text_hidden_size=6144,
        projector_hidden_size=6144,
        dtype=torch.float32,  # CPU smoke: don't require bf16 GPU
    )
    state_dict = dict(model.state_dict())

    # Patch embedding + pre-LN (the typo "pre_layrnorm" must match the
    # checkpoint).
    assert "vision_model.embeddings.patch_embedding.weight" in state_dict, state_dict.keys()
    assert "vision_model.pre_layrnorm.weight" in state_dict
    assert "vision_model.pre_layrnorm.bias" in state_dict

    # Each encoder layer must carry the canonical 16 weights (q/k/v/out *
    # weight+bias, layer_norm1/2 * weight+bias, mlp.fc1/2 * weight+bias).
    expected_per_layer = {
        "self_attn.q_proj.weight",
        "self_attn.q_proj.bias",
        "self_attn.k_proj.weight",
        "self_attn.k_proj.bias",
        "self_attn.v_proj.weight",
        "self_attn.v_proj.bias",
        "self_attn.out_proj.weight",
        "self_attn.out_proj.bias",
        "layer_norm1.weight",
        "layer_norm1.bias",
        "layer_norm2.weight",
        "layer_norm2.bias",
        "mlp.fc1.weight",
        "mlp.fc1.bias",
        "mlp.fc2.weight",
        "mlp.fc2.bias",
    }
    for i in range(cfg.num_hidden_layers):
        for suffix in expected_per_layer:
            key = f"vision_model.encoder.layers.{i}.{suffix}"
            assert key in state_dict, key

    # Projector + patch merger.
    assert "multi_modal_projector.linear_1.weight" in state_dict
    assert "multi_modal_projector.linear_1.bias" in state_dict
    assert "multi_modal_projector.linear_2.weight" in state_dict
    assert "multi_modal_projector.linear_2.bias" in state_dict
    assert "patch_merge_mlp.linear_1.weight" in state_dict
    assert "patch_merge_mlp.linear_1.bias" in state_dict
    assert "patch_merge_mlp.linear_2.weight" in state_dict
    assert "patch_merge_mlp.linear_2.bias" in state_dict


@pytest.mark.cpu_only
def test_minimax_vl_vision_model_param_shapes_match_checkpoint():
    cfg = CLIPVisionConfig.from_dict_or_obj(_make_vision_config_dict())
    model = MiniMaxVLVisionModel(
        config=cfg,
        text_hidden_size=6144,
        projector_hidden_size=6144,
        dtype=torch.float32,
    )
    state_dict = dict(model.state_dict())

    # Conv3d patch embedding: [hidden, channels, temporal_patch, patch, patch].
    pe = state_dict["vision_model.embeddings.patch_embedding.weight"]
    assert tuple(pe.shape) == (1280, 3, 2, 14, 14), pe.shape

    # Per-layer Q/K/V/out: square [hidden, hidden] = [1280, 1280].
    q = state_dict["vision_model.encoder.layers.0.self_attn.q_proj.weight"]
    assert tuple(q.shape) == (1280, 1280)
    out = state_dict["vision_model.encoder.layers.0.self_attn.out_proj.weight"]
    assert tuple(out.shape) == (1280, 1280)

    # MLP fc1: [intermediate=5120, hidden=1280]; fc2: [hidden=1280, 5120].
    fc1 = state_dict["vision_model.encoder.layers.0.mlp.fc1.weight"]
    fc2 = state_dict["vision_model.encoder.layers.0.mlp.fc2.weight"]
    assert tuple(fc1.shape) == (5120, 1280)
    assert tuple(fc2.shape) == (1280, 5120)

    # multi_modal_projector: vision_hidden(1280) -> mid(6144) -> text_hidden(6144).
    proj1 = state_dict["multi_modal_projector.linear_1.weight"]
    proj2 = state_dict["multi_modal_projector.linear_2.weight"]
    assert tuple(proj1.shape) == (6144, 1280)
    assert tuple(proj2.shape) == (6144, 6144)

    # patch_merge_mlp: text_hidden * spatial_merge_size^2 = 6144*4 -> mid -> text.
    pm1 = state_dict["patch_merge_mlp.linear_1.weight"]
    pm2 = state_dict["patch_merge_mlp.linear_2.weight"]
    assert tuple(pm1.shape) == (6144, 6144 * 4)
    assert tuple(pm2.shape) == (6144, 6144)


@pytest.mark.cpu_only
def test_minimax_vl_vision_model_supports_meta_init():
    config_dict = _make_vision_config_dict()
    config_dict["num_hidden_layers"] = 2
    cfg = CLIPVisionConfig.from_dict_or_obj(config_dict)

    with MetaInitMode():
        model = MiniMaxVLVisionModel(
            config=cfg,
            text_hidden_size=6144,
            projector_hidden_size=6144,
            dtype=torch.bfloat16,
        )

    parameters = list(model.parameters())
    assert parameters
    assert all(parameter.is_meta for parameter in parameters)


# ---------------------------------------------------------------------------
# Re-anchoring + split helpers.
# ---------------------------------------------------------------------------


@pytest.mark.cpu_only
def test_reanchor_multimodal_checkpoint_keys_maps_vision_branches():
    keys = [
        "vision_tower.vision_model.embeddings.patch_embedding.weight",
        "vision_tower.vision_model.encoder.layers.5.self_attn.q_proj.weight",
        "multi_modal_projector.linear_1.weight",
        "patch_merge_mlp.linear_2.bias",
        "language_model.model.norm.weight",  # NOT a vision key
        "language_model.lm_head.weight",
    ]
    plan = reanchor_multimodal_checkpoint_keys(keys)
    assert (
        plan["vision_tower.vision_model.embeddings.patch_embedding.weight"]
        == "vision_tower.vision_model.embeddings.patch_embedding.weight"
    )
    assert (
        plan["multi_modal_projector.linear_1.weight"]
        == "vision_tower.multi_modal_projector.linear_1.weight"
    )
    assert plan["patch_merge_mlp.linear_2.bias"] == "vision_tower.patch_merge_mlp.linear_2.bias"
    # Language-model keys are not in the mapping.
    assert "language_model.model.norm.weight" not in plan
    assert "language_model.lm_head.weight" not in plan


@pytest.mark.cpu_only
def test_split_multimodal_weights_partitions_text_and_vision():
    weights = {
        "language_model.model.norm.weight": torch.zeros(2),
        "language_model.lm_head.weight": torch.zeros(2, 2),
        "vision_tower.vision_model.pre_layrnorm.weight": torch.zeros(3),
        "multi_modal_projector.linear_1.weight": torch.zeros(4),
        "patch_merge_mlp.linear_2.weight": torch.zeros(5),
    }
    text, vision = split_multimodal_weights(weights)
    assert "language_model.model.norm.weight" in text
    assert "language_model.lm_head.weight" in text
    assert "vision_tower.vision_model.pre_layrnorm.weight" in vision
    assert "vision_tower.multi_modal_projector.linear_1.weight" in vision
    assert "vision_tower.patch_merge_mlp.linear_2.weight" in vision
    # No vision keys leaked into text.
    assert "vision_tower.vision_model.pre_layrnorm.weight" not in text
    assert "multi_modal_projector.linear_1.weight" not in text


@pytest.mark.cpu_only
def test_split_multimodal_weights_preserves_unknown_top_level_keys():
    weights = {
        "language_model.model.norm.weight": torch.zeros(2),
        "some_unknown.key": torch.zeros(2),
    }
    text, vision = split_multimodal_weights(weights)
    assert "some_unknown.key" in text
    assert vision == {}


# ---------------------------------------------------------------------------
# Vision state-dict loading.
# ---------------------------------------------------------------------------


@pytest.mark.cpu_only
def test_load_minimax_m3_vl_state_dict_fills_target_slots():
    # Use a tiny config so the test is fast on CPU.
    cfg = CLIPVisionConfig.from_dict_or_obj(
        {
            "hidden_size": 16,
            "num_attention_heads": 2,
            "num_hidden_layers": 1,
            "intermediate_size": 32,
            "patch_size": 2,
            "image_size": 4,
            "num_channels": 3,
            "position_embedding_type": "rope",
            "rope_mode": "3d",
            "rope_theta": 10000.0,
            "img_token_compression_config": {
                "spatial_merge_size": 2,
                "temporal_patch_size": 2,
            },
        }
    )
    model = MiniMaxVLVisionModel(
        config=cfg, text_hidden_size=32, projector_hidden_size=32, dtype=torch.float32
    )
    target = dict(model.state_dict())

    # Build a "checkpoint-shape" weights dict with sentinel values that we
    # can prove landed on the right parameter slots.
    raw_weights: Dict[str, torch.Tensor] = {}
    for k, v in target.items():
        raw_weights["vision_tower." + k] = torch.full_like(v, 0.5)
    # Re-anchored top-level projector / patch-merge keys must also land.
    raw_weights["multi_modal_projector.linear_1.weight"] = torch.full_like(
        target["multi_modal_projector.linear_1.weight"], 0.25
    )

    loaded, missing = load_minimax_m3_vl_state_dict(model, raw_weights, strict=False)
    assert missing == [], missing

    # Spot-check that values landed: the patched projector weight should
    # carry 0.25, while everything else carries 0.5.
    proj1 = model.multi_modal_projector.linear_1.weight.detach().cpu()
    assert torch.all(proj1 == 0.25)
    pre_ln = model.vision_model.pre_layrnorm.weight.detach().cpu()
    assert torch.all(pre_ln == 0.5)


@pytest.mark.cpu_only
def test_load_minimax_m3_vl_state_dict_strict_rejects_missing_keys():
    cfg = CLIPVisionConfig.from_dict_or_obj(
        {
            "hidden_size": 16,
            "num_attention_heads": 2,
            "num_hidden_layers": 1,
            "intermediate_size": 32,
            "patch_size": 2,
            "image_size": 4,
            "num_channels": 3,
            "position_embedding_type": "rope",
            "rope_mode": "3d",
            "rope_theta": 10000.0,
            "img_token_compression_config": {
                "spatial_merge_size": 2,
                "temporal_patch_size": 2,
            },
        }
    )
    model = MiniMaxVLVisionModel(config=cfg, text_hidden_size=32, dtype=torch.float32)
    # Provide only one weight; strict load must fail.
    raw_weights = {
        "vision_tower.vision_model.pre_layrnorm.weight": torch.zeros(16),
    }
    with pytest.raises(RuntimeError, match="missing"):
        load_minimax_m3_vl_state_dict(model, raw_weights, strict=True)


@pytest.mark.cpu_only
def test_load_minimax_m3_vl_state_dict_rejects_shape_mismatch():
    cfg = CLIPVisionConfig.from_dict_or_obj(
        {
            "hidden_size": 16,
            "num_attention_heads": 2,
            "num_hidden_layers": 1,
            "intermediate_size": 32,
            "patch_size": 2,
            "image_size": 4,
            "num_channels": 3,
            "position_embedding_type": "rope",
            "rope_mode": "3d",
            "rope_theta": 10000.0,
            "img_token_compression_config": {
                "spatial_merge_size": 2,
                "temporal_patch_size": 2,
            },
        }
    )
    model = MiniMaxVLVisionModel(config=cfg, text_hidden_size=32, dtype=torch.float32)
    # Wrong shape for pre_layrnorm.weight (expected [16], provided [8]).
    raw_weights = {
        "vision_tower.vision_model.pre_layrnorm.weight": torch.zeros(8),
    }
    with pytest.raises(ValueError, match="shape mismatch"):
        load_minimax_m3_vl_state_dict(model, raw_weights, strict=False)


# ---------------------------------------------------------------------------
# Real-checkpoint state-dict accounting (CUDA — / ).
# ---------------------------------------------------------------------------


@pytest.mark.gpu
@pytest.mark.skipif(not _has_cuda(), reason="MiniMax-M3 VL load needs CUDA")
def test_real_checkpoint_vision_state_dict_accounting():
    """every checkpoint vision key maps to a tower slot.

    Reads the safetensors index, collects every ``vision_tower.*``,
    ``multi_modal_projector.*``, and ``patch_merge_mlp.*`` key, and
    proves each one re-anchors to a real :class:`MiniMaxVLVisionModel`
    parameter (correct shape, dtype-compatible). Loading the full ~1 GB
    of vision tensors is skipped here to keep the smoke fast; the
    integration tests in / -#4 do that.
    """
    pytest.importorskip("transformers")

    checkpoint = _checkpoint_path()
    cfg = AutoConfig.from_pretrained(checkpoint, trust_remote_code=True)
    text_cfg = get_text_config(cfg)
    text_hidden_size = int(getattr(text_cfg, "hidden_size", 6144))
    projector_hidden_size = getattr(cfg, "projector_hidden_size", None)
    vision_config = CLIPVisionConfig.from_dict_or_obj(cfg.vision_config)

    model = MiniMaxVLVisionModel(
        config=vision_config,
        text_hidden_size=text_hidden_size,
        projector_hidden_size=projector_hidden_size,
        dtype=torch.bfloat16,
    )
    target_keys = set(model.state_dict().keys())

    with open(os.path.join(checkpoint, "model.safetensors.index.json")) as f:
        weight_map = json.load(f)["weight_map"]

    vision_source_keys = [
        k
        for k in weight_map.keys()
        if k.startswith("vision_tower.")
        or k.startswith("multi_modal_projector.")
        or k.startswith("patch_merge_mlp.")
    ]
    assert vision_source_keys, "no vision weights found in checkpoint index"

    plan = reanchor_multimodal_checkpoint_keys(vision_source_keys)
    # Every vision source key has a target.
    assert set(plan.keys()) == set(vision_source_keys)
    # Every target lands under vision_tower.* (since the module is exposed
    # as vision_tower on the parent wrapper).
    for v in plan.values():
        assert v.startswith("vision_tower."), v

    # Every target maps to a real module parameter slot.
    missing_targets: List[str] = []
    for source, target in plan.items():
        relative = target[len("vision_tower.") :]
        if relative not in target_keys:
            missing_targets.append(target)

    assert not missing_targets, (
        f"{len(missing_targets)} checkpoint vision keys have no parameter slot "
        f"in MiniMaxVLVisionModel; first 5: {missing_targets[:5]!r}"
    )

    # And every parameter slot is covered by a checkpoint key (no orphan
    # parameters in the module).
    target_relative = {plan[k][len("vision_tower.") :] for k in plan}
    orphan_params = sorted(target_keys - target_relative)
    assert not orphan_params, (
        f"{len(orphan_params)} MiniMaxVLVisionModel parameters have no "
        f"checkpoint key; first 5: {orphan_params[:5]!r}"
    )


@pytest.mark.gpu
@pytest.mark.skipif(not _has_cuda(), reason="MiniMax-M3 VL load needs CUDA")
def test_real_checkpoint_vision_weight_load_smoke():
    """load a sample of real vision tensors end-to-end.

    Reads a handful of representative vision-side tensors from the real
    checkpoint (Conv3d patch embedding, one encoder QKV stack, one mlp
    layer, projector linear_1, patch-merger linear_1) into a
    :class:`MiniMaxVLVisionModel` and confirms the bytes land in the
    right parameter slots with the right shapes. This is the lightweight
    runtime evidence half of / ; the full per-shard load is
    exercised by the production batch in MiniMax-M3 VL.
    """
    pytest.importorskip("safetensors")
    pytest.importorskip("transformers")

    checkpoint = _checkpoint_path()
    cfg = AutoConfig.from_pretrained(checkpoint, trust_remote_code=True)
    text_cfg = get_text_config(cfg)
    text_hidden_size = int(getattr(text_cfg, "hidden_size", 6144))
    projector_hidden_size = getattr(cfg, "projector_hidden_size", None)
    vision_config = CLIPVisionConfig.from_dict_or_obj(cfg.vision_config)

    model = MiniMaxVLVisionModel(
        config=vision_config,
        text_hidden_size=text_hidden_size,
        projector_hidden_size=projector_hidden_size,
        dtype=torch.bfloat16,
    )

    targets = [
        "vision_tower.vision_model.embeddings.patch_embedding.weight",
        "vision_tower.vision_model.pre_layrnorm.weight",
        "vision_tower.vision_model.pre_layrnorm.bias",
        "vision_tower.vision_model.encoder.layers.0.self_attn.q_proj.weight",
        "vision_tower.vision_model.encoder.layers.0.self_attn.k_proj.weight",
        "vision_tower.vision_model.encoder.layers.0.self_attn.v_proj.weight",
        "vision_tower.vision_model.encoder.layers.0.self_attn.out_proj.weight",
        "vision_tower.vision_model.encoder.layers.0.layer_norm1.weight",
        "vision_tower.vision_model.encoder.layers.0.mlp.fc1.weight",
        "vision_tower.vision_model.encoder.layers.0.mlp.fc2.weight",
        "multi_modal_projector.linear_1.weight",
        "multi_modal_projector.linear_2.weight",
        "patch_merge_mlp.linear_1.weight",
        "patch_merge_mlp.linear_2.weight",
    ]

    with open(os.path.join(checkpoint, "model.safetensors.index.json")) as f:
        weight_map = json.load(f)["weight_map"]
    by_shard: Dict[str, List[str]] = {}
    for key in targets:
        shard = weight_map[key]
        by_shard.setdefault(shard, []).append(key)

    raw_weights: Dict[str, torch.Tensor] = {}
    for shard, keys in by_shard.items():
        with safe_open(os.path.join(checkpoint, shard), framework="pt", device="cpu") as sf:
            for k in keys:
                raw_weights[k] = sf.get_tensor(k)

    loaded, missing = load_minimax_m3_vl_state_dict(model, raw_weights, strict=False)
    # Every key we requested must have been loaded into the corresponding slot.
    expected_loaded = set()
    for k in targets:
        if k.startswith("vision_tower."):
            expected_loaded.add(k[len("vision_tower.") :])
        else:
            expected_loaded.add(k)  # already in module-relative form
    assert expected_loaded.issubset(set(loaded)), (
        f"loaded={set(loaded)} expected_loaded={expected_loaded}"
    )

    # Spot-check the patch embedding lands with the right shape and dtype.
    pe = model.vision_model.embeddings.patch_embedding.weight.detach()
    assert tuple(pe.shape) == (1280, 3, 2, 14, 14)
    assert pe.dtype == torch.bfloat16
    # The first element of a bf16 weight read from disk should not be zero.
    assert pe.abs().sum().item() > 0, "patch_embedding loaded as zeros"

    # Projector linear_1 must be non-zero.
    p1 = model.multi_modal_projector.linear_1.weight.detach()
    assert p1.abs().sum().item() > 0, "multi_modal_projector loaded as zeros"

    # Patch merger linear_1 must be non-zero.
    pm1 = model.patch_merge_mlp.linear_1.weight.detach()
    assert pm1.abs().sum().item() > 0, "patch_merge_mlp loaded as zeros"


# ---------------------------------------------------------------------------
# multimodal token expansion + embedding merge helpers.
# ---------------------------------------------------------------------------


@pytest.mark.cpu_only
def test_compute_visual_token_count_matches_sglang_formula():
    # Single image, grid (1, 48, 48) with spatial_merge_size 2 -> 1 * 24 * 24 = 576.
    assert compute_visual_token_count(1, 48, 48, 2) == 576
    # Video with 4 frames at (4, 16, 16) with spatial_merge_size 2 -> 4 * 8 * 8 = 256.
    assert compute_visual_token_count(4, 16, 16, 2) == 256
    # Tiny grid used by the synthetic smoke: (1, 4, 4) merge 2 -> 4.
    assert compute_visual_token_count(1, 4, 4, 2) == 4


@pytest.mark.cpu_only
def test_compute_visual_token_counts_batched():
    # [1, 4, 4] merge=2 -> 1 * 2 * 2 = 4
    # [1, 8, 8] merge=2 -> 1 * 4 * 4 = 16
    # [2, 4, 6] merge=2 -> 2 * 2 * 3 = 12
    counts = compute_visual_token_counts([[1, 4, 4], [1, 8, 8], [2, 4, 6]], 2)
    assert counts == [4, 16, 12]


@pytest.mark.cpu_only
def test_compute_visual_token_count_rejects_non_divisible_grid():
    with pytest.raises(ValueError, match="multiple of spatial_merge_size"):
        compute_visual_token_count(1, 5, 4, 2)
    with pytest.raises(ValueError, match="multiple of spatial_merge_size"):
        compute_visual_token_count(1, 4, 5, 2)


@pytest.mark.cpu_only
def test_expand_multimodal_placeholders_image_only():
    # Two image placeholders with different repeat counts.
    input_ids = [1, 2, 200025, 3, 200025, 4]
    expanded, image_spans, video_spans = expand_multimodal_placeholders(
        input_ids, image_token_id=200025, image_repeats=[3, 2]
    )
    assert expanded == [1, 2, 200025, 200025, 200025, 3, 200025, 200025, 4]
    assert image_spans == [(2, 5), (6, 8)]
    assert video_spans == []


@pytest.mark.cpu_only
def test_expand_multimodal_placeholders_image_and_video():
    input_ids = [10, 200025, 20, 200026, 30]
    expanded, image_spans, video_spans = expand_multimodal_placeholders(
        input_ids,
        image_token_id=200025,
        image_repeats=[2],
        video_token_id=200026,
        video_repeats=[4],
    )
    assert expanded == [10, 200025, 200025, 20, 200026, 200026, 200026, 200026, 30]
    assert image_spans == [(1, 3)]
    assert video_spans == [(4, 8)]


@pytest.mark.cpu_only
def test_expand_multimodal_placeholders_count_mismatch_raises():
    with pytest.raises(ValueError, match="more image tokens"):
        expand_multimodal_placeholders([200025, 200025], image_token_id=200025, image_repeats=[2])
    with pytest.raises(ValueError, match="consumed 0 image"):
        expand_multimodal_placeholders([1, 2, 3], image_token_id=200025, image_repeats=[2])


@pytest.mark.cpu_only
def test_apply_multimodal_pad_values_rewrites_only_spans():
    input_ids = [1, 2, 200025, 200025, 200025, 4, 200026, 200026, 6]
    out = apply_multimodal_pad_values(
        input_ids,
        image_pad_values=[7777],
        image_spans=[(2, 5)],
        video_pad_values=[8888],
        video_spans=[(6, 8)],
    )
    assert out == [1, 2, 7777, 7777, 7777, 4, 8888, 8888, 6]


@pytest.mark.cpu_only
def test_merge_multimodal_embeddings_image_only_replaces_correct_positions():
    text_seq = 8
    hidden = 4
    # Use deterministic floats so the test reads cleanly.
    input_embeds = torch.zeros((text_seq, hidden))
    input_embeds[:] = torch.arange(text_seq, dtype=torch.float32).unsqueeze(-1) * 0.1
    image_embeds = torch.ones((5, hidden), dtype=torch.float32) * 9.0
    spans = [(2, 5), (6, 8)]
    out = merge_multimodal_embeddings(input_embeds, image_embeds=image_embeds, image_spans=spans)

    # Positions inside spans replaced by 9.0.
    for start, end in spans:
        assert torch.all(out[start:end] == 9.0), (start, end, out[start:end])
    # Positions outside spans untouched.
    for i in (0, 1, 5):
        assert torch.allclose(out[i], input_embeds[i])


@pytest.mark.cpu_only
def test_merge_multimodal_embeddings_rejects_mismatched_counts():
    input_embeds = torch.zeros((8, 4))
    image_embeds = torch.ones((4, 4))
    with pytest.raises(ValueError, match="sum of image_span lengths"):
        merge_multimodal_embeddings(
            input_embeds,
            image_embeds=image_embeds,
            image_spans=[(0, 3)],  # only 3 tokens but 4 embeds provided.
        )


@pytest.mark.cpu_only
def test_merge_multimodal_embeddings_negative_control_wrong_span():
    """Negative control — merging at the wrong span gives a different result.

    MiniMax-M3 VL will assert that wrong spans break parity against SGLang;
    this MiniMax-M3 VL unit-level mutation confirms the helper is sensitive
    to span placement so the parity assertion has a real failure path.
    """
    input_embeds = torch.zeros((8, 4))
    image_embeds = torch.arange(3 * 4, dtype=torch.float32).reshape(3, 4) + 1.0

    correct = merge_multimodal_embeddings(
        input_embeds, image_embeds=image_embeds, image_spans=[(2, 5)]
    )
    wrong = merge_multimodal_embeddings(
        input_embeds, image_embeds=image_embeds, image_spans=[(3, 6)]
    )
    assert not torch.equal(correct, wrong)


# ---------------------------------------------------------------------------
# vision tower forward (CPU, synthetic shapes).
# ---------------------------------------------------------------------------


def _tiny_vision_config():
    """Small vision config that still exercises 3D RoPE + merge math."""
    return {
        "hidden_size": 32,
        "num_attention_heads": 4,
        "num_hidden_layers": 2,
        "intermediate_size": 64,
        "patch_size": 4,
        "image_size": 8,
        "num_channels": 3,
        "position_embedding_type": "rope",
        "rope_mode": "3d",
        "rope_theta": 10000.0,
        "img_token_compression_config": {
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
        },
        "layer_norm_eps": 1e-5,
    }


@pytest.mark.cpu_only
def test_vision_tower_forward_shape_and_dtype_cpu():
    """Run the full vision tower on tiny synthetic data — CPU-only shape check."""
    cfg = CLIPVisionConfig.from_dict_or_obj(_tiny_vision_config())
    text_hidden_size = 16
    projector_hidden_size = 16
    model = MiniMaxVLVisionModel(
        config=cfg,
        text_hidden_size=text_hidden_size,
        projector_hidden_size=projector_hidden_size,
        dtype=torch.float32,
    )

    # Use grid (1, 4, 4): 16 patches before merge, 4 after merge.
    grid_thw = [[1, 4, 4]]
    n_patches = 1 * 4 * 4
    pixel_cols = (
        cfg.num_channels
        * cfg.img_token_compression_config["temporal_patch_size"]
        * cfg.patch_size
        * cfg.patch_size
    )
    pixel_values = torch.randn((n_patches, pixel_cols), dtype=torch.float32)

    out = model(pixel_values=pixel_values, grid_thw=grid_thw)
    expected_tokens = compute_visual_token_count(
        1, 4, 4, cfg.img_token_compression_config["spatial_merge_size"]
    )
    assert out.shape == (expected_tokens, text_hidden_size)
    assert out.dtype == torch.float32
    # The forward must not silently zero everything out under random init.
    assert torch.isfinite(out).all()


@pytest.mark.cpu_only
def test_vision_tower_forward_multi_image_concat():
    """Concatenated multi-image batch produces concatenated outputs."""
    cfg = CLIPVisionConfig.from_dict_or_obj(_tiny_vision_config())
    model = MiniMaxVLVisionModel(
        config=cfg, text_hidden_size=16, projector_hidden_size=16, dtype=torch.float32
    )

    grid_thws = [[1, 4, 4], [1, 4, 6]]
    spatial_merge = cfg.img_token_compression_config["spatial_merge_size"]
    expected_per_image = [
        compute_visual_token_count(*g, spatial_merge_size=spatial_merge) for g in grid_thws
    ]
    n_patches = sum(g[0] * g[1] * g[2] for g in grid_thws)
    pixel_cols = (
        cfg.num_channels
        * cfg.img_token_compression_config["temporal_patch_size"]
        * cfg.patch_size
        * cfg.patch_size
    )
    pixel_values = torch.randn((n_patches, pixel_cols), dtype=torch.float32)

    out = model(pixel_values=pixel_values, grid_thw=grid_thws)
    assert out.shape[0] == sum(expected_per_image)
    assert out.shape[1] == 16
    assert torch.isfinite(out).all()


@pytest.mark.cpu_only
def test_patch_embedding_shape_matches_conv3d_semantics():
    """Patch embedding produces ``[N, hidden]`` matching the Conv3d kernel."""
    cfg = CLIPVisionConfig.from_dict_or_obj(_tiny_vision_config())
    emb = MiniMaxVLPatchEmbedding(cfg, dtype=torch.float32)

    n = 16
    pixel_cols = (
        cfg.num_channels
        * cfg.img_token_compression_config["temporal_patch_size"]
        * cfg.patch_size
        * cfg.patch_size
    )
    pixel_values = torch.randn((n, pixel_cols), dtype=torch.float32)
    out = emb(pixel_values)
    assert out.shape == (n, cfg.hidden_size)


@pytest.mark.cpu_only
def test_patch_merger_forward_compresses_by_merge_factor():
    """``[N, hidden]`` -> ``[N / merge**2, hidden]`` after the merger."""
    cfg = CLIPVisionConfig.from_dict_or_obj(_tiny_vision_config())
    merge = MiniMaxVLPatchMerger(
        spatial_merge_size=cfg.img_token_compression_config["spatial_merge_size"],
        text_hidden_size=16,
        projector_hidden_size=16,
        dtype=torch.float32,
    )
    x = torch.randn((16, 16), dtype=torch.float32)
    out = merge(x)
    assert out.shape == (4, 16)


@pytest.mark.cpu_only
def test_patch_merger_rejects_unaligned_input():
    cfg = CLIPVisionConfig.from_dict_or_obj(_tiny_vision_config())
    merge = MiniMaxVLPatchMerger(
        spatial_merge_size=cfg.img_token_compression_config["spatial_merge_size"],
        text_hidden_size=16,
        projector_hidden_size=16,
        dtype=torch.float32,
    )
    x = torch.randn((7, 16), dtype=torch.float32)  # 7 not divisible by 4.
    with pytest.raises(ValueError, match="divisible by"):
        merge(x)


# ---------------------------------------------------------------------------
# full multimodal smoke (CUDA + real checkpoint).
# ---------------------------------------------------------------------------


@pytest.mark.gpu
@pytest.mark.skipif(not _has_cuda(), reason="MiniMax-M3 VL forward needs CUDA")
def test_real_checkpoint_vision_tower_forward_smoke():
    """vision tower forward on real BF16 checkpoint weights.

    Loads a representative subset of the real M3 vision-side weights
    (Conv3d patch embedding, pre-LN, the first 2 encoder layers,
    projector linear_1/2, patch-merger linear_1/2), runs a synthetic
    ``[N, C*T*P*P]`` pixel batch through the partial vision tower with
    only those layers, and asserts shape/dtype invariants. Loading the
    full 32-layer encoder is reserved for the MiniMax-M3 VL parity replay.

    The synthetic pixel input is deterministic (``torch.manual_seed``),
    so the output values are reproducible run-to-run.
    """
    pytest.importorskip("safetensors")
    pytest.importorskip("transformers")

    checkpoint = _checkpoint_path()
    cfg = AutoConfig.from_pretrained(checkpoint, trust_remote_code=True)
    text_cfg = get_text_config(cfg)
    text_hidden_size = int(getattr(text_cfg, "hidden_size", 6144))
    projector_hidden_size = getattr(cfg, "projector_hidden_size", None)
    vision_config_raw = cfg.vision_config
    if hasattr(vision_config_raw, "to_dict"):
        vision_config_dict = vision_config_raw.to_dict()
    else:
        vision_config_dict = dict(vision_config_raw)
    # Override num_hidden_layers to 2 to keep the smoke fast.
    vision_config_dict["num_hidden_layers"] = 2
    vision_config = CLIPVisionConfig.from_dict_or_obj(vision_config_dict)

    model = MiniMaxVLVisionModel(
        config=vision_config,
        text_hidden_size=text_hidden_size,
        projector_hidden_size=projector_hidden_size,
        dtype=torch.bfloat16,
    )

    targets = [
        "vision_tower.vision_model.embeddings.patch_embedding.weight",
        "vision_tower.vision_model.pre_layrnorm.weight",
        "vision_tower.vision_model.pre_layrnorm.bias",
    ]
    # First 2 encoder layers — q_proj / k_proj / v_proj / out_proj / LN / MLP.
    for i in range(2):
        for suffix in (
            "self_attn.q_proj.weight",
            "self_attn.q_proj.bias",
            "self_attn.k_proj.weight",
            "self_attn.k_proj.bias",
            "self_attn.v_proj.weight",
            "self_attn.v_proj.bias",
            "self_attn.out_proj.weight",
            "self_attn.out_proj.bias",
            "layer_norm1.weight",
            "layer_norm1.bias",
            "layer_norm2.weight",
            "layer_norm2.bias",
            "mlp.fc1.weight",
            "mlp.fc1.bias",
            "mlp.fc2.weight",
            "mlp.fc2.bias",
        ):
            targets.append(f"vision_tower.vision_model.encoder.layers.{i}.{suffix}")
    targets += [
        "multi_modal_projector.linear_1.weight",
        "multi_modal_projector.linear_1.bias",
        "multi_modal_projector.linear_2.weight",
        "multi_modal_projector.linear_2.bias",
        "patch_merge_mlp.linear_1.weight",
        "patch_merge_mlp.linear_1.bias",
        "patch_merge_mlp.linear_2.weight",
        "patch_merge_mlp.linear_2.bias",
    ]

    with open(os.path.join(checkpoint, "model.safetensors.index.json")) as f:
        weight_map = json.load(f)["weight_map"]
    by_shard: Dict[str, List[str]] = {}
    for key in targets:
        shard = weight_map[key]
        by_shard.setdefault(shard, []).append(key)

    raw_weights: Dict[str, torch.Tensor] = {}
    for shard, keys in by_shard.items():
        with safe_open(os.path.join(checkpoint, shard), framework="pt", device="cpu") as sf:
            for k in keys:
                raw_weights[k] = sf.get_tensor(k)

    load_minimax_m3_vl_state_dict(model, raw_weights, strict=False)
    model = model.to("cuda")
    model.eval()

    # Tiny synthetic image: grid (1, 4, 4) -> 16 patches before merge, 4 after.
    spatial_merge = vision_config.img_token_compression_config["spatial_merge_size"]
    grid_thw = [[1, 4, 4]]
    n_patches = 1 * 4 * 4
    pixel_cols = (
        vision_config.num_channels
        * vision_config.img_token_compression_config["temporal_patch_size"]
        * vision_config.patch_size
        * vision_config.patch_size
    )

    torch.manual_seed(20260604)
    pixel_values = torch.randn((n_patches, pixel_cols), dtype=torch.bfloat16, device="cuda")
    with torch.no_grad():
        out = model(pixel_values=pixel_values, grid_thw=grid_thw)

    expected_tokens = compute_visual_token_count(1, 4, 4, spatial_merge)
    assert out.shape == (expected_tokens, text_hidden_size), out.shape
    assert out.dtype == torch.bfloat16
    # The forward must produce finite, non-zero output under random pixels.
    assert torch.isfinite(out).all(), "vision tower output contains NaN/Inf"
    assert out.abs().sum().item() > 0, "vision tower output is all zeros"


@pytest.mark.gpu
@pytest.mark.skipif(not _has_cuda(), reason="MiniMax-M3 VL embedding merge needs CUDA")
def test_real_checkpoint_multimodal_embedding_merge_smoke():
    """end-to-end multimodal preprocessing + embedding merge.

    Builds an input_ids stream with a single image placeholder, expands
    the placeholder, applies the per-item ``pad_value`` rewrite, runs
    the partial vision tower on synthetic pixels, and merges the visual
    features into a synthetic text-embedding tensor at the placeholder
    span. Asserts:

    1. Expanded ``input_ids`` length matches the SGLang formula
       ``grid_t * (grid_h//merge) * (grid_w//merge)``.
    2. Image span starts/ends bracket exactly the expansion region.
    3. Placeholder positions in the rewritten ids carry the per-image
       ``pad_value`` (not the original image token id) — matches
       SGLang's ``pad_input_tokens`` semantics so the radix-attention
       prefix matcher sees a unique hash per image.
    4. ``merge_multimodal_embeddings`` replaces text embeddings at the
       span with the visual features and leaves the rest untouched.
    5. A negative control where the span is shifted by 1 yields a
       different merged tensor (the merge is span-sensitive).
    """
    pytest.importorskip("safetensors")
    pytest.importorskip("transformers")

    checkpoint = _checkpoint_path()
    cfg = AutoConfig.from_pretrained(checkpoint, trust_remote_code=True)
    text_cfg = get_text_config(cfg)
    text_hidden_size = int(getattr(text_cfg, "hidden_size", 6144))
    projector_hidden_size = getattr(cfg, "projector_hidden_size", None)
    image_token_id = int(getattr(cfg, "image_token_index", 200025))
    video_token_id = int(getattr(cfg, "video_token_index", 200026))
    vision_config_raw = cfg.vision_config
    if hasattr(vision_config_raw, "to_dict"):
        vision_config_dict = vision_config_raw.to_dict()
    else:
        vision_config_dict = dict(vision_config_raw)
    vision_config_dict["num_hidden_layers"] = 2
    vision_config = CLIPVisionConfig.from_dict_or_obj(vision_config_dict)

    model = MiniMaxVLVisionModel(
        config=vision_config,
        text_hidden_size=text_hidden_size,
        projector_hidden_size=projector_hidden_size,
        dtype=torch.bfloat16,
    )

    # Reuse the partial-checkpoint helper so the forward isn't on random init.
    targets = [
        "vision_tower.vision_model.embeddings.patch_embedding.weight",
        "vision_tower.vision_model.pre_layrnorm.weight",
        "vision_tower.vision_model.pre_layrnorm.bias",
    ]
    for i in range(2):
        for suffix in (
            "self_attn.q_proj.weight",
            "self_attn.q_proj.bias",
            "self_attn.k_proj.weight",
            "self_attn.k_proj.bias",
            "self_attn.v_proj.weight",
            "self_attn.v_proj.bias",
            "self_attn.out_proj.weight",
            "self_attn.out_proj.bias",
            "layer_norm1.weight",
            "layer_norm1.bias",
            "layer_norm2.weight",
            "layer_norm2.bias",
            "mlp.fc1.weight",
            "mlp.fc1.bias",
            "mlp.fc2.weight",
            "mlp.fc2.bias",
        ):
            targets.append(f"vision_tower.vision_model.encoder.layers.{i}.{suffix}")
    targets += [
        "multi_modal_projector.linear_1.weight",
        "multi_modal_projector.linear_1.bias",
        "multi_modal_projector.linear_2.weight",
        "multi_modal_projector.linear_2.bias",
        "patch_merge_mlp.linear_1.weight",
        "patch_merge_mlp.linear_1.bias",
        "patch_merge_mlp.linear_2.weight",
        "patch_merge_mlp.linear_2.bias",
    ]
    with open(os.path.join(checkpoint, "model.safetensors.index.json")) as f:
        weight_map = json.load(f)["weight_map"]
    by_shard: Dict[str, List[str]] = {}
    for key in targets:
        shard = weight_map[key]
        by_shard.setdefault(shard, []).append(key)
    raw_weights: Dict[str, torch.Tensor] = {}
    for shard, keys in by_shard.items():
        with safe_open(os.path.join(checkpoint, shard), framework="pt", device="cpu") as sf:
            for k in keys:
                raw_weights[k] = sf.get_tensor(k)
    load_minimax_m3_vl_state_dict(model, raw_weights, strict=False)
    model = model.to("cuda")
    model.eval()

    # Build a fake input stream with one image placeholder.
    raw_input_ids = [10, 11, image_token_id, 12, 13]
    spatial_merge = vision_config.img_token_compression_config["spatial_merge_size"]
    grid_thw = [1, 4, 4]
    repeat = compute_visual_token_count(*grid_thw, spatial_merge_size=spatial_merge)
    assert repeat == 4

    expanded_ids, image_spans, video_spans = expand_multimodal_placeholders(
        raw_input_ids,
        image_token_id=image_token_id,
        image_repeats=[repeat],
        video_token_id=video_token_id,
    )
    assert len(expanded_ids) == len(raw_input_ids) - 1 + repeat
    assert image_spans == [(2, 2 + repeat)]
    assert video_spans == []

    pad_value = 999_001
    padded_ids = apply_multimodal_pad_values(
        expanded_ids,
        image_pad_values=[pad_value],
        image_spans=image_spans,
    )
    for i in range(image_spans[0][0], image_spans[0][1]):
        assert padded_ids[i] == pad_value
    # Positions outside the span are untouched.
    for i, original in enumerate(expanded_ids):
        if i < image_spans[0][0] or i >= image_spans[0][1]:
            assert padded_ids[i] == original

    # Run the partial vision tower on synthetic deterministic pixels.
    n_patches = grid_thw[0] * grid_thw[1] * grid_thw[2]
    pixel_cols = (
        vision_config.num_channels
        * vision_config.img_token_compression_config["temporal_patch_size"]
        * vision_config.patch_size
        * vision_config.patch_size
    )
    torch.manual_seed(20260604)
    pixel_values = torch.randn((n_patches, pixel_cols), dtype=torch.bfloat16, device="cuda")
    with torch.no_grad():
        visual_embeds = model(pixel_values=pixel_values, grid_thw=[grid_thw])
    assert visual_embeds.shape == (repeat, text_hidden_size)
    assert torch.isfinite(visual_embeds).all()

    # Build a synthetic text-embedding tensor at the same hidden size and
    # mark each row with a unique sentinel for later equality checks.
    text_embeds = (
        torch.arange(len(expanded_ids) * text_hidden_size, dtype=torch.bfloat16, device="cuda")
        .reshape(len(expanded_ids), text_hidden_size)
        .contiguous()
    )

    merged = merge_multimodal_embeddings(
        text_embeds, image_embeds=visual_embeds, image_spans=image_spans
    )
    # Inside the span, merged carries the visual embeds.
    assert torch.allclose(
        merged[image_spans[0][0] : image_spans[0][1]].float(),
        visual_embeds.float(),
    ), "merge did not place visual embeddings at the image span"
    # Outside the span, merged carries the original text embeds.
    for i in range(len(expanded_ids)):
        if i < image_spans[0][0] or i >= image_spans[0][1]:
            assert torch.equal(merged[i], text_embeds[i]), f"merge mutated text-only position {i}"

    # Negative control: shift the span by 1 and confirm the merge differs.
    shifted_spans = [(image_spans[0][0] + 1, image_spans[0][1] + 1)]
    if shifted_spans[0][1] <= len(expanded_ids):
        merged_shifted = merge_multimodal_embeddings(
            text_embeds, image_embeds=visual_embeds, image_spans=shifted_spans
        )
        assert not torch.equal(merged, merged_shifted), (
            "merge is not span-sensitive: shifting the span left the result unchanged"
        )


# ---------------------------------------------------------------------------
# SGLang/HF processor contract: VISION_START + image_token +
# VISION_END expansion + inclusive-end offsets + pad_input_tokens parity.
# ---------------------------------------------------------------------------


def _sglang_build_input_ids_oracle(
    prompt: List[int],
    *,
    image_token_id: int,
    image_grid_thws: List[List[int]],
    video_token_id: int,
    video_grid_thws: List[List[int]],
    spatial_merge_size: int,
):
    """Standalone oracle mirroring SGLang ``base_processor.build_input_ids``.

    Re-implements the algorithm at
    ``sglang/srt/multimodal/processors/base_processor.py:289-354`` directly
    in the test so the helper under test cannot be passing just by sharing
    a bug with the production implementation.
    """
    vision_start_indices = []
    for i in range(len(prompt) - 1):
        nxt = prompt[i + 1]
        if image_token_id is not None and nxt == image_token_id:
            vision_start_indices.append((i, "image"))
        elif video_token_id is not None and nxt == video_token_id:
            vision_start_indices.append((i, "video"))
    modality_list = [m for _, m in vision_start_indices]

    cur_idx = 0
    input_ids = []
    image_offsets = []
    video_offsets = []
    img_idx = 0
    video_idx = 0
    for mm_start_idx, modality in vision_start_indices:
        if modality == "image":
            gt, gh, gw = image_grid_thws[img_idx]
            mm_token_num = gt * (gh // spatial_merge_size) * (gw // spatial_merge_size)
            mm_token_id = image_token_id
            img_idx += 1
        else:
            gt, gh, gw = video_grid_thws[video_idx]
            mm_token_num = gt * (gh // spatial_merge_size) * (gw // spatial_merge_size)
            mm_token_id = video_token_id
            video_idx += 1
        input_ids.extend(prompt[cur_idx : mm_start_idx + 1])
        mm_offset_start = len(input_ids)
        input_ids.extend([mm_token_id] * mm_token_num)
        cur_idx = mm_start_idx + 2
        offset = (mm_offset_start, len(input_ids) - 1)
        if modality == "image":
            image_offsets.append(offset)
        else:
            video_offsets.append(offset)
    input_ids.extend(prompt[cur_idx:])
    return input_ids, image_offsets, video_offsets, modality_list


# Canonical M3 VL ids from the checkpoint's ``added_tokens.json``.
# Cross-checked against HF config + tokenizer in the real-checkpoint
# smoke tests below.
_IMG_TOK = 200025
_VID_TOK = 200026
_VS_TOK = 200029  # ]<]start of image[>[  (used for BOTH images and videos)
_VE_TOK = 200030  # ]<]end of image[>[


@pytest.mark.cpu_only
def test_build_multimodal_input_ids_image_start_end_matches_sglang_oracle():
    """SGLang-style bracketed image expansion matches the oracle byte-for-byte."""
    # Prompt mirrors what the M3 tokenizer emits for "<sys> ]<]start of image[>[
    # ]<]image[>[ ]<]end of image[>[ describe it </sys>" before processor
    # expansion: BOS + sys + VS + IMG + VE + tail. The single IMG between VS
    # and VE is the SGLang/HF processor entry contract.
    prompt = [50, 51, _VS_TOK, _IMG_TOK, _VE_TOK, 60, 61, 62]
    image_grid_thws = [[1, 4, 4]]  # 1*2*2 = 4 placeholders
    spatial_merge = 2

    trt_ids, trt_img_off, trt_vid_off, trt_mods = build_multimodal_input_ids(
        prompt,
        image_token_id=_IMG_TOK,
        image_grid_thws=image_grid_thws,
        video_token_id=_VID_TOK,
        spatial_merge_size=spatial_merge,
    )
    oracle = _sglang_build_input_ids_oracle(
        prompt,
        image_token_id=_IMG_TOK,
        image_grid_thws=image_grid_thws,
        video_token_id=_VID_TOK,
        video_grid_thws=[],
        spatial_merge_size=spatial_merge,
    )
    assert trt_ids == oracle[0], (trt_ids, oracle[0])
    assert trt_img_off == oracle[1], (trt_img_off, oracle[1])
    assert trt_vid_off == oracle[2]
    assert trt_mods == oracle[3]

    # Explicit byte-level expectations: VS at idx=2, 4 IMG_TOKs at 3..6,
    # VE at 7. Offset is inclusive: (3, 6).
    assert trt_ids == [50, 51, _VS_TOK, _IMG_TOK, _IMG_TOK, _IMG_TOK, _IMG_TOK, _VE_TOK, 60, 61, 62]
    assert trt_img_off == [(3, 6)]
    assert trt_vid_off == []
    assert trt_mods == ["image"]


@pytest.mark.cpu_only
def test_build_multimodal_input_ids_video_uses_image_start_end_markers():
    """Videos use the IMAGE start/end markers (processing_minimax.py:54-62)."""
    # Video frame grid (2, 4, 6) merge=2 -> 2*2*3 = 12 placeholders.
    prompt = [10, _VS_TOK, _VID_TOK, _VE_TOK, 20]
    video_grid_thws = [[2, 4, 6]]
    spatial_merge = 2

    trt_ids, img_off, vid_off, mods = build_multimodal_input_ids(
        prompt,
        image_token_id=_IMG_TOK,
        image_grid_thws=[],
        video_token_id=_VID_TOK,
        video_grid_thws=video_grid_thws,
        spatial_merge_size=spatial_merge,
    )
    expected_ids = [10, _VS_TOK] + [_VID_TOK] * 12 + [_VE_TOK, 20]
    assert trt_ids == expected_ids
    assert img_off == []
    assert vid_off == [(2, 13)]  # inclusive
    assert mods == ["video"]


@pytest.mark.cpu_only
def test_build_multimodal_input_ids_image_and_video_interleaved():
    """Interleaved image + video preserves order in modality_list and offsets."""
    # image (1,4,4)=4 placeholders, then video (1,4,6)=6 placeholders.
    prompt = [1, _VS_TOK, _IMG_TOK, _VE_TOK, 2, _VS_TOK, _VID_TOK, _VE_TOK, 3]
    trt_ids, img_off, vid_off, mods = build_multimodal_input_ids(
        prompt,
        image_token_id=_IMG_TOK,
        image_grid_thws=[[1, 4, 4]],
        video_token_id=_VID_TOK,
        video_grid_thws=[[1, 4, 6]],
        spatial_merge_size=2,
    )
    oracle = _sglang_build_input_ids_oracle(
        prompt,
        image_token_id=_IMG_TOK,
        image_grid_thws=[[1, 4, 4]],
        video_token_id=_VID_TOK,
        video_grid_thws=[[1, 4, 6]],
        spatial_merge_size=2,
    )
    assert trt_ids == oracle[0]
    assert img_off == oracle[1]
    assert vid_off == oracle[2]
    assert mods == oracle[3]
    assert mods == ["image", "video"]


@pytest.mark.cpu_only
def test_build_multimodal_input_ids_rejects_count_mismatch():
    # Two placeholders but only one grid -> ValueError.
    prompt = [_VS_TOK, _IMG_TOK, _VE_TOK, _VS_TOK, _IMG_TOK, _VE_TOK]
    with pytest.raises(ValueError, match="image_grid_thws"):
        build_multimodal_input_ids(
            prompt,
            image_token_id=_IMG_TOK,
            image_grid_thws=[[1, 4, 4]],
            video_token_id=_VID_TOK,
            spatial_merge_size=2,
        )
    with pytest.raises(ValueError, match="video_grid_thws"):
        build_multimodal_input_ids(
            [_VS_TOK, _VID_TOK, _VE_TOK],
            image_token_id=_IMG_TOK,
            image_grid_thws=[],
            video_token_id=_VID_TOK,
            video_grid_thws=[],
            spatial_merge_size=2,
        )


@pytest.mark.cpu_only
def test_pad_multimodal_input_tokens_inclusive_matches_sglang_pad():
    """``pad_multimodal_input_tokens`` matches SGLang ``pad_input_tokens``.

    Mirror the SGLang inclusive-end pad algorithm and assert the helper
    rewrites *exactly* the multimodal positions (and only those) with the
    per-item ``pad_value``.
    """
    # image (1,4,4)=4 placeholders + video (1,4,6)=6 placeholders.
    prompt = [1, _VS_TOK, _IMG_TOK, _VE_TOK, 2, _VS_TOK, _VID_TOK, _VE_TOK, 3]
    expanded, img_off, vid_off, _ = build_multimodal_input_ids(
        prompt,
        image_token_id=_IMG_TOK,
        image_grid_thws=[[1, 4, 4]],
        video_token_id=_VID_TOK,
        video_grid_thws=[[1, 4, 6]],
        spatial_merge_size=2,
    )

    # SGLang pad values are unique per item; use distinct sentinels.
    img_pad = 777001
    vid_pad = 888001
    padded = pad_multimodal_input_tokens(
        expanded,
        image_pad_values=[img_pad],
        image_offsets=img_off,
        video_pad_values=[vid_pad],
        video_offsets=vid_off,
    )

    # Inclusive-end semantics: positions [start, end] are written.
    for start, end in img_off:
        for i in range(start, end + 1):
            assert padded[i] == img_pad, (i, padded[i])
    for start, end in vid_off:
        for i in range(start, end + 1):
            assert padded[i] == vid_pad, (i, padded[i])
    # Outside all spans -> untouched.
    span_set = set()
    for s, e in img_off + vid_off:
        span_set.update(range(s, e + 1))
    for i, original in enumerate(expanded):
        if i not in span_set:
            assert padded[i] == original, (i, padded[i], original)


@pytest.mark.cpu_only
def test_merge_multimodal_embeddings_inclusive_offsets_round_trip():
    """``merge_multimodal_embeddings_inclusive`` accepts SGLang inclusive offsets."""
    prompt = [9, _VS_TOK, _IMG_TOK, _VE_TOK, 10]
    expanded, img_off, _, _ = build_multimodal_input_ids(
        prompt,
        image_token_id=_IMG_TOK,
        image_grid_thws=[[1, 4, 4]],
        video_token_id=_VID_TOK,
        spatial_merge_size=2,
    )
    seq = len(expanded)
    hidden = 8
    text_embeds = torch.arange(seq * hidden, dtype=torch.float32).reshape(seq, hidden)
    visual_embeds = torch.full((img_off[0][1] - img_off[0][0] + 1, hidden), 5.0)

    merged = merge_multimodal_embeddings_inclusive(
        text_embeds, image_embeds=visual_embeds, image_offsets=img_off
    )
    for i in range(img_off[0][0], img_off[0][1] + 1):
        assert torch.all(merged[i] == 5.0), i
    for i in range(seq):
        if i < img_off[0][0] or i > img_off[0][1]:
            assert torch.equal(merged[i], text_embeds[i]), i


@pytest.mark.cpu_only
def test_negative_control_raw_exclusive_span_breaks_inclusive_pad():
    """Negative control — feeding raw exclusive spans into the SGLang-style
    inclusive pad/merge helpers clobbers the trailing VISION_END marker.

    The OLD raw-single-token expansion contract returns
    ``(start, start + repeat)`` exclusive-end spans (see
    :func:`expand_multimodal_placeholders`). The SGLang
    ``pad_input_tokens`` algorithm uses ``[start, end]`` inclusive-end
    indexing (``mm_utils.py:344-346``). Mixing the two writes one extra
    position past the placeholder run, which on a real bracketed prompt
    silently rewrites the VISION_END token.

    This test pins down that contract drift so a future regression that
    swaps the helper without updating the offset convention is caught
    here rather than at source-replay parity in MiniMax-M3 VL.
    """
    # Canonical M3 form: VS + IMG + VE around a placeholder.
    prompt = [50, 51, _VS_TOK, _IMG_TOK, _VE_TOK, 60]
    repeat = 4  # grid_thw (1, 4, 4) merge=2

    # Raw single-token expansion: returns *exclusive-end* spans.
    raw_ids, raw_spans, _ = expand_multimodal_placeholders(
        prompt, image_token_id=_IMG_TOK, image_repeats=[repeat]
    )
    # SGLang-aligned expansion: returns *inclusive-end* offsets.
    new_ids, new_off, _, _ = build_multimodal_input_ids(
        prompt,
        image_token_id=_IMG_TOK,
        image_grid_thws=[[1, 4, 4]],
        video_token_id=_VID_TOK,
        spatial_merge_size=2,
    )
    # Bytes happen to align because the M3 prompt places VS immediately
    # before the single placeholder. Confirm they agree at the byte level
    # so the negative control is really about offset convention drift.
    assert raw_ids == new_ids

    # But the offset conventions diverge: exclusive vs inclusive end.
    assert raw_spans[0] != new_off[0]
    assert raw_spans[0][1] == new_off[0][1] + 1

    # Feeding raw exclusive spans into the SGLang-inclusive pad helper
    # writes one extra position past the placeholder run, clobbering VE.
    pad_value = 999_999
    sgl_padded = pad_multimodal_input_tokens(
        new_ids, image_pad_values=[pad_value], image_offsets=new_off
    )
    misused_padded = pad_multimodal_input_tokens(
        new_ids,
        image_pad_values=[pad_value],
        image_offsets=[raw_spans[0]],  # convention mismatch
    )
    # The correct path leaves VE untouched.
    ve_pos = new_off[0][1] + 1
    assert sgl_padded[ve_pos] == _VE_TOK, sgl_padded[ve_pos]
    # The misuse overwrites VE with pad_value.
    assert misused_padded[ve_pos] == pad_value, (
        "Negative control regression: feeding raw exclusive spans into the "
        "SGLang-inclusive pad helper no longer rewrites the trailing "
        "VISION_END token. Update the discriminator."
    )
    assert sgl_padded != misused_padded


@pytest.mark.cpu_only
def test_negative_control_wrong_grid_order_breaks_parity():
    """Reversing the per-item grid order breaks parity at the placeholder count.

    MiniMax-M3 VL negative-control prerequisite for : a wrong grid_order
    must produce a different expansion than the SGLang oracle so the parity
    assertion has a real failure path.
    """
    prompt = [1, _VS_TOK, _IMG_TOK, _VE_TOK, 2, _VS_TOK, _IMG_TOK, _VE_TOK, 3]
    correct_ids, correct_off, _, _ = build_multimodal_input_ids(
        prompt,
        image_token_id=_IMG_TOK,
        image_grid_thws=[[1, 4, 4], [1, 4, 6]],  # 4 then 6 placeholders
        video_token_id=_VID_TOK,
        spatial_merge_size=2,
    )
    swapped_ids, swapped_off, _, _ = build_multimodal_input_ids(
        prompt,
        image_token_id=_IMG_TOK,
        image_grid_thws=[[1, 4, 6], [1, 4, 4]],  # swapped: 6 then 4
        video_token_id=_VID_TOK,
        spatial_merge_size=2,
    )
    assert correct_ids != swapped_ids
    assert correct_off != swapped_off


# ---------------------------------------------------------------------------
# CUDA + real-checkpoint processor parity smoke.
# ---------------------------------------------------------------------------


def _make_processor_image_grid_thw(height: int, width: int) -> List[List[int]]:
    """Predict ``image_grid_thw`` for a synthetic PIL image without invoking
    the processor — used to *also* assert grid prediction matches the
    actual processor output in the CUDA smoke."""
    # Replicate ``MiniMaxM3VLImageProcessor.get_number_of_image_patches`` logic:
    # height/width are already multiples of (patch=14 * merge=2 = 28); the
    # processor pads a single image to 2 temporal frames so grid_t=1.
    patch = 14
    merge = 2
    factor = patch * merge
    assert height % factor == 0 and width % factor == 0
    grid_h = height // patch
    grid_w = width // patch
    grid_t = 1
    return [[grid_t, grid_h, grid_w]]


@pytest.mark.gpu
@pytest.mark.skipif(not _has_cuda(), reason="processor parity smoke needs CUDA")
def _load_real_processor(checkpoint: str):
    """Best-effort load of the real ``MiniMaxVLProcessor``.

    Returns ``None`` if any dependency along the load path is missing
    (some lean container images strip ``torchvision`` or ship a
    ``transformers`` version older than what the checkpoint's
    ``image_processor.py`` imports). The caller then falls back to a
    hand-rolled-processor oracle that reimplements the exact string-level
    expansion contract from ``processing_minimax.py``.
    """
    try:
        return AutoProcessor.from_pretrained(checkpoint, trust_remote_code=True)
    except Exception:
        return None


def _hand_rolled_processor_input_ids(
    tokenizer,
    text: str,
    image_grid_thws: List[List[int]],
    *,
    spatial_merge_size: int,
) -> List[int]:
    """Reimplement ``MiniMaxVLProcessor.__call__`` string expansion.

    Used as a fallback oracle when ``AutoProcessor.from_pretrained``
    cannot be loaded in the current environment. The algorithm mirrors
    ``processing_minimax.py:191-207`` verbatim: for each image, replace
    the next ``]<]image[>[`` substring with
    ``]<]start of image[>[ + ]<]placeholder[>[ * N + ]<]end of image[>[``;
    after all replacements, swap ``]<]placeholder[>[`` back to
    ``]<]image[>[``. Then tokenize the resulting string.
    """
    image_token = "]<]image[>["
    vs_token = "]<]start of image[>["
    ve_token = "]<]end of image[>["
    placeholder = "]<]placeholder[>["
    merge_length = spatial_merge_size**2

    out = text
    for grid in image_grid_thws:
        gt, gh, gw = grid
        n = gt * gh * gw // merge_length
        replacement = vs_token + placeholder * n + ve_token
        out = out.replace(image_token, replacement, 1)
    out = out.replace(placeholder, image_token)
    return tokenizer.encode(out, add_special_tokens=False)


def test_real_checkpoint_processor_input_ids_match_trt_build_helper():
    """TRT helper matches the real HF/SGLang processor.

    Two oracle paths so the test runs in any container:
      * **Preferred**: ``AutoProcessor.from_pretrained(checkpoint, ...)``
        — the actual ``MiniMaxVLProcessor`` shipped with the checkpoint.
        Captures its ``input_ids`` (the expanded
        ``VISION_START + IMG_TOK * N + VISION_END`` form) and
        ``image_grid_thw``.
      * **Fallback**: hand-rolled-processor oracle (see
        :func:`_hand_rolled_processor_input_ids`) that reimplements the
        exact string-level expansion contract from
        ``processing_minimax.py``.

    In both cases:
      1. Build the SGLang-style pre-expansion prompt (with a single
         IMG_TOK between VS/VE) by replacing the oracle's
         ``[IMG_TOK] * N`` run with a single IMG_TOK.
      2. Call TRT's :func:`build_multimodal_input_ids` on that
         pre-expansion prompt with the matching ``image_grid_thw`` and
         assert the output is byte-for-byte equal to the oracle output,
         the per-image inclusive offset bounds the expanded placeholder
         run, and the bracketing VS/VE tokens straddle that offset.
    """
    pytest.importorskip("transformers")

    checkpoint = _checkpoint_path()
    cfg = AutoConfig.from_pretrained(checkpoint, trust_remote_code=True)
    image_token_id = int(cfg.image_token_index)
    video_token_id = int(cfg.video_token_index)
    assert image_token_id == _IMG_TOK
    assert video_token_id == _VID_TOK

    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
    vs_id = tokenizer.convert_tokens_to_ids("]<]start of image[>[")
    ve_id = tokenizer.convert_tokens_to_ids("]<]end of image[>[")
    assert vs_id == _VS_TOK, vs_id
    assert ve_id == _VE_TOK, ve_id

    # Fixed multimodal prompt — uses the canonical M3 IMAGE_TOKEN string
    # ``]<]image[>[`` exactly as a real client would.
    text_prompt = "Describe this image: ]<]image[>[ briefly."
    # 56x56 RGB image: with patch=14, merge=2, factor=28 -> divisible.
    expected_grid = _make_processor_image_grid_thw(56, 56)

    processor = _load_real_processor(checkpoint)

    if processor is not None:
        pytest.importorskip("PIL")
        rng = np.random.RandomState(20260604)
        pil_image = Image.fromarray(rng.randint(0, 255, (56, 56, 3), dtype=np.uint8))
        proc_out = processor(images=[pil_image], text=text_prompt, return_tensors="pt")
        assert "input_ids" in proc_out
        assert "image_grid_thw" in proc_out
        oracle_input_ids = proc_out["input_ids"][0].tolist()
        oracle_grid_thw = proc_out["image_grid_thw"].tolist()
        oracle_source = "AutoProcessor"
    else:
        oracle_grid_thw = expected_grid
        oracle_input_ids = _hand_rolled_processor_input_ids(
            tokenizer, text_prompt, oracle_grid_thw, spatial_merge_size=2
        )
        oracle_source = "hand-rolled MiniMaxVLProcessor"

    print(f"[goal22.2 processor-parity] oracle source: {oracle_source}")
    assert oracle_grid_thw == expected_grid, (oracle_grid_thw, expected_grid)
    grid_t, grid_h, grid_w = oracle_grid_thw[0]
    spatial_merge = 2
    expected_repeat = grid_t * (grid_h // spatial_merge) * (grid_w // spatial_merge)

    # Build the SGLang-style pre-expansion prompt: walk oracle_input_ids,
    # collapse each consecutive image_token_id run back to a single token.
    pre_expand: List[int] = []
    i = 0
    n_runs_collapsed = 0
    while i < len(oracle_input_ids):
        tok = oracle_input_ids[i]
        if tok == image_token_id:
            pre_expand.append(image_token_id)
            run_len = 0
            while i < len(oracle_input_ids) and oracle_input_ids[i] == image_token_id:
                run_len += 1
                i += 1
            n_runs_collapsed += 1
            assert run_len == expected_repeat, (run_len, expected_repeat)
        else:
            pre_expand.append(tok)
            i += 1
    assert n_runs_collapsed == 1, n_runs_collapsed

    # TRT helper must reproduce the oracle's input_ids byte-for-byte from
    # the pre-expansion form + image_grid_thw.
    trt_ids, trt_img_off, trt_vid_off, trt_mods = build_multimodal_input_ids(
        pre_expand,
        image_token_id=image_token_id,
        image_grid_thws=oracle_grid_thw,
        video_token_id=video_token_id,
        spatial_merge_size=spatial_merge,
    )
    if trt_ids != oracle_input_ids:
        # Be loud about where TRT diverges so a future failure is easy to
        # localise.
        first_diff = next(
            (
                j
                for j in range(min(len(trt_ids), len(oracle_input_ids)))
                if trt_ids[j] != oracle_input_ids[j]
            ),
            min(len(trt_ids), len(oracle_input_ids)),
        )
        raise AssertionError(
            f"TRT helper diverged from {oracle_source} oracle at index "
            f"{first_diff}; len_trt={len(trt_ids)} "
            f"len_oracle={len(oracle_input_ids)}"
        )
    assert trt_mods == ["image"]
    assert trt_vid_off == []

    # Image span (inclusive end) must point at the placeholder run, and
    # the bracketing VS/VE tokens must straddle that span.
    start, end = trt_img_off[0]
    assert end - start + 1 == expected_repeat
    for j in range(start, end + 1):
        assert oracle_input_ids[j] == image_token_id, (j, oracle_input_ids[j], image_token_id)
    assert oracle_input_ids[start - 1] == vs_id
    assert oracle_input_ids[end + 1] == ve_id


def test_real_checkpoint_processor_video_input_ids_match_trt_build_helper():
    """Video-capable prompts also go through the bracketed contract.

    ``processing_minimax.py:54-62`` documents that the M3 serving path
    uses the IMAGE start/end markers for BOTH images and videos. This
    test pins that contract: a fixed video-capable prompt with the
    canonical ``]<]video[>[`` literal expands into
    ``VISION_START + VIDEO_TOK * N + VISION_END`` (note: VISION_START is
    the IMAGE start marker), and TRT's
    :func:`build_multimodal_input_ids` reproduces the hand-rolled
    processor output byte-for-byte for that input.
    """
    pytest.importorskip("transformers")

    checkpoint = _checkpoint_path()
    cfg = AutoConfig.from_pretrained(checkpoint, trust_remote_code=True)
    image_token_id = int(cfg.image_token_index)
    video_token_id = int(cfg.video_token_index)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
    vs_id = tokenizer.convert_tokens_to_ids("]<]start of image[>[")
    ve_id = tokenizer.convert_tokens_to_ids("]<]end of image[>[")

    # Build the expected expanded string by hand using
    # ``processing_minimax.py``'s video-expansion algorithm but without
    # invoking the real video processor (which would need a real video
    # file). The processor's algorithm (lines 209-245) places
    # ``VISION_START + IMG_TOK * frame_seqlen + VISION_END`` per frame.
    # Here we set ``grid_t = 1`` so a single bracket is emitted, and the
    # placeholder count is ``frame_seqlen = (grid_h * grid_w) // merge ** 2``.
    text_prompt = "Watch: ]<]video[>[ then summarize."
    grid_t, grid_h, grid_w = 1, 4, 6
    spatial_merge = 2
    n_per_frame = (grid_h * grid_w) // (spatial_merge**2)
    vs_token = "]<]start of image[>["
    ve_token = "]<]end of image[>["
    video_token = "]<]video[>["
    placeholder = "]<]placeholder[>["
    # Mirror ``processing_minimax.py:209-245`` for grid_t=1, no timestamps.
    video_replacement = ""
    for _ in range(grid_t):
        video_replacement += vs_token + placeholder * n_per_frame + ve_token
    expanded_text = text_prompt.replace(video_token, video_replacement, 1)
    # Then swap placeholder back to VIDEO_TOK (videos differ from images
    # here: the processor swaps placeholder back to the *video* token id,
    # not image — but the original processing_minimax.py reuses the same
    # `]<]image[>[` swap for both modalities... let us check).
    # Actually `processing_minimax.py:245`:
    #     text[i] = text[i].replace(placeholder, self.VIDEO_TOKEN)
    # So video placeholders ARE swapped to VIDEO_TOKEN, not IMAGE_TOKEN.
    expanded_text = expanded_text.replace(placeholder, video_token)
    oracle_input_ids = tokenizer.encode(expanded_text, add_special_tokens=False)

    # Construct the SGLang-style pre-expansion stream: single VIDEO_TOK between
    # VISION_START and VISION_END for each grid_t bracket.
    pre_text = text_prompt.replace(
        video_token,
        (vs_token + video_token + ve_token) * grid_t,
        1,
    )
    pre_expand = tokenizer.encode(pre_text, add_special_tokens=False)

    # TRT helper expansion.
    trt_ids, img_off, vid_off, mods = build_multimodal_input_ids(
        pre_expand,
        image_token_id=image_token_id,
        image_grid_thws=[],
        video_token_id=video_token_id,
        video_grid_thws=[[grid_t, grid_h, grid_w]],
        spatial_merge_size=spatial_merge,
    )
    assert trt_ids == oracle_input_ids, (
        f"TRT helper diverged from hand-rolled processor video oracle; "
        f"len_trt={len(trt_ids)} len_oracle={len(oracle_input_ids)}"
    )
    assert img_off == []
    assert mods == ["video"]
    # Video span size matches grid product divided by merge**2.
    start, end = vid_off[0]
    assert end - start + 1 == n_per_frame
    # IMAGE start/end markers straddle the span (videos reuse them).
    assert oracle_input_ids[start - 1] == vs_id
    assert oracle_input_ids[end + 1] == ve_id
    # The placeholder run contains VIDEO_TOK ids (NOT IMAGE_TOK).
    for j in range(start, end + 1):
        assert oracle_input_ids[j] == video_token_id, (j, oracle_input_ids[j])


def _real_checkpoint_input_processor():
    """The registered input processor on the checkpoint's own tokenizer and HF processor."""
    checkpoint = _checkpoint_path()
    if not os.path.isdir(checkpoint):
        pytest.skip(f"checkpoint not found: {checkpoint}")
    pytest.importorskip("transformers")
    config = AutoConfig.from_pretrained(checkpoint, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
    cls = get_minimax_m3_vl_input_processor_cls()
    return cls(model_path=checkpoint, config=config, tokenizer=tokenizer), tokenizer


@pytest.mark.parametrize("timed", [False, True], ids=["untimed", "timed"])
@pytest.mark.parametrize("num_frames,side", [(3, 230), (4, 224), (7, 300)])
def test_real_checkpoint_video_fast_path_matches_processor_input_ids(
    num_frames: int, side: int, timed: bool
):
    """The token-ID fast path expands a video exactly as the checkpoint's processor does.

    The same frames go through the text-prompt path (the HF processor writes
    ``input_ids`` itself) and the token-ID path (``expand_prompt_token_ids_for_mm``
    derives the bracket count from the frame count); the two id sequences must
    match token for token, with one ``VISION_START`` bracket per temporal patch
    of the processor's ``video_grid_thw``. Odd and even frame counts cover the
    last-frame padding, and the frame sides are not multiples of the patch
    size, so the processor's resize is on the path.

    A ``VideoData`` with timing gets a ``]<]{t:.1f} seconds[>[`` timestamp
    before each bracket on both paths; a frame list gets none. The marker
    follows a newline, as in trtllm-serve's rendered prompts (see
    ``_video_timestamp_ids`` for text glued to the marker).
    """
    proc, tokenizer = _real_checkpoint_input_processor()
    rng = np.random.default_rng(num_frames)
    frames = [rng.integers(0, 256, size=(side, side, 3), dtype=np.uint8) for _ in range(num_frames)]
    video = _timed_video(frames) if timed else frames

    fast_ids, grid = _expand_video_prompt_both_ways(
        proc, tokenizer, "Watch:\n]<]video[>[ then summarize.", [video]
    )

    temporal = int(proc.processor.video_processor.temporal_patch_size)
    brackets = -(-num_frames // temporal)
    assert int(grid[0, 0]) == brackets
    vs_id = tokenizer.convert_tokens_to_ids("]<]start of image[>[")
    assert fast_ids.count(vs_id) == brackets
    assert tokenizer.decode(fast_ids).count(" seconds[>[") == (brackets if timed else 0)


@pytest.mark.parametrize(
    "timing",
    [(True, False), (False, True), (True, True)],
    ids=["timed_untimed", "untimed_timed", "timed_timed"],
)
def test_real_checkpoint_two_video_fast_path_matches_processor_input_ids(timing):
    """Two videos in one prompt expand on the token-ID path as the checkpoint's processor does.

    The processor gets one ``video_metadata`` entry per video, with
    ``fps=None`` for a frame list, and timestamps only the timed videos. The
    markers are adjacent, so the second video's first timestamp follows the
    first video's ``VISION_END`` rather than prompt text.
    """
    proc, tokenizer = _real_checkpoint_input_processor()
    rng = np.random.default_rng(len(timing))
    num_frames = (3, 4)
    videos = []
    for n, timed in zip(num_frames, timing):
        frames = [rng.integers(0, 256, size=(230, 230, 3), dtype=np.uint8) for _ in range(n)]
        videos.append(_timed_video(frames) if timed else frames)

    fast_ids, grid = _expand_video_prompt_both_ways(
        proc, tokenizer, "Watch:\n]<]video[>[]<]video[>[ then compare.", videos
    )

    temporal = int(proc.processor.video_processor.temporal_patch_size)
    brackets = [-(-n // temporal) for n in num_frames]
    assert grid[:, 0].tolist() == brackets
    vs_id = tokenizer.convert_tokens_to_ids("]<]start of image[>[")
    assert fast_ids.count(vs_id) == sum(brackets)
    assert tokenizer.decode(fast_ids).count(" seconds[>[") == sum(
        b for b, timed in zip(brackets, timing) if timed
    )


def _timed_video(frames: List[np.ndarray]) -> Any:
    """``VideoData`` of every 7th frame of a 3 fps source: 0.0, 4.7, 9.3, 14.0, ... seconds."""
    from tensorrt_llm.inputs.multimodal_data import VideoData

    n = len(frames)
    return VideoData(
        frames=frames,
        metadata={
            "total_num_frames": 7 * n,
            "fps": 3.0,
            "duration": 7 * n / 3.0,
            "frames_indices": list(range(0, 7 * n, 7)),
        },
    )


def _expand_video_prompt_both_ways(
    proc: Any, tokenizer: Any, prompt: str, videos: List[Any]
) -> Tuple[List[int], torch.Tensor]:
    """Run ``prompt`` through the text-prompt and the token-ID paths and check that they agree.

    Returns the token-ID path's ids and the ``video_grid_thw`` rows, which
    both paths must report identically.
    """
    sampling = SimpleNamespace(add_special_tokens=False, truncate_prompt_tokens=None)
    slow_ids, slow_extra = proc.call_with_text_prompt(
        {"prompt": prompt, "multi_modal_data": {"video": videos}, "chat_template_applied": True},
        sampling,
    )
    fast_ids, fast_extra = proc.call_with_token_ids(
        {
            "prompt_token_ids": tokenizer.encode(prompt, add_special_tokens=False),
            "multi_modal_data": {"video": videos},
        },
        sampling,
    )

    # The processor may prepend special tokens of its own; everything after them must match.
    prefix = slow_ids[: len(slow_ids) - len(fast_ids)]
    assert all(t in tokenizer.all_special_ids for t in prefix), prefix
    assert slow_ids[len(prefix) :] == fast_ids

    grid = torch.as_tensor(slow_extra["multimodal_data"]["video"]["video_grid_thw"]).reshape(-1, 3)
    assert torch.equal(
        grid,
        torch.as_tensor(fast_extra["multimodal_data"]["video"]["video_grid_thw"]).reshape(-1, 3),
    )
    return fast_ids, grid


def _smooth_uint8_image(side: int) -> np.ndarray:
    """A gradient picture: resizing it as float or as uint8 agrees to within rounding."""
    y = np.broadcast_to(np.linspace(0, 255, side)[:, None], (side, side))
    x = np.broadcast_to(np.linspace(0, 255, side)[None, :], (side, side))
    return np.stack([(x + y) / 2, x, y], axis=-1).astype(np.uint8)


@pytest.mark.parametrize("side", [224, 230])
def test_real_checkpoint_float_media_match_uint8_pixels(side: int):
    """Float [0, 1] media (media_io's ``pt`` format) yield the uint8 picture's pixels.

    The same picture goes through the checkpoint's processor as a uint8 HWC
    array (the preferred media_io format, rescaled by the processor) and as
    the ``ToTensor`` float tensor the LLM API default delivers (handed over
    with the processor's rescale turned off). Ids and grids must be identical
    and the pixels must agree to within the float-vs-uint8 rounding of the
    processor's resize plus the model dtype's, a few hundredths after
    normalization and far from the near-black result of rescaling twice. 224
    is already a multiple of the patch grid; 230 puts the resize on the path.
    """
    from torchvision.transforms import ToTensor

    proc, _ = _real_checkpoint_input_processor()
    array = _smooth_uint8_image(side)
    as_pt = ToTensor()(Image.fromarray(array))
    assert as_pt.dtype == torch.float32 and as_pt.shape == (3, side, side)
    sampling = SimpleNamespace(add_special_tokens=False, truncate_prompt_tokens=None)

    def run(media):
        return proc.call_with_text_prompt(
            {
                "prompt": "Look: ]<]image[>[ and watch: ]<]video[>[ then compare.",
                "multi_modal_data": {"image": [media], "video": [[media, media, media]]},
                "chat_template_applied": True,
            },
            sampling,
        )

    u8_ids, u8_extra = run(array)
    pt_ids, pt_extra = run(as_pt)
    assert pt_ids == u8_ids
    for modality, grid_key, pixel_key in (
        ("image", "image_grid_thw", "pixel_values"),
        ("video", "video_grid_thw", "pixel_values_videos"),
    ):
        u8 = u8_extra["multimodal_data"][modality]
        pt = pt_extra["multimodal_data"][modality]
        assert torch.equal(torch.as_tensor(u8[grid_key]), torch.as_tensor(pt[grid_key]))
        u8_pixels = u8[pixel_key].float()
        pt_pixels = pt[pixel_key].float()
        assert pt_pixels.shape == u8_pixels.shape
        assert (pt_pixels - u8_pixels).abs().max().item() < 0.05, modality
        # Rescaling twice would put every value near -mean/std (about -1.7).
        assert abs(pt_pixels.mean().item() - u8_pixels.mean().item()) < 0.01, modality


@pytest.mark.gpu
@pytest.mark.skipif(not _has_cuda(), reason="processor parity smoke needs CUDA")
def test_real_checkpoint_processor_full_pipeline_merge_and_pad_parity():
    """full pipeline against the SGLang/HF processor contract.

    Composes:
      - real ``MiniMaxVLProcessor`` (or hand-rolled-processor oracle if the
        real one cannot be loaded) -> ``input_ids`` + ``image_grid_thw``
      - TRT :func:`build_multimodal_input_ids` -> reproduces the oracle
        ``input_ids`` and computes inclusive offsets
      - TRT :func:`pad_multimodal_input_tokens` -> rewrites every offset
        position with a unique pad value and leaves everything else
        untouched
      - TRT vision tower (partial, first 2 encoder layers loaded from real
        BF16 checkpoint) -> visual embeddings on synthetic deterministic
        pixel tensors when the real ``pixel_values`` cannot be obtained
      - TRT :func:`merge_multimodal_embeddings_inclusive` -> places the
        visual embeddings at exactly the SGLang-style inclusive span and
        leaves every text-only position byte-equal to the original
        text-embedding row

    Plus negative controls:
      - raw expansion offset convention is NOT inclusive-end
      - shifting the merge offset by 1 changes the merged tensor
    """
    pytest.importorskip("safetensors")
    pytest.importorskip("transformers")

    checkpoint = _checkpoint_path()
    cfg = AutoConfig.from_pretrained(checkpoint, trust_remote_code=True)
    text_cfg = get_text_config(cfg)
    text_hidden_size = int(getattr(text_cfg, "hidden_size", 6144))
    projector_hidden_size = getattr(cfg, "projector_hidden_size", None)
    image_token_id = int(cfg.image_token_index)
    video_token_id = int(cfg.video_token_index)

    vision_config_raw = cfg.vision_config
    if hasattr(vision_config_raw, "to_dict"):
        vision_config_dict = vision_config_raw.to_dict()
    else:
        vision_config_dict = dict(vision_config_raw)
    vision_config_dict["num_hidden_layers"] = 2  # partial: keep the smoke fast
    vision_config = CLIPVisionConfig.from_dict_or_obj(vision_config_dict)
    spatial_merge = vision_config.img_token_compression_config["spatial_merge_size"]

    text_prompt = "What is shown? ]<]image[>[ Reply in one sentence."
    expected_grid = _make_processor_image_grid_thw(56, 56)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)

    processor = _load_real_processor(checkpoint)
    pixel_values = None
    if processor is not None:
        pytest.importorskip("PIL")
        rng = np.random.RandomState(20260604)
        pil_image = Image.fromarray(rng.randint(0, 255, (56, 56, 3), dtype=np.uint8))
        proc_out = processor(images=[pil_image], text=text_prompt, return_tensors="pt")
        processor_input_ids = proc_out["input_ids"][0].tolist()
        processor_grid_thw = proc_out["image_grid_thw"].tolist()
        pixel_values = proc_out["pixel_values"]
        oracle_source = "AutoProcessor"
    else:
        processor_grid_thw = expected_grid
        processor_input_ids = _hand_rolled_processor_input_ids(
            tokenizer, text_prompt, processor_grid_thw, spatial_merge_size=spatial_merge
        )
        oracle_source = "hand-rolled MiniMaxVLProcessor"

    print(f"[goal22.2 full-pipeline] oracle source: {oracle_source}")
    assert processor_grid_thw == expected_grid, (processor_grid_thw, expected_grid)

    # Build the pre-expansion prompt from the processor output by collapsing
    # the IMG_TOK run.
    grid_t, grid_h, grid_w = processor_grid_thw[0]
    expected_repeat = grid_t * (grid_h // spatial_merge) * (grid_w // spatial_merge)
    pre_expand: List[int] = []
    i = 0
    while i < len(processor_input_ids):
        tok = processor_input_ids[i]
        if tok == image_token_id:
            pre_expand.append(image_token_id)
            while i < len(processor_input_ids) and processor_input_ids[i] == image_token_id:
                i += 1
            continue
        pre_expand.append(tok)
        i += 1

    # TRT helper byte-for-byte parity with the real processor.
    trt_ids, trt_img_off, _, trt_mods = build_multimodal_input_ids(
        pre_expand,
        image_token_id=image_token_id,
        image_grid_thws=processor_grid_thw,
        video_token_id=video_token_id,
        spatial_merge_size=spatial_merge,
    )
    assert trt_ids == processor_input_ids, "TRT build helper != processor input_ids"
    assert trt_mods == ["image"]
    start, end = trt_img_off[0]
    assert end - start + 1 == expected_repeat

    # Negative control 1: the raw single-token expansion contract returns
    # exclusive-end spans, whereas the SGLang/HF processor contract uses
    # inclusive-end offsets. The expanded bytes happen to align (because the
    # processor wraps the placeholder with VS/VE adjacent to it) but the
    # offset conventions MUST diverge — otherwise a downstream consumer
    # using SGLang's inclusive pad/merge semantics with raw exclusive spans
    # would clobber the trailing VISION_END marker.
    raw_ids, raw_spans, _ = expand_multimodal_placeholders(
        pre_expand,
        image_token_id=image_token_id,
        image_repeats=[expected_repeat],
    )
    assert raw_ids == processor_input_ids  # bytes happen to match
    assert raw_spans[0] != trt_img_off[0], (
        "Raw single-token expansion and SGLang-aligned helper produced the "
        "SAME offset convention — the negative control is no longer "
        "discriminating."
    )
    assert raw_spans[0][1] == trt_img_off[0][1] + 1, (
        f"Expected exclusive-end raw span to be one past inclusive-end "
        f"SGLang offset; got raw={raw_spans[0]} sgl={trt_img_off[0]}"
    )

    # pad_input_tokens parity.
    img_pad = 555_001
    padded = pad_multimodal_input_tokens(
        processor_input_ids,
        image_pad_values=[img_pad],
        image_offsets=trt_img_off,
    )
    for j in range(start, end + 1):
        assert padded[j] == img_pad
    # Outside the span, untouched.
    for j in range(len(processor_input_ids)):
        if j < start or j > end:
            assert padded[j] == processor_input_ids[j], j

    # Vision tower forward against the real processor's pixel_values.
    model = MiniMaxVLVisionModel(
        config=vision_config,
        text_hidden_size=text_hidden_size,
        projector_hidden_size=projector_hidden_size,
        dtype=torch.bfloat16,
    )
    # Load only the partial first 2 encoder layers we need.
    targets = [
        "vision_tower.vision_model.embeddings.patch_embedding.weight",
        "vision_tower.vision_model.pre_layrnorm.weight",
        "vision_tower.vision_model.pre_layrnorm.bias",
    ]
    for li in range(2):
        for suffix in (
            "self_attn.q_proj.weight",
            "self_attn.q_proj.bias",
            "self_attn.k_proj.weight",
            "self_attn.k_proj.bias",
            "self_attn.v_proj.weight",
            "self_attn.v_proj.bias",
            "self_attn.out_proj.weight",
            "self_attn.out_proj.bias",
            "layer_norm1.weight",
            "layer_norm1.bias",
            "layer_norm2.weight",
            "layer_norm2.bias",
            "mlp.fc1.weight",
            "mlp.fc1.bias",
            "mlp.fc2.weight",
            "mlp.fc2.bias",
        ):
            targets.append(f"vision_tower.vision_model.encoder.layers.{li}.{suffix}")
    targets += [
        "multi_modal_projector.linear_1.weight",
        "multi_modal_projector.linear_1.bias",
        "multi_modal_projector.linear_2.weight",
        "multi_modal_projector.linear_2.bias",
        "patch_merge_mlp.linear_1.weight",
        "patch_merge_mlp.linear_1.bias",
        "patch_merge_mlp.linear_2.weight",
        "patch_merge_mlp.linear_2.bias",
    ]
    with open(os.path.join(checkpoint, "model.safetensors.index.json")) as f:
        weight_map = json.load(f)["weight_map"]
    by_shard: Dict[str, List[str]] = {}
    for k in targets:
        shard = weight_map[k]
        by_shard.setdefault(shard, []).append(k)
    raw_weights: Dict[str, torch.Tensor] = {}
    for shard, keys in by_shard.items():
        with safe_open(os.path.join(checkpoint, shard), framework="pt", device="cpu") as sf:
            for k in keys:
                raw_weights[k] = sf.get_tensor(k)
    load_minimax_m3_vl_state_dict(model, raw_weights, strict=False)
    model = model.to("cuda")
    model.eval()

    if pixel_values is None:
        # Synthesize deterministic BF16 pixel tensors matching the
        # processor's expected ``[N, C*T*P*P]`` shape.
        gt, gh, gw = processor_grid_thw[0]
        n_patches = gt * gh * gw
        pixel_cols = (
            vision_config.num_channels
            * vision_config.img_token_compression_config["temporal_patch_size"]
            * vision_config.patch_size
            * vision_config.patch_size
        )
        torch.manual_seed(20260604)
        pixel_values = torch.randn((n_patches, pixel_cols), dtype=torch.bfloat16, device="cuda")
    else:
        pixel_values = pixel_values.to("cuda", dtype=torch.bfloat16)
        if pixel_values.dim() == 3:
            pixel_values = pixel_values.reshape(pixel_values.shape[0] * pixel_values.shape[1], -1)

    with torch.no_grad():
        visual_embeds = model(
            pixel_values=pixel_values,
            grid_thw=processor_grid_thw,
        )
    assert visual_embeds.shape == (expected_repeat, text_hidden_size), visual_embeds.shape
    assert torch.isfinite(visual_embeds).all()
    assert visual_embeds.abs().sum().item() > 0

    # Merge step using inclusive offsets.
    text_embeds = (
        torch.arange(len(processor_input_ids) * text_hidden_size, dtype=torch.bfloat16)
        .reshape(len(processor_input_ids), text_hidden_size)
        .to("cuda")
    )
    merged = merge_multimodal_embeddings_inclusive(
        text_embeds, image_embeds=visual_embeds, image_offsets=trt_img_off
    )
    # Inside the span, merged carries the visual embeddings.
    assert torch.allclose(
        merged[start : end + 1].float(),
        visual_embeds.float(),
    ), "merge did not place visual embeddings at the SGLang inclusive offset"
    # Outside the span, merged matches the original text embeddings.
    for j in range(len(processor_input_ids)):
        if j < start or j > end:
            assert torch.equal(merged[j], text_embeds[j]), f"merge mutated text-only position {j}"

    # Negative control 2: shift the merge offset by 1 and confirm the merge
    # produces a different tensor — proves the merge is offset-sensitive.
    shifted_offsets = [(start + 1, end + 1)]
    if shifted_offsets[0][1] + 1 <= len(processor_input_ids):
        merged_shifted = merge_multimodal_embeddings_inclusive(
            text_embeds, image_embeds=visual_embeds, image_offsets=shifted_offsets
        )
        assert not torch.equal(merged, merged_shifted), (
            "Merge is not offset-sensitive — shifting the inclusive offset "
            "by 1 did not change the result."
        )


# ---------------------------------------------------------------------------
# Tokenized + MM fast-path hooks: regression coverage for the bug where
# Dynamo's pre-chat-templated prompt_token_ids were getting detokenized
# back to text and re-templated, doubling the image marker and crashing
# the HF MiniMaxVLProcessor at ``image_grid_thw[index].prod()``.
# ---------------------------------------------------------------------------


def _make_minimax_m3_vl_fast_path_processor():
    """Build a ``MiniMaxM3VLInputProcessor`` instance without invoking
    ``AutoProcessor.from_pretrained`` (which would require the full
    checkpoint). The instance has the constants/hooks needed to drive the
    tokenized fast-path expansion in isolation.
    """
    cls = get_minimax_m3_vl_input_processor_cls()
    obj = cls.__new__(cls)
    # Minimal config so get_vocab_size doesn't blow up.
    obj._config = SimpleNamespace(vocab_size=200_064)
    obj._model_path = "/dev/null"
    obj._tokenizer = None
    obj._use_fast = True
    obj._multimodal_hashing_supported = None
    # __init__ is bypassed via cls.__new__(cls); wire the special-token ids
    # the processor would otherwise resolve from config + tokenizer.
    obj._image_token_id = _IMG_TOK
    obj._video_token_id = _VID_TOK
    obj._vision_start_token_id = _VS_TOK
    obj._vision_end_token_id = _VE_TOK
    # Fake processor exposing the merge_size / temporal_patch_size that the
    # per-image / per-video token-count and bracket helpers consult. We don't
    # need the real processors here because the tests below feed precomputed
    # grid values.
    obj._processor = SimpleNamespace(
        image_processor=SimpleNamespace(merge_size=2),
        video_processor=SimpleNamespace(merge_size=2, temporal_patch_size=2),
    )
    return obj


@pytest.mark.cpu_only
def test_fast_path_get_text_with_mm_placeholders_returns_empty():
    """``get_text_with_mm_placeholders`` must return ``""`` so the chat
    template inserts exactly ``mm_counts['image']`` image markers via the
    ``{"type": "image"}`` content entries. Returning the markers ourselves
    would cause the template to double-insert them and re-trip the
    ``image_grid_thw[1]`` out-of-bounds crash that this fast path is
    designed to avoid.
    """
    proc = _make_minimax_m3_vl_fast_path_processor()
    assert proc.get_text_with_mm_placeholders({"image": 1}) == ""
    assert proc.get_text_with_mm_placeholders({"image": 3, "video": 0}) == ""
    assert proc.get_text_with_mm_placeholders({}) == ""


@pytest.mark.cpu_only
def test_fast_path_mm_token_and_special_ids_match_checkpoint_constants():
    """The MM fast-path expansion must report the M3 VL canonical token
    ids (200025/200026 placeholders + 200029/200030 framing) so the
    runtime's embed-mask + multimodal-hashing math classifies positions
    correctly.
    """
    proc = _make_minimax_m3_vl_fast_path_processor()
    mm_tokens = proc.get_mm_token_ids().tolist()
    specials = proc.get_mm_special_token_ids().tolist()
    assert mm_tokens == [_IMG_TOK, _VID_TOK]
    assert specials == [_VS_TOK, _VE_TOK]


@pytest.mark.cpu_only
def test_fast_path_expand_image_placeholders_adds_framing():
    """Each single ``IMAGE_TOKEN_ID`` in the Dynamo-style prompt must be
    rewritten to ``[VISION_START, IMAGE_TOKEN*N, VISION_END]`` where
    ``N = num_mm_tokens_per_placeholder[i] - 2``. This matches what the
    HF MiniMaxVLProcessor would produce in the slow path, so model
    forward sees the same input_ids shape regardless of which path the
    request takes.
    """
    proc = _make_minimax_m3_vl_fast_path_processor()
    img = _IMG_TOK
    vs = _VS_TOK
    ve = _VE_TOK

    # 1 placeholder, N = 5 inner image tokens (num_mm_tokens = 5 + 2).
    prompt_token_ids = [42, 43, img, 44, 45]
    expanded, mm_data_updates = proc.expand_prompt_token_ids_for_mm(
        prompt_token_ids,
        num_mm_tokens_per_placeholder=[7],
        mm_data={"image": [object()]},
    )
    assert mm_data_updates is None
    assert expanded == [42, 43, vs, img, img, img, img, img, ve, 44, 45]
    # Position invariant: the run is contiguous and totals exactly N+2 tokens.
    assert expanded.count(img) == 5
    assert expanded.count(vs) == 1 and expanded.count(ve) == 1


@pytest.mark.cpu_only
def test_fast_path_expand_multi_image_per_item_counts():
    """Two images with different per-item token counts must be expanded
    with their respective ``num_mm_tokens_per_placeholder`` entries.
    """
    proc = _make_minimax_m3_vl_fast_path_processor()
    img = _IMG_TOK

    prompt_token_ids = [1, img, 2, img, 3]
    # Image-0: N=3 inner placeholders (total 5). Image-1: N=4 inner (total 6).
    expanded, _ = proc.expand_prompt_token_ids_for_mm(
        prompt_token_ids,
        num_mm_tokens_per_placeholder=[5, 6],
        mm_data={"image": [object(), object()]},
    )
    assert expanded.count(img) == 3 + 4
    # Total length grew by (5 - 1) + (6 - 1) = 9 tokens.
    assert len(expanded) == len(prompt_token_ids) + 9


@pytest.mark.cpu_only
def test_fast_path_expand_video_placeholders_use_video_token_id():
    """Video-only requests must expand ``VIDEO_TOKEN_ID`` placeholders,
    not ``IMAGE_TOKEN_ID`` ones.
    """
    proc = _make_minimax_m3_vl_fast_path_processor()
    vid = _VID_TOK

    prompt_token_ids = [100, vid, 200]
    expanded, _ = proc.expand_prompt_token_ids_for_mm(
        prompt_token_ids,
        num_mm_tokens_per_placeholder=[4],
        mm_data={"video": [object()]},
    )
    # Inner count is 2, plus framing → total 4 added between text tokens.
    assert expanded.count(vid) == 2


@pytest.mark.cpu_only
def test_fast_path_expand_video_one_bracket_per_temporal_patch():
    """A video spans ``grid_t`` framed brackets, as the HF processor lays it out.

    Three frames pad to two temporal patches; a count of 8 is two brackets of
    VISION_START + 2 placeholders + VISION_END.
    """
    proc = _make_minimax_m3_vl_fast_path_processor()
    frames = [object()] * 3
    expanded, _ = proc.expand_prompt_token_ids_for_mm(
        [100, _VID_TOK, 200],
        num_mm_tokens_per_placeholder=[8],
        mm_data={"video": [SimpleNamespace(frames=frames)]},
    )
    bracket = [_VS_TOK, _VID_TOK, _VID_TOK, _VE_TOK]
    assert expanded == [100, *bracket, *bracket, 200]
    # A plain frame list works like the ``frames`` wrapper.
    expanded_list, _ = proc.expand_prompt_token_ids_for_mm(
        [100, _VID_TOK, 200], num_mm_tokens_per_placeholder=[8], mm_data={"video": [frames]}
    )
    assert expanded_list == expanded
    # A count that does not split into whole brackets is rejected.
    with pytest.raises(ValueError, match="2 x \\(VISION_START"):
        proc.expand_prompt_token_ids_for_mm(
            [100, _VID_TOK, 200], num_mm_tokens_per_placeholder=[7], mm_data={"video": [frames]}
        )


class _CharTokenizer:
    """Tokenizer stand-in: one id per character; ``encoded`` records every text."""

    def __init__(self):
        self.encoded = []

    def encode(self, text, add_special_tokens=True):
        assert not add_special_tokens
        self.encoded.append(text)
        return [ord(c) for c in text]


@pytest.mark.cpu_only
def test_fast_path_expand_timed_video_writes_timestamps():
    """A video with timing gets the processor's timestamp before each bracket.

    Three frames sampled at indices 0, 13 and 26 of a 10 fps source pad to
    two temporal patches, each stamped with the time of its first frame: 0.0
    and 2.6 seconds. A video without timing in the same request keeps
    adjacent brackets and never reaches the tokenizer. The timestamps are
    text outside the multimodal mask, so each video still counts 8 tokens and
    the timed one splits into one run per bracket.
    """
    from tensorrt_llm.inputs.multimodal import _compute_mm_masks, _find_mm_token_runs_from_mask

    proc = _make_minimax_m3_vl_fast_path_processor()
    proc._tokenizer = _CharTokenizer()
    frames = [object()] * 3
    timed = SimpleNamespace(frames=frames, metadata={"fps": 10.0, "frames_indices": [0, 13, 26]})
    expanded, _ = proc.expand_prompt_token_ids_for_mm(
        [100, _VID_TOK, 200, _VID_TOK, 300],
        num_mm_tokens_per_placeholder=[8, 8],
        mm_data={"video": [timed, frames]},
    )

    bracket = [_VS_TOK, _VID_TOK, _VID_TOK, _VE_TOK]
    stamps = ["]<]0.0 seconds[>[", "]<]2.6 seconds[>["]
    assert proc._tokenizer.encoded == stamps
    stamp_ids = [[ord(c) for c in stamp] for stamp in stamps]
    assert expanded == [
        100,
        *stamp_ids[0],
        *bracket,
        *stamp_ids[1],
        *bracket,
        200,
        *bracket,
        *bracket,
        300,
    ]
    mm_mask, _, _ = _compute_mm_masks(
        torch.tensor(expanded), None, proc.get_mm_token_ids(), proc.get_mm_special_token_ids()
    )
    item_run_cu_offsets, _, run_lengths = _find_mm_token_runs_from_mask(mm_mask, [8, 8])
    assert item_run_cu_offsets == [0, 2, 3]
    assert run_lengths == [4, 4, 8]


@pytest.mark.cpu_only
def test_video_token_count_is_one_bracket_per_temporal_patch():
    """``grid_t * (grid_h*grid_w/merge^2 + 2)`` for the processor's grid.

    ``find_mm_token_lengths`` passes the grid as a ``[1, 3]`` row; a 1-D
    tensor and a plain triple are accepted too. 10 frames at 224x224 give
    grid [5, 16, 16]: 5 brackets of 64 video tokens plus framing, 330.
    """
    proc = _make_minimax_m3_vl_fast_path_processor()
    assert (
        proc.get_num_tokens_per_video(video=[], video_grid_thw=torch.tensor([[5, 16, 16]])) == 330
    )
    assert proc.get_num_tokens_per_video(video=[], video_grid_thw=torch.tensor([5, 16, 16])) == 330
    assert proc.get_num_tokens_per_video(video=[], video_grid_thw=(5, 16, 16)) == 330
    assert proc.get_num_tokens_per_video(video=[], video_grid_thw=[[1, 4, 6]]) == 8
    with pytest.raises(RuntimeError, match="requires precomputed video_grid_thw"):
        proc.get_num_tokens_per_video(video=[])


@pytest.mark.cpu_only
def test_preferred_media_io_format_is_uint8_numpy():
    """The registered processor asks media_io for uint8 HWC arrays.

    The HF processors rescale by 1/255 themselves, so float tensors (the
    media_io default) would be rescaled twice; ``OpenAIServer`` seeds its
    media_io defaults from this method for ``BaseMultimodalInputProcessor``
    instances.
    """
    from tensorrt_llm.inputs.registry import BaseMultimodalInputProcessor

    proc = _make_minimax_m3_vl_fast_path_processor()
    assert isinstance(proc, BaseMultimodalInputProcessor)
    assert proc.get_preferred_media_io_kwargs() == {
        "image": {"format": "np"},
        "video": {"format": "np"},
    }


@pytest.mark.cpu_only
def test_fast_path_expand_rejects_mixed_image_and_video():
    """Single-modality only: the registry's ``_get_single_mm_token_lengths``
    returns only one modality's counts, so mixing image + video in one
    request must fail loudly rather than silently corrupting the prompt.
    """
    proc = _make_minimax_m3_vl_fast_path_processor()
    with pytest.raises(ValueError, match="mixed image \\+ video"):
        proc.expand_prompt_token_ids_for_mm(
            prompt_token_ids=[1, 2, 3],
            num_mm_tokens_per_placeholder=[3],
            mm_data={"image": [object()], "video": [object()]},
        )


@pytest.mark.cpu_only
def test_fast_path_expand_rejects_placeholder_count_mismatch():
    """If the prompt carries more (or fewer) placeholder tokens than
    ``num_mm_tokens_per_placeholder`` entries, we raise rather than
    producing a malformed prompt that fuse_input_embeds would reject
    deep inside the model.
    """
    proc = _make_minimax_m3_vl_fast_path_processor()
    img = _IMG_TOK

    # Two image placeholders in the prompt, but only one entry provided.
    with pytest.raises(ValueError, match="more placeholders than"):
        proc.expand_prompt_token_ids_for_mm(
            prompt_token_ids=[img, img],
            num_mm_tokens_per_placeholder=[5],
            mm_data={"image": [object(), object()]},
        )

    # One image placeholder in the prompt, but two entries provided.
    with pytest.raises(ValueError, match="prompt has 1 placeholders"):
        proc.expand_prompt_token_ids_for_mm(
            prompt_token_ids=[img],
            num_mm_tokens_per_placeholder=[5, 5],
            mm_data={"image": [object(), object()]},
        )


@pytest.mark.cpu_only
def test_token_id_request_takes_the_fast_path_without_detokenizing():
    """A ``prompt_token_ids`` + image request is expanded in token space.

    ``BaseMultimodalInputProcessor.__call__`` dispatches such a request to
    ``call_with_token_ids`` only when ``supports_token_id_mm_expansion`` is
    set; otherwise it detokenizes the ids and re-applies the chat template,
    which doubles the image marker and rejects the request. A pre-templated
    request from a front end such as Dynamo must therefore never reach the
    tokenizer's ``decode`` and must come back with the placeholder framed as
    ``VISION_START + IMAGE_TOKEN * N + VISION_END``.
    """
    from tensorrt_llm.sampling_params import SamplingParams

    proc = _make_minimax_m3_vl_fast_path_processor()

    class _NoDecodeTokenizer:
        def decode(self, *args, **kwargs):
            raise AssertionError("token-id request was detokenized")

    proc._tokenizer = _NoDecodeTokenizer()
    image = object()
    grid = torch.tensor([[1, 4, 4]])
    text_calls: List[Dict] = []

    def fake_call_with_text_prompt(inputs, sampling_params=None):
        text_calls.append(dict(inputs))
        return [], {"multimodal_data": {"image": {"image_grid_thw": grid}}}

    proc.call_with_text_prompt = fake_call_with_text_prompt
    # grid [1, 4, 4] at merge 2 -> 4 placeholders + the two framing tokens.
    proc.get_num_tokens_per_image = lambda *, image, **kwargs: 6

    ids, extra = proc(
        {"prompt_token_ids": [1, _IMG_TOK, 2], "multi_modal_data": {"image": [image]}},
        SamplingParams(),
    )

    assert ids == [1, _VS_TOK] + [_IMG_TOK] * 4 + [_VE_TOK, 2]
    assert torch.equal(extra["multimodal_data"]["image"]["image_grid_thw"], grid)
    # The vision tower ran once, on the dummy placeholder prompt with the
    # real image, not on a detokenized copy of the request.
    assert [c.get("prompt") for c in text_calls] == [""]
    assert text_calls[0]["multi_modal_data"] == {"image": [image]}


# ---------------------------------------------------------------------------
# Input contract without an image processor: text needs only the tokenizer;
# image requests load the HF processor on first use and reject a tokenizer.
# ---------------------------------------------------------------------------


class _WordTokenizer:
    """Tokenizer stand-in: one id per whitespace-separated word, BOS id 1."""

    unk_token_id = 0

    def __init__(self):
        self.template_calls = 0

    def convert_tokens_to_ids(self, token: str) -> int:
        return {"]<]start of image[>[": _VS_TOK, "]<]end of image[>[": _VE_TOK}.get(token, 0)

    def encode(self, text, add_special_tokens=True, truncation=False, max_length=None):
        ids = [100 + i for i, _ in enumerate(text.split())]
        if add_special_tokens:
            ids = [1] + ids
        return ids[:max_length] if truncation and max_length is not None else ids

    def apply_chat_template(
        self, messages, tokenize=False, add_generation_prompt=True, tools=None, **kwargs
    ):
        # Content is either the flattened text trtllm-serve renders (STRING
        # content format) or the content list the input processor builds.
        self.template_calls += 1
        content = messages[0]["content"]
        if isinstance(content, str):
            return "user: " + content + " assistant:"
        markers = {"image": "]<]image[>[", "video": "]<]video[>["}
        parts = [
            item["text"] if item["type"] == "text" else markers[item["type"]] for item in content
        ]
        return "user: " + " ".join(parts) + " assistant:"


class _ValidImageProcessor:
    """Processor stand-in with an image processor: one placeholder per item.

    Patch 14 / merge 2 / temporal 2 like the checkpoint; ``calls`` records
    the text of every call and ``call_kwargs`` the processor kwargs.
    """

    image_processor = SimpleNamespace(
        merge_size=2, get_number_of_image_patches=lambda h, w: (h // 14) * (w // 14)
    )
    video_processor = SimpleNamespace(merge_size=2, temporal_patch_size=2)

    def __init__(self):
        self.calls = []
        self.call_kwargs = []

    def __call__(self, text, images=None, videos=None, return_tensors=None, **kwargs):
        self.calls.append(text)
        self.call_kwargs.append(kwargs)
        num_images = len(images or [])
        num_videos = len(videos or [])
        input_ids = (
            [7]
            + [_VS_TOK, _IMG_TOK, _VE_TOK] * num_images
            + [_VS_TOK, _VID_TOK, _VE_TOK] * num_videos
        )
        out = {"input_ids": torch.tensor([input_ids])}
        if num_images:
            out["pixel_values"] = torch.ones(4 * num_images, 8)
            out["image_grid_thw"] = torch.tensor([[1, 2, 2]] * num_images)
        if num_videos:
            out["pixel_values_videos"] = torch.ones(4 * num_videos, 8)
            out["video_grid_thw"] = torch.tensor([[1, 2, 2]] * num_videos)
        return out


def _serve_style_prompt(tokenizer, text: str, media: Dict[str, list]) -> dict:
    """Build the request the way trtllm-serve does for a chat completion.

    Same steps as `parse_chat_messages_coroutines` + `apply_chat_template`
    (STRING content format) + the prompt assembly in `openai_server.py`:
    the registry supplies the placeholder per item, `apply_mm_placeholders`
    writes it into the message text, the tokenizer renders the chat
    template, and the rendered text is handed over with the media,
    `mm_item_order` and `chat_template_applied`.
    """
    from tensorrt_llm.inputs.utils import MultimodalDataTracker, apply_mm_placeholders

    tracker = MultimodalDataTracker("minimax_m3_vl")
    for modality, items in media.items():
        for item in items:
            tracker.add_data(modality, item)
    message = {"role": "user", "content": text, "media": [], "content_parts": []}
    apply_mm_placeholders("minimax_m3_vl", message, tracker.placeholder_counts(), tracker)
    rendered = tokenizer.apply_chat_template([message], tokenize=False, add_generation_prompt=True)
    mm_data, _ = tracker.retrieve_all_sync()
    return {
        "prompt": rendered,
        "multi_modal_data": mm_data,
        "mm_item_order": tracker.item_order(),
        "chat_template_applied": True,
    }


def _tokenizer_only_processor(monkeypatch, auto_processor_result):
    """Construct the input processor while recording AutoProcessor loads."""
    loads = []

    def fake_from_pretrained(*args, **kwargs):
        loads.append(args)
        return auto_processor_result

    monkeypatch.setattr(AutoProcessor, "from_pretrained", fake_from_pretrained)
    cls = get_minimax_m3_vl_input_processor_cls()
    config = SimpleNamespace(
        image_token_index=_IMG_TOK,
        video_token_index=_VID_TOK,
        vocab_size=200_064,
        text_config=SimpleNamespace(torch_dtype=torch.bfloat16),
    )
    proc = cls(model_path="/m3-model-dir", config=config, tokenizer=_WordTokenizer())
    return proc, loads


@pytest.mark.cpu_only
def test_text_request_needs_no_hf_processor(monkeypatch):
    """Text is tokenized without the HF processor; the first media request loads it once."""
    valid = _ValidImageProcessor()
    proc, loads = _tokenizer_only_processor(monkeypatch, valid)
    assert loads == []

    ids, extra = proc.call_with_text_prompt(
        {"prompt": "why is the sky blue"}, SimpleNamespace(add_special_tokens=True)
    )
    assert ids == [1, 100, 101, 102, 103, 104] and extra is None
    ids, _ = proc.call_with_text_prompt(
        {"prompt": "why is the sky blue"},
        SimpleNamespace(add_special_tokens=False, truncate_prompt_tokens=2),
    )
    assert ids == [100, 101]
    assert loads == []

    image = Image.new("RGB", (32, 32))
    for prompt in ("describe", "again"):
        ids, extra = proc.call_with_text_prompt(
            {"prompt": prompt, "multi_modal_data": {"image": [image]}}, SimpleNamespace()
        )
        assert ids == [7, _VS_TOK, _IMG_TOK, _VE_TOK]
        assert extra["multimodal_data"]["image"]["pixel_values"].dtype == torch.bfloat16
    assert len(loads) == 1 and proc.processor is valid


@pytest.mark.cpu_only
def test_tokenizer_from_auto_processor_is_not_image_support(monkeypatch):
    proc, loads = _tokenizer_only_processor(monkeypatch, _WordTokenizer())
    image = Image.new("RGB", (32, 32))
    with pytest.raises(ValueError, match="need an HF processor with an image processor") as error:
        proc.call_with_text_prompt(
            {"prompt": "describe", "multi_modal_data": {"image": [image]}}, SimpleNamespace()
        )
    assert "Text-only requests are unaffected" in str(error.value)
    assert "_WordTokenizer" in str(error.value)
    assert len(loads) == 1
    # Text keeps working after the rejected image request.
    ids, extra = proc.call_with_text_prompt({"prompt": "hello"}, SimpleNamespace())
    assert ids == [1, 100] and extra is None


@pytest.mark.cpu_only
def test_serve_rendered_image_prompt_is_used_verbatim(monkeypatch):
    """The prompt builder's rendered prompt reaches the processor as rendered.

    ``async_build_multimodal_prompt`` runs the steps trtllm-serve runs for a
    chat completion and marks the result ``chat_template_applied``; the
    registry placeholder is the bare marker, so the HF processor adds the only
    start/end framing.
    """
    from tensorrt_llm.inputs.utils import async_build_multimodal_prompt

    valid = _ValidImageProcessor()
    proc, _ = _tokenizer_only_processor(monkeypatch, valid)
    png = io.BytesIO()
    Image.new("RGB", (32, 32), "red").save(png, format="PNG")
    request = asyncio.run(
        async_build_multimodal_prompt(
            model_type="minimax_m3_vl",
            tokenizer=proc.tokenizer,
            processor=None,
            prompt="describe",
            media=["data:image/png;base64," + base64.b64encode(png.getvalue()).decode()],
        )
    )
    rendered = request["prompt"]
    assert rendered == "user: ]<]image[>[\ndescribe assistant:"
    assert request["chat_template_applied"] is True
    assert proc.tokenizer.template_calls == 1

    ids, extra = proc.call_with_text_prompt(request, SimpleNamespace())
    # Not rendered a second time; the processor sees the builder output as is.
    assert proc.tokenizer.template_calls == 1
    assert valid.calls == [[rendered]]
    assert ids == [7, _VS_TOK, _IMG_TOK, _VE_TOK]
    assert extra["multimodal_data"]["image"]["image_grid_thw"].tolist() == [[1, 2, 2]]


@pytest.mark.cpu_only
def test_serve_rendered_mixed_image_and_video_prompt(monkeypatch):
    valid = _ValidImageProcessor()
    proc, _ = _tokenizer_only_processor(monkeypatch, valid)
    video = SimpleNamespace(frames=[Image.new("RGB", (32, 32))] * 2)
    request = _serve_style_prompt(
        proc.tokenizer, "compare", {"image": [Image.new("RGB", (32, 32))], "video": [video]}
    )
    rendered = request["prompt"]
    assert rendered == "user: ]<]image[>[\n]<]video[>[\ncompare assistant:"

    ids, extra = proc.call_with_text_prompt(request, SimpleNamespace())
    assert proc.tokenizer.template_calls == 1
    assert valid.calls == [[rendered]]
    assert ids == [7, _VS_TOK, _IMG_TOK, _VE_TOK, _VS_TOK, _VID_TOK, _VE_TOK]
    assert extra["multimodal_data"]["image"]["image_grid_thw"].tolist() == [[1, 2, 2]]
    assert extra["multimodal_data"]["video"]["video_grid_thw"].tolist() == [[1, 2, 2]]


@pytest.mark.cpu_only
def test_framed_marker_is_reduced_to_the_bare_marker(monkeypatch):
    # A caller that frames the marker itself gets one frame, from the processor.
    valid = _ValidImageProcessor()
    proc, _ = _tokenizer_only_processor(monkeypatch, valid)
    framed = MINIMAX_M3_VL_VISION_START_TOKEN + "]<]image[>[" + MINIMAX_M3_VL_VISION_END_TOKEN
    ids, _ = proc.call_with_text_prompt(
        {
            "prompt": f"user: {framed}\ndescribe assistant:",
            "multi_modal_data": {"image": [Image.new("RGB", (32, 32))]},
            "chat_template_applied": True,
        },
        SimpleNamespace(),
    )
    assert valid.calls == [["user: ]<]image[>[\ndescribe assistant:"]]
    assert ids == [7, _VS_TOK, _IMG_TOK, _VE_TOK]


@pytest.mark.cpu_only
def test_marker_and_media_count_mismatch_is_rejected(monkeypatch):
    """More markers than media is a request error, before the HF processor runs.

    A rendered prompt is counted as written. A raw prompt gets one marker per
    item from the chat template, so marker text in it is one marker too many.
    """
    valid = _ValidImageProcessor()
    proc, _ = _tokenizer_only_processor(monkeypatch, valid)
    image = Image.new("RGB", (32, 32))
    mismatch = "2 ']<]image\\[>\\[' markers but the request carries 1"
    with pytest.raises(ValueError, match=mismatch):
        proc.call_with_text_prompt(
            {
                "prompt": "]<]image[>[ ]<]image[>[ compare",
                "multi_modal_data": {"image": [image]},
                "chat_template_applied": True,
            },
            SimpleNamespace(),
        )
    with pytest.raises(ValueError, match=mismatch):
        proc.call_with_text_prompt(
            {"prompt": "Explain ]<]image[>[", "multi_modal_data": {"image": [image]}},
            SimpleNamespace(),
        )
    assert proc.tokenizer.template_calls == 1
    assert valid.calls == []


@pytest.mark.cpu_only
def test_image_token_count_accepts_every_media_io_format(monkeypatch):
    """``get_num_tokens_per_image`` reads the size of PIL, CHW tensor and HWC array input.

    The hashing path (``find_mm_token_lengths``) hands over whatever
    media_io decoded; with the preferred "np" format that is a uint8 HWC
    array. 56x84 is a 4x6 patch grid: 24 patches / 4 + 2 framing = 8.
    """
    proc, _ = _tokenizer_only_processor(monkeypatch, _ValidImageProcessor())
    pil = Image.new("RGB", (84, 56))
    tensor = torch.zeros(3, 56, 84)
    array = np.zeros((56, 84, 3), dtype=np.uint8)
    assert [proc.get_num_tokens_per_image(image=m) for m in (pil, tensor, array)] == [8, 8, 8]


@pytest.mark.cpu_only
def test_video_request_lets_the_processor_resize_frames(monkeypatch):
    """Videos go to the HF processor with ``videos_kwargs={"do_resize": True}``.

    The processor's video default is ``do_resize=False`` and its patch
    view needs frame sides that are multiples of 28; the decoded frames
    are not pre-resized here. Image-only requests pass no such kwargs.
    """
    valid = _ValidImageProcessor()
    proc, _ = _tokenizer_only_processor(monkeypatch, valid)
    frames = [np.zeros((230, 230, 3), dtype=np.uint8)] * 3
    proc.call_with_text_prompt(
        {"prompt": "watch", "multi_modal_data": {"video": [SimpleNamespace(frames=frames)]}},
        SimpleNamespace(),
    )
    proc.call_with_text_prompt(
        {"prompt": "look", "multi_modal_data": {"image": [Image.new("RGB", (32, 32))]}},
        SimpleNamespace(),
    )
    assert valid.call_kwargs == [{"videos_kwargs": {"do_resize": True}}, {}]


class _MediaRecordingProcessor(_ValidImageProcessor):
    """``_ValidImageProcessor`` that records the media items it is handed."""

    def __init__(self):
        super().__init__()
        self.images = []
        self.videos = []

    def __call__(self, text, images=None, videos=None, return_tensors=None, **kwargs):
        self.images.append(images)
        self.videos.append(videos)
        return super().__call__(
            text, images=images, videos=videos, return_tensors=return_tensors, **kwargs
        )


@pytest.mark.cpu_only
def test_float_tensor_media_turn_off_the_processor_rescale(monkeypatch):
    """media_io's ``pt`` format (float [0, 1] CHW) goes in as is with ``do_rescale=False``.

    The checkpoint's processors rescale by 1/255 themselves, so a float
    [0, 1] input would be rescaled a second time and normalize to an almost
    black image. The kwarg is set per modality: uint8 arrays and PIL images
    keep the default rescale, and the video kwargs keep ``do_resize``.
    """
    from torchvision.transforms import ToTensor

    recorder = _MediaRecordingProcessor()
    proc, _ = _tokenizer_only_processor(monkeypatch, recorder)
    array = np.random.default_rng(0).integers(0, 256, size=(28, 42, 3), dtype=np.uint8)
    pil = Image.fromarray(array)
    as_pt = ToTensor()(pil)
    assert as_pt.dtype == torch.float32 and as_pt.shape == (3, 28, 42)

    proc.call_with_text_prompt(
        {"prompt": "look", "multi_modal_data": {"image": [as_pt]}}, SimpleNamespace()
    )
    assert recorder.images[0][0] is as_pt
    assert recorder.call_kwargs[0] == {"images_kwargs": {"do_rescale": False}}

    proc.call_with_text_prompt(
        {"prompt": "look", "multi_modal_data": {"image": [array, pil]}}, SimpleNamespace()
    )
    assert recorder.images[1][0] is array and recorder.images[1][1] is pil
    assert recorder.call_kwargs[1] == {}

    frames = [ToTensor()(Image.fromarray(array)) for _ in range(2)]
    proc.call_with_text_prompt(
        {"prompt": "watch", "multi_modal_data": {"video": [SimpleNamespace(frames=frames)]}},
        SimpleNamespace(),
    )
    (video_frames,) = recorder.videos[2]
    assert len(video_frames) == 2 and all(a is b for a, b in zip(video_frames, frames))
    assert recorder.call_kwargs[2] == {"videos_kwargs": {"do_resize": True, "do_rescale": False}}

    proc.call_with_text_prompt(
        {"prompt": "watch", "multi_modal_data": {"video": [SimpleNamespace(frames=[array] * 2)]}},
        SimpleNamespace(),
    )
    assert recorder.call_kwargs[3] == {"videos_kwargs": {"do_resize": True}}

    proc.call_with_text_prompt(
        {
            "prompt": "look watch",
            "multi_modal_data": {"image": [pil], "video": [SimpleNamespace(frames=frames)]},
        },
        SimpleNamespace(),
    )
    assert recorder.call_kwargs[4] == {"videos_kwargs": {"do_resize": True, "do_rescale": False}}


@pytest.mark.cpu_only
def test_float_media_outside_the_pt_contract_is_rejected(monkeypatch):
    """Only ``[C, H, W]`` float tensors with finite values in [0, 1] are accepted.

    An HWC float image, a float tensor holding [0, 255] values or one holding
    NaNs would be normalized silently into a garbage image, so all of them
    are rejected, for images and for video frames alike.
    """
    proc, _ = _tokenizer_only_processor(monkeypatch, _MediaRecordingProcessor())
    hwc = torch.rand(28, 42, 3)
    with pytest.raises(ValueError, match=r"\[C, H, W\] tensor with 1 or 3 channels"):
        proc.call_with_text_prompt(
            {"prompt": "look", "multi_modal_data": {"image": [hwc]}}, SimpleNamespace()
        )
    unscaled = torch.rand(3, 28, 42) * 255.0
    with pytest.raises(ValueError, match=r"scaled to \[0, 1\]"):
        proc.call_with_text_prompt(
            {"prompt": "look", "multi_modal_data": {"image": [unscaled]}}, SimpleNamespace()
        )
    with_nan = torch.rand(3, 28, 42)
    with_nan[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match=r"finite values"):
        proc.call_with_text_prompt(
            {"prompt": "look", "multi_modal_data": {"image": [with_nan]}}, SimpleNamespace()
        )
    with pytest.raises(ValueError, match=r"\[C, H, W\] tensor"):
        proc.call_with_text_prompt(
            {"prompt": "watch", "multi_modal_data": {"video": [[torch.rand(3, 3, 28, 42)]]}},
            SimpleNamespace(),
        )
    with pytest.raises(ValueError, match=r"scaled to \[0, 1\]"):
        proc.call_with_text_prompt(
            {"prompt": "watch", "multi_modal_data": {"video": [[unscaled]]}}, SimpleNamespace()
        )


@pytest.mark.cpu_only
def test_mixed_format_media_fall_back_to_uint8(monkeypatch):
    """A request mixing float and uint8 items in one modality is served by one call.

    The processor takes one ``do_rescale`` per modality, so the float items
    are converted back to the uint8 HWC arrays they came from (``ToTensor``
    divides by 255, so the conversion is exact) and the default rescale
    serves every item; the uint8 and PIL items pass through untouched.
    """
    from torchvision.transforms import ToTensor

    recorder = _MediaRecordingProcessor()
    proc, _ = _tokenizer_only_processor(monkeypatch, recorder)
    array = np.random.default_rng(1).integers(0, 256, size=(28, 42, 3), dtype=np.uint8)
    pil = Image.fromarray(array)
    as_pt = ToTensor()(pil)

    proc.call_with_text_prompt(
        {"prompt": "look", "multi_modal_data": {"image": [as_pt, array, pil]}},
        SimpleNamespace(),
    )
    converted, passed_array, passed_pil = recorder.images[0]
    assert isinstance(converted, np.ndarray) and converted.dtype == np.uint8
    np.testing.assert_array_equal(converted, array)
    assert passed_array is array and passed_pil is pil
    assert recorder.call_kwargs[0] == {}

    proc.call_with_text_prompt(
        {
            "prompt": "watch watch",
            "multi_modal_data": {
                "video": [SimpleNamespace(frames=[as_pt, as_pt]), SimpleNamespace(frames=[array])]
            },
        },
        SimpleNamespace(),
    )
    first, second = recorder.videos[1]
    assert len(first) == 2 and len(second) == 1 and second[0] is array
    for frame in first:
        assert isinstance(frame, np.ndarray) and frame.dtype == np.uint8
        np.testing.assert_array_equal(frame, array)
    assert recorder.call_kwargs[1] == {"videos_kwargs": {"do_resize": True}}


@pytest.mark.cpu_only
def test_video_timing_is_forwarded_to_the_processor(monkeypatch):
    """``VideoData.metadata`` reaches the HF processor as ``video_metadata``.

    With the source fps and the sampled frame indices, the processor writes a
    timestamp before each temporal group. Only the keys transformers'
    ``VideoMetadata`` accepts are passed, and ``total_num_frames`` counts the
    delivered frames. A video without timing in the same request gets
    ``fps=None``, so only the timed video gets timestamps.
    """
    recorder = _MediaRecordingProcessor()
    proc, _ = _tokenizer_only_processor(monkeypatch, recorder)
    frames = [np.zeros((28, 42, 3), dtype=np.uint8)] * 4
    timed = SimpleNamespace(
        frames=frames,
        metadata={
            "total_num_frames": 40,
            "fps": 10,
            "duration": 4.0,
            "frames_indices": [0, 13, 26, 39],
            "unrelated": "dropped",
        },
    )
    expected = {
        "total_num_frames": 4,
        "fps": 10.0,
        "duration": 4.0,
        "frames_indices": [0, 13, 26, 39],
    }

    proc.call_with_text_prompt(
        {"prompt": "watch", "multi_modal_data": {"video": [timed]}}, SimpleNamespace()
    )
    assert recorder.call_kwargs[0] == {
        "videos_kwargs": {"do_resize": True, "video_metadata": [expected]}
    }

    proc.call_with_text_prompt(
        {
            "prompt": "watch watch",
            "multi_modal_data": {"video": [timed, SimpleNamespace(frames=frames[:2])]},
        },
        SimpleNamespace(),
    )
    assert recorder.call_kwargs[1]["videos_kwargs"]["video_metadata"] == [
        expected,
        {"total_num_frames": 2, "fps": None, "duration": None, "frames_indices": [0, 1]},
    ]

    video_utils = pytest.importorskip("transformers.video_utils")
    assert video_utils.VideoMetadata(**expected).timestamps == pytest.approx([0.0, 1.3, 2.6, 3.9])


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "frames_indices",
    [
        torch.tensor([0, 13, 26, 39]),
        np.array([0, 13, 26, 39]),
        list(torch.tensor([0, 13, 26, 39])),
    ],
    ids=["tensor", "array", "list_of_tensors"],
)
def test_video_timing_accepts_index_tensors(monkeypatch, frames_indices):
    """Frame indices given as a tensor, an array or a list of 0-d tensors keep the video's timing, as plain ints."""
    recorder = _MediaRecordingProcessor()
    proc, _ = _tokenizer_only_processor(monkeypatch, recorder)
    frames = [np.zeros((28, 42, 3), dtype=np.uint8)] * 4
    video = SimpleNamespace(frames=frames, metadata={"fps": 10, "frames_indices": frames_indices})

    proc.call_with_text_prompt(
        {"prompt": "watch", "multi_modal_data": {"video": [video]}}, SimpleNamespace()
    )
    metadata = recorder.call_kwargs[0]["videos_kwargs"]["video_metadata"]
    assert metadata == [
        {"total_num_frames": 4, "fps": 10.0, "duration": None, "frames_indices": [0, 13, 26, 39]}
    ]
    assert all(type(i) is int for i in metadata[0]["frames_indices"])


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "metadata",
    [
        "frame_list",
        None,
        {"fps": 0, "frames_indices": [0, 1]},
        {"fps": float("nan"), "frames_indices": [0, 1]},
        {"fps": float("inf"), "frames_indices": [0, 1]},
        {"fps": "10", "frames_indices": [0, 1]},
        {"fps": 10.0},
        {"fps": 10.0, "frames_indices": []},
        {"fps": 10.0, "frames_indices": [0]},
        {"fps": 10.0, "frames_indices": [0, 5, 9]},
        {"fps": 10.0, "frames_indices": [0, None]},
        {"fps": 10.0, "frames_indices": [-10, 1]},
        {"fps": 10.0, "frames_indices": 2},
        {"fps": 10.0, "frames_indices": torch.tensor([0.0, 1.0])},
    ],
    ids=[
        "frame_list",
        "no_metadata",
        "zero_fps",
        "nan_fps",
        "inf_fps",
        "string_fps",
        "no_frame_indices",
        "empty_frame_indices",
        "fewer_frame_indices_than_frames",
        "more_frame_indices_than_frames",
        "none_frame_index",
        "negative_frame_index",
        "scalar_frame_indices",
        "float_tensor_frame_indices",
    ],
)
def test_video_without_timing_passes_no_metadata(monkeypatch, metadata):
    """Without usable timing the processor gets no ``video_metadata`` and adds no timestamps.

    Usable timing is a finite, positive ``fps`` and one frame index per
    delivered frame: a timestamp from a NaN or infinite rate, or from indices
    that do not line up with the frames, would be wrong.
    """
    recorder = _MediaRecordingProcessor()
    proc, _ = _tokenizer_only_processor(monkeypatch, recorder)
    frames = [np.zeros((28, 42, 3), dtype=np.uint8)] * 2
    if metadata == "frame_list":
        video = frames
    elif metadata is None:
        video = SimpleNamespace(frames=frames)
    else:
        video = SimpleNamespace(frames=frames, metadata=metadata)

    proc.call_with_text_prompt(
        {"prompt": "watch", "multi_modal_data": {"video": [video]}}, SimpleNamespace()
    )
    assert recorder.call_kwargs[0] == {"videos_kwargs": {"do_resize": True}}
