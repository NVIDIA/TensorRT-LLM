# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

from tensorrt_llm._torch.models.checkpoints.hf.cosmos3_edge_weight_mapper import (
    Cosmos3EdgeHfWeightMapper,
)
from tensorrt_llm._torch.models.modeling_cosmos3_edge import (
    merge_spatial_patches,
    resize_positional_embeddings_packed,
)
from tensorrt_llm._torch.models.modeling_qwen3vl import Qwen3VLVisionPatchMerger


def test_edge_mapper_drops_generator_weights_and_renames_reasoner_keys():
    mapper = Cosmos3EdgeHfWeightMapper()
    weights = {
        "layers.0.self_attn.to_q.weight": torch.empty(1),
        "layers.0.self_attn.to_k.weight": torch.empty(1),
        "layers.0.self_attn.to_v.weight": torch.empty(1),
        "layers.0.self_attn.to_out.weight": torch.empty(1),
        "layers.0.self_attn.k_norm_und_for_gen.weight": torch.empty(1),
        "layers.0.self_attn.add_q_proj.weight": torch.empty(1),
        "layers.0.mlp_moe_gen.up_proj.weight": torch.empty(1),
        "layers.0.mlp.up_proj.weight": torch.empty(1),
        "embed_tokens.weight": torch.empty(1),
        "norm.weight": torch.empty(1),
        "norm_moe_gen.weight": torch.empty(1),
        "lm_head.weight": torch.empty(1),
        "model.visual.encoder.layers.0.self_attn.out_proj.weight": torch.empty(1),
        "model.visual.encoder.layers.0.mlp.fc1.weight": torch.empty(1),
        "model.visual.encoder.layers.0.mlp.fc2.weight": torch.empty(1),
        "model.projector.linear_fc1.weight": torch.empty(1),
        "time_embedder.linear_1.weight": torch.empty(1),
    }
    mapped = mapper.preprocess_weights(weights)
    assert "model.layers.0.self_attn.q_proj.weight" in mapped
    assert "model.layers.0.self_attn.k_proj.weight" in mapped
    assert "model.layers.0.self_attn.v_proj.weight" in mapped
    assert "model.layers.0.self_attn.o_proj.weight" in mapped
    assert "model.layers.0.mlp.up_proj.weight" in mapped
    assert "model.embed_tokens.weight" in mapped
    assert "model.norm.weight" in mapped
    assert "lm_head.weight" in mapped
    assert "model.visual.encoder.layers.0.self_attn.o_proj.weight" in mapped
    assert "model.visual.encoder.layers.0.mlp.up_proj.weight" in mapped
    assert "model.visual.encoder.layers.0.mlp.down_proj.weight" in mapped
    assert mapped["model.projector.linear_fc1.weight"] is weights["model.projector.linear_fc1.weight"]
    dropped = [
        "layers.0.self_attn.k_norm_und_for_gen.weight",
        "layers.0.self_attn.add_q_proj.weight",
        "layers.0.mlp_moe_gen.up_proj.weight",
        "norm_moe_gen.weight",
        "time_embedder.linear_1.weight",
        "model.visual.encoder.layers.0.mlp.fc1.weight",
        "model.visual.encoder.layers.0.self_attn.out_proj.weight",
    ]
    assert all(key not in mapped for key in dropped)


def test_siglip_position_interpolation_keeps_packed_token_count():
    grid = 4
    hidden = 8
    positional = torch.arange(grid * grid * hidden, dtype=torch.float32).reshape(grid, grid, hidden)
    shapes = torch.tensor([[2, 2], [8, 4]], dtype=torch.int64)
    lengths = [4, 32]
    packed = resize_positional_embeddings_packed(positional, shapes, lengths)
    assert packed.shape == (36, hidden)
    assert packed.dtype == positional.dtype


def test_spatial_merge_packs_two_by_two_blocks():
    frames, height, width, hidden = 2, 4, 4, 3
    embeds = torch.arange(frames * height * width * hidden, dtype=torch.float32).reshape(
        frames * height * width, hidden
    )
    grid = torch.tensor([[frames, height, width]], dtype=torch.int64)
    merged = merge_spatial_patches(embeds, grid, merge_size=2)
    assert merged.shape == (frames * (height // 2) * (width // 2), 2 * 2 * hidden)


def test_patch_merger_intermediate_width_is_optional():
    assert Qwen3VLVisionPatchMerger.__init__.__defaults__ is not None
    # (use_postshuffle_norm, merger_intermediate_size, norm_eps)
    assert Qwen3VLVisionPatchMerger.__init__.__defaults__[-2:] == (None, None)
