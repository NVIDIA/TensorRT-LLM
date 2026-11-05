# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The vision weight walk into a built (tiny) tower.

`Linear.load_weights` places weights on the GPU, so this check needs a
device; the key accounting it builds on runs on CPU in
`test_musespark_open_omni_weights.py`.
"""

import pytest
import torch

from tensorrt_llm._torch.models.checkpoints.hf.musespark_open_weight_mapper import (
    CANONICAL_VISION_PREFIX,
    VISION_PROJECTION_KEY,
)


class _FakeConsumableWeights(dict):
    """The two release calls the loader makes, with the real semantics.

    `mark_consumed(prefix)` deletes `prefix + "."`, which silently deletes
    nothing for a bare tensor -- the trap the loader has to avoid for the
    positional embedding and the projection.
    """

    def mark_consumed(self, prefix: str) -> int:
        doomed = [k for k in self if k.startswith(prefix + ".")]
        for key in doomed:
            del self[key]
        return len(doomed)

    def mark_consumed_keys(self, keys) -> int:
        deleted = 0
        for key in keys:
            if key in self:
                del self[key]
                deleted += 1
        return deleted


def _synthetic_tower_weights(cfg, decoder_hidden, pos_embed_shape):
    """A checkpoint for the tiny tower, keyed the way the real one is."""
    import torch

    hidden, inter = cfg.hidden_size, cfg.intermediate_size
    patch_dim = cfg.num_channels * cfg.patch_temporal * cfg.patch_size**2
    weights = _FakeConsumableWeights()

    def put(name, *shape):
        weights[f"{CANONICAL_VISION_PREFIX}{name}"] = (
            torch.arange(int(torch.tensor(shape).prod()), dtype=torch.float32).reshape(*shape)
            * 1e-4
        )

    put("conv1_linear.weight", hidden, patch_dim)
    put("positional_embedding_vlm", *pos_embed_shape)
    for norm in ("ln_pre", "ln_post"):
        put(f"{norm}.weight", hidden)
        put(f"{norm}.bias", hidden)
    put("adapter.c_fc.weight", cfg.projector_input_dim, cfg.vision_output_dim)
    put("adapter.c_proj.weight", cfg.projector_input_dim, cfg.projector_input_dim)
    for i in range(cfg.num_hidden_layers):
        p = f"encoder.layers.{i}"
        for norm in ("ln_1", "ln_2"):
            put(f"{p}.{norm}.weight", hidden)
            put(f"{p}.{norm}.bias", hidden)
        for name in ("q", "k", "v", "out"):
            put(f"{p}.attn.{name}_proj.weight", hidden, hidden)
            put(f"{p}.attn.{name}_proj.bias", hidden)
        put(f"{p}.mlp.c_fc.weight", inter, hidden)
        put(f"{p}.mlp.c_fc.bias", inter)
        put(f"{p}.mlp.c_proj.weight", hidden, inter)
        put(f"{p}.mlp.c_proj.bias", hidden)
    weights[VISION_PROJECTION_KEY] = (
        torch.arange(decoder_hidden * cfg.projector_input_dim, dtype=torch.float32).reshape(
            decoder_hidden, cfg.projector_input_dim
        )
        * 1e-4
    )
    return weights


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Linear weight loading needs a GPU")
def test_a_real_tiny_load_leaves_nothing_unaccounted():
    """End to end over the seam: build, load, assert closure and values."""
    from test_modeling_musespark_open_vision import _tiny_model_config  # noqa: PLC0415
    from test_musespark_open_omni_weights import _build_tiny_tower  # noqa: PLC0415

    from tensorrt_llm._torch.models.checkpoints.hf.musespark_open_weight_mapper import (
        WeightAccounting,
        load_musespark_open_vision_weights,
    )

    model_config = _tiny_model_config()
    tower = _build_tiny_tower(model_config)
    cfg = model_config.pretrained_config.vision_config
    weights = _synthetic_tower_weights(
        cfg,
        model_config.pretrained_config.hidden_size,
        tuple(tower.pos_embed.shape),
    )
    expected_total = len(weights)

    accounting = WeightAccounting()
    load_musespark_open_vision_weights(tower, weights, accounting)

    # Every synthesized tensor was consumed and released.
    assert len(accounting.consumed) == expected_total
    assert not weights, f"{len(weights)} tensors were never released: {sorted(weights)[:5]}"

    # And the values actually landed: the positional embedding is the bare
    # tensor whose release call is the easiest one to get silently wrong.
    assert tower.pos_embed.abs().sum() > 0
