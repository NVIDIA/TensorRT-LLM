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
"""Key spelling of the Muse Spark Open export, mapped back to the source names.

The open-export checkpoint (`model_type: muse_spark`) stores the identical
tensors under HF-conventional names: the decoder under
`model.language_model.`, the towers under `model.vision_tower.` /
`model.audio_tower.`, and per-module renames (`feed_forward` -> `mlp`,
`add_attn` -> `post_attention_residual_gate`, ...). The loader in
`musespark_open_weight_mapper.py` walks the source spelling; this module is the
total, injective map from every open-export key back to that spelling, so one
in-place rename up front makes the open checkpoint indistinguishable from the
source export to everything downstream.
"""

import re
from typing import Dict, Iterable, List

#: Present exactly when a checkpoint uses the open-export spelling: the source
#: export stores the embedding at `model.embed_tokens.weight`.
OPEN_FORMAT_SENTINEL_KEY = "model.language_model.embed_tokens.weight"

_DECODER_LAYER_KEY = re.compile(r"^model\.language_model\.layers\.(\d+)\.(.+)$")
_VISION_LAYER_KEY = re.compile(r"^model\.vision_tower\.layers\.(\d+)\.(.+)$")
_AUDIO_STAGE_KEY = re.compile(r"^model\.audio_tower\.layers\.(\d+)\.(.+)$")
_AUDIO_CONV_BLOCK = re.compile(r"^conv_blocks\.(\d+)\.(.+)$")

# Per-decoder-layer tails, open spelling -> source spelling. The five
# `self_attn.{q,k,v,o,o_gate}` projections keep their names apart from
# `gate_proj` -> `o_gate_proj`; everything under `mlp.` moves to
# `feed_forward.` with the router/pooler renames; the stacked expert tensors
# recover their `w13`/`w2` names (same packing: gate rows then up rows,
# per-half scale columns).
_DECODER_TAILS = {
    "input_layernorm.weight": "self_attn.pre_qkv_proj_norm.weight",
    "self_attn.q_proj.weight": "self_attn.q_proj.weight",
    "self_attn.k_proj.weight": "self_attn.k_proj.weight",
    "self_attn.v_proj.weight": "self_attn.v_proj.weight",
    "self_attn.o_proj.weight": "self_attn.o_proj.weight",
    "self_attn.gate_proj.weight": "self_attn.o_gate_proj.weight",
    "post_attention_residual_gate.gate": "add_attn.gate",
    "post_feedforward_residual_gate.gate": "add_ffn.gate",
    "pre_feedforward_layernorm.weight": "pre_ffn_norm.weight",
    "post_feedforward_layernorm.weight": "post_ffn_norm.weight",
    "mlp.gate.weight": "feed_forward.router.weight",
    "mlp.gate.e_score_correction_bias": "feed_forward.expert_bias",
    "mlp.pre_expert_proj.weight": "feed_forward.pooler.weight",
    "mlp.post_expert_proj.weight": "feed_forward.unpooler.weight",
    "mlp.pre_expert_norm.weight": "feed_forward.pre_expert_norm.weight",
    # The open export nests the post-expert norm under `experts`; the source
    # keeps it beside the pre-expert norm.
    "mlp.experts.post_expert_norm.weight": "feed_forward.post_expert_norm.weight",
    "mlp.experts.gate_up_proj": "feed_forward.experts.w13_weight",
    "mlp.experts.down_proj": "feed_forward.experts.w2_weight",
    "mlp.experts.gate_up_proj_weight_scale": "feed_forward.experts.w13_weight_scale",
    "mlp.experts.gate_up_proj_weight_scale_2": "feed_forward.experts.w13_weight_scale_2",
    "mlp.experts.gate_up_proj_input_scale": "feed_forward.experts.w13_input_scale",
    "mlp.experts.down_proj_weight_scale": "feed_forward.experts.w2_weight_scale",
    "mlp.experts.down_proj_weight_scale_2": "feed_forward.experts.w2_weight_scale_2",
    "mlp.experts.down_proj_input_scale": "feed_forward.experts.w2_input_scale",
}

# Per-vision-layer tails. Same LayerNorm/MLP/attention tensors, HF names.
_VISION_TAILS = {
    "norm1.weight": "ln_1.weight",
    "norm1.bias": "ln_1.bias",
    "norm2.weight": "ln_2.weight",
    "norm2.bias": "ln_2.bias",
    "attn.q_proj.weight": "attn.q_proj.weight",
    "attn.q_proj.bias": "attn.q_proj.bias",
    "attn.k_proj.weight": "attn.k_proj.weight",
    "attn.k_proj.bias": "attn.k_proj.bias",
    "attn.v_proj.weight": "attn.v_proj.weight",
    "attn.v_proj.bias": "attn.v_proj.bias",
    "attn.proj.weight": "attn.out_proj.weight",
    "attn.proj.bias": "attn.out_proj.bias",
    "mlp.fc1.weight": "mlp.c_fc.weight",
    "mlp.fc1.bias": "mlp.c_fc.bias",
    "mlp.fc2.weight": "mlp.c_proj.weight",
    "mlp.fc2.bias": "mlp.c_proj.bias",
}

# Audio conv-block tails, shared by both blocks of a stage. The stage's
# `nn.Sequential` puts the patchifier at index 0, so open-export block `m`
# is source sub-module `m + 1`.
_AUDIO_BLOCK_TAILS = {
    "conv.weight": "conv.conv.weight",
    "conv.bias": "conv.conv.bias",
    "mlp.fc1.weight": "linear1.weight",
    "mlp.fc1.bias": "linear1.bias",
    "mlp.fc2.weight": "linear2.weight",
    "mlp.fc2.bias": "linear2.bias",
    "norm.weight": "norm.weight",
}

_AUDIO_PATCH_TAILS = {
    "patch_embedder.norm.weight": "0.norm.weight",
    "patch_embedder.proj.weight": "0.linear.weight",
    "patch_embedder.proj.bias": "0.linear.bias",
}

# Whole-key renames outside the three layer stacks.
_EXACT_KEYS = {
    "lm_head.weight": "lm_head.weight",
    "model.language_model.embed_tokens.weight": "model.embed_tokens.weight",
    "model.language_model.norm.weight": "model.norm.weight",
    # The patch embedding is the tower's pixel-unfold linear; the open export
    # names it after HF convention, the source after the CLIP lineage.
    "model.vision_tower.patch_embedder.patch_embedding.weight": (
        "vision_model.conv1_linear.weight"
    ),
    "model.vision_tower.patch_embedder.position_embedding_table.weight": (
        "vision_model.positional_embedding_vlm"
    ),
    "model.vision_tower.ln_pre.weight": "vision_model.ln_pre.weight",
    "model.vision_tower.ln_pre.bias": "vision_model.ln_pre.bias",
    "model.vision_tower.ln_post.weight": "vision_model.ln_post.weight",
    "model.vision_tower.ln_post.bias": "vision_model.ln_post.bias",
    "model.vision_adapter.fc1.weight": "vision_model.adapter.c_fc.weight",
    "model.vision_adapter.fc2.weight": "vision_model.adapter.c_proj.weight",
    # The bare-alias spelling the loader already accepts (the wrapped
    # `modalities.` spelling names a subtree the open export dropped).
    "model.vision_projection.weight": "vision_projection.weight",
    "model.audio_tower.norm.weight": (
        "modalities.soft_audio_perception.interaction_encoder.norm.weight"
    ),
    "model.audio_tower.post_proj_norm.weight": (
        "modalities.soft_audio_perception.interaction_encoder.dual_norm.weight"
    ),
    "model.audio_tower.proj.weight": (
        "modalities.soft_audio_perception.interaction_encoder.o_proj.weight"
    ),
    "model.audio_tower.proj.bias": (
        "modalities.soft_audio_perception.interaction_encoder.o_proj.bias"
    ),
}

_AUDIO_STAGE_PREFIX = "modalities.soft_audio_perception.interaction_encoder.stages"


def is_open_format(keys: Iterable[str]) -> bool:
    """Whether a key set uses the open-export spelling."""
    return OPEN_FORMAT_SENTINEL_KEY in set(keys)


def open_format_to_source_key(key: str) -> str:
    """Map one open-export key to the source-export spelling.

    Raises on any key with no mapping: an unknown tensor in a re-export must
    fail loudly rather than be skipped or half-loaded.
    """
    exact = _EXACT_KEYS.get(key)
    if exact is not None:
        return exact

    match = _DECODER_LAYER_KEY.match(key)
    if match is not None:
        tail = _DECODER_TAILS.get(match.group(2))
        if tail is not None:
            return f"model.layers.{match.group(1)}.{tail}"

    match = _VISION_LAYER_KEY.match(key)
    if match is not None:
        tail = _VISION_TAILS.get(match.group(2))
        if tail is not None:
            return f"vision_model.encoder.layers.{match.group(1)}.{tail}"

    match = _AUDIO_STAGE_KEY.match(key)
    if match is not None:
        stage, rest = match.group(1), match.group(2)
        patch_tail = _AUDIO_PATCH_TAILS.get(rest)
        if patch_tail is not None:
            return f"{_AUDIO_STAGE_PREFIX}.{stage}.{patch_tail}"
        block = _AUDIO_CONV_BLOCK.match(rest)
        if block is not None:
            block_tail = _AUDIO_BLOCK_TAILS.get(block.group(2))
            if block_tail is not None:
                return f"{_AUDIO_STAGE_PREFIX}.{stage}.{int(block.group(1)) + 1}.{block_tail}"

    raise ValueError(
        f"Muse Spark Open format: no source-spelling mapping for checkpoint key {key!r}. "
        "A re-export grew or renamed a tensor; map it in "
        "musespark_open_format.py rather than letting it be skipped."
    )


def build_open_format_rename(keys: Iterable[str]) -> Dict[str, str]:
    """Total, injective open-key -> source-key map for one checkpoint.

    Raises when two open keys land on one source key (the loader would silently
    read one tensor twice and drop the other) and when any key is unmapped.
    """
    rename: Dict[str, str] = {}
    claimed: Dict[str, str] = {}
    collisions: List[str] = []
    for key in keys:
        source = open_format_to_source_key(key)
        if source in claimed:
            collisions.append(f"{claimed[source]!r} and {key!r} -> {source!r}")
        claimed[source] = key
        rename[key] = source
    if collisions:
        raise ValueError(
            "Muse Spark Open format: key mapping is not injective: " + "; ".join(collisions)
        )
    return rename
