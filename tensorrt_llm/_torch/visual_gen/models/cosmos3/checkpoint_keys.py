# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint-key remap for Cosmos3 diffusion transformers.

Kept free of torch imports so the Edge-FP8 tensor manifest can be checked
without constructing the model or a GPU.
"""

from typing import Literal, Tuple

CheckpointKeyStatus = Literal["quantizer", "skip", "remap", "unmatched"]

# Source prefix and destination template. ``{layer_idx}`` comes from the key.
_LAYER_KEY_TEMPLATES: Tuple[Tuple[str, str], ...] = (
    ("self_attn.add_q_proj.", "gen_layers.{layer_idx}.cross_attention.to_q."),
    ("self_attn.add_k_proj.", "gen_layers.{layer_idx}.cross_attention.to_k."),
    ("self_attn.add_v_proj.", "gen_layers.{layer_idx}.cross_attention.to_v."),
    ("self_attn.to_add_out.", "gen_layers.{layer_idx}.cross_attention.to_out.0."),
    ("self_attn.norm_added_q.", "gen_layers.{layer_idx}.cross_attention.norm_q."),
    ("self_attn.norm_added_k.", "gen_layers.{layer_idx}.cross_attention.norm_k."),
    ("self_attn.to_q.", "language_model.layers.{layer_idx}.self_attn.to_q."),
    ("self_attn.to_k.", "language_model.layers.{layer_idx}.self_attn.to_k."),
    ("self_attn.to_v.", "language_model.layers.{layer_idx}.self_attn.to_v."),
    ("self_attn.to_out.", "language_model.layers.{layer_idx}.self_attn.to_out.0."),
    ("self_attn.norm_q.", "language_model.layers.{layer_idx}.self_attn.norm_q."),
    ("self_attn.norm_k.", "language_model.layers.{layer_idx}.self_attn.norm_k."),
    (
        "self_attn.k_norm_und_for_gen.",
        "language_model.layers.{layer_idx}.self_attn.k_norm_und_for_gen.",
    ),
    ("input_layernorm.", "language_model.layers.{layer_idx}.input_layernorm."),
    (
        "post_attention_layernorm.",
        "language_model.layers.{layer_idx}.post_attention_layernorm.",
    ),
    ("input_layernorm_moe_gen.", "gen_layers.{layer_idx}.input_layernorm."),
    (
        "post_attention_layernorm_moe_gen.",
        "gen_layers.{layer_idx}.post_attention_layernorm.",
    ),
    ("mlp.gate_proj.", "language_model.layers.{layer_idx}.mlp.gate_proj."),
    ("mlp.up_proj.", "language_model.layers.{layer_idx}.mlp.up_proj."),
    ("mlp.down_proj.", "language_model.layers.{layer_idx}.mlp.down_proj."),
    ("mlp_moe_gen.gate_proj.", "gen_layers.{layer_idx}.mlp.gate_proj."),
    ("mlp_moe_gen.up_proj.", "gen_layers.{layer_idx}.mlp.up_proj."),
    ("mlp_moe_gen.down_proj.", "gen_layers.{layer_idx}.mlp.down_proj."),
)


def remap_cosmos3_checkpoint_key(key: str) -> Tuple[CheckpointKeyStatus, str]:
    """Classify one diffusion-transformer checkpoint tensor.

    The second value is the normalized checkpoint key, except for ``"remap"``,
    where it is the translated name. ``"remap"`` means the name matches a
    known pattern. This helper does not check that the translated name is a
    parameter of the constructed module; the loader does.

    ``"quantizer"`` is a ModelOpt fake-quant buffer (``*_quantizer._amax`` /
    ``._scale``). The loader consumes ``weight``, ``weight_scale``, and
    ``input_scale`` only. ``"skip"`` is an intentionally unused model tensor.
    ``"unmatched"`` is a name outside this topology.
    """
    normalized = key[len("model.") :] if key.startswith("model.") else key
    if "quantizer" in normalized:
        return "quantizer", normalized
    if normalized.startswith(("lm_head.", "action_pos_embed.", "norm.")):
        return "skip", normalized

    for source, destination in (
        ("proj_in.", "vae2llm."),
        ("proj_out.", "llm2vae."),
        ("audio_proj_in.", "audio2llm."),
        ("audio_proj_out.", "llm2audio."),
    ):
        if normalized.startswith(source):
            return "remap", destination + normalized[len(source) :]

    if normalized.startswith(
        (
            "audio_modality_embed",
            "action_modality_embed",
            "action_proj_in.",
            "action_proj_out.",
            "norm_moe_gen.",
        )
    ):
        return "remap", normalized

    if normalized.startswith("time_embedder.linear"):
        mapped = normalized.replace("time_embedder.linear_1.", "time_embedder.mlp.linear_1.")
        mapped = mapped.replace("time_embedder.linear_2.", "time_embedder.mlp.linear_2.")
        return "remap", mapped

    if normalized.startswith("embed_tokens."):
        return "remap", f"language_model.{normalized}"

    if not normalized.startswith("layers."):
        return "unmatched", normalized

    parts = normalized.split(".", 2)
    if len(parts) < 3:
        return "unmatched", normalized
    layer_idx, rest = parts[1], parts[2]
    for source, destination in _LAYER_KEY_TEMPLATES:
        if rest.startswith(source):
            return "remap", destination.format(layer_idx=layer_idx) + rest[len(source) :]
    return "unmatched", normalized
