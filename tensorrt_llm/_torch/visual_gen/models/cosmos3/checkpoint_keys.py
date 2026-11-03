# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint-key remap for Cosmos3 diffusion transformers.

Kept free of torch imports so the Edge-FP8 tensor manifest can be checked
without constructing the model or a GPU.
"""

from typing import Optional, Tuple


def remap_cosmos3_checkpoint_key(key: str) -> Tuple[str, Optional[str]]:
    """Classify one diffusion-transformer checkpoint tensor.

    Returns ``("skip", None)`` for tensors the runtime does not load,
    ``("remap", runtime_name)`` for tensors that land on a constructed
    parameter, or ``("unmatched", normalized_key)`` when the name is not part
    of this topology. ModelOpt fake-quant buffers (``*_quantizer._amax`` /
    ``._scale``) are skipped: the loader consumes ``weight``, ``weight_scale``
    and ``input_scale`` only.
    """
    normalized = key[len("model.") :] if key.startswith("model.") else key
    if "quantizer" in normalized:
        return "skip", None
    if normalized.startswith(("lm_head.", "action_pos_embed.", "norm.")):
        return "skip", None

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
    understanding = f"language_model.layers.{layer_idx}"
    generation = f"gen_layers.{layer_idx}"
    # Generation patterns are matched first. ``add_*`` names share the
    # ``self_attn.`` prefix with the understanding projections but not their stems.
    mappings = (
        {
            "self_attn.add_q_proj.": f"{generation}.cross_attention.to_q.",
            "self_attn.add_k_proj.": f"{generation}.cross_attention.to_k.",
            "self_attn.add_v_proj.": f"{generation}.cross_attention.to_v.",
            "self_attn.to_add_out.": f"{generation}.cross_attention.to_out.0.",
            "self_attn.norm_added_q.": f"{generation}.cross_attention.norm_q.",
            "self_attn.norm_added_k.": f"{generation}.cross_attention.norm_k.",
        },
        {
            "self_attn.to_q.": f"{understanding}.self_attn.to_q.",
            "self_attn.to_k.": f"{understanding}.self_attn.to_k.",
            "self_attn.to_v.": f"{understanding}.self_attn.to_v.",
            "self_attn.to_out.": f"{understanding}.self_attn.to_out.0.",
            "self_attn.norm_q.": f"{understanding}.self_attn.norm_q.",
            "self_attn.norm_k.": f"{understanding}.self_attn.norm_k.",
            "self_attn.k_norm_und_for_gen.": f"{understanding}.self_attn.k_norm_und_for_gen.",
        },
        {
            "input_layernorm.": f"{understanding}.input_layernorm.",
            "post_attention_layernorm.": f"{understanding}.post_attention_layernorm.",
            "input_layernorm_moe_gen.": f"{generation}.input_layernorm.",
            "post_attention_layernorm_moe_gen.": f"{generation}.post_attention_layernorm.",
        },
        {
            "mlp.gate_proj.": f"{understanding}.mlp.gate_proj.",
            "mlp.up_proj.": f"{understanding}.mlp.up_proj.",
            "mlp.down_proj.": f"{understanding}.mlp.down_proj.",
            "mlp_moe_gen.gate_proj.": f"{generation}.mlp.gate_proj.",
            "mlp_moe_gen.up_proj.": f"{generation}.mlp.up_proj.",
            "mlp_moe_gen.down_proj.": f"{generation}.mlp.down_proj.",
        },
    )
    for mapping in mappings:
        for pattern, replacement in mapping.items():
            if rest.startswith(pattern):
                return "remap", replacement + rest[len(pattern) :]
    return "unmatched", normalized
