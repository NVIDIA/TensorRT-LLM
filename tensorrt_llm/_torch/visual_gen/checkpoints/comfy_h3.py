# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native Comfy H3 projections, including gate-first MLP weights.

Checkpoint names and row ordering are checked against SGLang's Apache-2.0
H3 configuration and model. See COMFY_IMPORT_NOTICE for pinned references.
"""

from collections.abc import Mapping

import torch

from .comfy_quant import ComfyQuantization, convert_comfy_nvfp4


def comfy_h3_quantization_config(
    checkpoint: ComfyQuantization, *, attention_dim: int, ffn_dim: int
) -> dict:
    """Derive the native H3 mixed-precision recipe using header shapes only."""
    shapes = {
        key: torch.empty(
            shape,
            device="meta",
            dtype=(
                torch.uint8
                if key.removesuffix(".weight") in checkpoint.nvfp4_layers
                else torch.float16
            ),
        )
        for key, shape in checkpoint.weight_shapes.items()
    }
    mapped = convert_comfy_h3_weights(shapes, attention_dim=attention_dim, ffn_dim=ffn_dim)
    modes: dict[str, set[bool]] = {}
    for key, value in mapped.items():
        if value.ndim != 2:
            continue
        module = key.removesuffix(".weight")
        # Exclude both separate and fused QKV modules (async Ulysses selects the former).
        names = [module]
        if module.endswith((".attn.to_q", ".attn.to_k", ".attn.to_v")):
            names.append(module.rsplit(".", 1)[0] + ".qkv_proj")
        elif module.endswith((".ff.gate", ".ff.up")):
            names.append(module.rsplit(".", 1)[0] + ".gate_up_proj")
        for name in names:
            modes.setdefault(name, set()).add(value.dtype == torch.uint8)
    for name, values in modes.items():
        if len(values) != 1:
            raise ValueError(f"{name}: fused H3 projections cannot mix NVFP4 and full precision")
    return {
        "quant_algo": "NVFP4",
        "ignore": sorted(name for name, values in modes.items() if values == {False}),
        "config_groups": {
            "comfy_nvfp4": {
                "weights": {"dynamic": False, "group_size": 16},
                "input_activations": {"dynamic": True},
            }
        },
    }


def convert_comfy_h3_weights(
    weights: Mapping[str, torch.Tensor], *, attention_dim: int, ffn_dim: int
) -> dict[str, torch.Tensor]:
    """Normalize quantization and map native H3 names to TRT-LLM projections.

    Native Comfy SwiGLU stores gate before up. Map those halves directly to
    the loader's gate/up entries, bypassing the Diffusers up/gate permutation.
    """
    source = {}
    for key, value in weights.items():
        key = key.removeprefix("model.diffusion_model.")
        if key in source:
            raise ValueError(f"Duplicate H3 tensor after prefix removal: {key}")
        source[key] = value
    source = convert_comfy_nvfp4(source)
    direct = {
        "video_patch_proj": "proj_in",
        "audio_patch_proj": "audio_proj_in",
        "condition_proj": "context_embedder",
        "time_embedder.proj_in": "time_embedder.linear_1",
        "time_embedder.proj_out": "time_embedder.linear_2",
        "final_layer.video_out": "proj_out",
        "final_layer.audio_out": "audio_proj_out",
        "final_layer.norm": "norm_out.norm",
        "final_layer.adaln_proj.linear": "norm_out.linear",
        "token_refiner.final_norm": "token_refiner.final_norm",
    }
    result = {}
    for key, value in source.items():
        if key in ("adaln_t_table", "rope.inv_freq"):
            result[key] = value
            continue
        module, suffix = key.rsplit(".", 1)
        if suffix not in ("weight", "bias", "weight_scale", "weight_scale_2"):
            raise ValueError(f"Unsupported native H3 checkpoint tensor: {key}")
        destinations = None
        sizes = None
        if module in direct:
            destinations = [direct[module]]
        else:
            if module.startswith("blocks."):
                module = "transformer_blocks." + module.removeprefix("blocks.")
            elif module.startswith("token_refiner.blocks."):
                module = "token_refiner.refiner_blocks." + module.removeprefix(
                    "token_refiner.blocks."
                )
            else:
                raise ValueError(f"Unsupported native H3 checkpoint tensor: {key}")
            if module.endswith(".attn.qkv_proj"):
                prefix = module.removesuffix("qkv_proj")
                destinations = [prefix + name for name in ("to_q", "to_k", "to_v")]
                sizes = [attention_dim] * 3
            elif module.endswith(".mlp.fc1"):
                prefix = module.removesuffix("mlp.fc1")
                destinations = [prefix + "ff.gate", prefix + "ff.up"]
                sizes = [ffn_dim] * 2
            else:
                replacements = {
                    ".attn.out_proj": ".attn.to_out.0",
                    ".attn.q_norm": ".attn.norm_q",
                    ".attn.k_norm": ".attn.norm_k",
                    ".mlp.fc2": ".ff.down_proj",
                    ".norm1": ".norm1",
                    ".norm2": ".norm2",
                    ".adaln_proj.linear": ".adaln_proj.linear",
                }
                for old, new in replacements.items():
                    if module.endswith(old):
                        destinations = [module.removesuffix(old) + new]
                        break
        if destinations is None:
            raise ValueError(f"Unsupported native H3 checkpoint tensor: {key}")
        if sizes is None or suffix == "weight_scale_2":
            pieces = [value] * len(destinations)
        else:
            if value.ndim == 0 or value.shape[0] != sum(sizes):
                raise ValueError(f"{key}: expected {sum(sizes)} fused projection rows")
            pieces = value.split(sizes, dim=0)
        for name, piece in zip(destinations, pieces):
            target = name + "." + suffix
            if target in result:
                raise ValueError(f"Duplicate converted H3 tensor: {target}")
            result[target] = piece.contiguous()
    return result
