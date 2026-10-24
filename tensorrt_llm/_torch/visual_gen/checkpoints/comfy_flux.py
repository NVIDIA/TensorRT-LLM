# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Map native FLUX checkpoint projections to Diffusers names without requantization.

Checkpoint names and row ordering are checked against SGLang's Apache-2.0
FLUX configuration and loaders. See COMFY_IMPORT_NOTICE for pinned references.
"""

from collections.abc import Mapping
from typing import Literal

import torch

from .comfy_quant import ComfyQuantization, convert_comfy_nvfp4


def comfy_flux_quantization_config(
    checkpoint: ComfyQuantization,
    *,
    version: Literal[1, 2],
    hidden_size: int,
    mlp_ratio: float = 4.0,
) -> dict:
    """Derive exclusions from checkpoint headers, before allocating model tensors.

    Meta tensors run the same name/row mapping as the weight importer without
    loading the weight payload. Their dtype carries the per-projection recipe.
    """
    shapes = {
        key: torch.empty(
            shape,
            dtype=torch.uint8
            if key.removesuffix(".weight") in checkpoint.nvfp4_layers
            else torch.float16,
            device="meta",
        )
        for key, shape in checkpoint.weight_shapes.items()
    }
    mapped = convert_comfy_flux_weights(
        shapes, version=version, hidden_size=hidden_size, mlp_ratio=mlp_ratio
    )
    recipes: dict[str, set[bool]] = {}
    for key, tensor in mapped.items():
        if tensor.ndim != 2:
            continue
        module = key.removesuffix(".weight")
        module = module.replace(".net.0.proj", ".up_proj").replace(".net.2", ".down_proj")
        if version == 2:
            module = module.replace(".linear_in", ".gate_up_proj").replace(
                ".linear_out", ".down_proj"
            )
        for original in ("to_q", "to_k", "to_v"):
            if module.endswith(".attn." + original):
                module = module.removesuffix(original) + "qkv_proj"
        for original in ("add_q_proj", "add_k_proj", "add_v_proj"):
            if module.endswith(".attn." + original):
                module = module.removesuffix(original) + "add_qkv_proj"
        recipes.setdefault(module, set()).add(tensor.dtype == torch.uint8)
    exclusions = []
    for module, modes in recipes.items():
        if len(modes) != 1:
            raise ValueError(f"{module}: fused projections cannot mix NVFP4 and full precision")
        if modes == {False}:
            # Wrapper projections contain child Linears at TP > 1.
            exclusions.extend((module, module + ".*"))
    return {
        "quant_algo": "NVFP4",
        "ignore": sorted(exclusions),
        "config_groups": {
            "comfy_nvfp4": {
                "weights": {"dynamic": False, "group_size": 16},
                "input_activations": {"dynamic": True},
            }
        },
    }


def convert_comfy_flux_weights(
    weights: Mapping[str, torch.Tensor],
    *,
    version: Literal[1, 2],
    hidden_size: int,
    mlp_ratio: float = 4.0,
) -> dict[str, torch.Tensor]:
    """Convert a transformer-only native FLUX state dict, including NVFP4 scales.

    Row partitions are applied identically to packed weights, biases and linear
    block scales. Each partition retains the original scalar global scale.
    FLUX.2 retains its fused single-stream attention/MLP projection.
    """
    if version not in (1, 2) or hidden_size <= 0 or mlp_ratio <= 0:
        raise ValueError("Expected FLUX version 1 or 2 and positive model dimensions")
    source = {}
    for key, value in weights.items():
        name = key.removeprefix("model.diffusion_model.")
        if name in source:
            raise ValueError(f"Duplicate FLUX checkpoint tensor after prefix removal: {name}")
        source[name] = value
    source = convert_comfy_nvfp4(source)
    embedding = "time_text_embed" if version == 1 else "time_guidance_embed"
    direct = {
        "img_in": "x_embedder",
        "txt_in": "context_embedder",
        "time_in.in_layer": f"{embedding}.timestep_embedder.linear_1",
        "time_in.out_layer": f"{embedding}.timestep_embedder.linear_2",
        "guidance_in.in_layer": f"{embedding}.guidance_embedder.linear_1",
        "guidance_in.out_layer": f"{embedding}.guidance_embedder.linear_2",
        "final_layer.linear": "proj_out",
        "final_layer.adaLN_modulation.1": "norm_out.linear",
    }
    if version == 1:
        direct.update(
            {
                "vector_in.in_layer": "time_text_embed.text_embedder.linear_1",
                "vector_in.out_layer": "time_text_embed.text_embedder.linear_2",
            }
        )
    else:
        direct.update(
            {
                "double_stream_modulation_img.lin": "double_stream_modulation_img.linear",
                "double_stream_modulation_txt.lin": "double_stream_modulation_txt.linear",
                "single_stream_modulation.lin": "single_stream_modulation.linear",
            }
        )
    double = {
        "img_attn.norm.query_norm": "attn.norm_q",
        "img_attn.norm.key_norm": "attn.norm_k",
        "txt_attn.norm.query_norm": "attn.norm_added_q",
        "txt_attn.norm.key_norm": "attn.norm_added_k",
        "img_attn.proj": "attn.to_out.0",
        "txt_attn.proj": "attn.to_add_out",
        "img_mlp.0": "ff.net.0.proj" if version == 1 else "ff.linear_in",
        "img_mlp.2": "ff.net.2" if version == 1 else "ff.linear_out",
        "txt_mlp.0": "ff_context.net.0.proj" if version == 1 else "ff_context.linear_in",
        "txt_mlp.2": "ff_context.net.2" if version == 1 else "ff_context.linear_out",
    }
    if version == 1:
        double.update({"img_mod.lin": "norm1.linear", "txt_mod.lin": "norm1_context.linear"})
    single = {
        "norm.query_norm": "attn.norm_q",
        "norm.key_norm": "attn.norm_k",
        "linear2": "proj_out" if version == 1 else "attn.to_out",
    }
    if version == 1:
        single["modulation.lin"] = "norm.linear"
    else:
        single["linear1"] = "attn.to_qkv_mlp_proj"

    result = {}
    for key, value in source.items():
        module, suffix = key.rsplit(".", 1)
        if suffix not in ("weight", "bias", "scale", "weight_scale", "weight_scale_2"):
            raise ValueError(f"Unsupported native FLUX checkpoint tensor: {key}")
        suffix = "weight" if suffix == "scale" else suffix
        destinations = None
        sizes = None
        if module in direct:
            destinations = [direct[module]]
        elif module.startswith(("double_blocks.", "single_blocks.")):
            block, index, inner = module.split(".", 2)
            if not index.isdigit():
                raise ValueError(f"Invalid FLUX block index: {key}")
            if block == "double_blocks":
                prefix = f"transformer_blocks.{index}."
                if inner in ("img_attn.qkv", "txt_attn.qkv"):
                    names = (
                        ("to_q", "to_k", "to_v")
                        if inner == "img_attn.qkv"
                        else ("add_q_proj", "add_k_proj", "add_v_proj")
                    )
                    destinations = [prefix + "attn." + name for name in names]
                    sizes = [hidden_size] * 3
                elif inner in double:
                    destinations = [prefix + double[inner]]
            else:
                prefix = f"single_transformer_blocks.{index}."
                if inner == "linear1" and version == 1:
                    destinations = [
                        prefix + name
                        for name in ("attn.to_q", "attn.to_k", "attn.to_v", "proj_mlp")
                    ]
                    sizes = [hidden_size] * 3 + [int(hidden_size * mlp_ratio)]
                elif inner in single:
                    destinations = [prefix + single[inner]]
        if destinations is None:
            raise ValueError(f"Unsupported native FLUX checkpoint tensor: {key}")
        if module == "final_layer.adaLN_modulation.1" and suffix != "weight_scale_2":
            if value.ndim == 0 or value.shape[0] != 2 * hidden_size:
                raise ValueError(f"{key}: expected {2 * hidden_size} modulation rows")
            # Diffusers uses scale, shift; the native checkpoint uses shift, scale.
            value = torch.cat((value[hidden_size:], value[:hidden_size]), dim=0)
        if sizes is None or suffix == "weight_scale_2":
            pieces = [value] * len(destinations)
        else:
            if value.ndim == 0 or value.shape[0] != sum(sizes):
                raise ValueError(f"{key}: expected {sum(sizes)} fused projection rows")
            pieces = value.split(sizes, dim=0)
        for name, piece in zip(destinations, pieces):
            target = f"{name}.{suffix}"
            if target in result:
                raise ValueError(f"Duplicate converted FLUX tensor: {target}")
            result[target] = piece.contiguous()
    return result
