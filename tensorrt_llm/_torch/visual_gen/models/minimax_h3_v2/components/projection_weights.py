# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn

from tensorrt_llm._torch.visual_gen.quantization.ops import (
    quantize_fp8_blockwise,
    quantize_fp8_per_tensor,
    quantize_nvfp4,
)
from tensorrt_llm.quantization.mode import QuantAlgo
from tensorrt_llm.quantization.utils import fp4_utils

if TYPE_CHECKING:
    from tensorrt_llm._torch.visual_gen.config import DiffusionModelConfig

    from .linear import Linear

_FIELDS = (
    "weight",
    "bias",
    "weight_scale",
    "weight_scale_inv",
    "input_scale",
    "weight_scale_2",
    "k_scale",
    "v_scale",
)
_HP_DTYPES = (torch.bfloat16, torch.float16, torch.float32)


def _quantize(parts, algo, block_size):
    if algo == QuantAlgo.NVFP4 and any(p["weight"].dtype in _HP_DTYPES for p in parts):
        # All packed QKV or gate/up rows must share one global multiplier, across ranks.
        values = [p["weight"].cuda() for p in parts]
        combined = torch.cat(values) if len(values) > 1 else values[0]
        weight, scales, global_scale = quantize_nvfp4(combined)
        sizes = [w.shape[0] for w in values]
        for p, w, s in zip(parts, weight.split(sizes), scales.split(sizes)):
            p.update(weight=w, weight_scale=s, weight_scale_2=global_scale)
    elif algo in (QuantAlgo.FP8, QuantAlgo.FP8_BLOCK_SCALES):
        for p in parts:
            if p["weight"].dtype in _HP_DTYPES:
                value = p["weight"].cuda()
                if algo == QuantAlgo.FP8:
                    p["weight"], p["weight_scale"] = quantize_fp8_per_tensor(value)
                else:
                    p["weight"], p["weight_scale"] = quantize_fp8_blockwise(
                        value, block_size=block_size
                    )


def _cat(parts):
    return torch.cat(parts, dim=0) if len(parts) > 1 else parts[0]


def _set_scale(module: Linear, name: str, value: torch.Tensor) -> None:
    scale = getattr(module, name, None)
    if scale is None:
        setattr(
            module,
            name,
            nn.Parameter(value.to(module.weight.device).float().reshape(()), requires_grad=False),
        )
    else:
        scale.copy_(value.reshape_as(scale))


@torch.no_grad()
def load_projection(
    module: Linear,
    name: str,
    weights: dict[str, torch.Tensor],
    model_config: DiffusionModelConfig | None,
) -> list[str]:
    fused = module.fused_weight_shard_indices_mapping
    labels = tuple(fused) if fused else (None,)
    parent = name.rpartition(".")[0]
    prefixes = (
        [name]
        if not fused
        else [
            ".".join(filter(None, (parent, "to_" + key if key in ("q", "k", "v") else key)))
            for key in labels
        ]
    )
    parts = [
        {field: weights[f"{prefix}.{field}"] for field in _FIELDS if f"{prefix}.{field}" in weights}
        for prefix in prefixes
    ]
    missing = [
        f"{name}.{field}"
        for field in (("weight", "bias") if module.has_bias else ("weight",))
        if any(field not in part for part in parts)
    ]
    if missing:
        return missing

    if model_config is not None and model_config.dynamic_weight_quant:
        config = module.quant_config or model_config.get_quant_config(name)
        excluded = (
            model_config.quant_config
            and model_config.quant_config.is_module_excluded_from_quantization(name)
        )
        if config is not None and not excluded:
            _quantize(
                parts,
                config.quant_algo,
                model_config.quant_config.group_size if model_config.quant_config else 128,
            )

    device = module.weight.device
    values = [
        module.load_shard(
            p["weight"],
            name=label,
            device=device,
            span=2 if module.has_nvfp4 and module.tp_mode == "row" else 1,
        )
        for label, p in zip(labels, parts)
    ]
    if module.has_fp8_qdq:
        scales = [p.get("weight_scale", module.weight_scale).reshape(()).to(device) for p in parts]
        largest = torch.stack(scales).max()
        module.weight_scale.copy_(largest)
        if fused:
            values = [
                w.to(module.dtype).mul_(s).div_(largest).to(torch.float8_e4m3fn)
                for w, s in zip(values, scales)
            ]
        inputs = [p["input_scale"].reshape(()).to(device) for p in parts if "input_scale" in p]
        if inputs:
            input_scale = torch.stack(inputs).max()
            _set_scale(module, "input_scale", input_scale)
            _set_scale(module, "inv_input_scale", input_scale.reciprocal())
        else:
            module.input_scale = module.inv_input_scale = None

    if module.has_fp8_block_scales:
        scales = []
        for label, p in zip(labels, parts):
            s = p.get("weight_scale_inv", p.get("weight_scale"))
            if s is None:
                raise ValueError(f"Missing FP8 block scales for {name}")
            if s.ndim == 4:
                s = s.squeeze(1).squeeze(-1)
            scales.append(module.load_shard(s, name=label, device=device, span=128))
        scales = _cat(scales)
        # Post-load packs these scales as int32. Reload starts from the checkpoint grid.
        scale_shape = ((module.out_features + 127) // 128, (module.in_features + 127) // 128)
        if scales.shape != scale_shape:
            raise ValueError(
                f"H3 block scales for {name} have shape {tuple(scales.shape)}, expected {scale_shape}"
            )
        module.weight_scale.data = scales.float().clone()
        if not fused and "input_scale" in parts[0]:
            _set_scale(module, "input_scale", parts[0]["input_scale"])
            _set_scale(module, "inv_input_scale", module.input_scale.reciprocal())
    elif module.has_nvfp4:
        scales = [
            module.load_shard(
                p["weight_scale"],
                device=device,
                name=label,
                span=16 if module.tp_mode == "row" else 1,
            )
            for label, p in zip(labels, parts)
        ]
        module.weight_scale.copy_(
            torch.ops.trtllm.block_scale_interleave(
                _cat(scales).contiguous().view(fp4_utils.float4_sf_dtype)
            )
        )
        globals_ = [p["weight_scale_2"].reshape(()).to(device) for p in parts]
        if any(not torch.allclose(globals_[0], s) for s in globals_[1:]):
            raise ValueError(f"Fused NVFP4 projection {name} needs one global weight scale")
        _set_scale(module, "weight_scale_2", globals_[0])
        input_scales = [
            p["input_scale"].reshape(()).to(device) for p in parts if "input_scale" in p
        ]
        if input_scales:
            if any(not torch.allclose(input_scales[0], s) for s in input_scales[1:]):
                raise ValueError(f"Fused NVFP4 projection {name} needs one global activation scale")
            _set_scale(module, "input_scale", input_scales[0].reciprocal())
            _set_scale(
                module,
                "inv_input_scale",
                module.input_scale / 6.0 if not fused else input_scales[0],
            )
            _set_scale(module, "alpha", input_scales[0] * globals_[0])

    value = _cat(values)
    if module.has_nvfp4 and module._weights_transformed:
        module.weight.data = value.to(dtype=module.weight.dtype, device=device).clone()
    else:
        module.weight.copy_(value)
    if module.has_bias:
        module.bias.copy_(
            _cat(
                [
                    module.load_shard(p["bias"], name=label, device=device)
                    for label, p in zip(labels, parts)
                ]
            )
        )
    if module.has_nvfp4 or module.has_fp8_qdq:
        keys = [p["k_scale"].reshape(()).to(device) for p in parts if "k_scale" in p]
        vals = [p["v_scale"].reshape(()).to(device) for p in parts if "v_scale" in p]
        if keys:
            k, v = torch.stack(keys).max(), torch.stack(vals).max()
            module.kv_scales.copy_(torch.stack((torch.ones_like(k), k, v)))
            module.inv_kv_scales.copy_(module.kv_scales.reciprocal())
    module._weights_transformed = False
    return []
