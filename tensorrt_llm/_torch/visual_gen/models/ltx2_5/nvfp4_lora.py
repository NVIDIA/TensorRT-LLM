# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Stage-specific dynamic NVFP4 weights built from pristine BF16 tensors."""

import math

import torch

from tensorrt_llm._torch.modules.linear import Linear
from tensorrt_llm._torch.visual_gen.quantization.ops import quantize_nvfp4
from tensorrt_llm._torch.visual_gen.runtime_lora import (
    _extract_lora_pairs,
    _load_lora_tensors,
    _normalize_common_name,
)
from tensorrt_llm.visual_gen.args import RuntimeLoRAConfig

from .fp8_lora import _StageLinear
from .lora import _target_name


class NVFP4StageLoRAWeights:
    """Reload complete raw packed/scale banks and rebuild native layouts together.

    Raw bank block scales use the loader's LINEAR E4M3 layout. Runtime Linear
    scales may be interleaved/padded and must never be stored as raw bank scales.
    Activation scales are recomputed from each input by the dynamic Linear.
    """

    @torch.no_grad()
    def __init__(self, transformer, weights: dict[str, torch.Tensor], path: str, scale: float):
        pairs, incomplete = _extract_lora_pairs(
            {
                key: value
                for key, value in _load_lora_tensors(path).items()
                if key.endswith((".lora_A.weight", ".lora_B.weight"))
            },
            RuntimeLoRAConfig(path=path, scale=scale),
        )
        if incomplete or not pairs:
            raise ValueError("IC-LoRA requires complete, nonempty A/B pairs")
        modules = dict(transformer.named_modules())
        source = {_normalize_common_name(key): value for key, value in weights.items()}
        targets, seen = [], set()
        for pair in pairs:
            name = _target_name(pair.name)
            module, base = modules.get(name), source.get(name + ".weight")
            if name in seen or not isinstance(module, Linear) or base is None:
                raise ValueError(f"Duplicate or unmatched IC-LoRA target: {pair.name}")
            seen.add(name)
            if (
                base.dtype != torch.bfloat16
                or base.ndim != 2
                or pair.a.ndim != 2
                or pair.b.ndim != 2
                or pair.a.shape[0] != pair.b.shape[1]
                or tuple(base.shape) != (pair.b.shape[0], pair.a.shape[1])
                or not math.isfinite(pair.scale)
            ):
                raise ValueError(f"Invalid BF16 IC-LoRA shape or scale: {name}")
            if module.weight.dtype not in (torch.bfloat16, torch.uint8):
                raise ValueError(f"Unsupported NVFP4 IC-LoRA target dtype: {name}")
            if module.weight.dtype == torch.uint8 and (
                module._weights_transformed
                or not module.force_dynamic_quantization
                or module.scaling_vector_size != 16
            ):
                raise ValueError(
                    f"NVFP4 bank requires pre-transform dynamic block16 Linear: {name}"
                )
            targets.append((name, module, base, pair))
        self._weights = []
        self.quantized_target_count = 0
        for name, module, base, pair in targets:
            device = module.weight.device
            a, b = pair.a.to(device, torch.bfloat16), pair.b.to(device, torch.bfloat16)
            merged = (b * pair.scale) @ a
            merged.add_(base.to(device))
            if not torch.isfinite(merged).all():
                raise ValueError(f"Non-finite IC-LoRA weight: {name}")
            if module.weight.dtype == torch.uint8:

                def pack(value):
                    payload, block_scale, global_scale = quantize_nvfp4(value)
                    return {
                        "weight": payload.cpu(),
                        "weight_scale": block_scale.cpu(),
                        "weight_scale_2": global_scale.cpu(),
                    }

                # Requantize the original BF16 source, never a transformed runtime bank.
                pristine, conditioned = pack(base.to(device)), pack(merged)
                self.quantized_target_count += 1
            else:
                pristine, conditioned = {"weight": base.cpu().clone()}, {"weight": merged.cpu()}
            if module.bias is not None:
                bias = source[name + ".bias"].cpu().clone()
                pristine["bias"] = conditioned["bias"] = bias
            self._weights.append(_StageLinear(name, module, pristine, conditioned))
        self._stage_one: bool | None = False

    @torch.no_grad()
    def select(self, stage_one: bool) -> None:
        if self._stage_one == stage_one:
            return
        self._stage_one = None
        for entry in self._weights:
            state = entry.conditioned if stage_one else entry.base
            module = entry.module
            if module.weight.dtype == torch.bfloat16:
                weight = module.weight
                if weight.ndim == 3 and weight.shape[0] == 1:
                    weight = weight[0].t()
                weight.copy_(state["weight"])
            else:
                module.pre_reload_weights()
                module.load_weights([state])
                module.post_load_weights()
        self._stage_one = stage_one
