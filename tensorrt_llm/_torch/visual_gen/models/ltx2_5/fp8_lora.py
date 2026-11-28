# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Stage-specific FP8 weights quantized from pristine BF16 checkpoint tensors."""

import math
from dataclasses import dataclass

import torch

from tensorrt_llm._torch.modules.linear import Linear
from tensorrt_llm._torch.visual_gen.quantization.ops import quantize_fp8_blockwise
from tensorrt_llm._torch.visual_gen.runtime_lora import (
    _extract_lora_pairs,
    _load_lora_tensors,
    _normalize_common_name,
)
from tensorrt_llm.visual_gen.args import RuntimeLoRAConfig

from .lora import _target_name


@dataclass
class _StageLinear:
    name: str
    module: Linear
    base: dict[str, torch.Tensor]
    conditioned: dict[str, torch.Tensor]


class FP8StageLoRAWeights:
    """Keep raw checkpoint-layout payloads; rebuild backend layouts on every switch.

    Each adapter is merged in BF16 before quantization. Both base and conditioned
    block scales are stored with their weights, independently of the backend's
    transformed scales, output-slice caches and small-M GEMM metadata.
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
            module = modules.get(name)
            base = source.get(name + ".weight")
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
            if module.weight.dtype not in (torch.bfloat16, torch.float8_e4m3fn):
                raise ValueError(f"Unsupported IC-LoRA target dtype: {name}")
            if module.weight.dtype == torch.float8_e4m3fn and (
                module._weights_transformed
                or module.weight_scale.dtype != torch.float32
                or tuple(module.weight_scale.shape)
                != (math.ceil(base.shape[0] / 128), math.ceil(base.shape[1] / 128))
            ):
                raise ValueError(
                    f"IC-LoRA FP8 bank requires raw pre-transform weights/scales: {name}"
                )
            targets.append((name, module, base, pair))
        self._weights = []
        self.quantized_target_count = 0
        for name, module, base, pair in targets:
            device = module.weight.device
            a = pair.a.to(device=device, dtype=torch.bfloat16)
            b = pair.b.to(device=device, dtype=torch.bfloat16)
            merged = (b * pair.scale) @ a
            merged.add_(base.to(device))
            if not torch.isfinite(merged).all():
                raise ValueError(f"Non-finite IC-LoRA weight: {name}")
            if module.weight.dtype == torch.float8_e4m3fn:
                qweight, scales = quantize_fp8_blockwise(merged)
                conditioned = {"weight": qweight.cpu(), "weight_scale": scales.cpu()}
                pristine = {
                    key: getattr(module, key).detach().to("cpu", copy=True)
                    for key in ("weight", "weight_scale")
                }
                self.quantized_target_count += 1
            else:
                conditioned = {"weight": merged.cpu()}
                pristine = {"weight": base.detach().to("cpu", copy=True)}
            if module.bias is not None:
                bias = source[name + ".bias"].detach().to("cpu", copy=True)
                pristine["bias"] = conditioned["bias"] = bias
            self._weights.append(_StageLinear(name, module, pristine, conditioned))
        self._stage_one: bool | None = False

    @torch.no_grad()
    def select(self, stage_one: bool) -> None:
        """Reload weights/scales together and invalidate all Linear-derived state."""
        if self._stage_one == stage_one:
            return
        self._stage_one = None
        for entry in self._weights:
            state = entry.conditioned if stage_one else entry.base
            if entry.module.weight.dtype == torch.bfloat16:
                weight = entry.module.weight
                if weight.ndim == 3 and weight.shape[0] == 1:
                    weight = weight[0].t()
                weight.copy_(state["weight"])
                continue
            entry.module.pre_reload_weights()
            entry.module.load_weights([state])
            entry.module.post_load_weights()
        self._stage_one = stage_one
