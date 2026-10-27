# SPDX-FileCopyrightText: Copyright (c) 2025-2026 Lightricks Ltd.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: LicenseRef-LTX-2x
"""Stage-specific BF16 weights with exact base-weight restoration."""

import math
from dataclasses import dataclass

import torch

from tensorrt_llm._torch.visual_gen.runtime_lora import (
    _extract_lora_pairs,
    _load_lora_tensors,
    _normalize_common_name,
)
from tensorrt_llm.visual_gen.args import RuntimeLoRAConfig


@dataclass
class _StageWeight:
    target: torch.Tensor
    base: torch.Tensor
    conditioned: torch.Tensor


def _target_name(name: str) -> str:
    for prefix in ("model.diffusion_model.", "diffusion_model."):
        if name.startswith(prefix):
            name = name.removeprefix(prefix)
            break
    return _normalize_common_name(name)


class StageLoRAWeights:
    """Keep CPU base/conditioned copies, transferring only at stage boundaries.

    The official BF16 merge rounds `(B * scale) @ A` before adding the base
    weight. An activation-side adapter is not numerically interchangeable.
    Host copies avoid keeping two full transformers in device memory.
    """

    @torch.no_grad()
    def __init__(self, transformer: torch.nn.Module, path: str, scale: float) -> None:
        # Official IC-LoRA uses the supplied coefficient directly, without
        # PEFT alpha/rank scaling or neighboring adapter_config.json discovery.
        pairs, incomplete = _extract_lora_pairs(
            {
                key: tensor
                for key, tensor in _load_lora_tensors(path).items()
                if key.endswith((".lora_A.weight", ".lora_B.weight"))
            },
            RuntimeLoRAConfig(path=path, scale=scale),
        )
        if incomplete or not pairs:
            raise ValueError("IC-LoRA requires complete, nonempty A/B pairs")
        modules = dict(transformer.named_modules())
        targets = []
        seen = set()
        # Validate the entire adapter before allocating copies or changing weights.
        for pair in pairs:
            name = _target_name(pair.name)
            if name in seen:
                raise ValueError(f"Duplicate or unmatched IC-LoRA target: {pair.name}")
            seen.add(name)
            module = modules.get(name)
            weight = getattr(module, "weight", None)
            # Upstream LTX-2 stores self-attention Q/K/V in one projection.
            # Merge each official adapter into its corresponding row slice.
            if module is None and name.endswith((".to_q", ".to_k", ".to_v")):
                parent, projection = name.rsplit(".", 1)
                attention = modules.get(parent)
                fused = modules.get(parent + ".qkv_proj")
                if attention is not None and fused is not None:
                    q, kv = attention.q_dim, attention.kv_dim
                    start, length = {"to_q": (0, q), "to_k": (q, kv), "to_v": (q + kv, kv)}[
                        projection
                    ]
                    weight = fused.weight.narrow(0, start, length)
            if weight is None:
                raise ValueError(f"unmatched IC-LoRA target: {pair.name}")
            if weight.is_meta or weight.dtype != torch.bfloat16:
                raise ValueError(f"IC-LoRA requires a loaded BF16 weight: {name}")
            if weight.ndim == 3 and weight.shape[0] == 1:
                weight = weight[0].t()
            if (
                pair.a.ndim != 2
                or pair.b.ndim != 2
                or pair.a.shape[0] != pair.b.shape[1]
                or tuple(weight.shape) != (pair.b.shape[0], pair.a.shape[1])
                or not math.isfinite(pair.scale)
            ):
                raise ValueError(f"Invalid IC-LoRA shape or scale: {name}")
            targets.append((weight, pair))
        self._weights = []
        for weight, pair in targets:
            a = pair.a.to(device=weight.device, dtype=torch.bfloat16)
            b = pair.b.to(device=weight.device, dtype=torch.bfloat16)
            merged = (b * pair.scale) @ a
            merged.add_(weight)
            if not torch.isfinite(merged).all():
                raise ValueError(f"Non-finite IC-LoRA weight: {pair.name}")
            self._weights.append(
                _StageWeight(weight, weight.detach().to("cpu", copy=True), merged.cpu())
            )
        self._stage_one: bool | None = False

    @torch.no_grad()
    def select(self, stage_one: bool) -> None:
        """Select conditioned Stage 1 or the pristine Stage 2 weights."""
        if self._stage_one == stage_one:
            return
        # A partial copy must never be mistaken for a completed stage switch.
        self._stage_one = None
        for entry in self._weights:
            entry.target.copy_(entry.conditioned if stage_one else entry.base)
        self._stage_one = stage_one
