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

from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

import safetensors.torch
import torch
from torch import nn

from tensorrt_llm.logger import logger
from tensorrt_llm.visual_gen.args import LoRAConfig

_COMMON_TRANSFORMER_PREFIXES = (
    "transformer.",
    "model.diffusion_model.",
    "diffusion_model.",
    "velocity_model.",
    "denoiser.",
    "dit.",
)
_QKV_SUFFIXES = (".to_q", ".to_k", ".to_v")


def _find_safetensors_files(path: str) -> List[Path]:
    lora_path = Path(path)
    if lora_path.is_file():
        return [lora_path]
    if lora_path.is_dir():
        return sorted(lora_path.glob("*.safetensors"))
    raise FileNotFoundError(f"LoRA path does not exist: {path}")


def _load_safetensors(path: str) -> Dict[str, torch.Tensor]:
    files = _find_safetensors_files(path)
    if not files:
        raise FileNotFoundError(f"No safetensors LoRA weights found at {path}")

    tensors: Dict[str, torch.Tensor] = {}
    for file_path in files:
        tensors.update(safetensors.torch.load_file(str(file_path), device="cpu"))
    return tensors


def _strip_lora_suffix(key: str) -> Optional[Tuple[str, str]]:
    suffixes = (
        (".lora_A.weight", "down"),
        (".lora_B.weight", "up"),
        (".lora_down.weight", "down"),
        (".lora_up.weight", "up"),
        (".alpha", "alpha"),
    )
    for suffix, kind in suffixes:
        if key.endswith(suffix):
            return key[: -len(suffix)], kind
    return None


def _normalize_module_name(
    name: str,
    strip_prefixes: Iterable[str],
    key_map: Mapping[str, str],
) -> str:
    prefixes = tuple(strip_prefixes) + _COMMON_TRANSFORMER_PREFIXES
    for prefix in sorted(set(prefixes), key=len, reverse=True):
        if prefix and name.startswith(prefix):
            name = name[len(prefix) :]
            break

    for source, target in key_map.items():
        name = name.replace(source, target)
    return name


def _normalize_param_name(name: str) -> str:
    return name.replace("._orig_mod.", ".")


def _named_parameters(module: nn.Module) -> Dict[str, torch.nn.Parameter]:
    return {_normalize_param_name(name): param for name, param in module.named_parameters()}


def _matching_weight_name(
    module_name: str,
    parameters: Mapping[str, torch.nn.Parameter],
) -> Optional[str]:
    if module_name in parameters:
        return module_name

    weight_name = f"{module_name}.weight"
    if weight_name in parameters:
        return weight_name

    return None


def _fuse_qkv_deltas(
    deltas: Dict[str, torch.Tensor],
    parameters: Mapping[str, torch.nn.Parameter],
) -> Dict[str, torch.Tensor]:
    fused: Dict[str, torch.Tensor] = {}
    consumed = set()

    for name in list(deltas):
        if name in consumed:
            continue

        matched_suffix = next((suffix for suffix in _QKV_SUFFIXES if name.endswith(suffix)), None)
        if matched_suffix is None:
            fused[name] = deltas[name]
            continue

        prefix = name[: -len(matched_suffix)]
        qkv_names = tuple(f"{prefix}{suffix}" for suffix in _QKV_SUFFIXES)
        if not all(qkv_name in deltas for qkv_name in qkv_names):
            fused[name] = deltas[name]
            continue

        fused_name = f"{prefix}.qkv_proj"
        if _matching_weight_name(fused_name, parameters) is None:
            fused[name] = deltas[name]
            continue

        fused[fused_name] = torch.cat([deltas[qkv_name] for qkv_name in qkv_names], dim=0)
        consumed.update(qkv_names)

    return fused


def load_lora_deltas(
    path: str,
    module: nn.Module,
    *,
    strength: float = 1.0,
    strip_prefixes: Iterable[str] = (),
    key_map: Optional[Mapping[str, str]] = None,
    fuse_qkv: bool = True,
) -> Dict[str, torch.Tensor]:
    """Load safetensors LoRA weights and materialize base-weight deltas."""

    if not path:
        raise ValueError("LoRAConfig.path must be set when LoRA is enabled")

    tensors = _load_safetensors(path)
    key_map = key_map or {}
    down_keys: Dict[str, torch.Tensor] = {}
    up_keys: Dict[str, torch.Tensor] = {}
    alpha_by_name: Dict[str, float] = {}

    for key, tensor in tensors.items():
        parsed = _strip_lora_suffix(key)
        if parsed is None:
            continue

        base_name, kind = parsed
        base_name = _normalize_module_name(base_name, strip_prefixes, key_map)
        if kind == "down":
            down_keys[base_name] = tensor
        elif kind == "up":
            up_keys[base_name] = tensor
        else:
            alpha_by_name[base_name] = float(tensor.reshape(-1)[0].item())

    deltas: Dict[str, torch.Tensor] = {}
    for base_name, down_weight in down_keys.items():
        up_weight = up_keys.get(base_name)
        if up_weight is None:
            continue

        if down_weight.ndim != 2 or up_weight.ndim != 2:
            raise ValueError(
                f"LoRA tensors for {base_name} must be rank-2, got "
                f"{tuple(down_weight.shape)} and {tuple(up_weight.shape)}"
            )

        rank = down_weight.shape[0]
        if rank == 0 or up_weight.shape[1] != rank:
            raise ValueError(
                f"LoRA tensors for {base_name} have incompatible ranks: "
                f"{tuple(down_weight.shape)} and {tuple(up_weight.shape)}"
            )

        alpha = alpha_by_name.get(base_name, float(rank))
        scale = float(strength) * alpha / rank
        deltas[base_name] = (up_weight.float() @ down_weight.float()) * scale

    if fuse_qkv:
        deltas = _fuse_qkv_deltas(deltas, _named_parameters(module))

    return deltas


def apply_static_lora(
    module: nn.Module,
    config: LoRAConfig,
    *,
    default_strip_prefixes: Iterable[str] = (),
    default_key_map: Optional[Mapping[str, str]] = None,
) -> int:
    """Merge a static BF16 LoRA adapter into a transformer module."""

    strip_prefixes = tuple(default_strip_prefixes) + tuple(config.strip_prefixes)
    key_map = {**(default_key_map or {}), **config.key_map}
    deltas = load_lora_deltas(
        config.path,
        module,
        strength=config.strength,
        strip_prefixes=strip_prefixes,
        key_map=key_map,
        fuse_qkv=config.fuse_qkv,
    )
    parameters = _named_parameters(module)
    errors = []
    applied = 0

    for delta_name, delta in deltas.items():
        weight_name = _matching_weight_name(delta_name, parameters)
        if weight_name is None:
            errors.append(f"{delta_name}: no matching transformer weight")
            continue

        weight = parameters[weight_name]
        if tuple(weight.shape) != tuple(delta.shape):
            errors.append(
                f"{delta_name}: shape mismatch, LoRA {tuple(delta.shape)} vs "
                f"target {tuple(weight.shape)}"
            )
            continue
        if weight.dtype != torch.bfloat16:
            errors.append(
                f"{delta_name}: static LoRA currently supports BF16 weights only, "
                f"got {weight.dtype}"
            )
            continue

        weight.data.add_(delta.to(device=weight.device, dtype=weight.dtype))
        applied += 1

    if config.strict and errors:
        formatted = "\n".join(errors[:20])
        if len(errors) > 20:
            formatted += f"\n... {len(errors) - 20} more errors"
        raise ValueError(f"Failed to apply static LoRA weights:\n{formatted}")
    if config.strict and applied == 0:
        raise ValueError(f"No LoRA weights from {config.path} matched transformer weights")

    if errors:
        logger.warning(f"Skipped {len(errors)} LoRA deltas while applying {config.path}")
    return applied
