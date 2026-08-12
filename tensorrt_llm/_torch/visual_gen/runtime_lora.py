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
"""Startup-preloaded LoRA adapters for VisualGen transformers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import safetensors.torch
import torch
import torch.nn as nn

from tensorrt_llm.logger import logger
from tensorrt_llm.visual_gen.args import RuntimeLoRAConfig

_QKV_SUFFIXES = (".to_q", ".to_k", ".to_v")
_DEFAULT_STRIP_PREFIXES = (
    "model.diffusion_model.",
    "diffusion_model.",
    "transformer.",
    "model.",
    "dit.",
)


@dataclass(frozen=True)
class RuntimeLoRAApplication:
    """Summary returned after applying one LoRA adapter."""

    path: str
    applied_modules: Tuple[str, ...]
    skipped_non_targets: int
    skipped_incomplete: int


@dataclass(frozen=True)
class _LoRAPair:
    name: str
    a: torch.Tensor
    b: torch.Tensor
    scale: float


@dataclass(frozen=True)
class _RuntimeLoRAAdapter:
    a: torch.Tensor
    b: torch.Tensor
    scale: float
    output_start: Optional[int] = None

    @property
    def output_end(self) -> Optional[int]:
        if self.output_start is None:
            return None
        return self.output_start + self.b.shape[0]


def apply_runtime_lora(
    transformer: nn.Module,
    config: RuntimeLoRAConfig,
    *,
    default_strip_prefixes: Iterable[str] = (),
    raise_on_no_matches: bool = True,
) -> RuntimeLoRAApplication:
    """Fuse startup LoRA deltas into matching transformer module weights."""

    if not isinstance(transformer, nn.Module):
        raise ValueError(
            f"Runtime LoRA requires an nn.Module target, got {type(transformer).__name__}"
        )

    raw_tensors = _load_lora_tensors(config.path)
    pairs, skipped_incomplete = _extract_lora_pairs(raw_tensors, config)
    modules = dict(transformer.named_modules())
    targets: Dict[str, List[_RuntimeLoRAAdapter]] = {}
    skipped_non_targets = 0

    strip_prefixes = _dedupe_prefixes(
        tuple(config.strip_prefixes) + tuple(default_strip_prefixes) + _DEFAULT_STRIP_PREFIXES
    )

    qkv_groups: Dict[str, Dict[str, _LoRAPair]] = {}
    incomplete_qkv_names: List[str] = []
    for pair in pairs:
        target_name, fused_suffix = _resolve_target_name(
            pair.name,
            modules,
            strip_prefixes=strip_prefixes,
            key_map=config.key_map,
            fuse_qkv=config.fuse_qkv,
        )
        if target_name is None:
            skipped_non_targets += 1
            continue

        if fused_suffix is None:
            targets.setdefault(target_name, []).append(
                _RuntimeLoRAAdapter(pair.a, pair.b, pair.scale)
            )
        else:
            qkv_groups.setdefault(target_name, {})[fused_suffix] = pair

    for target_name, parts in qkv_groups.items():
        if not all(suffix in parts for suffix in _QKV_SUFFIXES):
            skipped_incomplete += len(parts)
            incomplete_qkv_names.append(target_name)
            continue
        base_module = modules[target_name]
        fused_out = int(getattr(base_module, "out_features", 0))
        total_out = sum(parts[suffix].b.shape[0] for suffix in _QKV_SUFFIXES)
        if total_out != fused_out:
            raise ValueError(
                f"Runtime LoRA fused-QKV spans for '{target_name}' sum to {total_out} "
                f"but the layer output is {fused_out}. The adapter q/k/v split does "
                "not match the fused layer; disable fuse_qkv or use a matching adapter."
            )
        output_start = 0
        for suffix in _QKV_SUFFIXES:
            pair = parts[suffix]
            targets.setdefault(target_name, []).append(
                _RuntimeLoRAAdapter(pair.a, pair.b, pair.scale, output_start=output_start)
            )
            output_start += pair.b.shape[0]

    applied_modules: List[str] = []
    for module_name, adapters in targets.items():
        base_module = modules[module_name]
        if getattr(base_module, "_trtllm_runtime_lora_fused", False):
            raise ValueError(f"Runtime LoRA target '{module_name}' is already fused")
        if not _is_linear_like(base_module):
            if config.strict:
                raise ValueError(
                    f"Runtime LoRA target '{module_name}' is not linear-like: "
                    f"{type(base_module).__name__}"
                )
            skipped_non_targets += len(adapters)
            continue

        try:
            _fuse_lora_into_linear(base_module, adapters, module_name)
        except ValueError:
            if config.strict:
                raise
            skipped_non_targets += len(adapters)
            continue
        applied_modules.append(module_name)

    if config.strict and incomplete_qkv_names:
        raise ValueError(
            "Runtime LoRA found incomplete fused-QKV adapter groups for "
            f"{incomplete_qkv_names}. Provide all to_q/to_k/to_v tensors, "
            "disable fuse_qkv, or set strict=False."
        )

    if not applied_modules and raise_on_no_matches and config.strict:
        raise ValueError(
            f"No Runtime LoRA modules from {config.path!r} matched the transformer. "
            "Check target_components, strip_prefixes, and key_map."
        )

    logger.info(
        f"Runtime LoRA fused from {config.path}: applied={len(applied_modules)}, "
        f"skipped_non_targets={skipped_non_targets}, skipped_incomplete={skipped_incomplete}"
    )
    return RuntimeLoRAApplication(
        path=config.path,
        applied_modules=tuple(applied_modules),
        skipped_non_targets=skipped_non_targets,
        skipped_incomplete=skipped_incomplete,
    )


def _fuse_lora_into_linear(
    base_layer: nn.Module,
    adapters: Iterable[_RuntimeLoRAAdapter],
    module_name: str,
) -> None:
    weight = getattr(base_layer, "weight", None)
    if not isinstance(weight, torch.Tensor) or weight.is_meta:
        raise ValueError(f"Runtime LoRA target '{module_name}' has no loaded weight tensor")
    if weight.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        raise ValueError(
            f"Runtime LoRA target '{module_name}' cannot fuse into weight dtype {weight.dtype}"
        )

    in_features = int(getattr(base_layer, "in_features", 0))
    out_features = int(getattr(base_layer, "out_features", 0))
    device = weight.device
    prepared_adapters: List[_RuntimeLoRAAdapter] = []
    with torch.no_grad():
        for adapter in adapters:
            local_adapter = _shard_adapter(base_layer, adapter, module_name, device)
            _validate_adapter_for_fusion(local_adapter, module_name, in_features, out_features)
            _validate_lora_delta_shape_for_adapter(weight, local_adapter, module_name)
            prepared_adapters.append(local_adapter)
        for local_adapter in prepared_adapters:
            delta = _compute_lora_weight_delta(local_adapter, device, weight.dtype)
            delta = _prepare_lora_delta_for_weight(weight, delta, local_adapter, module_name)
            _add_lora_delta_to_weight(weight, delta, local_adapter, module_name)
    setattr(base_layer, "_trtllm_runtime_lora_fused", True)


def _validate_adapter_for_fusion(
    adapter: _RuntimeLoRAAdapter,
    module_name: str,
    in_features: int,
    out_features: int,
) -> None:
    if adapter.a.ndim != 2 or adapter.b.ndim != 2:
        raise ValueError(
            f"Runtime LoRA target '{module_name}' requires 2D A/B tensors, "
            f"got A={tuple(adapter.a.shape)}, B={tuple(adapter.b.shape)}"
        )
    if adapter.a.shape[0] != adapter.b.shape[1]:
        raise ValueError(
            f"Runtime LoRA rank mismatch for '{module_name}': "
            f"A rank {adapter.a.shape[0]} != B rank {adapter.b.shape[1]}"
        )
    if adapter.a.shape[1] != in_features:
        raise ValueError(
            f"Runtime LoRA input mismatch for '{module_name}': "
            f"A has {adapter.a.shape[1]} columns, layer expects {in_features}"
        )

    if adapter.output_start is None:
        if adapter.b.shape[0] != out_features:
            raise ValueError(
                f"Runtime LoRA output mismatch for '{module_name}': "
                f"B has {adapter.b.shape[0]} rows, expected {out_features}"
            )
        return

    if adapter.output_start < 0 or adapter.output_end is None:
        raise ValueError(f"Invalid Runtime LoRA output span for '{module_name}'")
    if adapter.output_end > out_features:
        raise ValueError(
            f"Runtime LoRA output span for '{module_name}' exceeds layer output: "
            f"end={adapter.output_end}, out_features={out_features}"
        )


def _compute_lora_weight_delta(
    adapter: _RuntimeLoRAAdapter,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    a = adapter.a.to(device=device, dtype=dtype)
    b = adapter.b.to(device=device, dtype=dtype)
    return torch.matmul(b, a).mul_(float(adapter.scale))


def _add_lora_delta_to_weight(
    weight: torch.Tensor,
    delta: torch.Tensor,
    adapter: _RuntimeLoRAAdapter,
    module_name: str,
) -> None:
    target = _lora_weight_target(weight, adapter, module_name)
    _validate_lora_delta_shape(tuple(delta.shape), tuple(target.shape), adapter, module_name)
    target.add_(delta)


def _prepare_lora_delta_for_weight(
    weight: torch.Tensor,
    delta: torch.Tensor,
    adapter: _RuntimeLoRAAdapter,
    module_name: str,
) -> torch.Tensor:
    if weight.ndim == 2:
        return delta
    if weight.ndim == 3 and weight.shape[0] == 1:
        return delta.t().unsqueeze(0)

    raise ValueError(
        f"Runtime LoRA target '{module_name}' has unsupported weight shape {tuple(weight.shape)}"
    )


def _validate_lora_delta_shape_for_adapter(
    weight: torch.Tensor,
    adapter: _RuntimeLoRAAdapter,
    module_name: str,
) -> None:
    target = _lora_weight_target(weight, adapter, module_name)
    if weight.ndim == 2:
        delta_shape = (adapter.b.shape[0], adapter.a.shape[1])
    elif weight.ndim == 3 and weight.shape[0] == 1:
        delta_shape = (1, adapter.a.shape[1], adapter.b.shape[0])
    else:
        raise ValueError(
            f"Runtime LoRA target '{module_name}' has unsupported weight shape "
            f"{tuple(weight.shape)}"
        )
    _validate_lora_delta_shape(delta_shape, tuple(target.shape), adapter, module_name)


def _lora_weight_target(
    weight: torch.Tensor,
    adapter: _RuntimeLoRAAdapter,
    module_name: str,
) -> torch.Tensor:
    output_start = adapter.output_start
    output_end = adapter.output_end
    if weight.ndim == 2:
        if output_start is None:
            return weight
        return weight[output_start:output_end, :]

    if weight.ndim == 3 and weight.shape[0] == 1:
        if output_start is None:
            return weight
        return weight[:, :, output_start:output_end]

    raise ValueError(
        f"Runtime LoRA target '{module_name}' has unsupported weight shape {tuple(weight.shape)}"
    )


def _validate_lora_delta_shape(
    delta_shape: Tuple[int, ...],
    target_shape: Tuple[int, ...],
    adapter: _RuntimeLoRAAdapter,
    module_name: str,
) -> None:
    if delta_shape == target_shape:
        return
    if adapter.output_start is None:
        raise ValueError(
            f"Runtime LoRA delta shape mismatch for '{module_name}': "
            f"delta={delta_shape}, weight={target_shape}"
        )
    raise ValueError(
        f"Runtime LoRA fused span mismatch for '{module_name}': "
        f"delta={delta_shape}, span={target_shape}"
    )


def _find_safetensors_files(path: str) -> List[Path]:
    lora_path = Path(path)
    if lora_path.is_file() and lora_path.suffix == ".safetensors":
        return [lora_path]
    if lora_path.is_dir():
        return sorted(lora_path.glob("*.safetensors"))
    return []


def _load_lora_tensors(path: str) -> Dict[str, torch.Tensor]:
    sft_paths = _find_safetensors_files(path)
    if not sft_paths:
        raise ValueError(f"No safetensors files found at {path}")

    raw: Dict[str, torch.Tensor] = {}
    for sft_path in sft_paths:
        with safetensors.torch.safe_open(sft_path, framework="pt") as f:
            for key in f.keys():
                raw[key] = f.get_tensor(key)
    return raw


def _extract_lora_pairs(
    raw: Dict[str, torch.Tensor],
    config: RuntimeLoRAConfig,
) -> Tuple[List[_LoRAPair], int]:
    down_suffixes = (".lora_A.weight", ".lora_down.weight")
    up_suffixes = (".lora_B.weight", ".lora_up.weight")

    down_tensors: Dict[str, torch.Tensor] = {}
    up_tensors: Dict[str, torch.Tensor] = {}
    alpha: Dict[str, float] = {}
    for key, tensor in raw.items():
        base = _strip_suffix(key, down_suffixes)
        if base is not None:
            down_tensors[base] = tensor
            continue
        base = _strip_suffix(key, up_suffixes)
        if base is not None:
            up_tensors[base] = tensor
            continue
        if key.endswith(".alpha"):
            alpha[key[: -len(".alpha")]] = float(tensor.item())

    pairs: List[_LoRAPair] = []
    skipped_incomplete = 0
    for base_name, a in down_tensors.items():
        b = up_tensors.get(base_name)
        if b is None:
            skipped_incomplete += 1
            continue
        if a.ndim != 2 or b.ndim != 2:
            raise ValueError(
                f"Runtime LoRA tensors for '{base_name}' must be 2D, "
                f"got A={tuple(a.shape)}, B={tuple(b.shape)}"
            )
        rank = a.shape[0]
        if rank <= 0 or b.shape[1] != rank:
            raise ValueError(
                f"Runtime LoRA rank mismatch for '{base_name}': "
                f"A={tuple(a.shape)}, B={tuple(b.shape)}"
            )
        pair_scale = float(config.scale) * alpha.get(base_name, float(rank)) / float(rank)
        pairs.append(_LoRAPair(base_name, a, b, pair_scale))

    skipped_incomplete += len(set(up_tensors) - set(down_tensors))
    return pairs, skipped_incomplete


def _resolve_target_name(
    base_name: str,
    modules: Dict[str, nn.Module],
    *,
    strip_prefixes: Tuple[str, ...],
    key_map: Dict[str, str],
    fuse_qkv: bool,
) -> Tuple[Optional[str], Optional[str]]:
    for candidate in _candidate_names(base_name, strip_prefixes, key_map):
        if candidate in modules:
            return candidate, None
        if not fuse_qkv:
            continue
        for suffix in _QKV_SUFFIXES:
            if candidate.endswith(suffix):
                attn_prefix = candidate[: -len(suffix)]
                fused_name = f"{attn_prefix}.qkv_proj"
                if fused_name in modules:
                    return fused_name, suffix
    return None, None


def _candidate_names(
    base_name: str,
    strip_prefixes: Tuple[str, ...],
    key_map: Dict[str, str],
) -> Iterable[str]:
    names = [base_name, _strip_known_prefix(base_name, strip_prefixes)]
    seen = set()
    for name in names:
        for normalized in (_normalize_common_name(name), _apply_key_map(name, key_map)):
            normalized = _normalize_common_name(normalized)
            if normalized not in seen:
                seen.add(normalized)
                yield normalized


def _normalize_common_name(name: str) -> str:
    normalized = name
    for ff_prefix in (".ff.", ".audio_ff.", ".ffn."):
        if ff_prefix + "net.0.proj" in normalized:
            normalized = normalized.replace(ff_prefix + "net.0.proj", ff_prefix + "up_proj")
        elif ff_prefix + "net.2" in normalized:
            normalized = normalized.replace(ff_prefix + "net.2", ff_prefix + "down_proj")
    normalized = normalized.replace(".q_norm.", ".norm_q.")
    normalized = normalized.replace(".k_norm.", ".norm_k.")
    return normalized


def _strip_known_prefix(name: str, strip_prefixes: Tuple[str, ...]) -> str:
    for prefix in strip_prefixes:
        if prefix and name.startswith(prefix):
            return name[len(prefix) :]
    return name


def _apply_key_map(name: str, key_map: Dict[str, str]) -> str:
    for source, target in key_map.items():
        if name == source:
            return target
        source_prefix = f"{source}."
        if name.startswith(source_prefix):
            return f"{target}{name[len(source) :]}"
    return name


def _strip_suffix(key: str, suffixes: Tuple[str, ...]) -> Optional[str]:
    for suffix in suffixes:
        if key.endswith(suffix):
            return key[: -len(suffix)]
    return None


def _dedupe_prefixes(prefixes: Iterable[str]) -> Tuple[str, ...]:
    return tuple(sorted({prefix for prefix in prefixes if prefix}, key=len, reverse=True))


def _is_linear_like(module: nn.Module) -> bool:
    return (
        callable(getattr(module, "forward", None))
        and hasattr(module, "in_features")
        and hasattr(module, "out_features")
    )


def _tp_mode_value(tp_mode) -> Optional[str]:
    if tp_mode is None:
        return None
    return getattr(tp_mode, "value", str(tp_mode))


def _shard_adapter(
    base_layer: nn.Module,
    adapter: _RuntimeLoRAAdapter,
    module_name: str,
    device: torch.device,
) -> _RuntimeLoRAAdapter:
    tp_size = int(getattr(base_layer, "tp_size", 1) or 1)
    tp_mode = _tp_mode_value(getattr(base_layer, "tp_mode", None))
    if tp_size <= 1 or tp_mode is None:
        return _RuntimeLoRAAdapter(
            adapter.a.to(device=device),
            adapter.b.to(device=device),
            adapter.scale,
            adapter.output_start,
        )

    if isinstance(getattr(base_layer, "tp_sharding", None), dict):
        raise RuntimeError(
            f"Runtime LoRA target '{module_name}' uses fused tensor-parallel sharding; "
            "fused TP runtime LoRA is not supported yet."
        )
    if adapter.output_start is not None:
        raise RuntimeError(
            f"Runtime LoRA target '{module_name}' uses fused QKV output spans with TP; "
            "fused QKV TP runtime LoRA is not supported yet."
        )
    if not hasattr(base_layer, "load_shard"):
        raise RuntimeError(
            f"Runtime LoRA target '{module_name}' is tensor-parallel but has no load_shard()"
        )

    a = adapter.a
    b = adapter.b
    if tp_mode == "row":
        a = base_layer.load_shard(adapter.a, device=device)
        b = adapter.b.to(device=device)
    elif tp_mode == "column":
        a = adapter.a.to(device=device)
        b = base_layer.load_shard(adapter.b, device=device)
    else:
        raise RuntimeError(f"Unsupported tensor-parallel mode for Runtime LoRA: {tp_mode}")

    return _RuntimeLoRAAdapter(a, b, adapter.scale, adapter.output_start)
