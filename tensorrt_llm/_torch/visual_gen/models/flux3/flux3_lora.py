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
"""FLUX.3 runtime LoRA: BFL-native/diffusers raw checkpoints."""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from tensorrt_llm.logger import logger
from tensorrt_llm.visual_gen.args import RuntimeLoRAConfig

from ...runtime_lora import (
    _QKV_SUFFIXES,
    RuntimeLoRAApplication,
    _candidate_names,
    _extract_lora_pairs,
    _fuse_lora_into_linear,
    _is_linear_like,
    _load_lora_tensors,
    _load_peft_lora_alpha,
    _LoRAPair,
    _prepare_lora_target_for_fusion,
    _resolve_target_name,
    _RuntimeLoRAAdapter,
)
from .transformer_flux3 import GATE_UP_ROLES, _remap_checkpoint_keys

_DEFAULT_STRIP_PREFIXES = ("transformer.", "model.transformer.", "dit.", "diffusion_model.")
_GATE_UP_SUFFIX = ".mlp_in"


def apply_flux3_runtime_lora(
    transformer: nn.Module,
    config: RuntimeLoRAConfig,
) -> RuntimeLoRAApplication:
    """Fuse a raw FLUX.3 LoRA checkpoint into the transformer.

    q/k/v, attn_out, and ff.down_proj resolve via the shared runtime_lora
    resolver; mlp_in (packed gate_up_proj) has no shared equivalent and is
    split below instead.
    """
    raw = _load_lora_tensors(config.path)
    peft_alpha = _load_peft_lora_alpha(config.path)

    strip_prefixes = tuple(config.strip_prefixes) + _DEFAULT_STRIP_PREFIXES
    stripped = {_strip_prefix(key, strip_prefixes): value for key, value in raw.items()}

    remap_fn = (
        _remap_diffusers_checkpoint_keys
        if _is_diffusers_checkpoint(stripped)
        else _remap_checkpoint_keys
    )
    remapped = _split_gate_up_lora(remap_fn(stripped, fuse_qkv=False))

    pairs, skipped_incomplete = _extract_lora_pairs(remapped, config, peft_alpha)
    modules = dict(transformer.named_modules())

    targets: Dict[str, List[_RuntimeLoRAAdapter]] = {}
    gate_up_targets: Dict[str, Dict[str, _LoRAPair]] = {}
    qkv_groups: Dict[str, Dict[str, _LoRAPair]] = {}
    skipped_non_targets = 0
    for pair in pairs:
        gate_up_target, role = None, None
        for candidate in _candidate_names(pair.name, (), config.key_map):
            gate_up_target, role = _resolve_gate_up_target(candidate, modules)
            if role is not None:
                break
        if role is not None:
            gate_up_targets.setdefault(gate_up_target, {})[role] = pair
            continue

        target_name, fused_suffix = _resolve_target_name(
            pair.name, modules, strip_prefixes=(), key_map=config.key_map, fuse_qkv=config.fuse_qkv
        )
        if target_name is None:
            skipped_non_targets += 1
            if config.strict:
                raise ValueError(
                    f"FLUX.3 runtime LoRA target '{pair.name}' not found in transformer"
                )
            continue
        if fused_suffix is None:
            targets.setdefault(target_name, []).append(
                _RuntimeLoRAAdapter(pair.a, pair.b, pair.scale)
            )
        else:
            qkv_groups.setdefault(target_name, {})[fused_suffix] = pair

    applied_modules: List[str] = []
    prepared_targets = []
    gate_up_deltas = []
    for module_name, adapters in targets.items():
        module = modules[module_name]
        if getattr(module, "_trtllm_runtime_lora_fused", False):
            raise ValueError(f"FLUX.3 runtime LoRA target '{module_name}' is already fused")
        if not _is_linear_like(module):
            if config.strict:
                raise ValueError(f"FLUX.3 runtime LoRA target '{module_name}' is not linear-like")
            skipped_non_targets += len(adapters)
            continue
        try:
            prepared = _prepare_lora_target_for_fusion(module, adapters, module_name)
        except ValueError:
            if config.strict:
                raise
            skipped_non_targets += len(adapters)
            continue
        prepared_targets.append(prepared)
        applied_modules.append(module_name)

    for target_name, parts in qkv_groups.items():
        if not all(suffix in parts for suffix in _QKV_SUFFIXES):
            skipped_incomplete += len(parts)
            if config.strict:
                raise ValueError(
                    f"FLUX.3 runtime LoRA incomplete fused-QKV group for '{target_name}'"
                )
            continue
        module = modules[target_name]
        if getattr(module, "_trtllm_runtime_lora_fused", False):
            raise ValueError(f"FLUX.3 runtime LoRA target '{target_name}' is already fused")
        fused_out = int(getattr(module, "out_features", 0))
        total_out = sum(parts[suffix].b.shape[0] for suffix in _QKV_SUFFIXES)
        if total_out != fused_out:
            raise ValueError(
                f"FLUX.3 runtime LoRA fused-QKV spans for '{target_name}' sum to {total_out} "
                f"but the layer output is {fused_out}. The adapter q/k/v split does "
                "not match the fused layer."
            )
        qkv_adapters: List[_RuntimeLoRAAdapter] = []
        output_start = 0
        for suffix in _QKV_SUFFIXES:
            pair = parts[suffix]
            qkv_adapters.append(
                _RuntimeLoRAAdapter(pair.a, pair.b, pair.scale, output_start=output_start)
            )
            output_start += pair.b.shape[0]
        try:
            prepared = _prepare_lora_target_for_fusion(module, qkv_adapters, target_name)
        except ValueError:
            if config.strict:
                raise
            skipped_non_targets += len(qkv_adapters)
            continue
        prepared_targets.append(prepared)
        applied_modules.append(target_name)

    for module_name, roles in gate_up_targets.items():
        if not all(role in roles for role in GATE_UP_ROLES):
            skipped_incomplete += len(roles)
            if config.strict:
                raise ValueError(
                    f"FLUX.3 runtime LoRA incomplete gate/up group for '{module_name}'"
                )
            continue
        module = modules[module_name]
        if getattr(module, "_trtllm_runtime_lora_fused", False):
            raise ValueError(f"FLUX.3 runtime LoRA target '{module_name}' is already fused")
        if not _is_linear_like(module):
            if config.strict:
                raise ValueError(f"FLUX.3 runtime LoRA target '{module_name}' is not linear-like")
            skipped_non_targets += len(GATE_UP_ROLES)
            continue
        try:
            delta = _prepare_gate_up_proj_lora(module, roles, module_name)
        except ValueError:
            if config.strict:
                raise
            skipped_non_targets += len(roles)
            continue
        gate_up_deltas.append((module, delta))
        applied_modules.append(module_name)

    if config.strict and not applied_modules:
        raise ValueError(
            f"No FLUX.3 runtime LoRA modules from {config.path!r} matched the transformer."
        )

    application = RuntimeLoRAApplication(
        path=config.path,
        applied_modules=tuple(applied_modules),
        skipped_non_targets=skipped_non_targets,
        skipped_incomplete=skipped_incomplete,
    )
    # Commit only after every target has been validated, including packed gate/up.
    for prepared in prepared_targets:
        _fuse_lora_into_linear(prepared.base_layer, prepared.deltas, prepared.module_name)
    with torch.no_grad():
        for module, delta in gate_up_deltas:
            module.weight.add_(delta)
            module._trtllm_runtime_lora_fused = True
    logger.info(
        f"FLUX.3 runtime LoRA fused from {application.path}: applied={len(application.applied_modules)}, "
        f"skipped_non_targets={application.skipped_non_targets}, skipped_incomplete={application.skipped_incomplete}"
    )
    return application


def _strip_prefix(key: str, prefixes: Tuple[str, ...]) -> str:
    for prefix in prefixes:
        if prefix and key.startswith(prefix):
            return key[len(prefix) :]
    return key


_GATE_UP_B_SUFFIXES = (f"{_GATE_UP_SUFFIX}.lora_B.weight", f"{_GATE_UP_SUFFIX}.lora_up.weight")
_GATE_UP_SHARED_SUFFIXES = (
    f"{_GATE_UP_SUFFIX}.lora_A.weight",
    f"{_GATE_UP_SUFFIX}.lora_down.weight",
    f"{_GATE_UP_SUFFIX}.alpha",
)


def _split_gate_up_lora(weights: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """mlp_in (fused gate+up) -> ff.gate_up_proj, splitting lora_B rows across GATE_UP_ROLES."""
    out: Dict[str, torch.Tensor] = {}
    n_roles = len(GATE_UP_ROLES)
    for key, value in weights.items():
        if key.endswith(_GATE_UP_B_SUFFIXES):
            if value.ndim != 2 or value.shape[0] % n_roles:
                raise ValueError(f"FLUX.3 runtime LoRA invalid packed gate/up shape for '{key}'")
            prefix, suffix = key.rsplit(_GATE_UP_SUFFIX, 1)
            chunk = value.shape[0] // n_roles
            for i, role in enumerate(GATE_UP_ROLES):
                out[f"{prefix}.ff.gate_up_proj.{role}{suffix}"] = value[i * chunk : (i + 1) * chunk]
        elif key.endswith(_GATE_UP_SHARED_SUFFIXES):
            prefix, suffix = key.rsplit(_GATE_UP_SUFFIX, 1)
            for role in GATE_UP_ROLES:
                out[f"{prefix}.ff.gate_up_proj.{role}{suffix}"] = value
        else:
            out[key] = value
    return out


def _resolve_gate_up_target(
    base_name: str, modules: Dict[str, nn.Module]
) -> Tuple[Optional[str], Optional[str]]:
    """Returns (target_module_name, role) for a packed gate_up_proj sub-pair
    (role is one of GATE_UP_ROLES), or (None, None) if base_name isn't one."""
    for role in GATE_UP_ROLES:
        leaf = f".gate_up_proj.{role}"
        if base_name.endswith(leaf):
            target_name = base_name[: -len(leaf)] + ".gate_up_proj"
            if target_name not in modules:
                return None, None
            return target_name, role
    return None, None


def _prepare_gate_up_proj_lora(
    module: nn.Module, roles: Dict[str, _LoRAPair], module_name: str
) -> torch.Tensor:
    """Prepare gate/up LoRA deltas via module.load_shard without mutating weights
    (a no-op at tp_size<=1, same primitive the checkpoint loader uses)."""
    weight = getattr(module, "weight", None)
    if not isinstance(weight, torch.Tensor) or weight.is_meta:
        raise ValueError(f"FLUX.3 runtime LoRA target '{module_name}' has no loaded weight tensor")
    if weight.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        raise ValueError(f"FLUX.3 runtime LoRA cannot fuse into weight dtype {weight.dtype}")

    with torch.no_grad():
        deltas = []
        for role in GATE_UP_ROLES:
            pair = roles[role]
            if pair.a.ndim != 2 or pair.b.ndim != 2 or pair.a.shape[0] != pair.b.shape[1]:
                raise ValueError(f"FLUX.3 runtime LoRA invalid gate/up rank for '{module_name}'")
            if pair.a.shape[1] != weight.shape[1]:
                raise ValueError(f"FLUX.3 runtime LoRA gate/up input mismatch for '{module_name}'")
            b_local = module.load_shard(pair.b, name=role, device=weight.device)
            a = pair.a.to(device=weight.device, dtype=weight.dtype)
            deltas.append(torch.matmul(b_local.to(weight.dtype), a).mul_(float(pair.scale)))
        delta = torch.cat(deltas, dim=0)
        if delta.shape != weight.shape:
            raise ValueError(
                f"FLUX.3 runtime LoRA gate/up delta shape mismatch for '{module_name}': "
                f"delta={tuple(delta.shape)}, weight={tuple(weight.shape)}"
            )
    return delta


_DIFFUSERS_MODE_BLOCK_RE = re.compile(r"^mode_blocks\.([a-z_]+)\.(\d+)\.")
_DIFFUSERS_TXT_BLOCK_RE = re.compile(r"^txt_mode_blocks\.(\d+)\.")
_DIFFUSERS_SINGLE_BLOCK_RE = re.compile(r"^single_transformer_blocks\.(\d+)\.")


def _is_diffusers_checkpoint(weights: dict) -> bool:
    """True if keys use diffusers' `.attn.to_q.` naming rather than BFL-native `.q_proj.`."""
    return any(".attn.to_q." in key for key in weights)


def _remap_diffusers_block_body(rest: str, q_name: str, k_name: str, v_name: str) -> str:
    # attn.norm_q.weight / attn.norm_k.weight are already diffusers-native in the
    # exact form `_remap_checkpoint_keys` targets
    rest = rest.replace("attn.to_q.", f"attn.{q_name}.")
    rest = rest.replace("attn.to_k.", f"attn.{k_name}.")
    rest = rest.replace("attn.to_v.", f"attn.{v_name}.")
    rest = rest.replace("attn.to_out.0.", "attn_out.")
    rest = rest.replace("ff.linear_in.", "mlp_in.")
    rest = rest.replace("ff.linear_out.", "ff.down_proj.")
    return rest


def _remap_diffusers_checkpoint_keys(weights: dict, fuse_qkv: bool = True) -> dict:
    """Remap diffusers-format FLUX.3 checkpoint keys onto the same target
    layout `_remap_checkpoint_keys` produces for BFL-native checkpoints."""
    q_name, k_name, v_name = (
        ("q_proj", "k_proj", "v_proj") if fuse_qkv else ("to_q", "to_k", "to_v")
    )

    remapped = {}
    for key, value in weights.items():
        new_key = key
        m = _DIFFUSERS_MODE_BLOCK_RE.match(key)
        if m:
            modality, index = m.groups()
            body = _remap_diffusers_block_body(key[m.end() :], q_name, k_name, v_name)
            new_key = f"content_mode_blocks.{modality}.{index}.{body}"
        elif m := _DIFFUSERS_TXT_BLOCK_RE.match(key):
            body = _remap_diffusers_block_body(key[m.end() :], q_name, k_name, v_name)
            new_key = f"txt_mode_blocks.{m.group(1)}.{body}"
        elif m := _DIFFUSERS_SINGLE_BLOCK_RE.match(key):
            body = _remap_diffusers_block_body(key[m.end() :], q_name, k_name, v_name)
            new_key = f"single_blocks.{m.group(1)}.{body}"
        elif key == "context_embedder.weight":
            new_key = "txt_in.weight"
        elif key.startswith("x_embedders."):
            new_key = key.replace("x_embedders.", "emb_in.", 1)
        elif key.startswith("timestep_embedder.linear_1."):
            new_key = key.replace("timestep_embedder.linear_1.", "time_in.in_layer.", 1)
        elif key.startswith("timestep_embedder.linear_2."):
            new_key = key.replace("timestep_embedder.linear_2.", "time_in.out_layer.", 1)
        elif key.startswith("vector_embedder.linear_1."):
            new_key = key.replace("vector_embedder.linear_1.", "vector_in.in_layer.", 1)
        elif key.startswith("vector_embedder.linear_2."):
            new_key = key.replace("vector_embedder.linear_2.", "vector_in.out_layer.", 1)
        elif key.startswith(("early_stream_modulations.", "single_stream_modulations.")) and (
            key.endswith(".linear.weight")
        ):
            new_key = key[: -len("linear.weight")] + "lin.weight"
        elif key.startswith("norm_out.") and (
            key.endswith(".linear.weight") or key.endswith(".linear.bias")
        ):
            modality = key.split(".")[1]
            suffix = "weight" if key.endswith(".weight") else "bias"
            new_key = f"final_layer.{modality}.adaLN_modulation.1.{suffix}"
            # diffusers orders the [shift, scale] chunks swapped relative to our layout.
            half = value.shape[0] // 2
            value = torch.cat([value[half:], value[:half]], dim=0)
        elif key.startswith("proj_out."):
            modality = key.split(".")[1]
            new_key = f"final_layer.{modality}.linear.weight"
        remapped[new_key] = value
    return remapped
