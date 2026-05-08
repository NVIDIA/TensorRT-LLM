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

import json
import os
from collections.abc import Iterable, Mapping
from pathlib import Path

import pytest
import torch

_DEEPSEEK_V4_FLASH_ENV_VARS = ("DEEPSEEK_V4_FLASH_MODEL_DIR", "DEEPSEEK_V4_MODEL_DIR")


def deepseek_v4_flash_checkpoint_or_skip() -> Path:
    candidates: list[Path] = []
    for env_var in _DEEPSEEK_V4_FLASH_ENV_VARS:
        value = os.environ.get(env_var)
        if value:
            candidate = Path(value)
            candidates.append(candidate)
            candidates.append(candidate / "DeepSeek-V4-Flash")

    models_root = os.environ.get("LLM_MODELS_ROOT")
    if models_root:
        root = Path(models_root)
        candidates.extend(
            (
                root / "DeepSeek-V4-Flash",
                root / "DeepSeek-V4" / "DeepSeek-V4-Flash",
                root / "deepseek-ai" / "DeepSeek-V4-Flash",
                root / "deepseek-ai__DeepSeek-V4-Flash",
            )
        )

    for candidate in candidates:
        has_safetensors = (candidate / "model.safetensors.index.json").is_file() or any(
            candidate.glob("*.safetensors")
        )
        if (candidate / "config.json").is_file() and has_safetensors:
            return candidate

    pytest.skip(
        "DeepSeek-V4-Flash checkpoint not found; set DEEPSEEK_V4_FLASH_MODEL_DIR, "
        "DEEPSEEK_V4_MODEL_DIR, or LLM_MODELS_ROOT to enable this real-checkpoint test."
    )


def load_safetensors_tensors_or_skip(
    checkpoint_dir: Path,
    tensor_names: Iterable[str],
) -> dict[str, torch.Tensor]:
    safetensors = pytest.importorskip("safetensors")
    safe_open = safetensors.safe_open

    names = tuple(tensor_names)
    shard_to_names = _safetensors_shards_for_tensors(
        checkpoint_dir,
        names,
        safe_open=safe_open,
    )

    tensors: dict[str, torch.Tensor] = {}
    for shard_path, shard_names in shard_to_names.items():
        if not shard_path.is_file():
            pytest.skip(f"DeepSeek-V4-Flash safetensors shard not found: {shard_path}")
        try:
            with safe_open(shard_path, framework="pt", device="cpu") as shard:
                for name in shard_names:
                    tensors[name] = shard.get_tensor(name)
        except (KeyError, OSError, RuntimeError, ValueError) as error:
            pytest.skip(f"Could not load DeepSeek-V4-Flash safetensors shard {shard_path}: {error}")
    return tensors


def _safetensors_shards_for_tensors(
    checkpoint_dir: Path,
    tensor_names: tuple[str, ...],
    *,
    safe_open: object,
) -> dict[Path, list[str]]:
    index_path = checkpoint_dir / "model.safetensors.index.json"
    if index_path.is_file():
        return _indexed_safetensors_shards_or_skip(checkpoint_dir, index_path, tensor_names)
    return _scan_safetensors_shards_or_skip(checkpoint_dir, tensor_names, safe_open=safe_open)


def _indexed_safetensors_shards_or_skip(
    checkpoint_dir: Path,
    index_path: Path,
    tensor_names: tuple[str, ...],
) -> dict[Path, list[str]]:
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        pytest.skip(f"Could not read DeepSeek-V4-Flash safetensors index {index_path}: {error}")

    weight_map = index.get("weight_map")
    if not isinstance(weight_map, Mapping):
        pytest.skip(f"DeepSeek-V4-Flash safetensors index has no weight_map: {index_path}")

    shard_to_names: dict[Path, list[str]] = {}
    for name in tensor_names:
        shard_name = weight_map.get(name)
        if not isinstance(shard_name, str):
            pytest.skip(f"DeepSeek-V4-Flash checkpoint is missing tensor {name}.")
        shard_to_names.setdefault(checkpoint_dir / shard_name, []).append(name)
    return shard_to_names


def _scan_safetensors_shards_or_skip(
    checkpoint_dir: Path,
    tensor_names: tuple[str, ...],
    *,
    safe_open: object,
) -> dict[Path, list[str]]:
    shard_paths = sorted(checkpoint_dir.glob("*.safetensors"))
    if not shard_paths:
        pytest.skip(f"DeepSeek-V4-Flash checkpoint has no safetensors shards: {checkpoint_dir}")

    remaining = set(tensor_names)
    shard_to_names: dict[Path, list[str]] = {}
    for shard_path in shard_paths:
        try:
            with safe_open(shard_path, framework="pt", device="cpu") as shard:
                shard_keys = set(shard.keys())
        except (OSError, RuntimeError, ValueError) as error:
            pytest.skip(f"Could not inspect DeepSeek-V4-Flash shard {shard_path}: {error}")
        found = [name for name in tensor_names if name in remaining and name in shard_keys]
        if found:
            shard_to_names[shard_path] = found
            remaining.difference_update(found)

    if remaining:
        missing = ", ".join(sorted(remaining))
        pytest.skip(f"DeepSeek-V4-Flash checkpoint is missing tensors: {missing}.")
    return shard_to_names
