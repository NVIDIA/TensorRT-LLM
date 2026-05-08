# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""AutoDeploy model registry — single source of truth for all AD entry points.

This registry maps model names to two layers of configs:

1. **AD defaults** (``ad_defaults``) — AD-internal knobs (transforms,
   compile_backend, etc.) shipped inside the package at
   ``_torch/auto_deploy/config/model_registry_internal/configs/``.
2. **User configs** (``user_configs``) — Common, user-facing knobs
   (max_batch_size, kv_cache_config, etc.) in
   ``examples/auto_deploy/model_registry/configs/``.

``build_and_run_ad.py`` and integration tests use
:func:`get_registry_yaml_extra` to get the combined list
``[*ad_defaults, *user_configs]`` which is passed as ``yaml_extra`` to AD's
``LlmArgs``. ``trtllm-serve`` uses :func:`get_ad_defaults` because the user's
``--config`` YAML is merged as init args with higher priority. AD defaults are
loaded first (lowest priority), user configs on top when included, and any
explicit CLI / init args win over both.
"""

from importlib.resources import files
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

_REGISTRY_PKG = files(__package__)
_REGISTRY_YAML = _REGISTRY_PKG / "models.yaml"
_AD_CONFIGS_DIR = Path(str(_REGISTRY_PKG / "configs"))

# User-facing configs live under examples/auto_deploy/model_registry/configs/.
# Resolve relative to the repo root.  This package lives at
# tensorrt_llm/_torch/auto_deploy/config/model_registry_internal/
# so the repo root is 4 parents up from the package directory.
_REPO_ROOT = Path(str(_REGISTRY_PKG)).parents[4]
_USER_CONFIGS_DIR = _REPO_ROOT / "examples" / "auto_deploy" / "model_registry" / "configs"


def _load_registry() -> dict:
    registry_text = _REGISTRY_YAML.read_text()
    return yaml.safe_load(registry_text)


def _find_entry(
    model_name: str,
    config_id: Optional[str] = None,
) -> dict:
    """Find a single registry entry for the given model name."""
    registry = _load_registry()
    matches = [entry for entry in registry.get("models", []) if entry.get("name") == model_name]

    if config_id is not None:
        matches = [entry for entry in matches if entry.get("config_id", "default") == config_id]
        if not matches:
            raise KeyError(
                f"Model '{model_name}' with config_id '{config_id}' not found "
                f"in the AutoDeploy model registry."
            )
    elif len(matches) > 1:
        default_matches = [
            entry for entry in matches if entry.get("config_id", "default") == "default"
        ]
        if len(default_matches) == 1:
            matches = default_matches
        else:
            available = sorted({entry.get("config_id", "default") for entry in matches})
            raise KeyError(
                f"Model '{model_name}' has multiple registry entries with "
                f"config_id values {available}. Provide config_id to select one."
            )

    if not matches:
        raise KeyError(f"Model '{model_name}' not found in the AutoDeploy model registry.")

    return matches[0]


def _resolve_paths(entry: dict) -> tuple[List[str], List[str]]:
    """Return (ad_paths, user_paths) for a registry entry."""
    ad_paths = [str(_AD_CONFIGS_DIR / cfg) for cfg in entry.get("ad_defaults", [])]
    user_paths = [str(_USER_CONFIGS_DIR / cfg) for cfg in entry.get("user_configs", [])]
    return ad_paths, user_paths


def _get_world_size(entry: dict) -> int:
    """Return the world_size for a registry entry."""
    return entry.get("world_size", 1)


def get_ad_defaults(
    model_name: str,
    config_id: Optional[str] = None,
) -> List[str]:
    """Return AD-internal default config paths for a model.

    Used by ``trtllm-serve`` where user-facing config values come from the
    explicit ``--config`` YAML / CLI options.

    Args:
        model_name: HuggingFace model id (e.g. ``deepseek-ai/DeepSeek-R1-0528``).
        config_id: Optional config variant selector when a model has multiple entries.

    Returns:
        List of absolute paths to AD-internal yaml config files.

    Raises:
        KeyError: If the model is not found or is ambiguous without ``config_id``.
    """
    entry = _find_entry(model_name, config_id)
    ad_paths, _ = _resolve_paths(entry)
    return ad_paths


def get_ad_defaults_with_world_size(
    model_name: str,
    config_id: Optional[str] = None,
) -> Tuple[List[str], int]:
    """Return AD-internal default config paths and registry world_size."""
    entry = _find_entry(model_name, config_id)
    ad_paths, _ = _resolve_paths(entry)
    return ad_paths, _get_world_size(entry)


def get_registry_yaml_extra(
    model_name: str,
    config_id: Optional[str] = None,
) -> List[str]:
    """Return the full yaml_extra paths: ``[*ad_defaults, *user_configs]``.

    Used by ``build_and_run_ad.py --use-registry`` and integration tests
    where the registry provides everything automatically.

    Args:
        model_name: HuggingFace model id (e.g. ``deepseek-ai/DeepSeek-R1-0528``).
        config_id: Optional config variant selector when a model has multiple entries.

    Returns:
        List of absolute paths to yaml config files (AD defaults first,
        user configs on top).

    Raises:
        KeyError: If the model is not found or is ambiguous without ``config_id``.
    """
    entry = _find_entry(model_name, config_id)
    ad_paths, user_paths = _resolve_paths(entry)
    return [*ad_paths, *user_paths]


def get_registry_yaml_extra_with_world_size(
    model_name: str,
    config_id: Optional[str] = None,
) -> Tuple[List[str], int]:
    """Return the full yaml_extra paths and registry world_size."""
    entry = _find_entry(model_name, config_id)
    ad_paths, user_paths = _resolve_paths(entry)
    return [*ad_paths, *user_paths], _get_world_size(entry)


def get_world_size(
    model_name: str,
    config_id: Optional[str] = None,
) -> int:
    """Return the world_size for a model (defaults to 1)."""
    entry = _find_entry(model_name, config_id)
    return _get_world_size(entry)


def list_registered_models() -> List[Dict]:
    """Return all models in the registry."""
    return _load_registry().get("models", [])
